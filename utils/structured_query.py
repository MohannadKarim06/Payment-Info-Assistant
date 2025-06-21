import os
import sys
import faiss
import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from rapidfuzz import fuzz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api.bedrock_client import call_llm
from utils.logger import log_event
from utils.config import PANDAS_QUERY_PROMPT

class StructuredDataSearcher:
    def __init__(self, 
                 index_path="data/column_index.faiss",
                 metadata_path="data/column_metadata.pkl",
                 data_path="data\structured_data.xlsx",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        
        # Initialize lightweight transformers model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            log_event("SUCCESS", f"Loaded lightweight model: {model_name}")
        except Exception as e:
            log_event("ERROR", f"Failed to load model: {e}")
            self.tokenizer = None
            self.model = None

        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.column_metadata = pickle.load(f)
            self.data = pd.read_excel(data_path)
            log_event("SUCCESS", "Structured data components loaded successfully")
        except Exception as e:
            log_event("ERROR", f"Failed to load structured data components: {e}")
            self.index = None
            self.column_metadata = []
            self.data = pd.DataFrame()

    def generate_embeddings(self, texts):
        """
        Generate embeddings using lightweight transformers
        """
        if self.tokenizer is None or self.model is None:
            log_event("ERROR", "Model not loaded")
            return None
        
        try:
            # Handle single string input
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            )
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state.mean(dim=1)
            
            return embeddings.numpy().astype("float32")
            
        except Exception as e:
            log_event("ERROR", f"Error generating embeddings: {e}")
            return None

    def search_relevant_columns(self, query, top_k=15, threshold=0.6):

        if self.index is None:
            log_event("ERROR", "FAISS index not loaded")
            return []
        
        try:
            query_embedding = self.generate_embeddings(query)
            if query_embedding is None:
                return []
            
            scores, indices = self.index.search(query_embedding, top_k)
            
            relevant_columns = []
            for score, idx in zip(scores[0], indices[0]):
                if score <= (2.0 - threshold):  
                    relevant_columns.append({
                        'metadata': self.column_metadata[idx],
                        'distance': score,
                        'similarity': 1 / (1 + score)  
                    })
            
            if not relevant_columns:
                log_event("INFO", "No relevant columns found above threshold")
                return []
            
            log_event("SUCCESS", f"Found {len(relevant_columns)} relevant columns")
            log_event("INFO", f"Column matches: {[col['metadata']['column_name'] for col in relevant_columns]}")
            return relevant_columns
            
        except Exception as e:
            log_event("ERROR", f"Error searching columns: {e}")
            return []

    def generate_pandas_query(self, query, relevant_columns):

        if not relevant_columns:
            return None
        

        columns_info = "\n".join([
            f"Column: {col['metadata']['column_name']}\n"
            f"column_name_details: {col['metadata']['column_name_details']}\n"
            f"Description: {col['metadata']['column_description']}\n"
            f"Data Type: {col['metadata']['data_type']}\n"
            f"Can be Empty: {col['metadata']['can_be_empty']}\n"
            f"Possible Values: {col['metadata']['possible_values']}\n"
            f"Sample Value: {col['metadata']['sample_value']}\n"
            for col in relevant_columns
        ])
        
        
        try:
            full_prompt = f"{PANDAS_QUERY_PROMPT}\n\nAvailable Columns:\n{columns_info}"
            pandas_query = call_llm(query=query, prompt=full_prompt, temp=0.1)

            pandas_query = pandas_query.strip()
            if pandas_query.startswith("```python"):
                pandas_query = pandas_query.replace("```python", "").replace("```", "").strip()
            elif pandas_query.startswith("```"):
                pandas_query = pandas_query.replace("```", "").strip()
            
            log_event("SUCCESS", f"Generated pandas query: {pandas_query}")
            return pandas_query
            
        except Exception as e:
            log_event("ERROR", f"Error generating pandas query: {e}")
            return None

    def execute_pandas_query(self, pandas_query):
        """
        Execute the generated pandas query safely
        """
        if not pandas_query or self.data.empty:
            return None
        
        try:
            # Create a safe execution environment
            safe_globals = {
                'df': self.data,
                'pd': pd,
                'np': np
            }
            
            # Execute the query
            result = eval(pandas_query, safe_globals)
            
            # Convert result to appropriate format
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    log_event("INFO", "Query executed but returned no data")
                    return None
                result_dict = result.to_dict('records')
                log_event("SUCCESS", f"Query executed successfully, returned {len(result_dict)} records")
                return result_dict
            elif isinstance(result, pd.Series):
                if result.empty:
                    log_event("INFO", "Query executed but returned no data")
                    return None
                result_dict = result.to_dict()
                log_event("SUCCESS", f"Query executed successfully, returned series data")
                return result_dict
            else:
                log_event("SUCCESS", f"Query executed successfully, returned: {result}")
                return str(result)
                
        except Exception as e:
            log_event("ERROR", f"Error executing pandas query: {e}")
            return None

def search_structured(query):
    """
    Main function to search structured data
    """
    searcher = StructuredDataSearcher()
    
    # Search for relevant columns
    relevant_columns = searcher.search_relevant_columns(query)
    
    if not relevant_columns:
        log_event("INFO", "No relevant structured data found")
        return "no data"
    
    # Generate pandas query
    pandas_query = searcher.generate_pandas_query(query, relevant_columns)
    
    if not pandas_query:
        log_event("ERROR", "Failed to generate pandas query")
        return "no data"
    
    # Execute query
    results = searcher.execute_pandas_query(pandas_query)
    
    if results is None:
        log_event("INFO", "Query executed but no results found")
        return "no data"
    
    return results