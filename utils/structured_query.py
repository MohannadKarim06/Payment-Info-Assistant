import os
import sys
import faiss
import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from rapidfuzz import fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

from app.api.bedrock_client import call_llm
from utils.logger import log_event
from utils.config import PANDAS_QUERY_PROMPT

# GLOBAL SINGLETON INSTANCE - LOADED ONCE
_searcher_instance = None

class StructuredDataSearcher:
    def __init__(self,
                 index_path="data/column_index.faiss",
                 metadata_path="data/column_metadata.pkl",
                 data_path="data/structured_data.xlsx",
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
            log_event("INFO", "Loading FAISS index and metadata...")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.column_metadata = pickle.load(f)

            log_event("INFO", f"Loading Excel file: {data_path} (this may take a moment...)")
            # CRITICAL: Load Excel only once during initialization
            self.data = pd.read_excel(data_path)

            log_event("SUCCESS", "Structured data components loaded successfully")
            log_event("INFO", f"Loaded {len(self.column_metadata)} columns, {self.data.shape[0]} rows")

            # Validate metadata against actual DataFrame columns
            self._validate_metadata()

        except Exception as e:
            log_event("ERROR", f"Failed to load structured data components: {e}")
            self.index = None
            self.column_metadata = []
            self.data = pd.DataFrame()

    def _validate_metadata(self):
        """Validate that metadata columns actually exist in DataFrame"""
        if not self.column_metadata or self.data.empty:
            return

        actual_columns = set(self.data.columns)
        metadata_columns = set(col['column_name'] for col in self.column_metadata if 'column_name' in col)

        missing_in_data = metadata_columns - actual_columns
        missing_in_metadata = actual_columns - metadata_columns

        if missing_in_data:
            log_event("WARNING", f"Columns in metadata but missing from DataFrame: {list(missing_in_data)[:5]}")

        if missing_in_metadata:
            log_event("INFO", f"Columns in DataFrame but missing from metadata: {len(missing_in_metadata)} columns")

        overlap = metadata_columns & actual_columns
        log_event("INFO", f"Column overlap: {len(overlap)}/{len(metadata_columns)} metadata columns exist in DataFrame")

    def generate_embeddings(self, texts):
        """Generate normalized embeddings for cosine similarity"""
        if self.tokenizer is None or self.model is None:
            log_event("ERROR", "Model not loaded")
            return None

        try:
            if isinstance(texts, str):
                texts = [texts]

            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state.mean(dim=1)

            # Normalize embeddings for cosine similarity
            embeddings = embeddings.numpy().astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
            
            return normalized_embeddings

        except Exception as e:
            log_event("ERROR", f"Error generating embeddings: {e}")
            return None

    def split_complex_query(self, query):
        """
        Split complex queries into meaningful parts for better retrieval
        """
        log_event("INFO", f"Splitting query: '{query}'")
        
        # Clean and normalize the query
        query = query.strip().lower()
        
        # Initialize query parts list
        query_parts = []
        
        # 1. Split by common conjunctions and separators
        conjunction_patterns = [
            r'\s+and\s+',
            r'\s+or\s+', 
            r'\s*,\s*',
            r'\s*;\s*',
            r'\s+also\s+',
            r'\s+plus\s+',
            r'\s+with\s+',
            r'\s+including\s+'
        ]
        
        parts = [query]  # Start with full query
        
        for pattern in conjunction_patterns:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(pattern, part))
            parts = new_parts
        
        # Clean and filter parts
        for part in parts:
            part = part.strip()
            if len(part) > 3:  # Ignore very short parts
                query_parts.append(part)
        
        # 2. Extract key phrases using simple NLP patterns
        key_phrase_patterns = [
            r'(?:show|find|get|display|list|count|sum|total|average|max|min|calculate)\s+(.+?)(?:\s+(?:by|for|in|from|where|group|order)|\s*$)',
            r'(?:number of|count of|total of|sum of|average of)\s+(.+?)(?:\s+(?:by|for|in|from|where|group|order)|\s*$)',
            r'(?:by|for|in|from|per)\s+(.+?)(?:\s+(?:and|or|by|for|in|from|where|group|order)|\s*$)',
            r'(?:where|when|which)\s+(.+?)(?:\s+(?:and|or|by|for|in|from|where|group|order)|\s*$)'
        ]
        
        for pattern in key_phrase_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if len(match) > 3 and match not in query_parts:
                    query_parts.append(match)
        
        # 3. Extract important keywords (nouns, adjectives)
        # Simple keyword extraction - remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', query)
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Group keywords into meaningful phrases
        if len(keywords) > 1:
            # Create 2-3 word combinations
            for i in range(len(keywords) - 1):
                phrase = ' '.join(keywords[i:i+2])
                if phrase not in query_parts:
                    query_parts.append(phrase)
                
                if i < len(keywords) - 2:
                    phrase3 = ' '.join(keywords[i:i+3])
                    if phrase3 not in query_parts:
                        query_parts.append(phrase3)
        
        # Always include the original full query
        if query not in query_parts:
            query_parts.insert(0, query)
        
        # Remove duplicates while preserving order
        unique_parts = []
        seen = set()
        for part in query_parts:
            if part not in seen:
                unique_parts.append(part)
                seen.add(part)
        
        log_event("INFO", f"Query split into {len(unique_parts)} parts: {unique_parts}")
        return unique_parts

    def search_with_cosine_similarity(self, query_parts, top_k_per_part=10):
        """
        Search using cosine similarity (FAISS Inner Product on normalized vectors)
        """
        if self.index is None:
            log_event("ERROR", "FAISS index not loaded")
            return []

        try:
            all_results = defaultdict(lambda: {'best_score': -1, 'metadata': None, 'query_matches': []})
            
            for query_part in query_parts:
                log_event("INFO", f"Searching for part: '{query_part}'")
                
                query_embedding = self.generate_embeddings(query_part)
                if query_embedding is None:
                    continue

                # Use inner product for cosine similarity (since embeddings are normalized)
                scores, indices = self.index.search(query_embedding, top_k_per_part)
                
                log_event("INFO", f"Part '{query_part}' - Top 3 scores: {scores[0][:3]}")
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.column_metadata):
                        column_name = self.column_metadata[idx].get('column_name', 'Unknown')
                        
                        # Update best score for this column
                        if score > all_results[column_name]['best_score']:
                            all_results[column_name]['best_score'] = score
                            all_results[column_name]['metadata'] = self.column_metadata[idx]
                        
                        # Track which query parts matched this column
                        all_results[column_name]['query_matches'].append({
                            'query_part': query_part,
                            'score': score
                        })
            
            # Convert to list and sort by best score
            final_results = []
            for column_name, data in all_results.items():
                if data['metadata'] is not None:
                    # Calculate aggregate score (max score + bonus for multiple matches)
                    num_matches = len(data['query_matches'])
                    aggregate_score = data['best_score'] + (num_matches - 1) * 0.1  # Bonus for multiple matches
                    
                    final_results.append({
                        'metadata': data['metadata'],
                        'cosine_similarity': data['best_score'],
                        'aggregate_score': aggregate_score,
                        'num_query_matches': num_matches,
                        'query_matches': data['query_matches']
                    })
                    
                    log_event("INFO", f"Column '{column_name}': best_score={data['best_score']:.3f}, matches={num_matches}, aggregate={aggregate_score:.3f}")
            
            # Sort by aggregate score (descending)
            final_results.sort(key=lambda x: x['aggregate_score'], reverse=True)
            
            log_event("SUCCESS", f"Found {len(final_results)} unique columns across all query parts")
            return final_results
            
        except Exception as e:
            log_event("ERROR", f"Error in cosine similarity search: {e}")
            return []

    def search_relevant_columns(self, query, top_k_final=15):
        """
        Enhanced search that splits complex queries and uses cosine similarity
        """
        # Split query into meaningful parts
        query_parts = self.split_complex_query(query)
        
        # Search with cosine similarity
        relevant_columns = self.search_with_cosine_similarity(query_parts, top_k_per_part=8)
        
        # Take top results
        final_results = relevant_columns[:top_k_final]
        
        # Add similarity field for backward compatibility
        for result in final_results:
            result['similarity'] = result['cosine_similarity']
            result['distance'] = 1.0 - result['cosine_similarity']  # Convert for compatibility
        
        log_event("SUCCESS", f"Final selection: {len(final_results)} columns for LLM")
        
        # Log top results
        for i, result in enumerate(final_results[:10]):
            col_name = result['metadata']['column_name']
            sim_score = result['cosine_similarity']
            matches = result['num_query_matches']
            log_event("INFO", f"Top {i+1}: '{col_name}' - similarity={sim_score:.3f}, matches={matches}")
        
        return final_results

    def fallback_fuzzy_search(self, query):
        """Simple fuzzy fallback - only as last resort"""
        if not self.column_metadata:
            return []

        log_event("INFO", "Using fuzzy matching as fallback")

        fuzzy_matches = []
        query_lower = query.lower()

        for col_meta in self.column_metadata:
            column_name = col_meta.get('column_name', '').lower()
            column_desc = col_meta.get('column_description', '').lower()

            # Check for meaningful keyword overlap
            name_ratio = fuzz.partial_ratio(query_lower, column_name)
            desc_ratio = fuzz.partial_ratio(query_lower, column_desc)
            max_ratio = max(name_ratio, desc_ratio)

            if max_ratio > 70:  # High threshold for fallback
                fuzzy_matches.append({
                    'metadata': col_meta,
                    'distance': 1.0 - (max_ratio / 100.0),
                    'similarity': max_ratio / 100.0,
                    'cosine_similarity': max_ratio / 100.0
                })

        return sorted(fuzzy_matches, key=lambda x: x['similarity'], reverse=True)[:3]

    def generate_pandas_query(self, query, relevant_columns):
        """Enhanced query generation with multi-step support"""
        if not relevant_columns:
            return None

        # Check for hardcoded patterns first (for demo reliability)
        hardcoded_result = self._check_hardcoded_patterns(query)
        if hardcoded_result:
            return hardcoded_result

        # Validate columns exist in actual DataFrame
        valid_columns = []
        invalid_columns = []

        for col in relevant_columns[:15]:  # Limit to top 15 most relevant columns
            col_name = col['metadata']['column_name']
            if col_name in self.data.columns:
                valid_columns.append(col)
                log_event("INFO", f"✓ VALIDATED: Column '{col_name}' exists in DataFrame (similarity: {col['cosine_similarity']:.3f})")
            else:
                invalid_columns.append(col_name)
                log_event("WARNING", f"✗ INVALID: Column '{col_name}' found in metadata but missing from DataFrame")

        if not valid_columns:
            log_event("ERROR", "No valid columns found in DataFrame after validation")
            log_event("ERROR", f"Invalid columns attempted: {invalid_columns}")

            # Try to find similar column names as a fallback
            available_columns = list(self.data.columns)
            log_event("INFO", f"Available DataFrame columns (first 10): {available_columns[:10]}")

            # Look for columns that might be similar to the invalid ones
            potential_matches = []
            for invalid_col in invalid_columns:
                for available_col in available_columns:
                    if fuzz.partial_ratio(invalid_col.lower(), available_col.lower()) > 80:
                        potential_matches.append((invalid_col, available_col))

            if potential_matches:
                log_event("INFO", f"Potential column matches found: {potential_matches}")

            return None

        # Enhanced column info with query matching details
        columns_info = "\n".join([
            f"Column: {col['metadata']['column_name']}\n"
            f"Description: {col['metadata'].get('column_description', 'No description')}\n"
            f"Data Type: {col['metadata'].get('data_type', 'Unknown')}\n"
            f"Sample Value: {col['metadata'].get('sample_value', 'No sample')}\n"
            f"Relevance Score: {col['cosine_similarity']:.3f}\n"
            f"Query Matches: {col.get('num_query_matches', 1)}\n"
            for col in valid_columns
        ])

        try:
            enhanced_prompt = f"""
{PANDAS_QUERY_PROMPT}

Available Columns (sorted by relevance using cosine similarity, VALIDATED to exist in DataFrame):
{columns_info}

User Query: {query}

IMPORTANT: 
- The columns are sorted by relevance using cosine similarity matching.
- Focus on columns with higher relevance scores and multiple query matches.
- For complex calculations requiring multiple steps, you can use multiple lines separated by newlines.
- Example multi-step format:
  total_payments = len(df)
  successful_payments = len(df[df['payment_status'] == 'success'])
  success_rate = (successful_payments / total_payments) * 100

MULTI-STEP CALCULATION EXAMPLES:
- Success Factor: Calculate total count, then successful count, then percentage
- Conversion Rate: Calculate total leads, then conversions, then rate
- Average by Group: Group data, then calculate averages

Generate a pandas query that comprehensively addresses the user's natural language request.
"""

            pandas_query = call_llm(query=query, prompt=enhanced_prompt, temp=0.1)

            pandas_query = pandas_query.strip()
            if pandas_query.startswith("```python"):
                pandas_query = pandas_query.replace("```python", "").replace("```", "").strip()
            elif pandas_query.startswith("```"):
                pandas_query = pandas_query.replace("```", "").strip()

            log_event("SUCCESS", f"Generated pandas query: {pandas_query}")
            log_event("INFO", f"Query uses {len(valid_columns)} validated columns")
            return pandas_query

        except Exception as e:
            log_event("ERROR", f"Error generating pandas query: {e}")
            return None

    def _check_hardcoded_patterns(self, query):
        """Check for hardcoded patterns that need special handling"""
        query_lower = query.lower()
        
        # Success factor/rate patterns
        success_patterns = ['success factor', 'success rate', 'conversion rate', 'completion rate']
        if any(pattern in query_lower for pattern in success_patterns):
            return self._generate_success_factor_query()
        
        # Add more patterns as needed
        return None

    def _generate_success_factor_query(self):
        """Generate a multi-step query for success factor calculation"""
        # Find relevant status columns
        status_columns = [col for col in self.data.columns 
                         if any(term in col.lower() for term in ['status', 'success', 'result', 'outcome', 'state'])]
        
        if not status_columns:
            return None
        
        status_col = status_columns[0]  # Use the first matching column
        
        # Generate multi-step query
        query = f"""total_count = len(df)
success_count = len(df[df['{status_col}'].astype(str).str.lower().str.contains('success|complete|paid|approved|done', na=False)])
success_factor = (success_count / total_count * 100) if total_count > 0 else 0
{{'total_records': total_count, 'successful_records': success_count, 'success_factor_percentage': round(success_factor, 2)}}"""
        
        log_event("INFO", f"Generated hardcoded success factor query using column: {status_col}")
        return query

    def execute_pandas_query(self, pandas_query):
        """Execute pandas query with support for multi-step calculations"""
        if not pandas_query or self.data.empty:
            return None

        try:
            log_event("INFO", f"Executing pandas query: {pandas_query}")

            # Create a safe execution environment
            safe_globals = {
                'df': self.data,
                'pd': pd,
                'np': np
            }

            # Check if this is a multi-step query
            if self._is_multi_step_query(pandas_query):
                return self._execute_multi_step_query(pandas_query, safe_globals)
            else:
                # Single-step execution (original logic)
                result = eval(pandas_query, safe_globals)
                return self._format_result(result)

        except Exception as e:
            log_event("ERROR", f"Error executing pandas query: {e}")
            log_event("ERROR", f"Failed query was: {pandas_query}")

            # Additional debugging: check if it's a column access error
            if "KeyError" in str(e) or "not found" in str(e).lower():
                log_event("ERROR", "This appears to be a column access error")
                log_event("INFO", f"Available columns in DataFrame: {list(self.data.columns)[:10]}...")

            return None

    def _is_multi_step_query(self, query):
        """Detect if query requires multiple steps"""
        multi_step_indicators = [
            '\n',  # Multiple lines
            ';',   # Multiple statements  
            'total_', 'successful_', 'temp_', 'step_', 'count_', 'rate_', 'factor_',  # Common intermediate variables
            '=.*\n',  # Assignment followed by newline
        ]
        
        return any(indicator in query for indicator in multi_step_indicators)

    def _execute_multi_step_query(self, pandas_query, safe_globals):
        """Execute multi-step pandas queries safely"""
        try:
            log_event("INFO", "Executing multi-step query")
            
            # Split query into individual statements
            statements = self._split_query_statements(pandas_query)
            
            result = None
            # Execute each statement
            for i, statement in enumerate(statements):
                statement = statement.strip()
                if not statement:
                    continue
                    
                log_event("INFO", f"Executing step {i+1}: {statement}")
                
                # Execute statement
                result = eval(statement, safe_globals)
                
                # Store intermediate results in safe_globals for next steps
                if '=' in statement and not statement.startswith('{'):
                    var_name = statement.split('=')[0].strip()
                    safe_globals[var_name] = result
                    log_event("INFO", f"Stored intermediate result: {var_name} = {result}")
            
            # Return the final result
            return self._format_result(result)
            
        except Exception as e:
            log_event("ERROR", f"Error in multi-step execution: {e}")
            return None

    def _split_query_statements(self, query):
        """Split multi-step query into individual statements"""
        # Handle different separators
        if '\n' in query:
            statements = query.split('\n')
        elif ';' in query:
            statements = query.split(';')
        else:
            statements = [query]
        
        return [stmt.strip() for stmt in statements if stmt.strip()]

    def _format_result(self, result):
        """Format the result consistently"""
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
            return result


def get_searcher_instance():
    """
    SINGLETON PATTERN: Get or create the searcher instance
    This ensures Excel is loaded only once across all requests
    """
    global _searcher_instance

    if _searcher_instance is None:
        log_event("INFO", "Creating new StructuredDataSearcher instance (first time)")
        _searcher_instance = StructuredDataSearcher()
    else:
        log_event("INFO", "Reusing existing StructuredDataSearcher instance (Excel already loaded)")

    return _searcher_instance

def search_structured(query):
    """
    Main function with enhanced query splitting and cosine similarity search
    Uses singleton pattern to avoid reloading Excel file
    """
    try:
        log_event("PROCESS", f"Processing complex query: '{query}'")

        # Get the singleton instance (Excel loaded only once)
        searcher = get_searcher_instance()

        # Enhanced search with query splitting and cosine similarity
        relevant_columns = searcher.search_relevant_columns(query)

        # FALLBACK: Only use fuzzy matching if embedding search completely fails
        if not relevant_columns:
            log_event("WARNING", "Embedding search failed, trying fuzzy fallback")
            relevant_columns = searcher.fallback_fuzzy_search(query)

            if not relevant_columns:
                log_event("ERROR", "Both embedding and fuzzy search failed")
                return "no data"

        log_event("SUCCESS", "Relevant columns found.")

        # Generate pandas query
        pandas_query = searcher.generate_pandas_query(query, relevant_columns)

        if not pandas_query:
            log_event("ERROR", "Failed to generate valid pandas query")
            return "no data"

        # Execute query
        log_event("PROCESS", "Executing query...")
        results = searcher.execute_pandas_query(pandas_query)

        if results is None:
            log_event("INFO", "Query executed but no results found")
            return "no data"

        log_event("SUCCESS", "Response generated successfully")
        return results

    except Exception as e:
        log_event("ERROR", f"Unexpected error in search_structured: {e}")
        return "no data"
