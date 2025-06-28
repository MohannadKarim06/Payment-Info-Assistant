import os
import sys
import faiss
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict
import re

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

from utils.logger import log_event

class UnstructuredRAGSearcher:
    def __init__(self,
                 index_path="data/unstructured_index.faiss",
                 chunks_path="data/unstructured_chunks.pkl",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.index_path = index_path
        self.chunks_path = chunks_path

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

        # Load FAISS index and chunks
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            log_event("SUCCESS", "Unstructured RAG components loaded successfully")
            log_event("INFO", f"Loaded {len(self.chunks)} chunks")
        except Exception as e:
            log_event("ERROR", f"Failed to load unstructured RAG components: {e}")
            self.index = None
            self.chunks = []

    def generate_embeddings(self, texts):
        """
        Generate NORMALIZED embeddings for cosine similarity (same as structured search)
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

            # Normalize embeddings for cosine similarity (CRITICAL for FAISS Inner Product)
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
        (Same logic as structured search for consistency)
        """
        log_event("INFO", f"Splitting unstructured query: '{query}'")
        
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
        
        log_event("INFO", f"Unstructured query split into {len(unique_parts)} parts: {unique_parts}")
        return unique_parts

    def search_with_cosine_similarity(self, query_parts, structured_results=None, top_k_per_part=8):
        """
        Search using cosine similarity (FAISS Inner Product on normalized vectors)
        Enhanced to also search for content related to structured results
        """
        if self.index is None:
            log_event("ERROR", "FAISS index not loaded")
            return []

        try:
            all_results = defaultdict(lambda: {'best_score': -1, 'chunk_data': None, 'query_matches': []})
            
            # Combine original query parts with structured results context
            all_query_parts = list(query_parts)
            
            # Add structured results as additional query context
            if structured_results and structured_results != "no data":
                log_event("INFO", "Adding structured results context to unstructured search")
                
                # Extract key terms from structured results
                if isinstance(structured_results, list):
                    for item in structured_results[:5]:  # Limit to top 5 results
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, str) and len(value) > 3:
                                    # Add key-value pairs as search terms
                                    search_term = f"{key} {value}"
                                    if search_term not in all_query_parts:
                                        all_query_parts.append(search_term)
                                        log_event("INFO", f"Added structured context: '{search_term}'")
                elif isinstance(structured_results, dict):
                    for key, value in structured_results.items():
                        if isinstance(value, str) and len(value) > 3:
                            search_term = f"{key} {value}"
                            if search_term not in all_query_parts:
                                all_query_parts.append(search_term)
                                log_event("INFO", f"Added structured context: '{search_term}'")
            
            # Search for each query part
            for query_part in all_query_parts:
                log_event("INFO", f"Searching unstructured for part: '{query_part}'")
                
                query_embedding = self.generate_embeddings(query_part)
                if query_embedding is None:
                    continue

                # Use inner product for cosine similarity (since embeddings are normalized)
                scores, indices = self.index.search(query_embedding, top_k_per_part)
                
                log_event("INFO", f"Unstructured part '{query_part}' - Top 3 scores: {scores[0][:3]}")
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunks):
                        chunk_id = f"chunk_{idx}"
                        
                        # Update best score for this chunk
                        if score > all_results[chunk_id]['best_score']:
                            all_results[chunk_id]['best_score'] = score
                            all_results[chunk_id]['chunk_data'] = self.chunks[idx]
                        
                        # Track which query parts matched this chunk
                        all_results[chunk_id]['query_matches'].append({
                            'query_part': query_part,
                            'score': score
                        })
            
            # Convert to list and sort by best score
            final_results = []
            for chunk_id, data in all_results.items():
                if data['chunk_data'] is not None:
                    # Calculate aggregate score (max score + bonus for multiple matches)
                    num_matches = len(data['query_matches'])
                    aggregate_score = data['best_score'] + (num_matches - 1) * 0.1  # Bonus for multiple matches
                    
                    final_results.append({
                        'content': data['chunk_data']['content'],
                        'metadata': data['chunk_data'].get('metadata', {}),
                        'source': data['chunk_data'].get('source', 'unknown'),
                        'cosine_similarity': data['best_score'],
                        'aggregate_score': aggregate_score,
                        'num_query_matches': num_matches,
                        'query_matches': data['query_matches'],
                        'similarity': data['best_score']  # For backward compatibility
                    })
                    
                    # Log chunk info
                    content_preview = data['chunk_data']['content'][:100] + "..." if len(data['chunk_data']['content']) > 100 else data['chunk_data']['content']
                    log_event("INFO", f"Chunk: best_score={data['best_score']:.3f}, matches={num_matches}, aggregate={aggregate_score:.3f}")
                    log_event("INFO", f"Content preview: '{content_preview}'")
            
            # Sort by aggregate score (descending)
            final_results.sort(key=lambda x: x['aggregate_score'], reverse=True)
            
            log_event("SUCCESS", f"Found {len(final_results)} unique chunks across all query parts")
            return final_results
            
        except Exception as e:
            log_event("ERROR", f"Error in unstructured cosine similarity search: {e}")
            return []

    def search_similar_chunks(self, query, structured_results=None, top_k=8):
        """
        Enhanced search that splits complex queries and uses cosine similarity
        Also considers structured results for better context
        """
        # Split query into meaningful parts
        query_parts = self.split_complex_query(query)
        
        # Search with cosine similarity, including structured results context
        relevant_chunks = self.search_with_cosine_similarity(query_parts, structured_results, top_k_per_part=6)
        
        # Take top results
        final_results = relevant_chunks[:top_k]
        
        log_event("SUCCESS", f"Final unstructured selection: {len(final_results)} chunks")
        
        # Log top results
        for i, result in enumerate(final_results[:5]):
            content_preview = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
            sim_score = result['cosine_similarity']
            matches = result['num_query_matches']
            log_event("INFO", f"Top {i+1}: similarity={sim_score:.3f}, matches={matches}")
            log_event("INFO", f"  Content: '{content_preview}'")
        
        return final_results

    def format_chunks_for_response(self, chunks):
        """
        Format retrieved chunks for the final LLM response
        """
        if not chunks:
            return "no data"

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunk = f"Document {i}:\n"
            formatted_chunk += f"Source: {chunk['source']}\n"
            formatted_chunk += f"Relevance Score: {chunk['cosine_similarity']:.3f}\n"
            formatted_chunk += f"Query Matches: {chunk['num_query_matches']}\n"
            if chunk['metadata']:
                for key, value in chunk['metadata'].items():
                    formatted_chunk += f"{key}: {value}\n"
            formatted_chunk += f"Content: {chunk['content']}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n" + "="*50 + "\n".join(formatted_chunks)

def search_unstructured(query, structured_results=None, top_k=8):
    """
    Main function to search unstructured data using cosine similarity
    Enhanced to consider structured results for better context
    """
    try:
        log_event("PROCESS", f"Processing unstructured query with cosine similarity: '{query}'")
        
        searcher = UnstructuredRAGSearcher()

        # Search for similar chunks with cosine similarity and structured context
        relevant_chunks = searcher.search_similar_chunks(query, structured_results, top_k)

        if not relevant_chunks:
            log_event("ERROR", "Failed to find any relevant unstructured data with cosine similarity")
            return "no data"

        # Format chunks for response
        formatted_result = searcher.format_chunks_for_response(relevant_chunks)

        log_event("SUCCESS", f"Retrieved {len(relevant_chunks)} unstructured chunks using cosine similarity")

        return {
            'chunks': relevant_chunks,
            'formatted_content': formatted_result,
            'total_chunks': len(relevant_chunks)
        }

    except Exception as e:
        log_event("ERROR", f"Error in unstructured cosine similarity search: {e}")
        return "no data"
