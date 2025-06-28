import os
import sys
import faiss
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

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

    def search_similar_chunks(self, query, top_k=5):
        """
        Search for similar chunks using FAISS similarity search with adaptive thresholding
        """
        if self.index is None:
            log_event("ERROR", "FAISS index not loaded")
            return []

        try:
            # Encode query
            query_embedding = self.generate_embeddings(query)
            if query_embedding is None:
                return []

            # Search FAISS index
            scores, indices = self.index.search(query_embedding, top_k)

            # Log raw results for debugging
            log_event("INFO", f"Raw search scores (top 5): {scores[0]}")

            # FIXED: Use adaptive thresholding similar to structured search
            score_list = scores[0].tolist()
            
            # Ensure we get at least some results for natural language queries
            min_results = min(3, len(score_list))  # At least top 3 or all available
            
            # Use relative thresholding - take results within reasonable range of the best score
            best_score = score_list[0]
            relative_threshold = best_score + 3.0  # Allow scores within 3 points of the best
            
            # Use percentile-based thresholding
            if len(score_list) >= 3:
                percentile_threshold = np.percentile(score_list, 60)  # More lenient than structured
            else:
                percentile_threshold = float('inf')

            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                # Multi-criteria acceptance
                accept_result = (
                    i < min_results or  # Top N results
                    score <= relative_threshold or  # Within range of best
                    score <= percentile_threshold  # Better than percentile
                )

                if accept_result and idx < len(self.chunks):
                    similarity = 1 / (1 + score)  # Convert distance to similarity
                    chunk_data = self.chunks[idx]
                    
                    relevant_chunks.append({
                        'content': chunk_data['content'],
                        'metadata': chunk_data.get('metadata', {}),
                        'similarity': similarity,
                        'distance': score,
                        'source': chunk_data.get('source', 'unknown')
                    })

                    # Log first 100 chars of chunk content for debugging
                    content_preview = chunk_data['content'][:100] + "..." if len(chunk_data['content']) > 100 else chunk_data['content']
                    log_event("INFO", f"✓ ACCEPTED: Chunk {i+1} (distance: {score:.3f}, similarity: {similarity:.3f})")
                    log_event("INFO", f"  Content preview: '{content_preview}'")
                else:
                    if idx < len(self.chunks):
                        content_preview = self.chunks[idx]['content'][:50] + "..." if len(self.chunks[idx]['content']) > 50 else self.chunks[idx]['content']
                        log_event("INFO", f"✗ REJECTED: Chunk {i+1} (distance: {score:.3f})")
                        log_event("INFO", f"  Content preview: '{content_preview}'")

            if not relevant_chunks:
                log_event("WARNING", "No chunks accepted by adaptive thresholding - this indicates a serious issue")
                log_event("INFO", f"Score range: {min(score_list):.3f} to {max(score_list):.3f}")
                
                # Emergency fallback - take the top result regardless
                if len(scores[0]) > 0 and indices[0][0] < len(self.chunks):
                    log_event("WARNING", "Using emergency fallback - taking top result")
                    best_idx = indices[0][0]
                    best_score = scores[0][0]
                    chunk_data = self.chunks[best_idx]
                    
                    relevant_chunks.append({
                        'content': chunk_data['content'],
                        'metadata': chunk_data.get('metadata', {}),
                        'similarity': 1 / (1 + best_score),
                        'distance': best_score,
                        'source': chunk_data.get('source', 'unknown')
                    })
                    log_event("INFO", f"Emergency fallback: Selected chunk with distance {best_score:.3f}")

            log_event("SUCCESS", f"Found {len(relevant_chunks)} relevant chunks")
            
            # Sort by similarity (highest first)
            relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            
            return relevant_chunks

        except Exception as e:
            log_event("ERROR", f"Error searching unstructured data: {e}")
            return []

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
            if chunk['metadata']:
                for key, value in chunk['metadata'].items():
                    formatted_chunk += f"{key}: {value}\n"
            formatted_chunk += f"Content: {chunk['content']}\n"
            formatted_chunk += f"Relevance Score: {chunk['similarity']:.3f}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n" + "="*50 + "\n".join(formatted_chunks)

def search_unstructured(query, top_k=5):
    """
    Main function to search unstructured data using RAG with adaptive thresholding
    """
    try:
        searcher = UnstructuredRAGSearcher()

        # Search for similar chunks with adaptive thresholding
        relevant_chunks = searcher.search_similar_chunks(query, top_k)

        if not relevant_chunks:
            log_event("ERROR", "Failed to find any relevant unstructured data even with adaptive thresholding")
            return "no data"

        # Format chunks for response
        formatted_result = searcher.format_chunks_for_response(relevant_chunks)

        log_event("SUCCESS", f"Retrieved {len(relevant_chunks)} unstructured chunks")

        return {
            'chunks': relevant_chunks,
            'formatted_content': formatted_result,
            'total_chunks': len(relevant_chunks)
        }

    except Exception as e:
        log_event("ERROR", f"Error in unstructured search: {e}")
        return "no data"
