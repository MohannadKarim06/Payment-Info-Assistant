import os
import sys
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_event

class UnstructuredRAGSearcher:
    def __init__(self, 
                 index_path="data/unstructured_index.faiss",
                 chunks_path="data/unstructured_chunks.pkl",
                 model_name="all-MiniLM-L6-v2"):
        
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index and chunks
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            log_event("SUCCESS", "Unstructured RAG components loaded successfully")
        except Exception as e:
            log_event("ERROR", f"Failed to load unstructured RAG components: {e}")
            self.index = None
            self.chunks = []

    def search_similar_chunks(self, query, top_k=3, threshold=0.7):
        """
        Search for similar chunks using FAISS similarity search
        """
        if self.index is None:
            log_event("ERROR", "FAISS index not loaded")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query]).astype('float32')
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Filter by threshold and return relevant chunks
            relevant_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                # Convert L2 distance to cosine similarity approximation
                # Lower L2 distance = higher similarity
                similarity = 1 / (1 + score)  # Simple transformation
                
                if similarity >= threshold:
                    chunk_data = self.chunks[idx]
                    relevant_chunks.append({
                        'content': chunk_data['content'],
                        'metadata': chunk_data.get('metadata', {}),
                        'similarity': similarity,
                        'source': chunk_data.get('source', 'unknown')
                    })
            
            if not relevant_chunks:
                log_event("INFO", f"No relevant chunks found above threshold {threshold}")
                return []
            
            log_event("SUCCESS", f"Found {len(relevant_chunks)} relevant chunks")
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
            formatted_chunk += f"Relevance Score: {chunk['similarity']:.2f}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n" + "="*50 + "\n".join(formatted_chunks)

def search_unstructured(query, top_k=3, threshold=0.7):
    """
    Main function to search unstructured data using RAG
    """
    try:
        searcher = UnstructuredRAGSearcher()
        
        # Search for similar chunks
        relevant_chunks = searcher.search_similar_chunks(query, top_k, threshold)
        
        if not relevant_chunks:
            log_event("INFO", "No relevant unstructured data found")
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

def create_unstructured_index(documents, output_dir="data"):
    """
    Utility function to create FAISS index from documents
    This should be run separately to prepare the index
    """
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Prepare chunks
        chunks = []
        embeddings = []
        
        for doc in documents:
            # Assuming documents have 'content', 'source', and optional 'metadata'
            content = doc['content']
            source = doc.get('source', 'unknown')
            metadata = doc.get('metadata', {})
            
            # Simple chunking (can be improved)
            chunk_size = 1000
            overlap = 100
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 50:  # Skip very short chunks
                    chunks.append({
                        'content': chunk_text,
                        'source': source,
                        'metadata': metadata
                    })
                    
                    # Create embedding
                    embedding = model.encode(chunk_text)
                    embeddings.append(embedding)
        
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        # Save index and chunks
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(index, f"{output_dir}/unstructured_index.faiss")
        
        with open(f"{output_dir}/unstructured_chunks.pkl", 'wb') as f:
            pickle.dump(chunks, f)
        
        log_event("SUCCESS", f"Created unstructured index with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        log_event("ERROR", f"Error creating unstructured index: {e}")
        return False