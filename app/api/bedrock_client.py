import boto3
import json
import os
import numpy as np
from botocore.exceptions import ClientError
from typing import List, Union

# Initialize Bedrock client with credentials support
def get_bedrock_client():
    """
    Initialize Bedrock client with proper credential handling
    """
    # Option 1: Use environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if aws_access_key_id and aws_secret_access_key:
        # Use explicit credentials from environment variables
        client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'eu-north-1'),
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token  # Will be None if not set
        )
    else:
        # Fall back to default credential chain (AWS CLI, credentials file, etc.)
        client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'eu-north-1'),
        )
    
    return client

# Initialize the client
bedrock_client = get_bedrock_client()

def call_llm(query, prompt=None, temp=0.7):
    """
    Call AWS Bedrock Claude model with the given query and optional system prompt
    
    Args:
        query (str): User query/message
        prompt (str, optional): System prompt to provide context
        temp (float): Temperature for response generation (0.0-1.0)
    
    Returns:
        str: Model response text
    """
    try:
        # Prepare messages (only user messages, no system role in messages array)
        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": query}]
            }
        ]
        
        # Prepare request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": temp,
            "messages": messages
        }
        
        # Add system prompt as top-level parameter if provided
        if prompt:
            body["system"] = prompt
        
        # Make API call using the inference profile ID
        response = bedrock_client.invoke_model(
            modelId="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Updated to use inference profile ID
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract text from response
        if 'content' in response_body and response_body['content']:
            return response_body['content'][0]['text']
        else:
            raise Exception("No content in response")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}")
    
    except Exception as e:
        raise Exception(f"Error calling LLM: {str(e)}")

def generate_embeddings(texts: Union[str, List[str]], model_id: str = "amazon.titan-embed-text-v2:0") -> np.ndarray:
    """
    Generate embeddings using AWS Bedrock embedding models
    
    Args:
        texts (str or List[str]): Text(s) to embed
        model_id (str): Bedrock embedding model ID
                       Options:
                       - "amazon.titan-embed-text-v2:0" (default, 1024 dimensions)
                       - "amazon.titan-embed-text-v1" (1536 dimensions)
                       - "cohere.embed-english-v3" (1024 dimensions)
                       - "cohere.embed-multilingual-v3" (1024 dimensions)
    
    Returns:
        np.ndarray: Embeddings array with shape (num_texts, embedding_dim)
    """
    try:
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Validate input
        if not texts or len(texts) == 0:
            raise ValueError("Input texts cannot be empty")
        
        embeddings_list = []
        
        # Process each text (some models have batch limits)
        for text in texts:
            if not text or not text.strip():
                # Handle empty strings by creating zero embeddings
                if "titan-embed-text-v2" in model_id:
                    embeddings_list.append(np.zeros(1024))
                elif "titan-embed-text-v1" in model_id:
                    embeddings_list.append(np.zeros(1536))
                elif "cohere.embed" in model_id:
                    embeddings_list.append(np.zeros(1024))
                else:
                    embeddings_list.append(np.zeros(1024))  # Default
                continue
            
            # Prepare request body based on model type
            if "titan-embed" in model_id:
                body = {
                    "inputText": text.strip()
                }
            elif "cohere.embed" in model_id:
                body = {
                    "texts": [text.strip()],
                    "input_type": "search_document"  # or "search_query" for queries
                }
            else:
                raise ValueError(f"Unsupported embedding model: {model_id}")
            
            # Make API call
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract embeddings based on model type
            if "titan-embed" in model_id:
                if 'embedding' in response_body:
                    embedding = response_body['embedding']
                else:
                    raise Exception("No embedding in Titan response")
            elif "cohere.embed" in model_id:
                if 'embeddings' in response_body and len(response_body['embeddings']) > 0:
                    embedding = response_body['embeddings'][0]
                else:
                    raise Exception("No embeddings in Cohere response")
            
            embeddings_list.append(embedding)
        
        # Convert to numpy array and ensure float32 type for FAISS compatibility
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        return embeddings_array
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        raise Exception(f"AWS Bedrock Embedding Error ({error_code}): {error_message}")
    
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

def generate_query_embedding(query: str, model_id: str = "amazon.titan-embed-text-v2:0") -> np.ndarray:
    """
    Generate embedding specifically for search queries
    
    Args:
        query (str): Search query text
        model_id (str): Bedrock embedding model ID
    
    Returns:
        np.ndarray: Query embedding array
    """
    try:
        # For Cohere models, use search_query input type for better query embeddings
        if "cohere.embed" in model_id:
            body = {
                "texts": [query.strip()],
                "input_type": "search_query"
            }
            
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'embeddings' in response_body and len(response_body['embeddings']) > 0:
                return np.array(response_body['embeddings'][0], dtype=np.float32).reshape(1, -1)
            else:
                raise Exception("No embeddings in Cohere response")
        else:
            # For Titan models, use the regular generate_embeddings function
            return generate_embeddings(query, model_id)
            
    except Exception as e:
        raise Exception(f"Error generating query embedding: {str(e)}")

def get_embedding_dimension(model_id: str = "amazon.titan-embed-text-v2:0") -> int:
    """
    Get the embedding dimension for a given model
    
    Args:
        model_id (str): Bedrock embedding model ID
    
    Returns:
        int: Embedding dimension
    """
    model_dimensions = {
        "amazon.titan-embed-text-v2:0": 1024,
        "amazon.titan-embed-text-v1": 1536,
        "cohere.embed-english-v3": 1024,
        "cohere.embed-multilingual-v3": 1024
    }
    
    return model_dimensions.get(model_id, 1024)  # Default to 1024
