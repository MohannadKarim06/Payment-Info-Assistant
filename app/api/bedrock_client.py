import boto3
import json
import os
from botocore.exceptions import ClientError, BotoCoreError
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_event

class BedrockClient:
    def __init__(self, region_name='us-east-1'):
        """
        Initialize the Bedrock client
        
        Args:
            region_name (str): AWS region name
        """
        self.region_name = region_name
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet
        
        try:
            # Initialize the Bedrock Runtime client
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
            log_event("SUCCESS", f"Bedrock client initialized for region: {region_name}")
        except Exception as e:
            log_event("ERROR", f"Failed to initialize Bedrock client: {str(e)}")
            self.bedrock_client = None

    def call_claude(self, prompt, query="", temperature=0.3, max_tokens=2000):
        """
        Call Claude model via Bedrock
        
        Args:
            prompt (str): The system prompt or instruction
            query (str): The user query/question
            temperature (float): Temperature for response generation (0.0-1.0)
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Claude's response
        """
        if not self.bedrock_client:
            log_event("ERROR", "Bedrock client not initialized")
            return "Error: Bedrock client not available"
        
        try:
            # Prepare the message
            if query:
                user_message = f"{prompt}\n\nUser Query: {query}"
            else:
                user_message = prompt
            
            # Prepare the request body for Claude 3
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            log_event("PROCESS", f"Calling Claude model: {self.model_id}")
            
            # Call the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract the text content
            if 'content' in response_body and len(response_body['content']) > 0:
                result = response_body['content'][0]['text']
                log_event("SUCCESS", f"Claude response received (length: {len(result)})")
                return result
            else:
                log_event("ERROR", "No content in Claude response")
                return "Error: No content in response"
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            log_event("ERROR", f"AWS ClientError: {error_code} - {error_message}")
            
            if error_code == 'ValidationException':
                return "Error: Invalid request parameters"
            elif error_code == 'ResourceNotFoundException':
                return "Error: Model not found"
            elif error_code == 'AccessDeniedException':
                return "Error: Access denied to Bedrock service"
            elif error_code == 'ThrottlingException':
                return "Error: Request throttled, please try again later"
            else:
                return f"Error: {error_message}"
                
        except BotoCoreError as e:
            log_event("ERROR", f"BotoCoreError: {str(e)}")
            return "Error: Connection issue with AWS service"
            
        except json.JSONDecodeError as e:
            log_event("ERROR", f"JSON decode error: {str(e)}")
            return "Error: Invalid response format"
            
        except Exception as e:
            log_event("ERROR", f"Unexpected error calling Claude: {str(e)}")
            return f"Error: {str(e)}"

    def test_connection(self):
        """
        Test the connection to Bedrock
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_response = self.call_claude(
                prompt="Please respond with 'Connection successful' if you can read this.",
                temperature=0.1,
                max_tokens=50
            )
            
            if "Connection successful" in test_response:
                log_event("SUCCESS", "Bedrock connection test passed")
                return True
            else:
                log_event("ERROR", f"Bedrock connection test failed: {test_response}")
                return False
                
        except Exception as e:
            log_event("ERROR", f"Bedrock connection test error: {str(e)}")
            return False

# Global instance
_bedrock_client = None

def get_bedrock_client():
    """
    Get singleton instance of BedrockClient
    """
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    return _bedrock_client

def call_llm(query="", prompt="", temp=0.3, max_tokens=2000):
    """
    Main function to call the LLM
    
    Args:
        query (str): User query
        prompt (str): System prompt
        temp (float): Temperature
        max_tokens (int): Maximum tokens
        
    Returns:
        str: LLM response
    """
    try:
        client = get_bedrock_client()
        response = client.call_claude(
            prompt=prompt,
            query=query,
            temperature=temp,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        log_event("ERROR", f"Error in call_llm: {str(e)}")
        return f"Error: {str(e)}"

def test_bedrock_connection():
    """
    Test Bedrock connection
    
    Returns:
        bool: Connection status
    """
    client = get_bedrock_client()
    return client.test_connection()

# Alternative model configurations
MODEL_CONFIGS = {
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-instant": "anthropic.claude-instant-v1",
    "claude-v2": "anthropic.claude-v2"
}

def set_model(model_name):
    """
    Set the model to use
    
    Args:
        model_name (str): Model name from MODEL_CONFIGS
    """
    global _bedrock_client
    if model_name in MODEL_CONFIGS:
        if _bedrock_client:
            _bedrock_client.model_id = MODEL_CONFIGS[model_name]
            log_event("SUCCESS", f"Model set to: {model_name}")
        else:
            log_event("ERROR", "Bedrock client not initialized")
    else:
        log_event("ERROR", f"Unknown model: {model_name}")
        log_event("INFO", f"Available models: {list(MODEL_CONFIGS.keys())}")

def get_available_models():
    """
    Get list of available models
    
    Returns:
        list: Available model names
    """
    return list(MODEL_CONFIGS.keys())