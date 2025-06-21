import boto3
import json
import os
from botocore.exceptions import ClientError

# Initialize Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

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
        # Prepare messages
        messages = []
        
        # Add system message if prompt is provided
        if prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": prompt}]
            })
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query}]
        })
        
        # Prepare request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": temp,
            "messages": messages
        }
        
        # Make API call
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
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