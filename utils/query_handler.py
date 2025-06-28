import os, sys

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

from app.api.bedrock_client import call_llm
from utils.logger import log_event
from utils.config import CLASS_INTENT_PROMPT, FINAL_STEP_PROMPT
from utils.structured_query import search_structured
from utils.unstructured_rag import search_unstructured

class PipelineReturn(Exception):
    def __init__(self, value):
        self.value = value
        super().__init__(f"Pipeline early return with value: {value}")

class QueryHandler:
    def classify_intent(self, query):
        try:
            response = call_llm(query=query, prompt=CLASS_INTENT_PROMPT, temp=0.1)
            
            if response == "0":
                result = "I only answer questions about payment transactions info. What is your question?"
                raise PipelineReturn(value=result)
            elif response == "1":
                return "payment_related"
            else:
                log_event("ERROR", f"Unexpected intent classification response: {response}")
                result = "I only answer questions about payment transactions info. What is your question?"
                raise PipelineReturn(value=result)
        except PipelineReturn:
            raise
        except Exception as e:
            log_event("ERROR", f"Error in intent classification: {e}")
            result = "I encountered an error while processing your question. Please try again."
            raise PipelineReturn(value=result)

    def search_data(self, query):
        try:
            structured_data = search_structured(query)
            unstructured_data = search_unstructured(query)
            
            data = f"Structured Data:\n{structured_data}\n\nUnstructured Data:\n{unstructured_data}"
            return data
        except Exception as e:
            log_event("ERROR", f"Error searching data: {e}")
            return "no data"

    def get_response(self, data, query):
        try:
            full_prompt = f"{FINAL_STEP_PROMPT}\n\nRetrieved data:\n\n{data}"
            response = call_llm(prompt=full_prompt, query=query, temp=0.5)
            return response
        except Exception as e:
            log_event("ERROR", f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."
