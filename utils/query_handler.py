import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api.bedrock_client import call_llm
from utils.logger import log_event
from utils.config import CLASS_INTENT_PROMPT, FINAL_STEP_PROMPT
from utils.structured_query import search_structured
from utils.unstructured_rag import search_unstructured

class QueryHandler:

    def classify_intent(self, query):

        response = call_llm(query=query, prompt=CLASS_INTENT_PROMPT, temp=0.1)

        if response == "0":
            result = "I only answer questions about payment transactions info, What is your question?"

        elif response == "1":
            return None

        else:
            result = "I only answer questions about payment transactions info, What is your question?"

    

    def search_data(self, query):

        structured_data = search_structured(query)

        unstructured_data = search_unstructured(query)

        data = f"Structured Data:\n{structured_data}\n\nUnstructured Data:\n{unstructured_data}"

        return data


    def get_response(data, query):

        full_prompt = f"{FINAL_STEP_PROMPT}\n\nRetrieved data:\n\n{data}"

        response = call_llm(prompt=full_prompt, query=query)

        return response
