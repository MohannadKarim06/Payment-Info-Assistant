import os, sys

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

from utils.query_handler import QueryHandler, PipelineReturn
from utils.synonym_resolver import simple_synonym_resolver as synonym_resolver
from utils.logger import log_event

query_handler = QueryHandler()

def query_pipeline(query):

    try:
        log_event("PROCESS", "Classifying intent...")
        query_type = query_handler.classify_intent(query)
        log_event("PROCESS", "Intent is classified.")
    except PipelineReturn as pr:
        return pr.value
    except Exception as e:
        log_event("ERROR",  f"An error occured while classifying intent: {e}")
        raise e

    try:
        log_event("PROCESS", "Resolving synonyms...")
        query = synonym_resolver(query)
        log_event("SUCCESS", "Synonyms are resolved.")
    except Exception as e:
        log_event("ERROR",  f"An error occured while resolving synonyms: {e}")
        raise e
    
    try:
        log_event("PROCESS", "Searching data...")
        data = query_handler.search_data(query=query)
        log_event("SUCCESS", f"Data is found.")
    except Exception as e:
        log_event("ERROR",  f"An error occured while searching data: {e}")
        raise e
    
    try:
        log_event("PROCESS", "Getting response...")
        response = query_handler.get_response(data=data, query=query)
        log_event("SUCCESS", "Response is generated successfully")
        return response
    except Exception as e:
        log_event("ERROR",  f"An error occured while getting response: {e}")
        raise e
    





        


