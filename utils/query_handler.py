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
            # First, search structured data
            log_event("PROCESS", "Searching structured data...")
            structured_data = search_structured(query)
            
            # Then, search unstructured data with structured results as context
            log_event("PROCESS", "Searching unstructured data with structured context...")
            unstructured_data = search_unstructured(query, structured_results=structured_data)
            
            # Format combined data for LLM
            formatted_data = self._format_combined_data(structured_data, unstructured_data)
            
            return formatted_data
        except Exception as e:
            log_event("ERROR", f"Error searching data: {e}")
            return "no data"

    def _format_combined_data(self, structured_data, unstructured_data):
        """
        Format structured and unstructured data for the final LLM response
        """
        try:
            formatted_sections = []
            
            # Format structured data section
            if structured_data and structured_data != "no data":
                formatted_sections.append("=== STRUCTURED DATA (Transaction Database) ===")
                
                if isinstance(structured_data, (list, dict)):
                    # Convert to string representation for the LLM
                    import json
                    try:
                        structured_str = json.dumps(structured_data, indent=2, default=str)
                        formatted_sections.append(structured_str)
                    except:
                        formatted_sections.append(str(structured_data))
                else:
                    formatted_sections.append(str(structured_data))
                
                log_event("SUCCESS", "Structured data formatted for LLM")
            else:
                formatted_sections.append("=== STRUCTURED DATA (Transaction Database) ===")
                formatted_sections.append("No relevant structured data found.")
                log_event("INFO", "No structured data available")
            
            # Format unstructured data section
            if unstructured_data and unstructured_data != "no data":
                formatted_sections.append("\n=== UNSTRUCTURED DATA (Documents & Policies) ===")
                
                if isinstance(unstructured_data, dict) and 'formatted_content' in unstructured_data:
                    formatted_sections.append(unstructured_data['formatted_content'])
                elif isinstance(unstructured_data, dict) and 'chunks' in unstructured_data:
                    # Format chunks manually if formatted_content is not available
                    chunks = unstructured_data['chunks']
                    for i, chunk in enumerate(chunks, 1):
                        chunk_text = f"\nDocument {i}:\n"
                        chunk_text += f"Source: {chunk.get('source', 'unknown')}\n"
                        chunk_text += f"Relevance Score: {chunk.get('cosine_similarity', 0):.3f}\n"
                        chunk_text += f"Content: {chunk.get('content', '')}\n"
                        formatted_sections.append(chunk_text)
                else:
                    formatted_sections.append(str(unstructured_data))
                
                log_event("SUCCESS", "Unstructured data formatted for LLM")
            else:
                formatted_sections.append("\n=== UNSTRUCTURED DATA (Documents & Policies) ===")
                formatted_sections.append("No relevant unstructured data found.")
                log_event("INFO", "No unstructured data available")
            
            # Combine all sections
            final_formatted_data = "\n".join(formatted_sections)
            
            log_event("SUCCESS", "Combined data formatted successfully")
            return final_formatted_data
            
        except Exception as e:
            log_event("ERROR", f"Error formatting combined data: {e}")
            return f"Structured Data: {structured_data}\n\nUnstructured Data: {unstructured_data}"

    def get_response(self, data, query):
        try:
            # Enhanced prompt with data source information
            full_prompt = f"""{FINAL_STEP_PROMPT}

USER QUERY: {query}

RETRIEVED DATA:
{data}
"""
            
            response = call_llm(prompt=full_prompt, query=query, temp=0.3)
            
            log_event("SUCCESS", "Enhanced response generated with source attribution")
            return response
            
        except Exception as e:
            log_event("ERROR", f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."
