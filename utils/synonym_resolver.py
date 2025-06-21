import json
import os
import sys
import re
from rapidfuzz import fuzz, process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_event

class SynonymResolver:
    def __init__(self, synonyms_path="data/synonyms.json"):
        self.synonyms_path = synonyms_path
        self.synonyms = {}
        self.load_synonyms()
    
    def load_synonyms(self):
        """
        Load synonyms from JSON file
        """
        try:
            with open(self.synonyms_path, 'r', encoding='utf-8') as f:
                self.synonyms = json.load(f)
            log_event("SUCCESS", f"Loaded {len(self.synonyms)} synonyms from {self.synonyms_path}")
        except FileNotFoundError:
            log_event("ERROR", f"Synonyms file not found: {self.synonyms_path}")
            self.synonyms = {}
        except json.JSONDecodeError as e:
            log_event("ERROR", f"Invalid JSON in synonyms file: {e}")
            self.synonyms = {}
        except Exception as e:
            log_event("ERROR", f"Error loading synonyms: {e}")
            self.synonyms = {}
    
    def fuzzy_match_keys(self, text, threshold=85):
        """
        Perform fuzzy matching against synonym keys
        """
        if not self.synonyms:
            return []
        
        matches = []
        words = text.split()
        
        # Try to match individual words and phrases
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                phrase = ' '.join(words[i:j])
                
                # Find best match among keys
                best_match = process.extractOne(
                    phrase, 
                    list(self.synonyms.keys()), 
                    scorer=fuzz.ratio
                )
                
                if best_match and best_match[1] >= threshold:
                    matches.append({
                        'original': phrase,
                        'matched_key': best_match[0],
                        'replacement': self.synonyms[best_match[0]],
                        'score': best_match[1],
                        'type': 'key_match'
                    })
        
        return matches
    
    def fuzzy_match_values(self, text, threshold=85):
        """
        Perform fuzzy matching against synonym values
        """
        if not self.synonyms:
            return []
        
        matches = []
        words = text.split()
        unique_values = list(set(self.synonyms.values()))
        
        # Try to match individual words and phrases
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                phrase = ' '.join(words[i:j])
                
                # Find best match among values
                best_match = process.extractOne(
                    phrase, 
                    unique_values, 
                    scorer=fuzz.ratio
                )
                
                if best_match and best_match[1] >= threshold:
                    matches.append({
                        'original': phrase,
                        'matched_value': best_match[0],
                        'replacement': best_match[0],  # Keep the same value
                        'score': best_match[1],
                        'type': 'value_match'
                    })
        
        return matches
    
    def resolve_synonyms(self, query, key_threshold=85, value_threshold=85):
        """
        Resolve synonyms in the query using fuzzy matching
        """
        if not query or not self.synonyms:
            return query
        
        original_query = query
        resolved_query = query
        replacements = []
        
        # First, try matching against keys
        key_matches = self.fuzzy_match_keys(query, key_threshold)
        
        # Then, try matching against values
        value_matches = self.fuzzy_match_values(query, value_threshold)
        
        # Combine and sort matches by score (highest first)
        all_matches = key_matches + value_matches
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply replacements, avoiding overlaps
        processed_positions = set()
        
        for match in all_matches:
            original_phrase = match['original']
            replacement = match['replacement']
            
            # Find all occurrences of the original phrase
            pattern = re.escape(original_phrase)
            matches_iter = re.finditer(pattern, resolved_query, re.IGNORECASE)
            
            for match_obj in matches_iter:
                start, end = match_obj.span()
                
                # Check if this position overlaps with already processed positions
                if not any(pos in range(start, end) for pos in processed_positions):
                    # Replace the phrase
                    resolved_query = (
                        resolved_query[:start] + 
                        replacement + 
                        resolved_query[end:]
                    )
                    
                    # Mark positions as processed
                    processed_positions.update(range(start, end))
                    
                    replacements.append({
                        'original': original_phrase,
                        'replacement': replacement,
                        'type': match['type'],
                        'score': match['score']
                    })
                    
                    # Break after first replacement to avoid position conflicts
                    break
        
        if replacements:
            log_event("SUCCESS", f"Applied {len(replacements)} synonym replacements")
            for replacement in replacements:
                log_event("INFO", f"Replaced '{replacement['original']}' with '{replacement['replacement']}' (score: {replacement['score']})")
        else:
            log_event("INFO", "No synonym replacements found")
        
        return resolved_query
    
    def add_synonym(self, key, value):
        """
        Add a new synonym to the dictionary
        """
        self.synonyms[key] = value
        try:
            with open(self.synonyms_path, 'w', encoding='utf-8') as f:
                json.dump(self.synonyms, f, indent=2, ensure_ascii=False)
            log_event("SUCCESS", f"Added synonym: {key} -> {value}")
        except Exception as e:
            log_event("ERROR", f"Error saving synonym: {e}")
    
    def get_all_synonyms(self):
        """
        Get all loaded synonyms
        """
        return self.synonyms.copy()

# Global instance
_synonym_resolver_instance = None

def get_synonym_resolver():
    """
    Get singleton instance of SynonymResolver
    """
    global _synonym_resolver_instance
    if _synonym_resolver_instance is None:
        _synonym_resolver_instance = SynonymResolver()
    return _synonym_resolver_instance

def synonym_resolver(query, key_threshold=85, value_threshold=85):
    """
    Main function to resolve synonyms in a query
    """
    try:
        resolver = get_synonym_resolver()
        resolved_query = resolver.resolve_synonyms(query, key_threshold, value_threshold)
        return resolved_query
    except Exception as e:
        log_event("ERROR", f"Error in synonym resolution: {e}")
        return query  # Return original query if resolution fails

