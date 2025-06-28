import os, sys

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

import json
import re

from utils.logger import log_event

class SimpleSynonymResolver:
    def __init__(self, synonyms_path="data/synonyms.json"):
        self.synonyms_path = synonyms_path
        self.synonyms = {}
        self.normalized_synonyms = {}  # For faster lookup
        self.load_synonyms()
    
    def load_synonyms(self):
        """
        Load synonyms from JSON file and create normalized lookup
        """
        try:
            with open(self.synonyms_path, 'r', encoding='utf-8') as f:
                raw_synonyms = json.load(f)
            
            # Process synonyms and handle list values
            self.synonyms = {}
            for key, value in raw_synonyms.items():
                if isinstance(value, list):
                    # Take first item if list, or join if multiple items
                    self.synonyms[key] = value[0] if value else ""
                else:
                    self.synonyms[key] = str(value)
            
            # Create normalized lookup for faster exact matching
            self.normalized_synonyms = {}
            for key, value in self.synonyms.items():
                # Normalize key for lookup (lowercase, no extra spaces)
                normalized_key = self._normalize_text(key)
                self.normalized_synonyms[normalized_key] = {
                    'original_key': key,
                    'replacement': value
                }
            
            log_event("SUCCESS", f"Loaded {len(self.synonyms)} synonyms (simple resolver)")
            
        except FileNotFoundError:
            log_event("ERROR", f"Synonyms file not found: {self.synonyms_path}")
            self.synonyms = {}
            self.normalized_synonyms = {}
        except json.JSONDecodeError as e:
            log_event("ERROR", f"Invalid JSON in synonyms file: {e}")
            self.synonyms = {}
            self.normalized_synonyms = {}
        except Exception as e:
            log_event("ERROR", f"Error loading synonyms: {e}")
            self.synonyms = {}
            self.normalized_synonyms = {}
    
    def _normalize_text(self, text):
        """
        Normalize text for matching (lowercase, remove extra spaces, basic cleanup)
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common punctuation that might interfere
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def resolve_synonyms(self, query):
        """
        Simple synonym resolution using exact matching on normalized text
        """
        if not query or not self.synonyms:
            return query
        
        original_query = query
        resolved_query = query
        replacements_made = 0
        
        try:
            # Split query into words for individual word matching
            words = query.split()
            resolved_words = []
            
            for word in words:
                normalized_word = self._normalize_text(word)
                
                # Check for exact match
                if normalized_word in self.normalized_synonyms:
                    replacement_info = self.normalized_synonyms[normalized_word]
                    replacement = replacement_info['replacement']
                    resolved_words.append(replacement)
                    replacements_made += 1
                    log_event("INFO", f"Replaced '{word}' with '{replacement}'")
                else:
                    resolved_words.append(word)
            
            # Also check for multi-word phrases (up to 3 words)
            resolved_query = ' '.join(resolved_words)
            resolved_query = self._resolve_phrases(resolved_query)
            
            if replacements_made > 0:
                log_event("SUCCESS", f"Applied {replacements_made} synonym replacements")
            else:
                log_event("INFO", "No exact synonym matches found")
            
            return resolved_query
            
        except Exception as e:
            log_event("ERROR", f"Error in simple synonym resolution: {e}")
            return original_query  # Return original if resolution fails
    
    def _resolve_phrases(self, query):
        """
        Resolve multi-word phrases (2-3 words)
        """
        words = query.split()
        if len(words) < 2:
            return query
        
        resolved_query = query
        replacements_made = 0
        
        # Check 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            normalized_phrase = self._normalize_text(phrase)
            
            if normalized_phrase in self.normalized_synonyms:
                replacement_info = self.normalized_synonyms[normalized_phrase]
                replacement = replacement_info['replacement']
                resolved_query = resolved_query.replace(phrase, replacement, 1)
                replacements_made += 1
                log_event("INFO", f"Replaced phrase '{phrase}' with '{replacement}'")
        
        # Check 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            normalized_phrase = self._normalize_text(phrase)
            
            if normalized_phrase in self.normalized_synonyms:
                replacement_info = self.normalized_synonyms[normalized_phrase]
                replacement = replacement_info['replacement']
                resolved_query = resolved_query.replace(phrase, replacement, 1)
                replacements_made += 1
                log_event("INFO", f"Replaced phrase '{phrase}' with '{replacement}'")
        
        return resolved_query
    
    def get_stats(self):
        """
        Get statistics about loaded synonyms
        """
        return {
            'total_synonyms': len(self.synonyms),
            'normalized_entries': len(self.normalized_synonyms),
            'file_path': self.synonyms_path,
            'loaded': len(self.synonyms) > 0
        }

# Global instance for singleton pattern
_simple_synonym_resolver_instance = None

def get_simple_synonym_resolver():
    """
    Get singleton instance of SimpleSynonymResolver
    """
    global _simple_synonym_resolver_instance
    if _simple_synonym_resolver_instance is None:
        _simple_synonym_resolver_instance = SimpleSynonymResolver()
    return _simple_synonym_resolver_instance

def simple_synonym_resolver(query):
    """
    Main function to resolve synonyms using simple exact matching
    Fast and lightweight - no fuzzy matching
    """
    try:
        resolver = get_simple_synonym_resolver()
        resolved_query = resolver.resolve_synonyms(query)
        return resolved_query
    except Exception as e:
        log_event("ERROR", f"Error in simple synonym resolution: {e}")
        return query  # Return original query if resolution fails

# Compatibility function to replace the old synonym_resolver
def synonym_resolver(query, key_threshold=85, value_threshold=85):
    """
    Lightweight replacement for the original synonym_resolver
    Ignores threshold parameters and uses exact matching instead
    """
    return simple_synonym_resolver(query)
