import logging
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))  # Current directory (/app/app)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Parent directory (/app)

# Default logs directory and file
LOGS_DIR = "logs"
LOGS_FILE = os.path.join(LOGS_DIR, "logs.log")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOGS_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # Append mode
)

# Also create a console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Get the root logger and add console handler
logger = logging.getLogger()
logger.addHandler(console_handler)

def log_event(event_type: str, details: str, return_timestamp: bool = False):
    """
    Log an event with timestamp and event type
    
    Args:
        event_type (str): Type of event (ERROR, SUCCESS, PROCESS, INFO, etc.)
        details (str): Event details
        return_timestamp (bool): Whether to return the timestamp
        
    Returns:
        str: Timestamp if return_timestamp is True, None otherwise
    """
    timestamp = datetime.now().isoformat()
    message = f"{event_type}: {details}"
    
    # Log to file
    logging.info(message)
    
    # Also print to console for development
    print(f"{timestamp} - {message}")
    
    if return_timestamp:
        return timestamp

def get_logs(limit: int = 100, level: Optional[str] = None, search: Optional[str] = None) -> List[Dict]:
    """
    Retrieve logs from the log file
    
    Args:
        limit (int): Maximum number of logs to return
        level (str): Filter by log level (ERROR, SUCCESS, PROCESS, INFO, etc.)
        search (str): Search term to filter logs
        
    Returns:
        List[Dict]: List of log entries
    """
    logs = []
    
    try:
        if not os.path.exists(LOGS_FILE):
            return []
        
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process lines in reverse order (newest first)
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse log line format: "timestamp - level - message"
                parts = line.split(' - ', 2)
                if len(parts) >= 3:
                    timestamp_str = parts[0]
                    log_level = parts[1]
                    message = parts[2]
                    
                    # Extract event type from message if it follows "EVENT_TYPE: details" format
                    event_type = None
                    if ':' in message:
                        potential_event_type = message.split(':', 1)[0]
                        # Check if it looks like an event type (uppercase, underscores)
                        if potential_event_type.isupper() and '_' in potential_event_type:
                            event_type = potential_event_type
                            message = message.split(':', 1)[1].strip()
                    
                    log_entry = {
                        'timestamp': timestamp_str,
                        'level': log_level,
                        'event_type': event_type,
                        'message': message,
                        'raw_line': line
                    }
                    
                    # Apply filters
                    include_log = True
                    
                    # Filter by level
                    if level and level.upper() != log_level.upper():
                        # Also check event_type for backward compatibility
                        if not event_type or level.upper() != event_type.upper():
                            include_log = False
                    
                    # Filter by search term
                    if search and include_log:
                        search_lower = search.lower()
                        if (search_lower not in message.lower() and 
                            search_lower not in (event_type or '').lower() and
                            search_lower not in log_level.lower()):
                            include_log = False
                    
                    if include_log:
                        logs.append(log_entry)
                        
                        # Stop if we've reached the limit
                        if len(logs) >= limit:
                            break
                            
            except Exception as e:
                # If we can't parse a line, include it as-is
                logs.append({
                    'timestamp': 'unknown',
                    'level': 'PARSE_ERROR',
                    'event_type': None,
                    'message': line,
                    'raw_line': line,
                    'parse_error': str(e)
                })
                
                if len(logs) >= limit:
                    break
    
    except Exception as e:
        # Return error log if we can't read the file
        return [{
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'event_type': 'LOG_READ_ERROR',
            'message': f"Error reading log file: {str(e)}",
            'raw_line': ''
        }]
    
    return logs

def get_log_stats() -> Dict:
    """
    Get basic statistics about logs
    
    Returns:
        Dict: Log statistics
    """
    try:
        logs = get_logs(limit=1000)  # Get more logs for better stats
        
        stats = {
            'total_logs': len(logs),
            'by_level': {},
            'by_event_type': {},
            'recent_errors': 0,
            'recent_successes': 0
        }
        
        # Calculate statistics
        for log in logs:
            level = log.get('level', 'UNKNOWN')
            event_type = log.get('event_type', 'UNKNOWN')
            
            # Count by level
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # Count by event type
            if event_type:
                stats['by_event_type'][event_type] = stats['by_event_type'].get(event_type, 0) + 1
        
        # Count recent errors and successes (last 100 logs)
        recent_logs = logs[:100]
        stats['recent_errors'] = len([log for log in recent_logs if 'ERROR' in log.get('level', '')])
        stats['recent_successes'] = len([log for log in recent_logs if 'SUCCESS' in log.get('event_type', '')])
        
        return stats
        
    except Exception as e:
        return {
            'error': f"Error calculating log stats: {str(e)}",
            'total_logs': 0,
            'by_level': {},
            'by_event_type': {}
        }

def clear_logs():
    """
    Clear all logs (use with caution)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'w') as f:
                f.write('')
            log_event("SUCCESS", "Log file cleared")
            return True
        return True
    except Exception as e:
        log_event("ERROR", f"Error clearing logs: {str(e)}")
        return False

def export_logs_json(output_file: str = None, limit: int = 1000) -> str:
    """
    Export logs to JSON format
    
    Args:
        output_file (str): Output file path (optional)
        limit (int): Maximum number of logs to export
        
    Returns:
        str: Path to exported file or JSON string
    """
    try:
        logs = get_logs(limit=limit)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            log_event("SUCCESS", f"Logs exported to {output_file}")
            return output_file
        else:
            return json.dumps(logs, indent=2, ensure_ascii=False)
            
    except Exception as e:
        log_event("ERROR", f"Error exporting logs: {str(e)}")
        return f"Error: {str(e)}"

# Convenience functions for different log levels
def log_error(message: str):
    """Log an error message"""
    log_event("ERROR", message)

def log_success(message: str):
    """Log a success message"""
    log_event("SUCCESS", message)

def log_info(message: str):
    """Log an info message"""
    log_event("INFO", message)

def log_process(message: str):
    """Log a process message"""
    log_event("PROCESS", message)

def log_api_request(message: str):
    """Log an API request"""
    log_event("API_REQUEST", message)

def log_api_success(message: str):
    """Log an API success"""
    log_event("API_SUCCESS", message)

def log_api_error(message: str):
    """Log an API error"""
    log_event("API_ERROR", message)
