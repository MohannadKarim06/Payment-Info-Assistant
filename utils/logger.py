import logging
import os, sys 
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import LOGS_FILE

logging.basicConfig(
    filename=LOGS_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_event(event_type: str, details: str):

    message = f"{event_type}: {details}"
    logging.info(message)
    print(message)
