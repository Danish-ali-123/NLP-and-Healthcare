# src/utils/logging.py
import logging

def setup_logging(log_level=logging.INFO):
    """Set up logging for the project."""
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message: str):
    """Log an info message."""
    logging.info(message)

def log_warning(message: str):
    """Log a warning message."""
    logging.warning(message)

def log_error(message: str):
    """Log an error message."""
    logging.error(message)

def log_exception(exception: Exception):
    """Log an exception message."""
    logging.exception(exception)
