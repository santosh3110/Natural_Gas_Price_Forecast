import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name="gaspriceforecastLogger", log_file="running_logs.log"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

    # Get or create a named logger
    logger = logging.getLogger(name)

    # To avoid duplicate logs, remove all handlers if re-called
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(logging.Formatter(logging_str))

    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(logging_str))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
