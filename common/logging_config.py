import logging
import logging.config
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")  # Store logs in /logs/
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "application.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": LOG_FILE,
            "maxBytes": 5*1024*1024,  # 5MB file size limit
            "backupCount": 3,  # Keep last 3 log files
            "level": "DEBUG"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("AI-ML-XOOPER")
