import os
import logging
from configs.development import DevelopmentConfig
from configs.staging import StagingConfig
from configs.production import ProductionConfig

logging.basicConfig(level=logging.INFO)

def get_config():
    environment = os.getenv("ENV", "development").lower()
    if environment == "staging":
        logging.info("Using Staging configuration.")
        return StagingConfig()
    elif environment == "production":
        logging.info("Using Production configuration.")
        return ProductionConfig()
    elif environment == "development":
        logging.info("Using Development configuration.")
        return DevelopmentConfig()
    else:
        logging.warning(f"Invalid environment '{environment}', falling back to Development configuration.")
        return DevelopmentConfig()
