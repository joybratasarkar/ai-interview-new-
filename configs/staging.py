from configs.base_config import BaseConfig

class StagingConfig(BaseConfig):
    DATABASE_URL = "postgresql://staging_user:staging_password@staging-db"
    LOG_LEVEL = "INFO"
    SECRET_KEY = "staging-secret-key"
