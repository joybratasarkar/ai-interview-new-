from configs.base_config import BaseConfig

class ProductionConfig(BaseConfig):
    DATABASE_URL = "postgresql://prod_user:prod_password@prod-db"
    LOG_LEVEL = "ERROR"
    SECRET_KEY = "prod-secret-key"
