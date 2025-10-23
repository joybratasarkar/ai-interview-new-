from configs.base_config import BaseConfig

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    DATABASE_URL = "postgresql://dev_user:dev_password@localhost/xooper_dev"
    LOG_LEVEL = "DEBUG"
    SECRET_KEY = "dev-secret-key"
