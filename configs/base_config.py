class BaseConfig:
    APP_NAME = "Xooper"
    DEBUG = False
    DATABASE_URL = "postgresql://user:password@localhost/xooper"
    LOG_LEVEL = "INFO"
    SECRET_KEY = "default-secret-key"
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"