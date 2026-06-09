from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "file:./mlruns" #localhost:5000 for cloud
    ALLOWED_ORIGINS: str = "*"
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
