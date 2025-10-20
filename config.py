from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Stock Price Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model Settings
    model_path: str = "models/lstm_model.h5"
    default_sequence_length: int = 60
    max_prediction_batch_size: int = 100
    
    # CORS Settings
    cors_origins: list = ["*"]  # Configure properly for production
    
    # Logging Settings
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
