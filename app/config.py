"""Configuration management for the SAM 3D pipeline."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Storage
    STORAGE_BACKEND: str = Field(default="local", description="Storage backend: local|s3")
    DATA_DIR: str = Field(default="/data", description="Local data directory")
    S3_BUCKET: str = Field(default="", description="S3 bucket name (if using S3)")
    AWS_DEFAULT_REGION: str = Field(default="us-east-1", description="AWS region")

    # Model paths
    MODELS_DIR: str = Field(default="/models", description="Directory for model weights")

    # Hugging Face
    HUGGINGFACE_HUB_TOKEN: str = Field(default="", description="HF token for gated models")

    # API
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")

    # Device
    DEVICE: str = Field(default="cuda", description="Device for inference: cuda|cpu")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
