"""
core/config.py — Environment Configuration via Pydantic Settings
=================================================================
Centralizes all application configuration, supporting both local
(offline) and cloud-based (OpenAI) embedding models.

Usage:
    from core.config import settings
    print(settings.EMBEDDING_PROVIDER)  # "local" or "openai"
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project metadata
    PROJECT_NAME: str = "VectorGallery API"
    VERSION: str = "1.0.0"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Embedding configuration
    EMBEDDING_PROVIDER: str = "local"  # "local" or "openai"
    EMBEDDING_MODEL: str = "clip-ViT-B-32"
    VECTOR_DIMENSION: int = 512

    # OpenAI API (only used if EMBEDDING_PROVIDER == "openai")
    OPENAI_API_KEY: str = None

    # Data storage
    # Storage layout (mounted as a single directory in Docker)
    DATA_PATH: str = "./storage/data.jsonl"
    VECTOR_DB_PATH: str = "./storage/vector_db"
    SEED_IMAGES_DIR: str = "./storage/seed_images"
    PROCESSED_IMAGES_DIR: str = "./storage/processed_images"

    class Config:
        """Load settings from .env file."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
