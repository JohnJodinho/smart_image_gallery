"""
utils/embedders/factory.py — Embedder Factory
==============================================
Factory pattern for creating and managing embedder instances based on configuration.
"""

import logging
from typing import Optional

from core.config import settings
from utils.embedders.base import BaseEmbedder
from utils.embedders.local_clip import LocalCLIPEmbedder
from utils.embedders.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)

_embedder_instance: Optional[BaseEmbedder] = None


def get_embedder() -> BaseEmbedder:
    """
    Get or create the embedder instance based on settings.EMBEDDING_PROVIDER.

    Returns
    -------
    BaseEmbedder
        Either a LocalCLIPEmbedder (for offline) or OpenAIEmbedder (for cloud).

    Raises
    ------
    ValueError
        If EMBEDDING_PROVIDER is not "local" or "openai", or if OpenAI 
        provider is selected but OPENAI_API_KEY is missing.
    """
    global _embedder_instance

    if _embedder_instance is not None:
        return _embedder_instance

    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        logger.info("Creating LocalCLIPEmbedder with model: %s",
                    settings.EMBEDDING_MODEL)
        _embedder_instance = LocalCLIPEmbedder(
            model_name=settings.EMBEDDING_MODEL)

    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "EMBEDDING_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set. "
                "Please provide OPENAI_API_KEY in your .env file."
            )
        logger.info("Creating OpenAIEmbedder with model: %s",
                    settings.EMBEDDING_MODEL)
        _embedder_instance = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider}. "
            f"Must be 'local' or 'openai'."
        )

    return _embedder_instance
