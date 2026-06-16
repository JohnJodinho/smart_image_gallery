"""
utils/embedders/local_clip.py — Local Offline CLIP Embedder
==========================================================
Implements the Singleton pattern for `clip-ViT-B-32` via SentenceTransformer
with offline-first caching from HuggingFace.
"""

import os
import logging
from typing import Any, List, Union

from sentence_transformers import SentenceTransformer

from utils.embedders.base import BaseEmbedder

# Enforce offline mode before any transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

_model_instance = None


class LocalCLIPEmbedder(BaseEmbedder):
    """
    Offline-first CLIP embedder using SentenceTransformer.

    This embedder:
    - Loads `clip-ViT-B-32` model from local cache on first use.
    - Supports encoding both text and images.
    - Implements the Singleton pattern to avoid redundant loads.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the LocalCLIPEmbedder.

        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model (default: "clip-ViT-B-32").
        """
        self.model_name = model_name
        self._model = self._get_model()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the model using the Singleton pattern."""
        global _model_instance
        if _model_instance is None:
            logger.info(
                "Initializing CLIP model (%s) with HF_HUB_OFFLINE=1…",
                self.model_name
            )
            _model_instance = SentenceTransformer(self.model_name)
            logger.info("CLIP model loaded successfully into memory.")
        return _model_instance

    def encode(self, inputs: Union[str, List[str], Any]) -> Any:
        """
        Encode text or images to embedding vectors.

        Parameters
        ----------
        inputs
            String, list of strings, PIL Image, or list of PIL Images.

        Returns
        -------
        numpy.ndarray
            Embeddings with shape (dimension,) for single inputs or 
            (batch_size, dimension) for batch inputs.
        """
        return self._model.encode(inputs)
