"""
utils/embedders/openai_embedder.py — Cloud-Based OpenAI Embedder
===============================================================
Implements embedding via the OpenAI API (requires OPENAI_API_KEY).
"""

import logging
from typing import Any, List, Union

import numpy as np
from openai import OpenAI

from utils.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """
    Cloud-based embedder using the OpenAI Embeddings API.

    This embedder:
    - Requires a valid OPENAI_API_KEY.
    - Supports text embeddings via the OpenAI API.
    - Does NOT support image inputs (images must be described as text).
    """

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """
        Initialize the OpenAIEmbedder.

        Parameters
        ----------
        api_key : str
            OpenAI API key for authentication.
        model_name : str
            OpenAI embedding model (default: "text-embedding-3-small").

        Raises
        ------
        ValueError
            If api_key is None or empty.
        """
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI embeddings. "
                "Set it in your .env file."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI Embedder initialized with model: %s", model_name)

    def encode(self, inputs: Union[str, List[str], Any]) -> Any:
        """
        Encode text to embedding vectors via OpenAI API.

        Parameters
        ----------
        inputs
            String or list of strings (text queries or descriptions).
            Note: PIL Images are not supported; convert to text descriptions.

        Returns
        -------
        numpy.ndarray
            Embeddings with shape (dimension,) for single text or 
            (batch_size, dimension) for batch text.

        Raises
        ------
        TypeError
            If inputs is not text-based (e.g., PIL Image).
        """
        # Normalize input to list of strings
        if isinstance(inputs, str):
            texts = [inputs]
            is_single = True
        elif isinstance(inputs, list) and all(isinstance(t, str) for t in inputs):
            texts = inputs
            is_single = False
        else:
            raise TypeError(
                "OpenAI embeddings only support text input (str or List[str]). "
                "For images, use the local CLIP embedder or convert images to descriptions."
            )

        # Call OpenAI API
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )

        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])

        # Return single embedding if input was single string
        if is_single:
            return embeddings[0]
        return embeddings
