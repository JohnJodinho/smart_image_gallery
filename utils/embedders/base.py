"""
utils/embedders/base.py — Abstract Embedder Interface
======================================================
Defines the contract all embedder implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding strategies.

    Implementations must return embeddings as a numpy array or list-compatible
    object with a .tolist() method for seamless downstream processing.
    """

    @abstractmethod
    def encode(
        self, inputs: Union[str, List[str], Any]
    ) -> Any:
        """
        Encode text or image(s) into embedding vector(s).

        Parameters
        ----------
        inputs
            Either a string (single text), list of strings (batch text),
            or PIL Image(s) for image embeddings.

        Returns
        -------
        Any
            A numpy array or similar object with a .tolist() method.
            Shape depends on input and model:
            - Single text: (dimension,)
            - Batch text or images: (batch_size, dimension)
        """
        pass
