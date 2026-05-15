
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_model_instance = None


def get_model():
    """
    Returns the loaded CLIP model.
    Initializes it if not already loaded.
    """
    global _model_instance

    if _model_instance is None:
        logger.info("Initializing CLIP model (clip-ViT-B-32)...")
        logger.info("This should only happen ONCE per process.")

        _model_instance = SentenceTransformer("clip-ViT-B-32")

        logger.info("CLIP Model loaded successfully into memory.")

    return _model_instance


# ====================================================================
# Test Block: Allows the agent to verify the singleton behavior
# ====================================================================
# if __name__ == "__main__":
#     print("\n--- Test 1: First Call ---")
#     model_1 = get_model()

#     print("\n--- Test 2: Second Call ---")
#     model_2 = get_model()

#     # Verify both variables point to the exact same object in memory
#     if model_1 is model_2:
#         print("\nSuccess: Both calls returned the exact same model instance.")
#     else:
#         print("\nError: Multiple model instances were created.")
