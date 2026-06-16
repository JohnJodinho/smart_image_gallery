import os

os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer, util
from PIL import Image


import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


model = SentenceTransformer("clip-ViT-B-32")

logger.info("Model successfully loaded!")

# text_prompt = "A beautiful dog"
# text_vector = model.encode(text_prompt, show_progress_bar=True)

# # logger.info(f"Vector: {text_vector}")
# logger.info(f"Vector shape: {text_vector.shape}")

image_path = "image1.jpeg"

logger.info("Loading image")
image_obj = Image.open(image_path)


# logger.info("Getting vectors...")
# image_vector = model.encode(image_obj)

# logger.info(f"Vector shape: {image_vector.shape}")


# # Calculating similarity (-1 < score < 1)
# sim_score = util.cos_sim(image_vector, text_vector)[0][0]

# logger.info(f"Match Score: {sim_score:.4f}")


def zero_shot_classifier(img: Image.Image, candidate_labels: List[str]):
    image_emb = model.encode(img)
    text_embs = model.encode(candidate_labels)

    scores = util.cos_sim(image_emb, text_embs)[0]

    results = list(zip(candidate_labels, scores.tolist()))

    results.sort(key=lambda x: x[1], reverse=True)

    return results


labels = [
    "A picture of a cat",
    "A picture of a dog",
    "A picture of a car",
    "A landscape photo of a mountain",
]

class_results = zero_shot_classifier(image_obj, labels)

logger.info("Class results:")
for label, score in class_results:
    logger.info(f"{label}: {score:.4f}")
