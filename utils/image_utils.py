"""
utils/image_utils.py
====================
Utilities for processing incoming images — whether they arrive as raw
UploadFile objects or as already-encoded base64 strings.

Public API
----------
process_bulk_images(images)
    Accepts a mixed list of UploadedImage | Base64Image dicts.
    Returns List[{"id", "base64_image", "embedding", "category"}].

Matches the field names used in services/search_service.py's zvec schema:
    fields  → base64_image, category
    vectors → embedding
"""


import base64
import io
from typing import Any, Dict, List, Union

from fastapi import UploadFile
from PIL import Image
from typing import TypedDict

from utils.embedder import get_model


# ---------------------------------------------------------------------------
# Input type definitions
# ---------------------------------------------------------------------------


class UploadedImage(TypedDict):
    """An image that arrives as a multipart file upload."""

    id: str
    category: str
    file: UploadFile


class Base64Image(TypedDict):
    """An image that arrives already base64-encoded (with or without data-URI prefix)."""

    id: str
    category: str
    base64_image: str


# Output shape — mirrors zvec Doc fields in search_service.py
class ProcessedImage(TypedDict):
    id: str
    base64_image: str  # data-URI: "data:image/jpeg;base64,..."
    embedding: List[float]
    category: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_data_uri(b64_str: str, content_type: str = "image/jpeg") -> str:
    """Guarantee the string is a well-formed data-URI."""
    if b64_str.startswith("data:"):
        return b64_str
    return f"data:{content_type};base64,{b64_str}"


def _strip_data_uri_prefix(b64_str: str) -> str:
    """Return the raw base64 payload, removing any data-URI prefix."""
    if "," in b64_str:
        return b64_str.split(",", 1)[1]
    return b64_str


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


async def process_bulk_images(
    images: List[Union[UploadedImage, Base64Image]],
) -> List[ProcessedImage]:
    """
    Process a mixed list of UploadedImage and/or Base64Image objects.

    For each item the function:
      1. Reads/decodes the image data exactly once.
      2. Builds a data-URI base64 string for UI display.
      3. Converts to a PIL Image for CLIP embedding.

    All PIL images are then encoded in a single batch call for speed.

    Parameters
    ----------
    images : list
        Any combination of UploadedImage (has "file" key) and
        Base64Image (has "base64_image" key). Must not be empty.

    Returns
    -------
    list of dicts, each with keys:
        id           : str
        base64_image : str  (data-URI ready for <img src="...">)
        embedding    : list[float]
        category     : str
    """
    if not images:
        return []

    # Parallel lists — one entry per input image, built in a single pass.
    metadata: List[Dict[str, Any]] = []  # id, category, base64_image per item
    pil_images: List[Image.Image] = []  # PIL images for batch encoding

    for item in images:
        item_id: str = item.get("id", "")
        category: str = item.get("category", "")

        # ── Branch 1: UploadFile ────────────────────────────────────────────
        if "file" in item:
            upload: UploadFile = item["file"]

            # Read bytes exactly once
            image_bytes = await upload.read()

            # Build data-URI
            content_type = upload.content_type or "image/jpeg"
            raw_b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:{content_type};base64,{raw_b64}"

            # Build PIL image for embedding
            pil_img = Image.open(io.BytesIO(image_bytes))

        # ── Branch 2: Already base64 ────────────────────────────────────────
        elif "base64_image" in item:
            raw_b64_input: str = item["base64_image"]

            # Normalise to data-URI for storage; strip prefix for decoding
            data_uri = _ensure_data_uri(raw_b64_input)
            raw_b64 = _strip_data_uri_prefix(raw_b64_input)

            image_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(image_bytes))

        else:
            raise ValueError(
                f"Each item must have either a 'file' (UploadFile) or "
                f"'base64_image' (str) key. Got keys: {list(item.keys())}"
            )

        # Ensure RGB — CLIP requires 3-channel input
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        pil_images.append(pil_img)
        metadata.append(
            {"id": item_id, "base64_image": data_uri, "category": category})

    # ── Batch encode all images in one forward pass ─────────────────────────
    model = get_model()
    embeddings: List[List[float]] = model.encode(pil_images).tolist()

    # ── Assemble final results ───────────────────────────────────────────────
    results: List[ProcessedImage] = [
        {
            "id": metadata[i]["id"],
            "base64_image": metadata[i]["base64_image"],
            "embedding": embeddings[i],
            "category": metadata[i]["category"],
        }
        for i in range(len(pil_images))
    ]

    return results
