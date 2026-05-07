"""
services/search_service.py — Core Search Logic
===============================================
Images are batch-encoded at startup with the local clip-ViT-B-32 model
(512-dimensional vectors) via utils.image_utils.process_bulk_images.

Search is fully dynamic: any free-text query is embedded live by the
same CLIP model, then queried against the zvec collection.
No predefined query list or exact-match dictionary is required.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import zvec

from utils.embedder import get_model
from utils.image_utils import Base64Image, process_bulk_images, ProcessedImage

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self) -> None:
        self._images: List[Dict[str, Any]] = []
        self._categories: List[str] = []
        self.collection = None

    # -----------------------------------------------------------------------
    # Startup: data loading + CLIP embedding
    # -----------------------------------------------------------------------

    async def load_data(self, data_path: str, queries_path: str = "") -> None:
        """
        Initialises the zvec collection, loads images into memory, and —
        when the DB is empty — batch-encodes all images with CLIP and inserts
        them.

        `queries_path` is accepted for call-site compatibility but is no
        longer used; text query pre-computation has been removed in favour
        of live dynamic embedding in search().
        """

        # 1. Initialise zvec collection (512-dim, clip-ViT-B-32)
        # ── Changed from ./image_search_auth_db so zvec creates a fresh DB
        #    that matches the new 512-dim vector size (was 1024).
        db_path = "./image_search_clip_db"

        image_field = zvec.FieldSchema(
            name="base64_image", data_type=zvec.DataType.STRING
        )
        cat_field = zvec.FieldSchema(name="category", data_type=zvec.DataType.STRING)
        embedding_field = zvec.VectorSchema(
            name="embedding",
            data_type=zvec.DataType.VECTOR_FP32,
            dimension=512,  # clip-ViT-B-32 outputs exactly 512 dims
            index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
        )
        schema = zvec.CollectionSchema(
            name="gallery",
            fields=[image_field, cat_field],
            vectors=[embedding_field],
        )

        if os.path.exists(db_path):
            self.collection = zvec.open(path=db_path)
        else:
            self.collection = zvec.create_and_open(path=db_path, schema=schema)

        # 2. Load images into memory & batch-encode for zvec when DB is empty
        p_data = Path(data_path)
        if not p_data.exists():
            logger.error(
                "data.jsonl not found at '%s'. Image store is empty.", data_path
            )
            return

        zvec_stats = json.loads(str(self.collection.stats))
        needs_insert = zvec_stats.get("doc_count", 0) == 0

        # Collect Base64Image dicts for batch CLIP processing (only when needed).
        images_to_embed: List[Base64Image] = []

        with p_data.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)

                # Normalise base64 encoding for UI display
                enc = record.get("image_encoding", "")
                if enc and not enc.startswith("data:"):
                    enc = f"data:image/jpeg;base64,{enc}"
                record["image_encoding"] = enc

                cat = record.get("category", "")
                if cat and cat not in self._categories:
                    self._categories.append(cat)

                # Always build in-memory list for the default gallery view
                self._images.append(record)

                # Package for batch CLIP encoding only when the DB is empty
                if needs_insert:
                    images_to_embed.append(
                        Base64Image(
                            id=record["id"],
                            category=cat,
                            base64_image=enc,
                        )
                    )

        # Batch-encode with CLIP → insert into zvec → build HNSW index
        if needs_insert and images_to_embed:
            logger.info(
                "Batch-encoding %d images with clip-ViT-B-32 (512-dim)…",
                len(images_to_embed),
            )
            # process_bulk_images reads each image once, encodes in one
            # forward pass, and returns {id, base64_image, embedding, category}
            processed = await process_bulk_images(images_to_embed)

            for item in processed:
                doc = zvec.Doc(
                    id=item["id"],
                    fields={
                        "base64_image": item["base64_image"],
                        "category": item["category"],
                    },
                    vectors={"embedding": item["embedding"]},  # 512-dim CLIP vector
                )
                self.collection.insert(doc)

            self.collection.optimize()
            logger.info("Inserted and indexed %d documents into zvec.", len(processed))

        logger.info(
            "SearchService ready: %d images | categories: %s",
            len(self._images),
            sorted(self._categories),
        )

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------

    @property
    def image_count(self) -> int:
        return len(self._images)

    @property
    def categories(self) -> List[str]:
        return sorted(self._categories)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_images(self, processed_images: List[ProcessedImage]) -> None:
        """Adds new images to the zvec collection and memory."""
        for item in processed_images:
            doc = zvec.Doc(
                id=item["id"],
                fields={
                    "base64_image": item["base64_image"],
                    "category": item["category"],
                },
                vectors={"embedding": item["embedding"]},
            )
            self.collection.insert(doc)

            record = {
                "id": item["id"],
                "image_encoding": item["base64_image"],
                "category": item["category"],
            }
            self._images.append(record)

            if item["category"] and item["category"] not in self._categories:
                self._categories.append(item["category"])
                self._categories.sort()

    def search_by_vector(
        self, embedding: List[float], topk: int = 12
    ) -> List[Dict[str, Any]]:
        """Executes a zvec VectorQuery using a provided image embedding."""
        result = self.collection.query(
            zvec.VectorQuery(
                field_name="embedding",
                vector=embedding,
            ),
            topk=topk,
            include_vector=False,
        )

        images = []
        for doc in result:
            img_cat = doc.fields.get("category", "")
            images.append(
                {
                    "id": doc.id,
                    "image_encoding": doc.fields.get("base64_image", ""),
                    "category": img_cat,
                    "score": doc.score,
                }
            )
        return images

    def get_all_images(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all images for the default gallery view."""
        if not category:
            return list(self._images)
        cat = category.lower()
        return [
            img for img in self._images if (img.get("category") or "").lower() == cat
        ]

    def search(
        self, query: str, topk: int = 12, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed `query` dynamically with the local CLIP model, then run
        nearest-neighbour search in zvec.

        Any free-text concept is supported — no predefined query list
        or exact-match dictionary lookup required.
        """
        # 1. Generate a live 512-dim CLIP embedding from the raw query text
        model = get_model()
        text_embedding: List[float] = model.encode(query).tolist()

        # 2. Nearest-neighbour search in zvec
        result = self.collection.query(
            zvec.VectorQuery(
                field_name="embedding",
                vector=text_embedding,
            ),
            topk=topk,
            include_vector=False,
        )

        # 3. Format zvec documents back into UI-friendly dicts
        images = []
        for doc in result:
            img_cat = doc.fields.get("category", "")

            # Apply dynamic category filter if the user clicked a sidebar pill
            if category and category.lower() != img_cat.lower():
                continue

            images.append(
                {
                    "id": doc.id,
                    "image_encoding": doc.fields.get("base64_image", ""),
                    "category": img_cat,
                    "score": doc.score,
                }
            )

        return images
