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
import re
import base64
import uuid
import shutil
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import zvec

from core.config import settings
from utils.embedders.factory import get_embedder
from utils.image_utils import Base64Image, process_bulk_images, ProcessedImage

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self) -> None:
        self._images: List[Dict[str, Any]] = []
        self._categories: List[str] = []
        self.collection = None
        self._data_path: str = ""

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
        self._data_path = data_path

        # Ensure storage directories exist (storage root + seed/processed dirs)
        Path(settings.DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.SEED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.PROCESSED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

        # 1. Initialise zvec collection with configurable dimension
        # ── Database path is namespaced by dimension to avoid conflicts
        #    when switching between embedding models.
        db_path = f"{settings.VECTOR_DB_PATH}_{settings.VECTOR_DIMENSION}d"

        image_field = zvec.FieldSchema(
            name="base64_image", data_type=zvec.DataType.STRING
        )
        cat_field = zvec.FieldSchema(
            name="category", data_type=zvec.DataType.STRING)
        embedding_field = zvec.VectorSchema(
            name="embedding",
            data_type=zvec.DataType.VECTOR_FP32,
            dimension=settings.VECTOR_DIMENSION,
            index_param=zvec.HnswIndexParam(
                metric_type=zvec.MetricType.COSINE),
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

        # 2. Load images into memory
        p_data = Path(data_path)
        # Ensure an empty data.jsonl exists so fresh clones don't crash
        if not p_data.exists():
            logger.info(
                "data.jsonl not found at '%s'; creating empty file.", data_path)
            p_data.parent.mkdir(parents=True, exist_ok=True)
            p_data.touch()

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

        # ------------------------------------------------------------------
        # Hot-folder ingestion: scan SEED_IMAGES_DIR for raw .jpg/.png files
        # and auto-ingest them into the system on startup.
        # ------------------------------------------------------------------
        seed_dir = Path(settings.SEED_IMAGES_DIR)
        seed_files = [p for p in seed_dir.iterdir(
        ) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

        if seed_files:
            logger.info(
                "Found %d files in seed folder; beginning batch ingest...", len(seed_files))
            seed_items: List[Base64Image] = []
            for f in seed_files:
                try:
                    raw = f.read_bytes()
                    raw_b64 = base64.b64encode(raw).decode("utf-8")
                    ctype = mimetypes.guess_type(str(f))[0] or "image/jpeg"
                    data_uri = f"data:{ctype};base64,{raw_b64}"

                    # sanitize filename and create a unique id
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', f.stem)
                    unique_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"

                    seed_items.append(Base64Image(
                        id=unique_id, category="", base64_image=data_uri))
                except Exception as exc:
                    logger.error("Failed to read seed file %s: %s", f, exc)

            if seed_items:
                try:
                    processed = await process_bulk_images(seed_items)
                    # Persist via add_images() which inserts into zvec and appends to JSONL
                    self.add_images(processed)

                    # Move processed files to processed_images dir
                    for f in seed_files:
                        try:
                            dest = Path(settings.PROCESSED_IMAGES_DIR) / f.name
                            shutil.move(str(f), str(dest))
                        except Exception as exc:
                            logger.warning(
                                "Failed to move processed seed file %s: %s", f, exc)

                    logger.info(
                        "Seed folder ingestion complete. Processed %d files.", len(processed))
                except Exception as exc:
                    logger.error("Seed folder ingestion failed: %s", exc)

        # 3. Auto-Healing Sync Check
        zvec_stats = json.loads(str(self.collection.stats))
        doc_count = zvec_stats.get("doc_count", 0)

        if doc_count == len(self._images):
            logger.info("Database perfectly synchronized (Zvec: %d, JSONL: %d). Skipping embedding phase.",
                        doc_count, len(self._images))
        else:
            logger.warning("Desync detected: JSONL has %d, Zvec has %d. Rebuilding index...", len(
                self._images), doc_count)

            if doc_count > 0:
                self.collection.destroy()
                self.collection = zvec.create_and_open(
                    path=db_path, schema=schema)

            images_to_embed: List[Base64Image] = []
            for record in self._images:
                images_to_embed.append(
                    Base64Image(
                        id=record["id"],
                        category=record.get("category", ""),
                        base64_image=record["image_encoding"],
                    )
                )

            # Batch-encode with embedder → insert into zvec → build HNSW index
            if images_to_embed:
                logger.info(
                    "Batch-encoding %d images with %s (%d-dim)…",
                    len(images_to_embed),
                    settings.EMBEDDING_MODEL,
                    settings.VECTOR_DIMENSION,
                )
                processed = await process_bulk_images(images_to_embed)

                for item in processed:
                    doc = zvec.Doc(
                        id=item["id"],
                        fields={
                            "base64_image": item["base64_image"],
                            "category": item["category"],
                        },
                        # 512-dim CLIP vector
                        vectors={"embedding": item["embedding"]},
                    )
                    self.collection.insert(doc)

                self.collection.optimize()
                logger.info(
                    "Inserted and indexed %d documents into zvec.", len(processed))

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
        """Adds new images to the zvec collection, memory, and persists to disk."""
        records_to_save = []

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
            records_to_save.append(record)

            if item["category"] and item["category"] not in self._categories:
                self._categories.append(item["category"])
                self._categories.sort()

        if self._data_path and records_to_save:
            with open(self._data_path, "a", encoding="utf-8") as f:
                for rec in records_to_save:
                    f.write(json.dumps(rec) + "\n")
            logger.info(
                f"Persisted {len(records_to_save)} new images to {self._data_path}")

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
        Embed `query` dynamically with the configured embedder, then run
        nearest-neighbour search in zvec.

        Any free-text concept is supported — no predefined query list
        or exact-match dictionary lookup required.
        """
        # 1. Generate embedding from the raw query text
        embedder = get_embedder()
        embedding_result = embedder.encode(query)
        if hasattr(embedding_result, 'tolist'):
            text_embedding: List[float] = embedding_result.tolist()
        else:
            text_embedding: List[float] = list(embedding_result)

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
