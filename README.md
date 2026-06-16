# VectorGallery

A fully local multimodal semantic image search engine built for offline AI inference and production-grade persistence. VectorGallery pairs a lightweight FastAPI backend with a local CLIP embedding pipeline and a fast HNSW vector index, enabling text-to-image and image-to-image search without cloud dependencies.

## Features

- Offline AI inference via `clip-ViT-B-32` with offline Hugging Face caching
- Text-to-image semantic search using live CLIP embeddings
- Image-to-image visual search via uploaded query images
- Write-through persistence to `data.jsonl` and a local `zvec` vector database
- Auto-healing startup sync between JSONL source records and the vector index
- Vanilla JS + Tailwind UI with masonry gallery rendering and category filters

## Tech Stack

- FastAPI for the HTTP API and app lifecycle
- `zvec` for HNSW vector search over 512-dimensional embeddings
- `sentence-transformers` / CLIP for multimodal semantic embeddings
- Tailwind CSS and plain JavaScript for the frontend gallery UI
- PIL / Pillow for image decoding and RGB normalization

## System Architecture

This repository maintains two persistence layers:

1. `data.jsonl`: the primary document store and source of truth for every image record. It persists base64-encoded image data and category metadata as newline-delimited JSON.
2. `zvec` collection: a 512-dimensional vector index built for fast semantic similarity search. It stores the same image metadata plus the CLIP embedding vector.

The separation is intentional:

- `data.jsonl` is authoritative, durable, and easy to append to.
- `zvec` is optimized for nearest-neighbor search and cannot safely act as the source-of-truth by itself.

At startup, `SearchService.load_data()` compares `data.jsonl` against the `zvec` doc count. If the two stores diverge, the vector collection is rebuilt from the JSONL file, preventing stale index state or split-brain duplication after restarts.

## Local Setup & Installation

```bash
cd smart_image_gallery
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS / Linux
# source venv/bin/activate
pip install -r requirements.txt
```

### Run the app

```bash
uvicorn main:app --reload
```

Then navigate to `http://127.0.0.1:8000`.

> **Note:** the first execution will download the CLIP model weights for `clip-ViT-B-32` (approximately 398MB). After the initial download, `HF_HUB_OFFLINE=1` ensures the model is reused from cache and future starts are offline.

## Data Storage & Bootstrapping

The application uses a single `./storage/` directory to hold all persistent state. This design avoids mounting individual files into Docker (which can cause Docker to create empty files or directories unexpectedly).

Important: Do NOT mount `./storage/data.jsonl` directly as a file into the container. Always mount the whole `./storage` directory. Example `docker-compose` already maps `./storage:/app/storage`.

On first run the app will automatically create the storage layout:

- `./storage/data.jsonl` — primary JSONL document store (source of truth)
- `./storage/vector_db_*d` — zvec vector index folders (namespaced by vector dimension)
- `./storage/seed_images/` — hot-folder for raw image ingestion
- `./storage/processed_images/` — archive for images already ingested

Bootstrapping options:

- **The Blank Slate:** A fresh clone begins empty. The app creates `./storage/` and necessary subfolders on startup.

- **Option A — API & UI Uploads:** Start the app and use the frontend to drag-and-drop files, or call the `/api/images/upload` endpoint with multipart uploads. The server will batch-embed and persist the images.

- **Option B — The "Hot Folder" (Raw Images):** Drop `.jpg`, `.jpeg`, or `.png` files into `./storage/seed_images/` before starting the server (or while restarting). On startup the backend will batch-process all files, append records to `data.jsonl`, insert vectors into zvec, and move originals to `./storage/processed_images/` to avoid re-processing.

- **Option C — Preconfigured Dataset:** If you already have a `data.jsonl` file, place it at `./storage/data.jsonl`. On startup the app will detect it and rebuild the vector index from the JSONL records if needed.

These ingestion paths are additive and interoperable: you can seed with raw files, later augment via API uploads, or restore from a `data.jsonl` export.

## API Reference

| Endpoint             | Method | Purpose                                       |
| -------------------- | ------ | --------------------------------------------- |
| `/api/images`        | GET    | List all saved images or filter by category   |
| `/api/search`        | GET    | Semantic free-text search via CLIP embedding  |
| `/api/images/upload` | POST   | Bulk upload new images with category metadata |
| `/api/search/image`  | POST   | Image-to-image similarity search              |
| `/api/health`        | GET    | Health and loaded image metadata              |

### Example requests

- `/api/search?q=sunset&topk=12`
- `/api/images?category=animal`

## How the backend avoids split-brain duplication

- New uploads are written through immediately to both `zvec` and `data.jsonl`.
- On startup, the service rebuilds the vector index if the saved JSONL count does not match `zvec`.
- This keeps the document store and vector store aligned, even after a crash or server restart.

## Project Layout

- `main.py`: FastAPI app startup, lifespan, and static frontend serving
- `api/routers/gallery.py`: API routes and request handling
- `services/search_service.py`: vector index management, query logic, and persistence
- `utils/image_utils.py`: base64 image normalization, PIL conversion, and batch embedding
- `utils/embedder.py`: offline-first singleton CLIP model loader
- `static/index.html`: frontend gallery UI and search/upload interactions
- `models/schemas.py`: Pydantic response shapes

## Notes for developers

- The model is loaded once per process via `utils.embedder.get_model()`.
- `process_bulk_images()` decodes each image exactly once and performs a single batch embedding call, minimizing memory and CPU overhead.
- `data.jsonl` is append-only for new images, making it easy to restore state and rebuild the search index.
