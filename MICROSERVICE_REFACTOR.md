# Microservice Refactor: Migration Guide

## Overview

VectorGallery has been refactored into a highly configurable, model-agnostic microservice API. The hardcoded CLIP logic is now decoupled via a **strategy pattern**, allowing any developer to fork this repo, configure a `.env` file, and instantly deploy it supporting either local models or frontier cloud models.

## What Changed

### 1. **Configuration Management** (`core/config.py`)

All environment variables are now centralized in a `Settings` class via `pydantic-settings`:

```python
from core.config import settings

print(settings.EMBEDDING_PROVIDER)  # "local" or "openai"
print(settings.VECTOR_DIMENSION)    # 512 or 1536, etc.
```

**Key Settings:**

- `EMBEDDING_PROVIDER`: `"local"` (offline CLIP) or `"openai"` (cloud)
- `EMBEDDING_MODEL`: Model name (e.g., `"clip-ViT-B-32"` or `"text-embedding-3-small"`)
- `VECTOR_DIMENSION`: Must match your model's output dimensionality
- `OPENAI_API_KEY`: Only needed if using OpenAI provider

See `.env.example` for a complete reference.

### 2. **Embedder Strategy Pattern** (`utils/embedders/`)

The monolithic `utils/embedder.py` has been replaced with a pluggable architecture:

```
utils/embedders/
├── __init__.py
├── base.py                  # Abstract BaseEmbedder interface
├── local_clip.py            # LocalCLIPEmbedder (offline CLIP)
├── openai_embedder.py       # OpenAIEmbedder (cloud)
└── factory.py               # Factory to select embedder based on config
```

**Usage:**

```python
from utils.embedders.factory import get_embedder

embedder = get_embedder()  # Returns LocalCLIPEmbedder or OpenAIEmbedder
embeddings = embedder.encode(["text", "or PIL images"])  # Returns numpy array
```

All embedders return `.tolist()`-compatible objects, so downstream code (image processing, search) remains unchanged.

### 3. **Dynamic Vector Database** (`services/search_service.py`)

The `zvec` database is now dynamically named based on vector dimension:

```python
db_path = f"./zvec_db_{settings.VECTOR_DIMENSION}d"
```

This prevents dimension-mismatch crashes when switching models.

### 4. **Dockerization**

- **`Dockerfile`**: Production-ready slim Python 3.10+ image with health checks
- **`docker-compose.yml`**: Persistent volumes for `data.jsonl` and vector databases

**Deploy Locally:**

```bash
cp .env.example .env
# Edit .env as needed
docker-compose up --build
```

**Deploy with OpenAI:**

```bash
# In .env:
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSION=1536
OPENAI_API_KEY=sk-...

docker-compose up --build
```

## Switching Embedding Providers

### Local (Offline CLIP)

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=clip-ViT-B-32
VECTOR_DIMENSION=512
```

### OpenAI

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSION=1536
OPENAI_API_KEY=sk-...
```

## Deprecated Files

- `utils/embedder.py`: No longer used. Can be removed after confirming no external code imports it.

All imports have been migrated to `from utils.embedders.factory import get_embedder`.

## Key Architectural Improvements

1. **Single-Pass Memory Optimization Preserved**: `process_bulk_images()` still decodes each image once and performs one batch embedding call.
2. **Singleton Model Management**: Both `LocalCLIPEmbedder` and `OpenAIEmbedder` are singletons, preventing redundant loads.
3. **Auto-Healing Sync**: The dual-database pattern (JSONL + zvec) remains unchanged and still validates consistency at startup.
4. **Zero Downtime**: Switch providers by updating `.env` and restarting the container.

## Testing the Refactor

```bash
# Test local CLIP
export EMBEDDING_PROVIDER=local
uvicorn main:app --reload

# Test OpenAI
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

Both should behave identically from the API perspective—the backend abstraction is transparent to the frontend and API consumers.

## Notes for Deployment

- The first run with a new model/dimension will rebuild the vector index.
- The `VECTOR_DIMENSION` must match the model's output size exactly.
- Switching between 512-dim and 1536-dim models creates separate databases (`zvec_db_512d/` and `zvec_db_1536d/`), so you don't lose existing search indices.
