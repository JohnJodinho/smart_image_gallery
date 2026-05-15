"""
api/routers/gallery.py — Gallery API Routes
============================================
Thin route layer: validate inputs → call service → return response.
Zero business logic lives here.
"""


import logging
import re
import uuid
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Request, File, Form, UploadFile

from utils.image_utils import process_bulk_images

from models.schemas import GalleryResponse, HealthResponse, ImageItem
from services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Gallery"])


def _svc(request: Request) -> SearchService:
    """Retrieve the shared SearchService from app state."""
    return request.app.state.search_service


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, summary="Health / readiness check")
async def health(request: Request):
    svc = _svc(request)
    return HealthResponse(
        status="ok",
        images_loaded=svc.image_count,
        categories=svc.categories,
    )


# ---------------------------------------------------------------------------
# GET /api/images
# ---------------------------------------------------------------------------

@router.get("/images", response_model=GalleryResponse, summary="List all images")
async def get_images(
    request: Request,
    category: Optional[str] = Query(
        None, description="Filter by category (case-insensitive)"),
):
    """Return all images, optionally filtered by category."""
    svc = _svc(request)
    try:
        raw = svc.get_all_images(category=category)
        items = [
            ImageItem(
                id=img.get("id", ""),
                image_encoding=img.get("image_encoding", ""),
                category=img.get("category"),
            )
            for img in raw
        ]
        return GalleryResponse(images=items, total=len(items))
    except Exception as exc:
        logger.error("GET /api/images error: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve images.")


# ---------------------------------------------------------------------------
# GET /api/search
# ---------------------------------------------------------------------------

@router.get("/search", response_model=GalleryResponse, summary="Semantic image search (CLIP)")
async def search_images(
    request: Request,
    q: str = Query(..., min_length=1,
                   description="Free-text query — embedded live with CLIP"),
    topk: int = Query(default=12, ge=1, le=50,
                      description="Maximum number of images to return"),
    category: Optional[str] = Query(
        None, description="Optional category filter"),
):
    """
    Embed `q` dynamically with the local clip-ViT-B-32 model and return
    the `topk` most semantically similar images from the zvec collection.
    
    """
    svc = _svc(request)
    try:
        raw = svc.search(query=q, topk=topk, category=category)
        items = [
            ImageItem(
                id=img.get("id", ""),
                image_encoding=img.get("image_encoding", ""),
                category=img.get("category"),
            )
            for img in raw
        ]
        return GalleryResponse(images=items, total=len(items), query=q)
    except Exception as exc:
        logger.error("GET /api/search error (q=%r): %s", q, exc)
        raise HTTPException(status_code=500, detail="Search failed.")


# ---------------------------------------------------------------------------
# POST /api/images/upload
# ---------------------------------------------------------------------------

@router.post("/images/upload", summary="Bulk Image Upload")
async def upload_images(
    request: Request,
    files: List[UploadFile] = File(...),
    category: str = Form(...)
):
    svc = _svc(request)
    try:
        images_data = []
        for f in files:
            # Clean filename to pass zvec ID regex validation
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', f.filename or "upload")
            unique_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"
            images_data.append(
                {"id": unique_id, "category": category, "file": f})

        processed_images = await process_bulk_images(images_data)
        svc.add_images(processed_images)
        return {"message": "Images added successfully", "count": len(processed_images)}
    except Exception as exc:
        logger.error("POST /api/images/upload error: %s", exc)
        raise HTTPException(status_code=500, detail="Upload failed.")


# ---------------------------------------------------------------------------
# POST /api/search/image
# ---------------------------------------------------------------------------

@router.post("/search/image", response_model=GalleryResponse, summary="Image-to-Image Search")
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    topk: int = Form(12)
):
    svc = _svc(request)
    try:
        image_data = [{"id": "search_query",
                       "category": "query", "file": file}]
        processed_images = await process_bulk_images(image_data)

        if not processed_images:
            raise HTTPException(
                status_code=400, detail="Failed to process image.")

        embedding = processed_images[0]["embedding"]
        raw = svc.search_by_vector(embedding=embedding, topk=topk)

        items = [
            ImageItem(
                id=img.get("id", ""),
                image_encoding=img.get("image_encoding", ""),
                category=img.get("category"),
            )
            for img in raw
        ]
        return GalleryResponse(images=items, total=len(items))
    except Exception as exc:
        logger.error("POST /api/search/image error: %s", exc)
        raise HTTPException(status_code=500, detail="Image search failed.")
