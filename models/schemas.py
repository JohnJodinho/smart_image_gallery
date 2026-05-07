"""
models/schemas.py — Pydantic Response Models
=============================================
All API response shapes are defined here.
Keeping schemas separate from routes allows independent versioning.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ImageItem(BaseModel):
    """A single image record as returned by the API."""
    id: str = Field(..., description="Unique image identifier")
    image_encoding: str = Field(..., description="Base64 data-URI — plug directly into <img src>")
    category: Optional[str] = Field(None, description="Category label, e.g. 'animal', 'car'")


class GalleryResponse(BaseModel):
    """Wraps a list of images with search metadata."""
    images: List[ImageItem] = Field(default_factory=list)
    total: int = Field(..., description="Number of images returned")
    query: Optional[str] = Field(None, description="The search query (if any)")
    matched: bool = Field(
        False,
        description="True when the search returned at least one result"
    )


class HealthResponse(BaseModel):
    """Operational status and data-load statistics."""
    status: str
    images_loaded: int
    categories: List[str]
