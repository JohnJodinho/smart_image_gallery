"""
main.py — FastAPI Application Entry Point
==========================================
Responsibilities:
  • App factory with lifespan (startup data load, shutdown cleanup)
  • CORS middleware
  • Global exception handler + structured logging
  • Router inclusion
  • Static-file mounting + root HTML route

Business logic:  services/search_service.py
API contracts:   models/schemas.py
Route handlers:  api/routers/gallery.py
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers import gallery
from services.search_service import SearchService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared service singleton — created once, lives for the process lifetime
# ---------------------------------------------------------------------------
_search_service = SearchService()

# ---------------------------------------------------------------------------
# Lifespan: load data at startup, release resources at shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("  Image Gallery API starting …")
    await _search_service.load_data(
        data_path="./data.jsonl",
        # queries_path removed — live CLIP embedding replaces text query lookup
    )
    app.state.search_service = _search_service
    logger.info("  Ready — %d images loaded.", _search_service.image_count)
    yield
    # --- shutdown ---
    # 🔌 FUTURE: close zvec collection / model sessions here
    logger.info("  Application shutdown.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Gallery API",
    description="Text-to-image search powered by live CLIP embeddings + zvec.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_exc_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled %s on %s %s", type(exc).__name__, request.method, request.url
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(gallery.router)

# ---------------------------------------------------------------------------
# Static files + frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the single-page gallery application."""
    html_path = Path("static/index.html")
    if not html_path.exists():
        return HTMLResponse(
            "<h1>Frontend not found. Place index.html in ./static/</h1>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
