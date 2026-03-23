"""FastAPI application for the image embedding search service.

Environment variables
---------------------
INDEX_PATH   Directory where the FAISS index is persisted (default: ``data/index``).
CLIP_MODEL   sentence-transformers CLIP model name (default: ``clip-ViT-B-32``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.embedder import CLIPEmbedder
from src.indexer import ImageIndex
from src.searcher import ImageSearcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_PATH: str = os.getenv("INDEX_PATH", "data/index")
MODEL_NAME: str = os.getenv("CLIP_MODEL", "clip-ViT-B-32")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

# ---------------------------------------------------------------------------
# Application & shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Embedding Search",
    description=(
        "Semantic search over a local image collection using CLIP embeddings "
        "stored in a FAISS vector index."
    ),
    version="1.0.0",
)

embedder = CLIPEmbedder(model_name=MODEL_NAME)
index = ImageIndex(embedder=embedder, index_path=INDEX_PATH)
searcher = ImageSearcher(index=index, embedder=embedder)

# Try to load an existing index on startup (non-fatal if missing).
try:
    index.load()
    print(f"[startup] Loaded index with {len(index)} images from '{INDEX_PATH}'.")
except FileNotFoundError:
    print(f"[startup] No existing index at '{INDEX_PATH}'. Index images first.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    folder: str
    recursive: bool = True


class SearchResult(BaseModel):
    path: str
    score: float


class IndexResponse(BaseModel):
    indexed: int
    total: int


class StatsResponse(BaseModel):
    indexed_images: int
    model: str
    index_path: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/index",
    response_model=IndexResponse,
    summary="Index images from a local folder",
)
def index_images(request: IndexRequest) -> IndexResponse:
    """Scan *folder* for images, embed them with CLIP and add them to the index.

    The index is persisted to disk after every call so it survives restarts.
    """
    folder = Path(request.folder)
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    if request.recursive:
        paths = [str(p) for p in folder.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS]
    else:
        paths = [str(p) for p in folder.glob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS]

    count = index.add_images(paths)
    index.save()
    return IndexResponse(indexed=count, total=len(index))


@app.post(
    "/search/text",
    response_model=List[SearchResult],
    summary="Search images by text query",
)
def search_by_text(
    query: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results to return"),
) -> List[SearchResult]:
    """Embed *query* with CLIP and return the *top_k* most similar indexed images."""
    results = searcher.search_by_text(query, top_k=top_k)
    return [SearchResult(path=path, score=score) for path, score in results]


@app.post(
    "/search/image",
    response_model=List[SearchResult],
    summary="Search images by uploading a query image",
)
def search_by_image(
    file: UploadFile = File(..., description="Query image file"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results to return"),
) -> List[SearchResult]:
    """Embed the uploaded image with CLIP and return the *top_k* most similar indexed images."""
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        results = searcher.search_by_image(tmp_path, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return [SearchResult(path=path, score=score) for path, score in results]


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Return index statistics",
)
def stats() -> StatsResponse:
    """Return the number of indexed images and current model configuration."""
    return StatsResponse(
        indexed_images=len(index),
        model=MODEL_NAME,
        index_path=INDEX_PATH,
    )
