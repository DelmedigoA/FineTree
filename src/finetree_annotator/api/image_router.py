"""Page image serving endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .deps import get_data_root
from ..workspace import IMAGE_SUFFIXES, page_image_paths, pdf_images_root

router = APIRouter(prefix="/api/images", tags=["images"])


def _images_dir(doc_id: str) -> Path:
    return pdf_images_root(get_data_root()) / doc_id


@router.get("/{doc_id}/pages")
def list_pages(doc_id: str) -> list[str]:
    images_dir = _images_dir(doc_id)
    if not images_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No images directory for {doc_id}")
    return [p.name for p in page_image_paths(images_dir)]


@router.get("/{doc_id}/pages/{page_name}")
def get_page_image(doc_id: str, page_name: str) -> FileResponse:
    image_path = _images_dir(doc_id) / page_name
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail=f"Image not found: {page_name}")
    return FileResponse(
        image_path,
        media_type=f"image/{image_path.suffix.lstrip('.').lower()}",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{doc_id}/thumbnails/{page_name}")
def get_thumbnail(doc_id: str, page_name: str) -> FileResponse:
    image_path = _images_dir(doc_id) / page_name
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail=f"Image not found: {page_name}")
    # Serve the full image for now; the frontend handles scaling.
    # A future optimization can generate and cache smaller thumbnails server-side.
    return FileResponse(
        image_path,
        media_type=f"image/{image_path.suffix.lstrip('.').lower()}",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
