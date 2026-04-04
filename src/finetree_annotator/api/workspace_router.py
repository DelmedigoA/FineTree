"""Workspace document management endpoints."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

from .deps import get_data_root
from ..workspace import (
    WorkspaceDocumentSummary,
    build_document_summary,
    delete_workspace_document,
    discover_workspace_documents,
    import_pdf_to_workspace,
    reset_document_approved_pages,
    set_document_checked,
    set_document_reviewed,
)

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


def _summary_to_dict(summary: WorkspaceDocumentSummary) -> dict[str, Any]:
    data = asdict(summary)
    for key in ("source_pdf", "images_dir", "annotations_path", "thumbnail_path"):
        val = data.get(key)
        data[key] = str(val) if val is not None else None
    return data


@router.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    summaries = discover_workspace_documents(data_root=get_data_root())
    return [_summary_to_dict(s) for s in summaries]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    try:
        summary = build_document_summary(doc_id, data_root=get_data_root())
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _summary_to_dict(summary)


@router.post("/import-pdf")
async def import_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)
        try:
            result = import_pdf_to_workspace(tmp_path, data_root=get_data_root())
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "document": _summary_to_dict(result.document),
        "copied_pdf": result.copied_pdf,
        "opened_existing": result.opened_existing,
    }


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, bool]:
    try:
        delete_workspace_document(doc_id, data_root=get_data_root())
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/documents/{doc_id}/checked")
def mark_checked(doc_id: str, checked: bool = True) -> dict[str, bool]:
    set_document_checked(doc_id, checked, data_root=get_data_root())
    return {"ok": True}


@router.post("/documents/{doc_id}/reviewed")
def mark_reviewed(doc_id: str, reviewed: bool = True) -> dict[str, bool]:
    set_document_reviewed(doc_id, reviewed, data_root=get_data_root())
    return {"ok": True}


@router.post("/documents/{doc_id}/reset-approved")
def reset_approved(doc_id: str) -> dict[str, Any]:
    data_root = get_data_root()
    annotations_path = data_root / "annotations" / f"{doc_id}.json"
    if not annotations_path.is_file():
        raise HTTPException(status_code=404, detail=f"No annotations found for {doc_id}")
    count = reset_document_approved_pages(annotations_path)
    return {"ok": True, "reset_count": count}
