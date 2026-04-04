"""AI inference endpoints with SSE streaming."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .deps import get_data_root
from .sse_helpers import sse_event, sse_response
from ..workspace import page_image_paths, pdf_images_root

router = APIRouter(prefix="/api/ai", tags=["ai"])

_cancel_events: dict[str, threading.Event] = {}


def _images_dir(doc_id: str) -> Path:
    return pdf_images_root(get_data_root()) / doc_id


def _page_image_path(doc_id: str, page_name: str) -> Path:
    path = _images_dir(doc_id) / page_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Page image not found: {page_name}")
    return path


class ExtractRequest(BaseModel):
    doc_id: str
    page_name: str
    provider: str = "gemini"
    action: str = "gt"
    config: dict[str, Any] = {}


class DetectBboxRequest(BaseModel):
    doc_id: str
    page_name: str
    backend: str = "merged"


class FillRequest(BaseModel):
    doc_id: str
    page_name: str
    facts: list[dict[str, Any]]
    fields: list[str]
    provider: str = "gemini"
    config: dict[str, Any] = {}


class FixSpellingRequest(BaseModel):
    doc_id: str
    page_name: str


class AlignBboxesRequest(BaseModel):
    doc_id: str
    page_name: str
    facts: list[dict[str, Any]]


class CancelRequest(BaseModel):
    session_id: str


@router.post("/extract")
async def extract_stream(request: ExtractRequest):
    image_path = _page_image_path(request.doc_id, request.page_name)
    session_id = f"extract:{request.doc_id}:{request.page_name}"
    cancel_event = threading.Event()
    _cancel_events[session_id] = cancel_event

    def generate() -> Iterator[str]:
        try:
            if request.provider == "gemini":
                from ..gemini_vlm import stream_page_extraction_response
                for chunk in stream_page_extraction_response(
                    image_path=image_path,
                    model_name=request.config.get("model"),
                    temperature=request.config.get("temperature"),
                    enable_thinking=request.config.get("enable_thinking", False),
                ):
                    if cancel_event.is_set():
                        yield sse_event({"type": "cancelled"}, event="cancelled")
                        return
                    yield sse_event({"type": "chunk", "text": chunk})
            elif request.provider == "qwen":
                from ..qwen_vlm import stream_content_from_image
                config = request.config or {}
                for chunk in stream_content_from_image(
                    image_path=image_path,
                    model_id=config.get("model_id"),
                    enable_thinking=config.get("enable_thinking", False),
                ):
                    if cancel_event.is_set():
                        yield sse_event({"type": "cancelled"}, event="cancelled")
                        return
                    yield sse_event({"type": "chunk", "text": chunk})
            else:
                yield sse_event({"type": "error", "message": f"Unknown provider: {request.provider}"}, event="error")
                return
            yield sse_event({"type": "done"}, event="done")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="error")
        finally:
            _cancel_events.pop(session_id, None)

    return sse_response(generate())


@router.post("/detect-bbox")
async def detect_bbox_stream(request: DetectBboxRequest):
    image_path = _page_image_path(request.doc_id, request.page_name)
    session_id = f"detect:{request.doc_id}:{request.page_name}"
    cancel_event = threading.Event()
    _cancel_events[session_id] = cancel_event

    def generate() -> Iterator[str]:
        try:
            from ..local_doctr import detect_bboxes_on_image
            results = detect_bboxes_on_image(image_path, backend=request.backend)
            for bbox_data in results:
                if cancel_event.is_set():
                    yield sse_event({"type": "cancelled"}, event="cancelled")
                    return
                yield sse_event({"type": "bbox", "data": bbox_data})
            yield sse_event({"type": "done"}, event="done")
        except Exception as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="error")
        finally:
            _cancel_events.pop(session_id, None)

    return sse_response(generate())


@router.post("/fill")
async def fill_facts(request: FillRequest) -> dict[str, Any]:
    image_path = _page_image_path(request.doc_id, request.page_name)
    try:
        if request.provider == "gemini":
            from ..gemini_vlm import fill_fact_fields
            result = fill_fact_fields(
                image_path=image_path,
                facts=request.facts,
                fields=request.fields,
                model_name=request.config.get("model"),
            )
            return {"patch": result}
        elif request.provider == "qwen":
            from ..qwen_vlm import fill_fact_fields as qwen_fill
            result = qwen_fill(
                image_path=image_path,
                facts=request.facts,
                fields=request.fields,
                model_id=request.config.get("model_id"),
            )
            return {"patch": result}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/fix-spelling")
async def fix_spelling(request: FixSpellingRequest) -> dict[str, Any]:
    image_path = _page_image_path(request.doc_id, request.page_name)
    try:
        from ..gemini_vlm import correct_page_json_hebrew_spelling
        corrected = correct_page_json_hebrew_spelling(image_path=image_path)
        return {"corrected_page": corrected}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/align-bboxes")
async def align_bboxes(request: AlignBboxesRequest) -> dict[str, Any]:
    image_path = _page_image_path(request.doc_id, request.page_name)
    try:
        from ..qwen_import_matcher import match_qwen_import_payloads
        aligned = match_qwen_import_payloads(
            image_path=image_path,
            import_facts=request.facts,
        )
        return {"aligned_facts": aligned}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cancel")
async def cancel_session(request: CancelRequest) -> dict[str, bool]:
    event = _cancel_events.get(request.session_id)
    if event is not None:
        event.set()
    return {"ok": True}
