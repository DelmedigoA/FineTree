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
from ..annotation_core import normalize_fact_data
from ..workspace import page_image_paths, pdf_images_root, annotations_root

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


def _stream_custom_endpoint_extract(
    *,
    image_path: Path,
    config: dict[str, Any],
    cancel_event: threading.Event,
) -> Iterator[str]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: openai") from exc

    from ..batch_qwen_inference import (
        DEFAULT_MAX_PIXELS,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        _extract_delta_text,
        _load_prompt_pair,
        _normalize_base_url,
        _prepare_image_bytes,
    )

    base_url = str(config.get("base_url") or "").strip()
    model_id = str(config.get("model_id") or "").strip()
    if not base_url or not model_id:
        raise RuntimeError("Custom endpoint requires both base_url and model_id.")

    system_prompt, instruction = _load_prompt_pair()
    image_data_uri, _original_size, _prepared_size = _prepare_image_bytes(
        image_path,
        max_pixels=int(config.get("max_pixels") or DEFAULT_MAX_PIXELS),
    )
    client = OpenAI(
        base_url=_normalize_base_url(base_url),
        api_key="unused",
        timeout=300.0,
    )
    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
        max_tokens=int(config.get("max_tokens") or DEFAULT_MAX_TOKENS),
        temperature=float(config.get("temperature") or DEFAULT_TEMPERATURE),
        stream=True,
    )

    for chunk in stream:
        if cancel_event.is_set():
            yield sse_event({"type": "cancelled"}, event="cancelled")
            return
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        text = _extract_delta_text(getattr(delta, "content", None))
        if text:
            yield sse_event({"type": "chunk", "text": text})


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
            elif request.provider == "custom_endpoint":
                yield from _stream_custom_endpoint_extract(
                    image_path=image_path,
                    config=request.config or {},
                    cancel_event=cancel_event,
                )
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
        if request.provider in (None, "gemini"):
            from ..gemini_vlm import fill_fact_fields
            result = fill_fact_fields(
                image_path=image_path,
                facts=request.facts,
                fields=request.fields,
                model_name=request.config.get("model"),
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
        from ..local_doctr import extract_numeric_bbox_facts
        from ..qwen_import_matcher import match_qwen_import_payloads

        imported_payloads: list[dict[str, Any]] = []
        for fallback_index, raw_payload in enumerate(request.facts):
            payload = dict(raw_payload)
            try:
                payload["page_index"] = int(payload.get("page_index", fallback_index))
            except (TypeError, ValueError):
                payload["page_index"] = fallback_index
            imported_payloads.append(payload)

        # Load reading_direction from document annotations.
        data_root = get_data_root()
        annotations_path = annotations_root(data_root) / f"{request.doc_id}.json"
        reading_direction = "rtl"
        if annotations_path.is_file():
            raw = json.loads(annotations_path.read_text(encoding="utf-8"))
            reading_direction = (raw.get("document_meta") or {}).get("reading_direction") or "rtl"

        # Run local detector to get candidate bboxes.
        detector_payloads = extract_numeric_bbox_facts(image_path)

        # Align imported facts against detected bboxes.
        aligned, _ = match_qwen_import_payloads(
            page_name=request.page_name,
            imported_payloads=imported_payloads,
            detector_payloads=detector_payloads,
            reading_direction=reading_direction,
        )
        aligned_facts = [
            {
                "page_index": int(payload.get("page_index", index)),
                "bbox": payload.get("bbox"),
                "fact": normalize_fact_data(payload),
            }
            for index, payload in enumerate(aligned)
        ]
        return {"aligned_facts": aligned_facts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cancel")
async def cancel_session(request: CancelRequest) -> dict[str, bool]:
    event = _cancel_events.get(request.session_id)
    if event is not None:
        event.set()
    return {"ok": True}
