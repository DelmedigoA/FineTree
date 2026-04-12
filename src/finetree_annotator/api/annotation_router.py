"""Annotation CRUD, validation, and normalization endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .deps import get_data_root
from ..annotation_backups import atomic_write_text, create_annotation_backup
from ..annotation_core import (
    PageState,
    build_annotations_payload_with_findings,
    default_page_meta,
    extract_document_meta,
    load_page_states,
    serialize_annotations_json,
)
from ..page_issues import validate_document_issues
from ..schema_io import EquationIntegrityError, canonicalize_with_findings, load_any_schema
from ..workspace import annotations_root, page_image_paths, pdf_images_root

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


def _resolve_paths(doc_id: str) -> tuple[Path, Path]:
    data_root = get_data_root()
    images_dir = pdf_images_root(data_root) / doc_id
    annotations_path = annotations_root(data_root) / f"{doc_id}.json"
    return images_dir, annotations_path


def _load_document_payload(annotations_path: Path) -> dict[str, Any]:
    if not annotations_path.is_file():
        return {"pages": []}
    text = annotations_path.read_text(encoding="utf-8")
    raw = json.loads(text)
    return load_any_schema(raw)


@router.get("/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    images_dir, annotations_path = _resolve_paths(doc_id)
    payload = _load_document_payload(annotations_path)
    page_paths = page_image_paths(images_dir)
    page_names = [p.name for p in page_paths]
    states = load_page_states(payload, page_names, placeholder_missing_bboxes=True)
    document_meta = extract_document_meta(payload)

    # Guarantee an entry for EVERY page image on disk, even when the annotation
    # file is missing or incomplete. Without this the frontend inspector hides
    # Page/Facts/Edit sections entirely and the document becomes uneditable.
    page_states_response: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(page_names):
        state = states.get(name)
        if state is None:
            page_states_response[name] = {
                "meta": default_page_meta(index),
                "facts": [],
            }
        else:
            page_states_response[name] = {
                "meta": state.meta,
                "facts": [
                    {"bbox": box.bbox, "fact": box.fact}
                    for box in state.facts
                ],
            }

    return {
        "images_dir": str(images_dir),
        "page_images": page_names,
        "document_meta": document_meta,
        "page_states": page_states_response,
    }


class SaveDocumentRequest(BaseModel):
    document_meta: dict[str, Any] | None = None
    page_states: dict[str, Any] | None = None
    raw_payload: dict[str, Any] | None = None


@router.put("/{doc_id}")
def save_document(doc_id: str, request: SaveDocumentRequest) -> dict[str, Any]:
    images_dir, annotations_path = _resolve_paths(doc_id)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    if request.raw_payload is not None:
        payload = request.raw_payload
    else:
        page_paths = page_image_paths(images_dir)
        page_names = [p.name for p in page_paths]
        page_states: dict[str, PageState] = {}
        raw_states = request.page_states or {}
        for page_name, state_data in raw_states.items():
            if not isinstance(state_data, dict):
                continue
            meta = state_data.get("meta", {})
            facts_raw = state_data.get("facts", [])
            from ..annotation_core import BoxRecord, normalize_bbox_data, normalize_fact_data
            facts = []
            for entry in facts_raw:
                if not isinstance(entry, dict):
                    continue
                bbox = normalize_bbox_data(entry.get("bbox"))
                fact = normalize_fact_data(entry.get("fact"))
                facts.append(BoxRecord(bbox=bbox, fact=fact))
            page_states[page_name] = PageState(meta=meta, facts=facts)
        payload, equation_findings = build_annotations_payload_with_findings(
            images_dir,
            page_paths,
            page_states,
            document_meta=request.document_meta,
        )

    try:
        canonical, equation_findings = canonicalize_with_findings(
            payload,
            strict_equation_guards=True,
        )
    except EquationIntegrityError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "findings": exc.findings},
        ) from exc

    json_text = serialize_annotations_json(canonical)
    if annotations_path.is_file():
        create_annotation_backup(
            get_data_root(),
            annotations_path,
            reason="web_save",
            algo_version="web_annotator_v1",
        )
    atomic_write_text(annotations_path, json_text)
    return {"ok": True, "warnings": equation_findings if equation_findings else []}


@router.get("/{doc_id}/pages/{page_name}")
def get_page(doc_id: str, page_name: str) -> dict[str, Any]:
    images_dir, annotations_path = _resolve_paths(doc_id)
    payload = _load_document_payload(annotations_path)
    page_paths = page_image_paths(images_dir)
    page_names = [p.name for p in page_paths]
    states = load_page_states(payload, page_names, placeholder_missing_bboxes=True)
    state = states.get(page_name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Page {page_name} not found")
    return {
        "meta": state.meta,
        "facts": [{"bbox": box.bbox, "fact": box.fact} for box in state.facts],
    }


class ValidateRequest(BaseModel):
    document_meta: dict[str, Any] | None = None
    page_states: dict[str, Any] | None = None


@router.post("/{doc_id}/validate")
def validate_document(doc_id: str) -> dict[str, Any]:
    images_dir, annotations_path = _resolve_paths(doc_id)
    payload = _load_document_payload(annotations_path)
    page_paths = page_image_paths(images_dir)
    page_names = [p.name for p in page_paths]
    states = load_page_states(payload, page_names, placeholder_missing_bboxes=True)
    document_meta = extract_document_meta(payload)
    ordered_states = [
        (name, states.get(name, PageState(meta={}, facts=[])))
        for name in page_names
    ]
    issue_summary = validate_document_issues(ordered_states)
    return {
        "reg_flag_count": issue_summary.reg_flag_count,
        "warning_count": issue_summary.warning_count,
        "pages": {
            page_name: {
                "reg_flags": [
                    {"code": issue.code, "message": issue.message, "fact_index": issue.fact_index}
                    for issue in page_summary.issues if issue.severity == "reg_flag"
                ],
                "warnings": [
                    {"code": issue.code, "message": issue.message, "fact_index": issue.fact_index}
                    for issue in page_summary.issues if issue.severity == "warning"
                ],
            }
            for page_name, page_summary in issue_summary.page_summaries.items()
        },
    }


@router.post("/{doc_id}/normalize")
def normalize_document(doc_id: str) -> dict[str, Any]:
    _images_dir, annotations_path = _resolve_paths(doc_id)
    payload = _load_document_payload(annotations_path)
    normalized = load_any_schema(payload)
    return normalized
