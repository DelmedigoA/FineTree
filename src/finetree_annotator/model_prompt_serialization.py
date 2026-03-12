from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Optional

SchemaMode = Literal["model_prompt", "session_storage", "legacy_import"]

MODEL_PROMPT_MODE: SchemaMode = "model_prompt"
SESSION_STORAGE_MODE: SchemaMode = "session_storage"
LEGACY_IMPORT_MODE: SchemaMode = "legacy_import"


def build_schema_mode_payload(
    *,
    pages: Iterable[dict[str, Any]],
    mode: SchemaMode,
    images_dir: Optional[str] = None,
    document_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    page_list = [dict(page) for page in pages if isinstance(page, dict)]
    if mode == MODEL_PROMPT_MODE:
        return {"pages": page_list}
    if mode == LEGACY_IMPORT_MODE:
        return {
            "document_meta": dict(document_meta or {}),
            "pages": page_list,
        }
    return {
        "images_dir": str(images_dir).strip() or None if images_dir is not None else None,
        "metadata": dict(document_meta or {}),
        "pages": page_list,
    }


def build_single_page_payload(
    *,
    page_name: str,
    page_meta: dict[str, Any],
    facts: list[dict[str, Any]],
    mode: SchemaMode = MODEL_PROMPT_MODE,
    images_dir: Optional[str] = None,
    document_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return build_schema_mode_payload(
        pages=[
            {
                "image": str(page_name or "").strip() or None,
                "meta": dict(page_meta or {}),
                "facts": [dict(fact) for fact in facts if isinstance(fact, dict)],
            }
        ],
        mode=mode,
        images_dir=images_dir,
        document_meta=document_meta,
    )


def serialize_schema_mode_payload(payload: dict[str, Any], *, indent: Optional[int] = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def coerce_payload_for_schema_mode(payload: Any, *, mode: SchemaMode, default_page_name: Optional[str] = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    pages = payload.get("pages")
    if isinstance(pages, list):
        page_dicts = [dict(page) for page in pages if isinstance(page, dict)]
    else:
        page_dicts = []
    if not page_dicts and ("meta" in payload or "facts" in payload):
        page_dicts = [
            {
                "image": str(payload.get("image") or default_page_name or "").strip() or None,
                "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
                "facts": payload.get("facts") if isinstance(payload.get("facts"), list) else [],
            }
        ]
    if not page_dicts:
        raise ValueError("payload has no pages to serialize")
    return build_schema_mode_payload(
        pages=page_dicts,
        mode=mode,
        images_dir=str(payload.get("images_dir") or "").strip() or None,
        document_meta=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload.get("document_meta"),
    )


__all__ = [
    "LEGACY_IMPORT_MODE",
    "MODEL_PROMPT_MODE",
    "SESSION_STORAGE_MODE",
    "SchemaMode",
    "build_schema_mode_payload",
    "build_single_page_payload",
    "coerce_payload_for_schema_mode",
    "serialize_schema_mode_payload",
]
