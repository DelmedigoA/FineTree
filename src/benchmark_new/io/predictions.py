from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_page_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("pages"), list) and payload.get("pages"):
        first = payload["pages"][0]
        return dict(first) if isinstance(first, dict) else None
    if isinstance(payload, dict) and ("meta" in payload or "facts" in payload):
        return dict(payload)
    return None


def load_prediction_page_payload(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    parsed_page = _coerce_page_payload(raw.get("parsed_page"))
    if parsed_page is not None:
        return parsed_page
    assistant_text = raw.get("assistant_text")
    if isinstance(assistant_text, str) and assistant_text.strip():
        parsed = json.loads(assistant_text)
        page = _coerce_page_payload(parsed)
        if page is not None:
            return page
    parsed_json = _coerce_page_payload(raw.get("parsed_json"))
    if parsed_json is not None:
        return parsed_json
    raise RuntimeError(f"Could not parse prediction page payload: {path}")
