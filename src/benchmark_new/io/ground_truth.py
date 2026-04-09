from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workspace_paths import DEFAULT_DATA_ROOT, resolve_annotations_path


def load_ground_truth_document(doc_id: str, *, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    path = resolve_annotations_path(doc_id, data_root=Path(data_root))
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth annotations not found for {doc_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_ground_truth_page(doc_id: str, page_index: int, *, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    document = load_ground_truth_document(doc_id, data_root=data_root)
    pages = document.get("pages") if isinstance(document.get("pages"), list) else []
    if page_index < 1 or page_index > len(pages):
        raise IndexError(f"Ground-truth page {page_index} is out of range for {doc_id}")
    page = pages[page_index - 1]
    if not isinstance(page, dict):
        raise TypeError(f"Ground-truth page {page_index} is not an object for {doc_id}")
    return page
