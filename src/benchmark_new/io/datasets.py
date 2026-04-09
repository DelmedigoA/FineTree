from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import DatasetVersionInfo, DocumentInput, PageInput
from .workspace_paths import DEFAULT_DATA_ROOT, page_image_paths


@dataclass(frozen=True)
class DatasetSelection:
    dataset: DatasetVersionInfo
    split: str
    documents: tuple[DocumentInput, ...]


def _datasets_dir(data_root: Path) -> Path:
    return Path(data_root) / "datasets"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_dataset_versions(data_root: Path = DEFAULT_DATA_ROOT) -> list[DatasetVersionInfo]:
    versions: list[DatasetVersionInfo] = []
    for path in sorted(_datasets_dir(Path(data_root)).glob("*.json")):
        raw = _load_json(path)
        versions.append(
            DatasetVersionInfo(
                version_id=str(raw.get("version_id") or path.stem),
                name=str(raw.get("name") or path.stem),
                created_at=float(raw.get("created_at") or 0.0),
                updated_at=float(raw["updated_at"]) if raw.get("updated_at") is not None else None,
                split_assignments={str(key): str(value) for key, value in dict(raw.get("split_assignments") or {}).items()},
                split_stats={str(key): dict(value or {}) for key, value in dict(raw.get("split_stats") or {}).items()},
                export_config=dict(raw.get("export_config") or {}),
                path=path,
            )
        )
    versions.sort(key=lambda item: (item.updated_at or item.created_at, item.name), reverse=True)
    return versions


def _resolve_dataset_version(version_id: str, data_root: Path) -> DatasetVersionInfo:
    for version in list_dataset_versions(data_root=data_root):
        if version.version_id == version_id:
            return version
    raise FileNotFoundError(f"Dataset version not found: {version_id}")


def _annotation_status(page_payload: dict[str, Any]) -> str:
    meta = page_payload.get("meta") if isinstance(page_payload.get("meta"), dict) else {}
    return str(meta.get("annotation_status") or "").strip().lower()


def _load_document_input(doc_id: str, *, data_root: Path, approved_pages_only: bool) -> DocumentInput | None:
    annotation_path = Path(data_root) / "annotations" / f"{doc_id}.json"
    images_dir = Path(data_root) / "pdf_images" / doc_id
    if not annotation_path.is_file() or not images_dir.is_dir():
        return None
    payload = _load_json(annotation_path)
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    image_paths = {path.name: path for path in page_image_paths(images_dir)}
    selected_pages: list[PageInput] = []
    for page_index, page_payload in enumerate(pages, start=1):
        if not isinstance(page_payload, dict):
            continue
        image_name = str(page_payload.get("image") or "").strip()
        if not image_name or image_name not in image_paths:
            continue
        if approved_pages_only and _annotation_status(page_payload) != "approved":
            continue
        selected_pages.append(
            PageInput(
                page_index=page_index,
                image_name=image_name,
                image_path=image_paths[image_name],
            )
        )
    if not selected_pages:
        return None
    return DocumentInput(
        doc_id=doc_id,
        annotation_path=annotation_path,
        images_dir=images_dir,
        pages=tuple(selected_pages),
    )


def load_dataset_selection(
    version_id: str,
    *,
    split: str = "val",
    data_root: Path = DEFAULT_DATA_ROOT,
) -> DatasetSelection:
    dataset = _resolve_dataset_version(version_id, data_root=Path(data_root))
    approved_pages_only = bool((dataset.export_config or {}).get("approved_pages_only", True))
    documents: list[DocumentInput] = []
    for doc_id, assigned_split in dataset.split_assignments.items():
        if assigned_split != split:
            continue
        document = _load_document_input(
            doc_id,
            data_root=Path(data_root),
            approved_pages_only=approved_pages_only,
        )
        if document is not None:
            documents.append(document)
    documents.sort(key=lambda item: item.doc_id)
    return DatasetSelection(dataset=dataset, split=split, documents=tuple(documents))
