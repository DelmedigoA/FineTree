from __future__ import annotations

import re
import unicodedata
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def page_image_paths(images_dir: Path) -> list[Path]:
    if not Path(images_dir).is_dir():
        return []
    return sorted(path for path in Path(images_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sanitize_doc_id(raw_name: str) -> str:
    text = unicodedata.normalize("NFC", str(raw_name or "").strip())
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "document"


def resolve_annotations_path(doc_id: str, *, data_root: Path) -> Path:
    root = Path(data_root) / "annotations"
    direct = root / f"{doc_id}.json"
    if direct.is_file():
        return direct
    normalized = sanitize_doc_id(doc_id)
    normalized_path = root / f"{normalized}.json"
    if normalized_path.is_file():
        return normalized_path
    stem_matches = sorted(path for path in root.glob("*.json") if path.stem == doc_id or path.stem == normalized)
    if stem_matches:
        return stem_matches[0]
    return direct
