from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def create_annotation_backup(
    root: Path,
    source_path: Path,
    *,
    reason: str,
    algo_version: str,
    direction_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    source = source_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cannot backup missing source file: {source}")
    try:
        relative_source = source.relative_to(root)
    except ValueError:
        relative_source = source

    source_text = source.read_text(encoding="utf-8")
    digest8 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:8]
    stamp = utc_timestamp_compact()
    backup_base = root / "data" / "annotations" / "_backup"
    backup_dir = backup_base / source.stem
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{source.stem}.{stamp}.{digest8}.json"
    backup_path.write_text(source_text, encoding="utf-8")

    manifest_path = backup_base / "manifest.jsonl"
    relative_backup = str(backup_path.relative_to(root))
    relative_original = str(relative_source)
    record: dict[str, Any] = {
        # Spec-aligned keys.
        "timestamp": stamp,
        "original_path": relative_original,
        "backup_path": relative_backup,
        "reason": reason,
        "algo_version": algo_version,
        # Backward-compatible aliases for older tooling.
        "timestamp_utc": stamp,
        "source_path": relative_original,
    }
    if direction_source:
        record["direction_source"] = direction_source
    if extra:
        record.update(extra)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "timestamp_utc": stamp,
        "timestamp": stamp,
        "source_path": relative_original,
        "original_path": relative_original,
        "backup_path": str(backup_path),
        "manifest_path": str(manifest_path),
    }


__all__ = [
    "atomic_write_text",
    "create_annotation_backup",
    "utc_timestamp_compact",
]
