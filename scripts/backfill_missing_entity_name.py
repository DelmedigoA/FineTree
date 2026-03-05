#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _iter_annotation_files(root: Path, pattern: str) -> list[Path]:
    try:
        candidates = [Path(p) for p in root.glob(pattern)]
    except (ValueError, NotImplementedError):
        candidates = [Path(p) for p in glob.glob(str(root / pattern))]
    return sorted([p for p in candidates if p.is_file()])


def _pick_source_entity(payload: dict[str, Any], override_entity: str | None) -> str | None:
    if override_entity:
        value = override_entity.strip()
        return value or None

    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    for page in pages:
        if not isinstance(page, dict):
            continue
        meta = page.get("meta")
        if not isinstance(meta, dict):
            continue
        entity = str(meta.get("entity_name") or "").strip()
        if entity:
            return entity
    return None


def _backfill_missing_entity_in_payload(payload: dict[str, Any], source_entity: str) -> int:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return 0

    updated = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        meta = page.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            page["meta"] = meta
        entity = str(meta.get("entity_name") or "").strip()
        if entity:
            continue
        meta["entity_name"] = source_entity
        updated += 1
    return updated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill: set page.meta.entity_name only for pages where it is empty. "
            "Existing non-empty entity_name values are never overwritten."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json")
    parser.add_argument(
        "--entity-name",
        default=None,
        help="Optional fixed entity_name to apply. If omitted, uses first non-empty entity_name found per file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print report without writing files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    files = _iter_annotation_files(root, str(args.annotations_glob))

    files_changed = 0
    pages_updated = 0
    files_skipped_no_source = 0
    per_file: list[dict[str, Any]] = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            per_file.append({"file": str(path), "error": f"parse_error: {exc}"})
            continue
        if not isinstance(payload, dict):
            per_file.append({"file": str(path), "error": "not_a_json_object"})
            continue

        source_entity = _pick_source_entity(payload, args.entity_name)
        if not source_entity:
            files_skipped_no_source += 1
            per_file.append({"file": str(path), "updated_pages": 0, "skipped": "no_source_entity"})
            continue

        updated = _backfill_missing_entity_in_payload(payload, source_entity)
        pages_updated += updated
        if updated > 0:
            files_changed += 1
            if not args.dry_run:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        per_file.append({"file": str(path), "updated_pages": updated, "source_entity": source_entity})

    report = {
        "root": str(root),
        "annotations_glob": str(args.annotations_glob),
        "dry_run": bool(args.dry_run),
        "files_scanned": len(files),
        "files_changed": files_changed,
        "pages_updated": pages_updated,
        "files_skipped_no_source": files_skipped_no_source,
        "per_file": per_file,
    }
    print("ENTITY_NAME_BACKFILL_REPORT:", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
