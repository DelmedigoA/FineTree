from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from ..annotation_core import normalize_bbox_data
from ..fact_normalization import normalize_fact_payload


def _resolve_annotation_files(root: Path, annotations_glob: str) -> list[Path]:
    root = root.resolve()
    pattern = Path(annotations_glob).expanduser()
    search_pattern = str(pattern) if pattern.is_absolute() else str(root / annotations_glob)
    resolved = sorted(Path(path).resolve() for path in glob.glob(search_pattern, recursive=True))
    return [path for path in resolved if path.is_file()]


def _iter_pages(payload: Any, *, default_page_name: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    pages = payload.get("pages")
    if isinstance(pages, list):
        return [page for page in pages if isinstance(page, dict)]

    facts = payload.get("facts")
    if isinstance(facts, list):
        return [{"image": default_page_name, "facts": facts}]

    return []


def fact_uniqueness_key(fact_payload: dict[str, Any]) -> tuple[Any, ...]:
    normalized_fact, _warnings = normalize_fact_payload(fact_payload, include_bbox=False)
    bbox = normalize_bbox_data(fact_payload.get("bbox"))
    path = tuple(str(p) for p in (normalized_fact.get("path") or []))
    return (
        round(float(bbox["x"]), 2),
        round(float(bbox["y"]), 2),
        round(float(bbox["w"]), 2),
        round(float(bbox["h"]), 2),
        str(normalized_fact.get("value") or ""),
        str(normalized_fact.get("comment") or ""),
        str(normalized_fact.get("is_note") if normalized_fact.get("is_note") is not None else ""),
        str(normalized_fact.get("note") or ""),
        str(normalized_fact.get("note_reference") or ""),
        str(normalized_fact.get("date") or ""),
        path,
    )


def duplicate_facts_report(root: Path, *, annotations_glob: str = "data/annotations/*.json") -> dict[str, Any]:
    root = root.resolve()
    findings: list[dict[str, Any]] = []
    pages_scanned = 0
    facts_scanned = 0
    duplicate_groups = 0
    duplicate_rows = 0

    files = _resolve_annotation_files(root, annotations_glob)
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse annotation JSON: {file_path}") from exc

        pages = _iter_pages(payload, default_page_name=file_path.name)
        for page_idx, page in enumerate(pages):
            page_name = str(page.get("image") or f"page_{page_idx + 1}")
            facts = page.get("facts")
            if not isinstance(facts, list):
                continue

            pages_scanned += 1
            facts_scanned += len(facts)

            grouped_indexes: dict[tuple[Any, ...], list[int]] = {}
            for idx, fact in enumerate(facts):
                if not isinstance(fact, dict):
                    continue
                key = fact_uniqueness_key(fact)
                grouped_indexes.setdefault(key, []).append(idx)

            for key, indexes in grouped_indexes.items():
                if len(indexes) <= 1:
                    continue
                duplicate_groups += 1
                duplicate_rows += len(indexes) - 1
                findings.append(
                    {
                        "file": str(file_path.relative_to(root)),
                        "page": page_name,
                        "indexes": indexes,
                        "bbox": [key[0], key[1], key[2], key[3]],
                        "value": key[4],
                        "comment": key[5],
                        "is_note": key[6],
                        "note": key[7],
                        "note_reference": key[8],
                        "date": key[9],
                        "path": list(key[10]),
                    }
                )

    return {
        "annotations_glob": annotations_glob,
        "files_scanned": len(files),
        "pages_scanned": pages_scanned,
        "facts_scanned": facts_scanned,
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "findings": findings,
    }


def assert_no_duplicate_facts(
    root: Path,
    *,
    annotations_glob: str = "data/annotations/*.json",
    fail_on_duplicates: bool = True,
) -> dict[str, Any]:
    report = duplicate_facts_report(root, annotations_glob=annotations_glob)
    summary = {
        "annotations_glob": report["annotations_glob"],
        "files_scanned": report["files_scanned"],
        "pages_scanned": report["pages_scanned"],
        "facts_scanned": report["facts_scanned"],
        "duplicate_groups": report["duplicate_groups"],
        "duplicate_rows": report["duplicate_rows"],
        "findings_preview": report["findings"][:5],
    }
    print("DUPLICATE_FACTS_AUDIT:", json.dumps(summary, ensure_ascii=False))

    if fail_on_duplicates and int(report["duplicate_rows"]) > 0:
        raise RuntimeError(
            "Exact duplicate facts detected in annotation JSON. "
            f"duplicate_rows={report['duplicate_rows']} duplicate_groups={report['duplicate_groups']}. "
            "Run scripts/check_duplicate_facts.py for details, or pass --allow-duplicate-facts to bypass."
        )
    return report


__all__ = [
    "assert_no_duplicate_facts",
    "duplicate_facts_report",
    "fact_uniqueness_key",
]
