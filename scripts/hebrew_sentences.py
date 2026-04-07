#!/usr/bin/env python3
"""
List all Hebrew strings per page across the workspace.

Output format (tab-separated):
    doc_id  page_name  hebrew_text

Run:
    python scripts/hebrew_sentences.py
    python scripts/hebrew_sentences.py --data-root /path/to/data
    python scripts/hebrew_sentences.py --doc my_doc_id   # single doc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_DATA_ROOT = REPO_ROOT / "data"
HEBREW_RANGE = range(0x0590, 0x0600)  # Unicode Hebrew block


def _has_hebrew(text: str) -> bool:
    return any(ord(c) in HEBREW_RANGE for c in text)


def _collect_strings(obj: object) -> list[str]:
    """Recursively collect all non-empty string values from any JSON structure."""
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        result: list[str] = []
        for item in obj:
            result.extend(_collect_strings(item))
        return result
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(_collect_strings(v))
        return result
    return []


def hebrew_strings_for_page(page: dict) -> list[str]:
    """Extract every Hebrew string found anywhere in a page dict."""
    seen: set[str] = set()
    result: list[str] = []
    for s in _collect_strings(page):
        s = s.strip()
        if s and _has_hebrew(s) and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def process_annotation_file(path: Path) -> list[tuple[str, str, list[str]]]:
    """
    Returns list of (doc_id, page_name, hebrew_list) for every page that has Hebrew text.
    """
    doc_id = path.stem
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return []

    pages = raw.get("pages", [])
    rows: list[tuple[str, str, list[str]]] = []
    for idx, page in enumerate(pages):
        # Use page_num from meta if available, otherwise fall back to index
        meta = page.get("meta") or {}
        page_num = meta.get("page_num") or str(idx + 1)
        page_name = f"page_{int(page_num):04d}" if str(page_num).isdigit() else str(page_num)

        hebrew = hebrew_strings_for_page(page)
        if hebrew:
            rows.append((doc_id, page_name, hebrew))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="List Hebrew strings per page.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--doc",
        metavar="DOC_ID",
        help="Process a single document only.",
    )
    args = parser.parse_args()

    ann_root: Path = args.data_root / "annotations"
    if not ann_root.is_dir():
        print(f"[ERROR] Annotations directory not found: {ann_root}", file=sys.stderr)
        sys.exit(1)

    if args.doc:
        files = [ann_root / f"{args.doc}.json"]
    else:
        files = sorted(ann_root.glob("*.json"))

    total_pages = 0
    for path in files:
        if not path.is_file():
            print(f"[WARN] Not found: {path}", file=sys.stderr)
            continue
        rows = process_annotation_file(path)
        for doc_id, page_name, hebrew_list in rows:
            total_pages += 1
            print(f"{doc_id}\t{page_name}")
            for text in hebrew_list:
                print(f"    • {text}")
            print()

    print(f"— {total_pages} pages with Hebrew text across {len(files)} document(s) —", file=sys.stderr)


if __name__ == "__main__":
    main()
