#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.annotation_backups import atomic_write_text, create_annotation_backup  # noqa: E402
from finetree_annotator.fact_ordering import (  # noqa: E402
    canonical_fact_order_indices,
    resolve_reading_direction,
)

ALGO_VERSION = "fact_ordering_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite annotation JSON with canonical reading-order for facts.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
    parser.add_argument("--default-direction", choices=["rtl", "ltr"], default="rtl", help="Fallback direction on uncertain language.")
    parser.add_argument("--row-tolerance-ratio", type=float, default=0.35, help="Row grouping tolerance ratio based on median bbox height.")
    parser.add_argument("--row-tolerance-min-px", type=float, default=6.0, help="Minimum row grouping tolerance in pixels.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files.")
    return parser.parse_args(argv)


def _resolve_files(root: Path, annotations_glob: str) -> list[Path]:
    pattern = Path(annotations_glob).expanduser()
    query = str(pattern) if pattern.is_absolute() else str(root / annotations_glob)
    return [Path(path).resolve() for path in sorted(glob.glob(query, recursive=True)) if Path(path).is_file()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    files = _resolve_files(root, args.annotations_glob)
    files_changed = 0
    pages_changed = 0
    backups_written = 0

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        direction_info = resolve_reading_direction(
            payload.get("metadata", payload.get("document_meta")),
            payload=payload,
            default_direction=args.default_direction,
        )
        direction = "rtl" if str(direction_info.get("direction")) == "rtl" else "ltr"
        pages = payload.get("pages")
        if not isinstance(pages, list):
            continue

        changed_here = 0
        for page in pages:
            if not isinstance(page, dict):
                continue
            facts = page.get("facts")
            if not isinstance(facts, list) or len(facts) <= 1:
                continue
            ordered_indexes = canonical_fact_order_indices(
                facts,  # type: ignore[arg-type]
                direction=direction,
                row_tolerance_ratio=args.row_tolerance_ratio,
                row_tolerance_min_px=args.row_tolerance_min_px,
            )
            if ordered_indexes == list(range(len(facts))):
                continue
            page["facts"] = [facts[idx] for idx in ordered_indexes if 0 <= idx < len(facts)]
            changed_here += 1

        if changed_here == 0:
            continue

        files_changed += 1
        pages_changed += changed_here
        if args.dry_run:
            print(f"DRY_RUN_CHANGE: file={file_path} pages_changed={changed_here} direction={direction}")
            continue

        backup_info = create_annotation_backup(
            root,
            file_path,
            reason="fact_order_fix",
            algo_version=ALGO_VERSION,
            direction_source=str(direction_info.get("source") or ""),
            extra={
                "direction_used": direction,
                "direction_uncertain": bool(direction_info.get("uncertain")),
                "pages_changed": changed_here,
            },
        )
        backups_written += 1
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(file_path, serialized)
        print(
            f"FIXED: file={file_path} pages_changed={changed_here} direction={direction} "
            f"backup={backup_info['backup_path']}"
        )

    print(
        "FIX_FACT_ORDER_SUMMARY: "
        f"files_scanned={len(files)} files_changed={files_changed} pages_changed={pages_changed} "
        f"backups_written={backups_written} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
