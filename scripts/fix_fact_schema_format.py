#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.annotation_backups import atomic_write_text, create_annotation_backup  # noqa: E402
from finetree_annotator.fact_normalization import normalize_annotation_payload  # noqa: E402

ALGO_VERSION = "fact_schema_format_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite annotation JSON into canonical fact schema with normalized date/value fields."
    )
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
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
    facts_changed = 0
    backups_written = 0

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        normalized, findings = normalize_annotation_payload(payload)
        if normalized == payload:
            continue

        files_changed += 1
        facts_changed += len(findings)
        if args.dry_run:
            print(f"DRY_RUN_CHANGE: file={file_path} fact_issues={len(findings)}")
            continue

        backup_info = create_annotation_backup(
            root,
            file_path,
            reason="fact_schema_format_fix",
            algo_version=ALGO_VERSION,
            direction_source="schema_normalization",
            extra={"facts_with_issues": len(findings)},
        )
        backups_written += 1
        serialized = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(file_path, serialized)
        print(
            f"FIXED: file={file_path} fact_issues={len(findings)} "
            f"backup={backup_info['backup_path']}"
        )

    print(
        "FIX_FACT_SCHEMA_FORMAT_SUMMARY: "
        f"files_scanned={len(files)} files_changed={files_changed} "
        f"facts_changed={facts_changed} backups_written={backups_written} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

