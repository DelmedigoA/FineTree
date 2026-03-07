#!/usr/bin/env python3
from __future__ import annotations

"""
One-time migration: add period_* fields for facts with legacy "date".

Behavior:
- Preserve "date" in all facts.
- Add period_type/period_start/period_end (null if missing).
- If period_* are missing, fill them heuristically from "date".
- Remove suggested_period_* keys when present.

The migration is deterministic and idempotent.
"""

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
from finetree_annotator.migrations.date_to_period import migrate_payload  # noqa: E402

ALGO_VERSION = "date_to_period_v1"
DEFAULT_JSON_GLOB = "**/*.json"
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    ".env",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy date fields to period_* fields.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--json-glob", default=DEFAULT_JSON_GLOB, help="Glob pattern for JSON files to scan.")
    parser.add_argument(
        "--exclude-backups",
        action="store_true",
        help="Exclude data/annotations/_backup from the scan.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files.")
    return parser.parse_args(argv)


def _resolve_files(root: Path, json_glob: str, *, exclude_backups: bool) -> list[Path]:
    pattern = Path(json_glob).expanduser()
    query = str(pattern) if pattern.is_absolute() else str(root / json_glob)
    candidates = [Path(path).resolve() for path in glob.glob(query, recursive=True) if Path(path).is_file()]
    filtered: list[Path] = []
    for path in candidates:
        if exclude_backups and "_backup" in path.parts:
            continue
        if any(part in DEFAULT_EXCLUDE_PARTS for part in path.parts):
            continue
        filtered.append(path)
    return sorted(set(filtered))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    files = _resolve_files(root, args.json_glob, exclude_backups=args.exclude_backups)

    files_changed = 0
    facts_scanned = 0
    facts_with_date = 0
    facts_updated = 0
    fields_added = 0
    fields_removed = 0
    suggestions_added = 0
    backups_written = 0

    for file_path in files:
        try:
            payload = _load_json(file_path)
        except Exception as exc:
            print(f"SKIP_PARSE_ERROR: file={file_path} error={exc}")
            continue

        stats = migrate_payload(payload)
        facts_scanned += stats.facts_scanned
        facts_with_date += stats.facts_with_date
        facts_updated += stats.facts_updated
        fields_added += stats.fields_added
        fields_removed += stats.fields_removed
        suggestions_added += stats.suggestions_added

        if not stats.changed:
            continue

        files_changed += 1
        if args.dry_run:
            print(f"DRY_RUN_CHANGE: file={file_path} facts_updated={stats.facts_updated}")
            continue

        backup_info = create_annotation_backup(
            root,
            file_path,
            reason="date_to_period_migration",
            algo_version=ALGO_VERSION,
            direction_source="date_to_period",
            extra={
                "facts_updated": stats.facts_updated,
                "fields_added": stats.fields_added,
                "fields_removed": stats.fields_removed,
                "suggestions_added": stats.suggestions_added,
            },
        )
        backups_written += 1
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(file_path, serialized)
        print(
            f"MIGRATED: file={file_path} facts_updated={stats.facts_updated} "
            f"backup={backup_info['backup_path']}"
        )

    print(
        "MIGRATE_DATE_TO_PERIOD_SUMMARY: "
        f"files_scanned={len(files)} files_changed={files_changed} "
        f"facts_scanned={facts_scanned} facts_with_date={facts_with_date} "
        f"facts_updated={facts_updated} fields_added={fields_added} "
        f"fields_removed={fields_removed} suggestions_added={suggestions_added} "
        f"backups_written={backups_written} "
        f"dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
