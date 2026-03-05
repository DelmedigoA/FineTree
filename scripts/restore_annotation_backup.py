#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.annotation_backups import atomic_write_text, create_annotation_backup  # noqa: E402

ALGO_VERSION = "restore_annotation_backup_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore an annotation JSON file from a backup snapshot.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--source", required=True, help="Annotation JSON path to restore.")
    parser.add_argument("--backup", required=True, help="Backup JSON snapshot path to restore from.")
    return parser.parse_args(argv)


def _resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    source_path = _resolve_path(root, args.source)
    backup_path = _resolve_path(root, args.backup)
    if not backup_path.is_file():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    pre_backup = create_annotation_backup(
        root,
        source_path,
        reason="restore_pre_state",
        algo_version=ALGO_VERSION,
        direction_source="manual_restore",
        extra={"restore_from": str(backup_path)},
    )
    restored_text = backup_path.read_text(encoding="utf-8")
    atomic_write_text(source_path, restored_text)
    print(
        "RESTORED: "
        f"source={source_path} from_backup={backup_path} "
        f"pre_restore_backup={pre_backup['backup_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
