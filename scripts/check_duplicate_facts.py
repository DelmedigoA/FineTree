#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.finetune.duplicate_facts import assert_no_duplicate_facts  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check annotation JSON files for exact duplicate facts.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Exit successfully even when duplicates are found (still prints report).",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    report = assert_no_duplicate_facts(
        root,
        annotations_glob=args.annotations_glob,
        fail_on_duplicates=not args.allow_duplicates,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "DUPLICATE_FACTS_REPORT: "
            f"files={report['files_scanned']} pages={report['pages_scanned']} facts={report['facts_scanned']} "
            f"duplicate_groups={report['duplicate_groups']} duplicate_rows={report['duplicate_rows']}"
        )
        for finding in report["findings"]:
            print(
                f"- {finding['file']} | page={finding['page']} | indexes={finding['indexes']} "
                f"| bbox={finding['bbox']} | value={finding['value']!r}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
