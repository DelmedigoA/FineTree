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

from finetree_annotator.fact_normalization import assert_fact_format, fact_format_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check annotation JSON files for fact schema/date/value format issues.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
    parser.add_argument(
        "--allow-format-issues",
        action="store_true",
        help="Exit successfully even when format issues are found.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if args.json:
        report = fact_format_report(root, annotations_glob=args.annotations_glob)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if (not args.allow_format_issues) and int(report["facts_with_issues"]) > 0:
            return 1
        return 0

    report = assert_fact_format(
        root,
        annotations_glob=args.annotations_glob,
        fail_on_issues=not args.allow_format_issues,
    )
    print(
        "FACT_FORMAT_REPORT: "
        f"files={report['files_scanned']} pages={report['pages_scanned']} "
        f"facts={report['facts_scanned']} facts_with_issues={report['facts_with_issues']}"
    )
    for finding in report["findings"][:200]:
        print(
            f"- {finding['file']} | page={finding['page']} | fact_index={finding['fact_index']} "
            f"| issues={finding['issue_codes']} | raw_value={finding['raw_value']!r} "
            f"| normalized_value={finding['normalized_value']!r}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

