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

from finetree_annotator.fact_ordering import assert_fact_order, fact_order_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check annotation JSON files for reading-order violations.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
    parser.add_argument("--default-direction", choices=["rtl", "ltr"], default="rtl", help="Fallback direction on uncertain language.")
    parser.add_argument("--row-tolerance-ratio", type=float, default=0.35, help="Row grouping tolerance ratio based on median bbox height.")
    parser.add_argument("--row-tolerance-min-px", type=float, default=6.0, help="Minimum row grouping tolerance in pixels.")
    parser.add_argument(
        "--allow-ordering-issues",
        action="store_true",
        help="Exit successfully even when ordering issues are found.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if args.json:
        report = fact_order_report(
            root,
            annotations_glob=args.annotations_glob,
            default_direction=args.default_direction,
            row_tolerance_ratio=args.row_tolerance_ratio,
            row_tolerance_min_px=args.row_tolerance_min_px,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if (not args.allow_ordering_issues) and int(report["pages_with_order_issues"]) > 0:
            return 1
        return 0

    report = assert_fact_order(
        root,
        annotations_glob=args.annotations_glob,
        default_direction=args.default_direction,
        row_tolerance_ratio=args.row_tolerance_ratio,
        row_tolerance_min_px=args.row_tolerance_min_px,
        fail_on_issues=not args.allow_ordering_issues,
    )
    print(
        "FACT_ORDER_REPORT: "
        f"files={report['files_scanned']} pages={report['pages_scanned']} "
        f"pages_with_order_issues={report['pages_with_order_issues']} "
        f"uncertain_documents={len(report['uncertain_documents'])}"
    )
    for finding in report["findings"]:
        first = finding.get("first_violation") or {}
        print(
            f"- {finding['file']} | page={finding['page']} | direction={finding['direction']} "
            f"| source={finding['direction_source']} | violations={finding['violations_count']} "
            f"| first_position={first.get('position')} "
            f"| expected_index={first.get('expected_index')} "
            f"| current_index={first.get('current_index')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
