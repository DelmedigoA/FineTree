#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.schema_guardrails import scan_raw_schema_key_literals, schema_keys_for_literal_scan  # noqa: E402

DEFAULT_ALLOW_PATHS: tuple[str, ...] = (
    "src/finetree_annotator/app.py",
    "src/finetree_annotator/schemas.py",
    "src/finetree_annotator/schema_registry.py",
    "src/finetree_annotator/schema_io.py",
    "src/finetree_annotator/schema_contract.py",
    "src/finetree_annotator/schema_guardrails.py",
    "src/finetree_annotator/fact_normalization.py",
    "src/finetree_annotator/fact_ordering.py",
    "src/finetree_annotator/annotation_core.py",
    "src/finetree_annotator/page_issues.py",
    "src/finetree_annotator/gemini_few_shot.py",
    "src/finetree_annotator/gemini_vlm.py",
    "src/finetree_annotator/finetune/push_dataset_hub.py",
    "scripts/fix_fact_reading_order.py",
    "tests/test_annotation_core.py",
    "tests/test_fact_normalization.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when legacy schema key literals appear outside adapter modules.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument(
        "--include-glob",
        action="append",
        default=["src/**/*.py", "scripts/**/*.py"],
        help="Glob(s) to scan, relative to root.",
    )
    parser.add_argument(
        "--allow-path",
        action="append",
        default=list(DEFAULT_ALLOW_PATHS),
        help="Relative paths allowed to contain legacy schema key literals.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    findings = scan_raw_schema_key_literals(
        root,
        include_globs=args.include_glob,
        allow_relative_paths=args.allow_path,
        scanned_keys=schema_keys_for_literal_scan(),
    )
    if findings:
        print("SCHEMA_KEY_LITERAL_VIOLATIONS:")
        for path, line_no, key in findings[:200]:
            print(f"- {path}:{line_no}: {key}")
        if len(findings) > 200:
            print(f"... {len(findings) - 200} more")
        return 1
    print("SCHEMA_KEY_LITERAL_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
