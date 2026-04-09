#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetree_annotator.finetune import push_dataset_hub  # noqa: E402


DEFAULT_MAX_PIXELS = 1_400_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push the compact FineTree HF dataset using only approved pages, "
            "with bbox pixel adjustment and the no-date Gemini-GT-style prompt schema."
        )
    )
    parser.add_argument("--config", default="configs/finetune_qwen35a3_vl.yaml")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/FineTree-annotated-pages")
    parser.add_argument("--token", default=None, help="HF token override.")
    parser.add_argument("--export-dir", default="artifacts/hf_dataset_export_compact_approved")
    parser.add_argument("--public", action="store_true", help="Push dataset publicly instead of privately.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum image pixel budget.")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Maximum image pixel budget. Defaults to {DEFAULT_MAX_PIXELS}.",
    )
    parser.add_argument("--include-doc-ids", default=None, help="Optional comma-separated doc ids to include.")
    parser.add_argument("--validation-doc-ids", default=None, help="Optional comma-separated doc ids for validation.")
    parser.add_argument("--exclude-doc-ids", default=None, help="Optional comma-separated doc ids to exclude.")
    parser.add_argument(
        "--aggressive-compact-tokens",
        "--aggressive_compact_tokens",
        action="store_true",
        dest="aggressive_compact_tokens",
        help="Also shorten JSON keys in the exported assistant payload.",
    )
    parser.add_argument("--allow-duplicate-facts", action="store_true")
    parser.add_argument("--allow-ordering-issues", action="store_true")
    parser.add_argument("--allow-format-issues", action="store_true")
    parser.add_argument("--push-train-val-separately", action="store_true")
    parser.add_argument("--repo-id-train", default=None)
    parser.add_argument("--repo-id-validation", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    push_argv: list[str] = [
        "--config",
        args.config,
        "--export-dir",
        args.export_dir,
        "--instruction-mode",
        "source",
        "--approved-pages-only",
        "--drop-date",
    ]
    if args.aggressive_compact_tokens:
        push_argv.append("--aggressive-compact-tokens")
    else:
        push_argv.append("--compact_tokens")
    if args.repo_id:
        push_argv.extend(["--repo-id", args.repo_id])
    if args.token:
        push_argv.extend(["--token", args.token])
    if args.public:
        push_argv.append("--public")
    if args.min_pixels is not None:
        push_argv.extend(["--min-pixels", str(args.min_pixels)])
    if args.max_pixels is not None:
        push_argv.extend(["--max-pixels", str(args.max_pixels)])
    if args.include_doc_ids:
        push_argv.extend(["--include-doc-ids", args.include_doc_ids])
    if args.validation_doc_ids:
        push_argv.extend(["--validation-doc-ids", args.validation_doc_ids])
    if args.exclude_doc_ids:
        push_argv.extend(["--exclude-doc-ids", args.exclude_doc_ids])
    if args.allow_duplicate_facts:
        push_argv.append("--allow-duplicate-facts")
    if args.allow_ordering_issues:
        push_argv.append("--allow-ordering-issues")
    if args.allow_format_issues:
        push_argv.append("--allow-format-issues")
    if args.push_train_val_separately:
        push_argv.append("--push-train-val-separately")
    if args.repo_id_train:
        push_argv.extend(["--repo-id-train", args.repo_id_train])
    if args.repo_id_validation:
        push_argv.extend(["--repo-id-validation", args.repo_id_validation])
    return push_dataset_hub.main(push_argv)


if __name__ == "__main__":
    raise SystemExit(main())
