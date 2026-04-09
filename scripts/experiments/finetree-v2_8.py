#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huggingface_hub import HfApi  # noqa: E402

from finetree_annotator.finetune.push_dataset_hub import build_dataset, resolve_hf_token  # noqa: E402
from finetree_annotator.finetune.push_dataset_hub_no_bbox import (  # noqa: E402
    build_hf_dataset_no_bbox_from_export,
    export_for_hf_no_bbox,
    push_train_validation_separately_no_bbox,
)
from finetree_annotator.schema_contract import PROMPT_PAGE_META_KEYS, build_custom_extraction_prompt_template  # noqa: E402


DEFAULT_CONFIG_PATH = "configs/finetune_qwen35a3_vl.yaml"
DEFAULT_EXPORT_DIR = "artifacts/hf_finetree_2_8"
DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_VALIDATION_DOC_IDS: tuple[str, ...] = ("pdf_4",)
DEFAULT_PAGE_META_KEYS: tuple[str, ...] = PROMPT_PAGE_META_KEYS
DEFAULT_FACT_KEYS: tuple[str, ...] = (
    "value",
    "fact_num",
    "comment_ref",
    "note_flag",
    "note_name",
    "note_num",
    "note_ref",
    "period_type",
    "period_start",
    "period_end",
    "path",
    "path_source",
    "currency",
    "scale",
    "value_type",
    "value_context",
)
TRAIN_REPO_BASENAME = "FineTree-2.8-train"
VALIDATION_REPO_BASENAME = "FineTree-2.8-validation"

_SIGNED_NUMERIC_RE = re.compile(
    r"""
    ^
    (?P<sign>[+-])?
    \s*
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:
      \.
      (?P<frac>\d+)
    )?
    $
    """,
    re.VERBOSE,
)
_ACCOUNTING_NUMERIC_RE = re.compile(
    r"""
    ^
    \(
    \s*
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:
      \.
      (?P<frac>\d+)
    )?
    \s*
    \)
    $
    """,
    re.VERBOSE,
)


@dataclass
class ValueNormalizationStats:
    normalized_values: int = 0
    unchanged_values: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push the public FineTree 2.8 split datasets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--token", default=None, help="HF token override.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--include_tabular_mixed_only",
        action="store_true",
        help="Exclude facts whose value_context is textual, keeping only tabular/mixed/unspecified facts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and export locally without pushing to HF.")
    return parser.parse_args(argv)


def _resolve_repo_ids(api: HfApi) -> tuple[str, str]:
    owner = str(api.whoami().get("name") or "").strip() or "user"
    return f"{owner}/{TRAIN_REPO_BASENAME}", f"{owner}/{VALIDATION_REPO_BASENAME}"


def _build_prompt_template() -> str:
    base_prompt = build_custom_extraction_prompt_template(
        page_meta_keys=DEFAULT_PAGE_META_KEYS,
        fact_keys=DEFAULT_FACT_KEYS,
        include_bbox=False,
    )
    formatting_rules = (
        "\n\n"
        "Output formatting:\n"
        "1. Return the final JSON as a single compact line.\n"
        "2. Do not pretty-print, indent, add line breaks, or add extra spaces between JSON tokens.\n"
        "3. Do not add any prefix, suffix, explanation, or surrounding text."
    )
    return base_prompt + formatting_rules


def _valid_grouped_integer(text: str) -> bool:
    if "," not in text:
        return True
    groups = text.split(",")
    if not groups[0].isdigit() or len(groups[0]) > 3:
        return False
    return all(group.isdigit() and len(group) == 3 for group in groups[1:])


def _format_integer_part(text: str) -> str | None:
    if not _valid_grouped_integer(text):
        return None
    digits_only = text.replace(",", "")
    if not digits_only.isdigit():
        return None
    if len(digits_only) > 1 and digits_only.startswith("0"):
        return None
    return f"{int(digits_only):,}"


def _normalize_export_value(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw

    accounting_match = _ACCOUNTING_NUMERIC_RE.fullmatch(raw)
    if accounting_match is not None:
        formatted_int = _format_integer_part(accounting_match.group("int"))
        if formatted_int is None:
            return raw
        frac = accounting_match.group("frac")
        return f"({formatted_int}{('.' + frac) if frac else ''})"

    signed_match = _SIGNED_NUMERIC_RE.fullmatch(raw)
    if signed_match is None:
        return raw
    formatted_int = _format_integer_part(signed_match.group("int"))
    if formatted_int is None:
        return raw
    frac = signed_match.group("frac")
    sign = signed_match.group("sign") or ""
    return f"{sign}{formatted_int}{('.' + frac) if frac else ''}"


def _iter_fact_dicts(payload: Any) -> list[dict[str, Any]]:
    facts_out: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return facts_out

    top_level_facts = payload.get("facts")
    if isinstance(top_level_facts, list):
        for fact in top_level_facts:
            if isinstance(fact, dict):
                facts_out.append(fact)

    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_facts = page.get("facts")
            if not isinstance(page_facts, list):
                continue
            for fact in page_facts:
                if isinstance(fact, dict):
                    facts_out.append(fact)

    return facts_out


def _normalize_text_payload_values(text: str) -> tuple[str, ValueNormalizationStats]:
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise RuntimeError("Failed to parse exported assistant payload while normalizing values.") from exc

    stats = ValueNormalizationStats()
    for fact in _iter_fact_dicts(payload):
        if "value" not in fact:
            continue
        original = str(fact.get("value") or "")
        normalized = _normalize_export_value(original)
        if normalized != original:
            fact["value"] = normalized
            stats.normalized_values += 1
        else:
            stats.unchanged_values += 1
    return json.dumps(payload, ensure_ascii=False), stats


def _normalize_export_jsonl_values(split_path: Path) -> ValueNormalizationStats:
    stats = ValueNormalizationStats()
    if not split_path.is_file():
        return stats

    updated_lines: list[str] = []
    for line_num, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse exported row JSON in {split_path}:{line_num}.") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected JSON object row in {split_path}:{line_num}.")
        text = row.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Missing string `text` payload in {split_path}:{line_num}.")

        normalized_text, row_stats = _normalize_text_payload_values(text)
        row["text"] = normalized_text
        stats.normalized_values += row_stats.normalized_values
        stats.unchanged_values += row_stats.unchanged_values
        updated_lines.append(json.dumps(row, ensure_ascii=False))

    split_path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
    return stats


def _normalize_export_dir_values(export_dir: Path) -> ValueNormalizationStats:
    stats = ValueNormalizationStats()
    for split_name in ("train.jsonl", "val.jsonl"):
        split_stats = _normalize_export_jsonl_values(export_dir / split_name)
        stats.normalized_values += split_stats.normalized_values
        stats.unchanged_values += split_stats.unchanged_values
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd().resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token or export FINETREE_HF_TOKEN/HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        )

    api = HfApi(token=token)
    repo_id_train, repo_id_validation = _resolve_repo_ids(api)
    config_path = (root / args.config).resolve()
    export_dir = (root / args.export_dir).resolve()
    excluded_value_contexts = ("textual",) if args.include_tabular_mixed_only else None

    prompt_template = _build_prompt_template()

    build_dataset(
        config_path,
        validation_doc_ids=set(DEFAULT_VALIDATION_DOC_IDS),
        approved_pages_only=True,
        prompt_template_override=prompt_template,
        selected_page_meta_keys=DEFAULT_PAGE_META_KEYS,
        selected_fact_keys=DEFAULT_FACT_KEYS,
        page_only_wrapper=True,
        excluded_value_contexts=excluded_value_contexts,
        include_empty_pages_override=True,
        dedupe_exact_facts=True,
    )

    train_rows, val_rows = export_for_hf_no_bbox(
        root,
        export_dir,
        instruction_mode="source",
        max_pixels=args.max_pixels,
    )
    normalization_stats = _normalize_export_dir_values(export_dir)
    dataset, _, _ = build_hf_dataset_no_bbox_from_export(export_dir, instruction_mode="source")

    if args.dry_run:
        print("DRY_RUN: true")
    else:
        pushed = push_train_validation_separately_no_bbox(
            dataset,
            token=token,
            base_repo_id=f"{repo_id_train}-unused-base",
            private=False,
            repo_id_train=repo_id_train,
            repo_id_validation=repo_id_validation,
        )
        print(f"PUSHED_TRAIN_REPO: {pushed.get('train')}")
        print(f"PUSHED_VALIDATION_REPO: {pushed.get('validation')}")

    print(f"CONFIG: {config_path}")
    print(f"EXPORT_DIR: {export_dir}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"PAGE_META_KEYS: {list(DEFAULT_PAGE_META_KEYS)}")
    print(f"FACT_KEYS: {list(DEFAULT_FACT_KEYS)}")
    print(f"VALIDATION_DOC_IDS: {list(DEFAULT_VALIDATION_DOC_IDS)}")
    print(f"REPO_ID_TRAIN: {repo_id_train}")
    print(f"REPO_ID_VALIDATION: {repo_id_validation}")
    print(f"MAX_PIXELS: {args.max_pixels}")
    print(f"EXCLUDED_VALUE_CONTEXTS: {list(excluded_value_contexts) if excluded_value_contexts is not None else []}")
    print(f"NORMALIZED_VALUES: {normalization_stats.normalized_values}")
    print(f"UNCHANGED_VALUES: {normalization_stats.unchanged_values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
