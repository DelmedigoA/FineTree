#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huggingface_hub import HfApi  # noqa: E402

from finetree_annotator.finetune.push_dataset_hub import (  # noqa: E402
    build_dataset,
    build_hf_dataset_from_export,
    export_for_hf,
    push_to_hf,
    push_train_validation_separately,
    resolve_hf_token,
)
from finetree_annotator.schema_contract import PROMPT_PAGE_META_KEYS, build_custom_extraction_prompt_template  # noqa: E402


DEFAULT_CONFIG_PATH = "configs/finetune_qwen35a3_vl.yaml"
DEFAULT_EXPORT_DIR = "artifacts/hf_finetree_3_0_value_bbox_factnum"
DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_VALIDATION_DOC_IDS: tuple[str, ...] = ("pdf_4",)
DEFAULT_PAGE_META_KEYS: tuple[str, ...] = PROMPT_PAGE_META_KEYS
DEFAULT_FACT_KEYS: tuple[str, ...] = (
    "value",
    "fact_num",
)
TRAIN_REPO_BASENAME = "FineTree-3.0-value-bbox-factnum-train"
VALIDATION_REPO_BASENAME = "FineTree-3.0-value-bbox-factnum-validation"

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
    parser = argparse.ArgumentParser(
        description="Build and push the public FineTree 3.0 split datasets with bbox, value, and fact_num only."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--token", default=None, help="HF token override.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional single HF dataset repo id. When set, pushes train and validation as splits inside one repo.",
    )
    parser.add_argument(
        "--base-repo-id",
        default=None,
        help="Optional HF dataset base repo id. When set, pushes to <base>-train and <base>-validation.",
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Keep exported images at full resolution instead of resizing to a max pixel budget.",
    )
    parser.add_argument(
        "--merge-hf-dataset",
        default=None,
        help="Optional HF dataset repo id to append to the locally approved export before pushing.",
    )
    parser.add_argument(
        "--merge-hf-train-split",
        default="train",
        help="Split name to read from --merge-hf-dataset for training rows.",
    )
    parser.add_argument(
        "--merge-hf-validation-split",
        default="validation",
        help="Split name to read from --merge-hf-dataset for validation rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and export locally without pushing to HF.")
    return parser.parse_args(argv)


def _resolve_repo_ids(api: HfApi, *, base_repo_id: str | None = None) -> tuple[str, str]:
    resolved_base = str(base_repo_id or "").strip()
    if resolved_base:
        return f"{resolved_base}-train", f"{resolved_base}-validation"
    owner = str(api.whoami().get("name") or "").strip() or "user"
    return f"{owner}/{TRAIN_REPO_BASENAME}", f"{owner}/{VALIDATION_REPO_BASENAME}"


def _build_prompt_template() -> str:
    base_prompt = build_custom_extraction_prompt_template(
        page_meta_keys=DEFAULT_PAGE_META_KEYS,
        fact_keys=DEFAULT_FACT_KEYS,
        include_bbox=True,
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
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text_out, row_stats = _normalize_text_payload_values(str(row.get("text") or ""))
        row["text"] = text_out
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


def _ensure_required_columns(split_name: str, dataset: Dataset) -> None:
    required_columns = ("image", "system", "instruction", "text")
    missing = [column for column in required_columns if column not in dataset.column_names]
    if missing:
        raise RuntimeError(
            f"Merged HF dataset split '{split_name}' is missing required columns: {missing}. "
            "Expected columns compatible with FineTree export rows."
        )


def _align_split_to_features(split_name: str, dataset: Dataset, target: Dataset) -> Dataset:
    _ensure_required_columns(split_name, dataset)
    ordered = dataset.select_columns(list(target.column_names))
    if ordered.features == target.features:
        return ordered
    return ordered.cast(target.features)


def _load_merge_split(repo_id: str, split_name: str, target: Dataset) -> Dataset:
    loaded = load_dataset(repo_id, split=split_name)
    if not isinstance(loaded, Dataset):
        raise RuntimeError(
            f"Expected split '{split_name}' from merged HF dataset {repo_id} to load as a Dataset, got {type(loaded)!r}."
        )
    return _align_split_to_features(split_name, loaded, target)


def _merge_with_hf_dataset(
    dataset: DatasetDict,
    *,
    repo_id: str,
    train_split: str,
    validation_split: str,
) -> tuple[DatasetDict, dict[str, int]]:
    local_train = dataset["train"]
    local_validation = dataset["validation"]
    hf_train = _load_merge_split(repo_id, train_split, local_train)
    hf_validation = _load_merge_split(repo_id, validation_split, local_validation)
    merged = DatasetDict(
        {
            "train": concatenate_datasets([local_train, hf_train]),
            "validation": concatenate_datasets([local_validation, hf_validation]),
        }
    )
    stats = {
        "local_train": len(local_train),
        "local_validation": len(local_validation),
        "merged_hf_train": len(hf_train),
        "merged_hf_validation": len(hf_validation),
        "final_train": len(merged["train"]),
        "final_validation": len(merged["validation"]),
    }
    return merged, stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd().resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token or export FINETREE_HF_TOKEN/HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        )

    api = HfApi(token=token)
    repo_id_train, repo_id_validation = _resolve_repo_ids(api, base_repo_id=args.base_repo_id)
    config_path = (root / args.config).resolve()
    export_dir = (root / args.export_dir).resolve()
    resolved_max_pixels = None if args.full_resolution else int(args.max_pixels)

    prompt_template = _build_prompt_template()

    build_dataset(
        config_path,
        validation_doc_ids=set(DEFAULT_VALIDATION_DOC_IDS),
        approved_pages_only=True,
        prompt_template_override=prompt_template,
        selected_page_meta_keys=DEFAULT_PAGE_META_KEYS,
        selected_fact_keys=DEFAULT_FACT_KEYS,
        page_only_wrapper=True,
        include_empty_pages_override=True,
        dedupe_exact_facts=True,
    )

    train_rows, val_rows = export_for_hf(
        root,
        export_dir,
        instruction_mode="source",
        max_pixels=resolved_max_pixels,
    )
    normalization_stats = _normalize_export_dir_values(export_dir)
    dataset, _, _ = build_hf_dataset_from_export(export_dir)
    merge_stats: dict[str, int] | None = None
    merged_repo_id = str(args.merge_hf_dataset or "").strip() or None
    single_repo_id = str(args.repo_id or "").strip() or None
    if merged_repo_id is not None:
        dataset, merge_stats = _merge_with_hf_dataset(
            dataset,
            repo_id=merged_repo_id,
            train_split=str(args.merge_hf_train_split),
            validation_split=str(args.merge_hf_validation_split),
        )

    if args.dry_run:
        print("DRY_RUN: true")
    else:
        if single_repo_id is not None:
            pushed_repo = push_to_hf(
                dataset,
                token=token,
                repo_id=single_repo_id,
                private=False,
            )
            print(f"PUSHED_REPO: {pushed_repo}")
        else:
            pushed = push_train_validation_separately(
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
    print("INCLUDE_BBOX: true")
    print(f"VALIDATION_DOC_IDS: {list(DEFAULT_VALIDATION_DOC_IDS)}")
    print(f"REPO_ID: {single_repo_id or 'unset'}")
    print(f"REPO_ID_TRAIN: {repo_id_train}")
    print(f"REPO_ID_VALIDATION: {repo_id_validation}")
    print(f"MAX_PIXELS: {resolved_max_pixels if resolved_max_pixels is not None else 'unset'}")
    print(f"FULL_RESOLUTION: {args.full_resolution}")
    print(f"NORMALIZED_VALUES: {normalization_stats.normalized_values}")
    print(f"UNCHANGED_VALUES: {normalization_stats.unchanged_values}")
    print(f"MERGE_HF_DATASET: {merged_repo_id or 'unset'}")
    if merge_stats is not None:
        print(f"MERGE_TRAIN_SPLIT: {args.merge_hf_train_split}")
        print(f"MERGE_VALIDATION_SPLIT: {args.merge_hf_validation_split}")
        print(f"LOCAL_TRAIN_ROWS: {merge_stats['local_train']}")
        print(f"LOCAL_VAL_ROWS: {merge_stats['local_validation']}")
        print(f"MERGED_HF_TRAIN_ROWS: {merge_stats['merged_hf_train']}")
        print(f"MERGED_HF_VAL_ROWS: {merge_stats['merged_hf_validation']}")
        print(f"FINAL_PUSH_TRAIN_ROWS: {merge_stats['final_train']}")
        print(f"FINAL_PUSH_VAL_ROWS: {merge_stats['final_validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
