from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Literal, Tuple

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi

from ..fact_normalization import assert_fact_format
from ..fact_ordering import assert_fact_order
from .config import load_finetune_config
from .duplicate_facts import assert_no_duplicate_facts
from .push_dataset_hub import (
    _DEFAULT_SYSTEM_PROMPT,
    _compact_token_text_payload,
    _copy_or_resize_image,
    _dataset_for_push,
    _parse_excluded_doc_ids,
    _resolve_system_prompt,
    _resolve_resize_bounds,
    assert_source_instruction_schema,
    assert_no_train_val_contamination,
    build_dataset,
    resolve_hf_token,
)

InstructionMode = Literal["source", "minimal"]

_NO_BBOX_FALLBACK_INSTRUCTION = "Extract page JSON using the FineTree schema. Do not include bbox fields."
_MINIMAL_INSTRUCTION = "Extract metadata and financial facts from the provided image."
_BBOX_WORD_RE = re.compile(r"\bbbox\b", flags=re.IGNORECASE)


def _normalize_instruction_mode(value: str) -> InstructionMode:
    mode = (value or "").strip().lower()
    if mode not in {"source", "minimal"}:
        raise ValueError("--instruction-mode must be one of: source, minimal")
    return mode  # type: ignore[return-value]


def _strip_bbox_from_text_payload(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return text

    if isinstance(payload, dict):
        facts = payload.get("facts")
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict):
                    fact.pop("bbox", None)
    return json.dumps(payload, ensure_ascii=False)


def _sanitize_instruction_no_bbox(instruction: str) -> str:
    filtered_lines = [line for line in instruction.splitlines() if _BBOX_WORD_RE.search(line) is None]
    sanitized = "\n".join(filtered_lines).strip()
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized or _NO_BBOX_FALLBACK_INSTRUCTION


def _instruction_for_mode(source_instruction: str, instruction_mode: InstructionMode) -> str:
    if instruction_mode == "minimal":
        return _MINIMAL_INSTRUCTION
    return _sanitize_instruction_no_bbox(source_instruction)


def _rows_from_chat_jsonl_no_bbox(
    split_path: Path,
    *,
    instruction_mode: InstructionMode = "source",
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not split_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
        if len(messages) < 2:
            continue

        user_content = messages[0].get("content") if isinstance(messages[0], dict) else None
        assistant_content = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(user_content, list) or not isinstance(assistant_content, list):
            continue

        image_path = ""
        instruction = ""
        text = ""

        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and isinstance(part.get("image"), str):
                image_path = part["image"]
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                instruction = part["text"]

        if assistant_content and isinstance(assistant_content[0], dict):
            text_val = assistant_content[0].get("text")
            if isinstance(text_val, str):
                text = _strip_bbox_from_text_payload(text_val)

        if not image_path or not text:
            continue

        rows.append(
            {
                "image": image_path,
                "system": system_prompt,
                "instruction": _instruction_for_mode(instruction, instruction_mode),
                "text": text,
            }
        )
    return rows


def _rows_from_export_jsonl_no_bbox(
    split_path: Path,
    export_dir: Path,
    *,
    default_system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not split_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if not isinstance(sample, dict):
            continue
        image_path = sample.get("image")
        system_prompt = sample.get("system")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            system_prompt = sample.get("system_prompt")
        instruction = sample.get("instruction")
        text = sample.get("text")
        if not isinstance(image_path, str) or not isinstance(instruction, str) or not isinstance(text, str):
            continue
        resolved_system_prompt = (
            str(system_prompt).strip()
            if isinstance(system_prompt, str) and str(system_prompt).strip()
            else default_system_prompt
        )
        abs_img = (export_dir / image_path).resolve()
        rows.append({"image": str(abs_img), "system": resolved_system_prompt, "instruction": instruction, "text": text})
    return rows


def _build_hf_dataset_no_bbox_from_rows(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> tuple[DatasetDict, int, int]:
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    empty_payload = {"image": [], "system": [], "instruction": [], "text": []}
    train_ds = Dataset.from_list(train_rows, features=features) if train_rows else Dataset.from_dict(empty_payload, features=features)
    val_ds = Dataset.from_list(val_rows, features=features) if val_rows else Dataset.from_dict(empty_payload, features=features)
    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
        }
    )
    return dataset, len(train_rows), len(val_rows)


def build_hf_dataset_no_bbox(root: Path, *, instruction_mode: InstructionMode = "source") -> tuple[DatasetDict, int, int]:
    system_prompt = _resolve_system_prompt(root)
    train_rows = _rows_from_chat_jsonl_no_bbox(
        root / "data/finetune/train.jsonl",
        instruction_mode=instruction_mode,
        system_prompt=system_prompt,
    )
    val_rows = _rows_from_chat_jsonl_no_bbox(
        root / "data/finetune/val.jsonl",
        instruction_mode=instruction_mode,
        system_prompt=system_prompt,
    )
    return _build_hf_dataset_no_bbox_from_rows(train_rows, val_rows)


def build_hf_dataset_no_bbox_from_export(
    export_dir: Path,
    *,
    instruction_mode: InstructionMode = "source",
) -> tuple[DatasetDict, int, int]:
    _ = instruction_mode
    default_system_prompt = _resolve_system_prompt(Path(".").resolve())
    train_rows = _rows_from_export_jsonl_no_bbox(
        export_dir / "train.jsonl",
        export_dir,
        default_system_prompt=default_system_prompt,
    )
    val_rows = _rows_from_export_jsonl_no_bbox(
        export_dir / "val.jsonl",
        export_dir,
        default_system_prompt=default_system_prompt,
    )
    return _build_hf_dataset_no_bbox_from_rows(train_rows, val_rows)


def export_for_hf_no_bbox(
    root: Path,
    export_dir: Path,
    *,
    instruction_mode: InstructionMode = "source",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    exclude_doc_ids: set[str] | None = None,
    compact_tokens: bool = False,
    aggressive_compact_tokens: bool = False,
) -> Tuple[int, int]:
    system_prompt = _resolve_system_prompt(root)
    resize_bounds = _resolve_resize_bounds(min_pixels, max_pixels)
    resolved_min = resize_bounds[0] if resize_bounds else None
    resolved_max = resize_bounds[1] if resize_bounds else None
    resize_enabled = resize_bounds is not None
    excluded = set(exclude_doc_ids or set())

    train_in = root / "data/finetune/train.jsonl"
    val_in = root / "data/finetune/val.jsonl"

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    resized_images = 0
    unchanged_images = 0
    skipped_excluded_docs = 0

    def export_split(src_path: Path, dst_name: str) -> int:
        nonlocal resized_images, unchanged_images, skipped_excluded_docs
        dst_path = export_dir / dst_name
        if not src_path.exists():
            dst_path.write_text("", encoding="utf-8")
            return 0

        out_lines: list[str] = []
        for row in _rows_from_chat_jsonl_no_bbox(
            src_path,
            instruction_mode=instruction_mode,
            system_prompt=system_prompt,
        ):
            src_img = Path(row["image"])
            if not src_img.is_absolute():
                src_img = (root / src_img).resolve()
            if not src_img.exists():
                raise FileNotFoundError(f"Missing source image: {src_img}")

            doc_id = src_img.parent.name or "unknown_doc"
            if doc_id in excluded:
                skipped_excluded_docs += 1
                continue
            rel_img = Path("images") / doc_id / src_img.name
            dst_img = export_dir / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            resize_stats = _copy_or_resize_image(
                src_img,
                dst_img,
                min_pixels=resolved_min,
                max_pixels=resolved_max,
            )
            if resize_stats["resized"]:
                resized_images += 1
            else:
                unchanged_images += 1

            payload = {
                "image": rel_img.as_posix(),
                "system": row["system"],
                "instruction": row["instruction"].replace(str(src_img), rel_img.as_posix()),
                "text": (
                    _compact_token_text_payload(row["text"], remap_keys=aggressive_compact_tokens)
                    if (compact_tokens or aggressive_compact_tokens)
                    else row["text"]
                ),
            }
            out_lines.append(json.dumps(payload, ensure_ascii=False))

        dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        return len(out_lines)

    train_rows = export_split(train_in, "train.jsonl")
    val_rows = export_split(val_in, "val.jsonl")

    (export_dir / "README.md").write_text(
        "# FineTree Annotated Dataset (No BBox)\n\n"
        "Generated from current repository annotations.\n\n"
        "- Assistant JSON payload has bbox removed from all facts.\n\n"
        "- Includes `system` column from `system_prompt.txt`.\n"
        f"- Instruction mode: {instruction_mode}\n"
        f"- Resize enabled: {'yes' if resize_enabled else 'no'}\n"
        f"- Min pixels: {resolved_min if resolved_min is not None else 'unset'}\n"
        f"- Max pixels: {resolved_max if resolved_max is not None else 'unset'}\n"
        f"- Resized images: {resized_images}\n"
        f"- Unchanged images: {unchanged_images}\n"
        f"- Skipped rows by excluded docs: {skipped_excluded_docs}\n"
        f"- Excluded doc ids: {sorted(excluded)}\n"
        f"- Compact tokens: {'yes' if compact_tokens else 'no'}\n"
        f"- Aggressive compact tokens: {'yes' if aggressive_compact_tokens else 'no'}\n"
        f"- Train rows: {train_rows}\n"
        f"- Val rows: {val_rows}\n",
        encoding="utf-8",
    )
    print(
        "EXPORT_STATS_NO_BBOX:",
        json.dumps(
            {
                "instruction_mode": instruction_mode,
                "resize_enabled": resize_enabled,
                "min_pixels": resolved_min,
                "max_pixels": resolved_max,
                "resized_images": resized_images,
                "unchanged_images": unchanged_images,
                "skipped_excluded_docs": skipped_excluded_docs,
                "excluded_doc_ids": sorted(excluded),
                "compact_tokens": compact_tokens,
                "aggressive_compact_tokens": aggressive_compact_tokens,
            },
            ensure_ascii=False,
        ),
    )
    return train_rows, val_rows


def _default_repo_id_no_bbox(api: HfApi, *, instruction_mode: InstructionMode = "source") -> str:
    username = api.whoami().get("name") or "user"
    suffix = "" if instruction_mode == "source" else "-minimal-instruction"
    return f"{username}/FineTree-annotated-pages-no-bbox{suffix}"


def push_to_hf_no_bbox(
    dataset: DatasetDict,
    token: str,
    repo_id: str | None,
    private: bool = True,
    *,
    instruction_mode: InstructionMode = "source",
) -> str:
    api = HfApi(token=token)
    resolved_repo_id = repo_id or _default_repo_id_no_bbox(api, instruction_mode=instruction_mode)
    dataset_to_push, kept_splits, dropped_splits = _dataset_for_push(dataset)
    api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset_to_push.push_to_hub(repo_id=resolved_repo_id, token=token, private=private)
    print(f"PUSH_SPLITS: kept={kept_splits} dropped={dropped_splits}")
    return resolved_repo_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and push FineTree no-bbox dataset with columns: image, instruction, text (bbox removed)."
    )
    parser.add_argument("--config", default="configs/finetune_qwen35a3_vl.yaml")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HF dataset repo id, e.g. username/FineTree-annotated-pages-no-bbox",
    )
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN/HF_TOKEN/Doppler)")
    parser.add_argument("--export-dir", default="artifacts/hf_dataset_export_no_bbox")
    parser.add_argument("--public", action="store_true", help="Create/push dataset as public.")
    parser.add_argument(
        "--instruction-mode",
        default="source",
        choices=["source", "minimal"],
        help="Use source sanitized instruction or fixed minimal instruction.",
    )
    parser.add_argument(
        "--omit-instruction",
        action="store_true",
        help="Deprecated alias; maps to --instruction-mode minimal.",
    )
    parser.add_argument(
        "--exclude-doc-ids",
        default=None,
        help="Comma-separated document/image folder ids to exclude, e.g. pdf_4,test",
    )
    parser.add_argument(
        "--compact_tokens",
        action="store_true",
        help="Compact assistant JSON payload formatting and numeric separators without renaming keys.",
    )
    parser.add_argument(
        "--aggressive-compact-tokens",
        "--aggressive_compact_tokens",
        action="store_true",
        dest="aggressive_compact_tokens",
        help="Aggressive compaction: enables key shortening in addition to --compact_tokens behavior.",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional minimum image pixel budget for export.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional maximum image pixel budget for export.")
    parser.add_argument(
        "--allow-duplicate-facts",
        action="store_true",
        help="Allow HF export/push to continue even when exact duplicate facts are detected in annotations.",
    )
    parser.add_argument(
        "--allow-ordering-issues",
        action="store_true",
        help="Allow HF export/push to continue when fact reading-order violations are detected in annotations.",
    )
    parser.add_argument(
        "--allow-format-issues",
        action="store_true",
        help="Allow HF export/push to continue when fact schema/date/value format issues are detected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(".").resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token, export FINETREE_HF_TOKEN/HF_TOKEN, or configure HF_TOKEN in Doppler."
        )

    instruction_mode = _normalize_instruction_mode(args.instruction_mode)
    if args.omit_instruction:
        print("WARNING: --omit-instruction is deprecated; using --instruction-mode minimal.")
        instruction_mode = "minimal"
    excluded_doc_ids = _parse_excluded_doc_ids(args.exclude_doc_ids)
    compact_tokens = bool(args.compact_tokens or args.aggressive_compact_tokens)

    config_path = (root / args.config).resolve()
    cfg = load_finetune_config(config_path)
    export_dir = (root / args.export_dir).resolve()

    build_dataset(config_path, allow_format_issues=args.allow_format_issues)
    if instruction_mode == "source":
        assert_source_instruction_schema(root, fail_on_issues=True)
    assert_no_train_val_contamination(root)
    duplicate_report = assert_no_duplicate_facts(
        root,
        annotations_glob=cfg.data.annotations_glob,
        fail_on_duplicates=not args.allow_duplicate_facts,
    )
    if args.allow_duplicate_facts and int(duplicate_report["duplicate_rows"]) > 0:
        print(
            "WARNING: continuing with duplicate facts because --allow-duplicate-facts was set. "
            f"duplicate_rows={duplicate_report['duplicate_rows']}"
        )
    if cfg.data.fact_order_enforce:
        ordering_report = assert_fact_order(
            root,
            annotations_glob=cfg.data.annotations_glob,
            default_direction=cfg.data.fact_order_default_on_uncertain,
            row_tolerance_ratio=cfg.data.fact_order_row_tolerance_ratio,
            row_tolerance_min_px=cfg.data.fact_order_row_tolerance_min_px,
            fail_on_issues=not args.allow_ordering_issues,
        )
        if args.allow_ordering_issues and int(ordering_report["pages_with_order_issues"]) > 0:
            print(
                "WARNING: continuing with ordering issues because --allow-ordering-issues was set. "
                f"pages_with_order_issues={ordering_report['pages_with_order_issues']}"
            )
    else:
        print("FACT_ORDER_AUDIT: skipped (data.fact_order_enforce=false)")
    if cfg.data.fact_format_enforce:
        format_report = assert_fact_format(
            root,
            annotations_glob=cfg.data.annotations_glob,
            fail_on_issues=not args.allow_format_issues,
        )
        if args.allow_format_issues and int(format_report["facts_with_issues"]) > 0:
            print(
                "WARNING: continuing with format issues because --allow-format-issues was set. "
                f"facts_with_issues={format_report['facts_with_issues']}"
            )
    else:
        print("FACT_FORMAT_AUDIT: skipped (data.fact_format_enforce=false)")
    train_rows, val_rows = export_for_hf_no_bbox(
        root,
        export_dir,
        instruction_mode=instruction_mode,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        exclude_doc_ids=excluded_doc_ids,
        compact_tokens=compact_tokens,
        aggressive_compact_tokens=args.aggressive_compact_tokens,
    )
    dataset, _, _ = build_hf_dataset_no_bbox_from_export(export_dir, instruction_mode=instruction_mode)
    repo = push_to_hf_no_bbox(
        dataset,
        token=token,
        repo_id=args.repo_id,
        private=not args.public,
        instruction_mode=instruction_mode,
    )

    print(f"PUSHED: {repo}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"INSTRUCTION_MODE: {instruction_mode}")
    print(f"COMPACT_TOKENS: {compact_tokens}")
    print(f"AGGRESSIVE_COMPACT_TOKENS: {args.aggressive_compact_tokens}")
    print(f"EXPORT_DIR: {export_dir}")
    return 0


__all__ = [
    "build_hf_dataset_no_bbox",
    "build_hf_dataset_no_bbox_from_export",
    "export_for_hf_no_bbox",
    "push_to_hf_no_bbox",
    "_sanitize_instruction_no_bbox",
    "_MINIMAL_INSTRUCTION",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
