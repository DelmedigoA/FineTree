from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence

from ..annotation_core import bbox_to_list
from ..fact_normalization import assert_fact_format
from ..fact_normalization import normalize_fact_payload
from ..fact_ordering import normalize_document_meta, reorder_facts, resolve_reading_direction
from ..schema_contract import default_extraction_prompt_template
from ..schema_io import load_any_schema
from ..schemas import PageMeta
from .duplicate_facts import fact_uniqueness_key
from .config import FinetuneConfig, load_finetune_config


@dataclass
class DatasetBuildStats:
    annotation_files: int = 0
    pages_seen: int = 0
    samples_written_train: int = 0
    samples_written_val: int = 0
    samples_written_test: int = 0
    pages_skipped_empty: int = 0
    pages_skipped_missing_image: int = 0
    pages_skipped_unapproved: int = 0
    facts_deduped: int = 0


def _resolve_prompt_template(cfg: FinetuneConfig) -> str:
    if cfg.prompt.use_custom_prompt:
        prompt_path = cfg.prompt.prompt_path
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. Set prompt.prompt_path or disable prompt.use_custom_prompt."
            )
        return prompt_path.read_text(encoding="utf-8")
    fallback = str(cfg.prompt.fallback_template or "").strip()
    return fallback or default_extraction_prompt_template()


def _iter_annotation_files(pattern: str) -> Iterable[Path]:
    root = Path()
    try:
        candidates = [Path(p) for p in root.glob(pattern)]
    except (ValueError, NotImplementedError):
        candidates = [Path(p) for p in glob.glob(pattern)]
    for p in sorted(candidates):
        if p.is_file():
            yield p


def _doc_in_val_split(doc_id: str, val_ratio: float) -> bool:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < val_ratio


def _doc_split_map(
    annotation_files: List[Path],
    val_ratio: float,
    *,
    forced_val_doc_ids: set[str] | None = None,
    force_explicit_val_doc_ids: bool = False,
    forced_test_doc_ids: set[str] | None = None,
) -> Dict[str, Literal["train", "val", "test"]]:
    doc_ids = [p.stem for p in annotation_files]
    if not doc_ids:
        return {}

    forced_test = {doc_id.strip() for doc_id in (forced_test_doc_ids or set()) if str(doc_id).strip()}
    forced_val = {doc_id.strip() for doc_id in (forced_val_doc_ids or set()) if str(doc_id).strip()}

    if forced_test:
        result: Dict[str, Literal["train", "val", "test"]] = {}
        for doc_id in doc_ids:
            if doc_id in forced_test:
                result[doc_id] = "test"
            elif doc_id in forced_val:
                result[doc_id] = "val"
            else:
                result[doc_id] = "train"
        return result

    if force_explicit_val_doc_ids:
        return {doc_id: ("val" if doc_id in forced_val else "train") for doc_id in doc_ids}
    if forced_val:
        return {doc_id: ("val" if doc_id in forced_val else "train") for doc_id in doc_ids}

    if val_ratio <= 0.0:
        return {doc_id: "train" for doc_id in doc_ids}

    if len(doc_ids) == 1:
        # Keep single-doc datasets in train by default. Validation requirement is
        # enforced later in training when needed.
        return {doc_ids[0]: "train"}

    in_val = {doc_id for doc_id in doc_ids if _doc_in_val_split(doc_id, val_ratio)}
    if not in_val:
        in_val = {sorted(doc_ids)[0]}
    if len(in_val) == len(doc_ids):
        in_val.remove(sorted(doc_ids)[0])

    return {doc_id: ("val" if doc_id in in_val else "train") for doc_id in doc_ids}


def _resolve_page_image_path(
    cfg: FinetuneConfig,
    annotation_path: Path,
    payload: Dict[str, Any],
    page_image_name: str,
) -> Path:
    images_dir_raw = str(payload.get("images_dir") or "").strip()
    if not images_dir_raw:
        raise ValueError(f"Missing images_dir in annotation file: {annotation_path}")
    images_dir = Path(images_dir_raw)
    if not images_dir.is_absolute():
        images_dir = (cfg.data.images_root / images_dir).resolve()
    return (images_dir / page_image_name).resolve()


def _transform_page_for_target(
    cfg: FinetuneConfig,
    *,
    images_dir: str,
    metadata: dict[str, Any],
    page: Dict[str, Any],
    direction: str,
    drop_date: bool = False,
    selected_page_meta_keys: Sequence[str] | None = None,
    selected_fact_keys: Sequence[str] | None = None,
    page_only_wrapper: bool = False,
    excluded_value_contexts: Sequence[str] | None = None,
    dedupe_exact_facts: bool = False,
) -> tuple[Dict[str, Any], int]:
    facts = page.get("facts") if isinstance(page.get("facts"), list) else []
    excluded_value_context_set = {
        str(value_context).strip().lower()
        for value_context in (excluded_value_contexts or ())
        if str(value_context).strip()
    }
    seen_fact_keys: set[tuple[Any, ...]] = set()
    deduped_count = 0
    typed_facts: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        normalized_fact, _warnings = normalize_fact_payload(fact, include_bbox=("bbox" in fact))
        value_context = str(normalized_fact.get("value_context") or "").strip().lower()
        if value_context and value_context in excluded_value_context_set:
            continue
        if dedupe_exact_facts:
            duplicate_key = fact_uniqueness_key(fact)
            if duplicate_key in seen_fact_keys:
                deduped_count += 1
                continue
            seen_fact_keys.add(duplicate_key)
        typed_facts.append(normalized_fact)
    ordered_facts = typed_facts
    if cfg.data.fact_order_enforce:
        ordered_facts = reorder_facts(
            typed_facts,
            direction="rtl" if str(direction).lower() == "rtl" else "ltr",
            row_tolerance_ratio=cfg.data.fact_order_row_tolerance_ratio,
            row_tolerance_min_px=cfg.data.fact_order_row_tolerance_min_px,
        )
    out_facts: List[Dict[str, Any]] = []
    for fact in ordered_facts:
        item = dict(fact)
        if drop_date:
            item.pop("date", None)
        if cfg.data.bbox_policy == "drop_all":
            item.pop("bbox", None)
        elif "bbox" in item:
            item["bbox"] = bbox_to_list(item.get("bbox"))
        out_facts.append(item)

    page_meta_full = PageMeta.model_validate(page.get("meta") if isinstance(page.get("meta"), dict) else {}).model_dump(
        mode="json"
    )
    if selected_page_meta_keys is None:
        page_meta = {
            key: page_meta_full.get(key)
            for key in ("entity_name", "page_num", "page_type", "statement_type", "title")
        }
    else:
        page_meta = {str(key): page_meta_full.get(str(key)) for key in selected_page_meta_keys}

    selected_fact_key_set = {str(key) for key in selected_fact_keys} if selected_fact_keys is not None else None
    page_facts = [
        {
            key: value
            for key, value in fact.items()
            if key == "bbox" or selected_fact_key_set is None or key in selected_fact_key_set
        }
        for fact in out_facts
    ]
    if page_only_wrapper:
        return {
            "meta": page_meta,
            "facts": page_facts,
        }, deduped_count
    page_payload = {
        "image": str(page.get("image") or "").strip() or None,
        "meta": page_meta,
        "facts": page_facts,
    }
    return (
        {
            "images_dir": images_dir or None,
            "metadata": dict(metadata),
            "pages": [page_payload],
        },
        deduped_count,
    )


def build_unsloth_chat_datasets(
    cfg: FinetuneConfig,
    *,
    include_doc_ids: set[str] | None = None,
    forced_val_doc_ids: set[str] | None = None,
    forced_test_doc_ids: set[str] | None = None,
    force_explicit_val_doc_ids: bool = False,
    approved_pages_only: bool = False,
    drop_date: bool = False,
    prompt_template_override: str | None = None,
    selected_page_meta_keys: Sequence[str] | None = None,
    selected_fact_keys: Sequence[str] | None = None,
    page_only_wrapper: bool = False,
    excluded_value_contexts: Sequence[str] | None = None,
    include_empty_pages_override: bool | None = None,
    dedupe_exact_facts: bool = False,
) -> DatasetBuildStats:
    stats = DatasetBuildStats()

    prompt_template = prompt_template_override if isinstance(prompt_template_override, str) and prompt_template_override.strip() else _resolve_prompt_template(cfg)
    include_empty_pages = cfg.data.include_empty_pages if include_empty_pages_override is None else bool(include_empty_pages_override)
    annotation_files = list(_iter_annotation_files(cfg.data.annotations_glob))
    included = {doc_id.strip() for doc_id in (include_doc_ids or set()) if str(doc_id).strip()}
    if include_doc_ids is not None:
        annotation_files = [path for path in annotation_files if path.stem in included]
    resolved_forced_val_doc_ids = (
        {doc_id.strip() for doc_id in forced_val_doc_ids if str(doc_id).strip()}
        if forced_val_doc_ids is not None
        else set(cfg.data.val_doc_ids)
    )
    resolved_forced_test_doc_ids = (
        {doc_id.strip() for doc_id in forced_test_doc_ids if str(doc_id).strip()}
        if forced_test_doc_ids is not None
        else None
    )
    split_map = _doc_split_map(
        annotation_files,
        cfg.data.val_ratio,
        forced_val_doc_ids=resolved_forced_val_doc_ids,
        force_explicit_val_doc_ids=force_explicit_val_doc_ids,
        forced_test_doc_ids=resolved_forced_test_doc_ids,
    )
    train_path = cfg.data.output_train_jsonl
    val_path = cfg.data.output_val_jsonl
    test_path = cfg.data.output_test_jsonl
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        train_path.open("w", encoding="utf-8") as train_f,
        val_path.open("w", encoding="utf-8") as val_f,
        test_path.open("w", encoding="utf-8") as test_f,
    ):
        for annotation_path in annotation_files:
            stats.annotation_files += 1
            doc_id = annotation_path.stem
            doc_split = split_map.get(doc_id, "train")
            if doc_split == "val":
                out_f = val_f
            elif doc_split == "test":
                out_f = test_f
            else:
                out_f = train_f

            payload = load_any_schema(json.loads(annotation_path.read_text(encoding="utf-8")))
            metadata = normalize_document_meta(payload.get("metadata"))
            direction_info = resolve_reading_direction(
                metadata if isinstance(payload, dict) else None,
                payload=payload,
                default_direction=cfg.data.fact_order_default_on_uncertain,
            )
            direction = str(direction_info["direction"])
            images_dir = str(payload.get("images_dir") or "").strip()
            pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                stats.pages_seen += 1

                meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
                page_status = str(meta.get("annotation_status") or "").strip().lower()
                if approved_pages_only and page_status != "approved":
                    stats.pages_skipped_unapproved += 1
                    continue

                image_name = str(page.get("image") or "").strip()
                if not image_name:
                    continue

                image_path = _resolve_page_image_path(cfg, annotation_path, payload, image_name)
                if not image_path.is_file():
                    stats.pages_skipped_missing_image += 1
                    continue

                prompt_text = prompt_template.replace("{{PAGE_IMAGE}}", str(image_path))
                prompt_text = prompt_text.replace("{{IMAGE_NAME}}", image_path.name)

                target_obj, deduped_count = _transform_page_for_target(
                    cfg,
                    images_dir=images_dir,
                    metadata=metadata,
                    page=page,
                    direction=direction,
                    drop_date=drop_date,
                    selected_page_meta_keys=selected_page_meta_keys,
                    selected_fact_keys=selected_fact_keys,
                    page_only_wrapper=page_only_wrapper,
                    excluded_value_contexts=excluded_value_contexts,
                    dedupe_exact_facts=dedupe_exact_facts,
                )
                stats.facts_deduped += deduped_count
                page_payload = target_obj if page_only_wrapper else (target_obj.get("pages") or [{}])[0]
                target_facts = page_payload.get("facts") if isinstance(page_payload, dict) else None
                if not include_empty_pages and not isinstance(target_facts, list):
                    stats.pages_skipped_empty += 1
                    continue
                if not include_empty_pages and not target_facts:
                    stats.pages_skipped_empty += 1
                    continue
                assistant_text = json.dumps(target_obj, ensure_ascii=False)

                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": prompt_text},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": assistant_text},
                            ],
                        },
                    ],
                    "metadata": {
                        "document_id": doc_id,
                        "annotation_file": str(annotation_path),
                        "image": image_name,
                        "reading_direction": direction,
                        "reading_direction_source": str(direction_info.get("source") or ""),
                        "reading_direction_uncertain": bool(direction_info.get("uncertain")),
                    },
                }

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if doc_split == "val":
                    stats.samples_written_val += 1
                elif doc_split == "test":
                    stats.samples_written_test += 1
                else:
                    stats.samples_written_train += 1

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Unsloth chat datasets from FineTree annotations.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    parser.add_argument(
        "--allow-format-issues",
        action="store_true",
        help="Continue dataset build even when fact schema/date/value format issues are found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    if cfg.data.fact_format_enforce:
        format_report = assert_fact_format(
            Path(".").resolve(),
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
    stats = build_unsloth_chat_datasets(cfg)
    print(f"Annotation files: {stats.annotation_files}")
    print(f"Pages seen: {stats.pages_seen}")
    print(f"Train samples: {stats.samples_written_train}")
    print(f"Val samples: {stats.samples_written_val}")
    print(f"Test samples: {stats.samples_written_test}")
    print(f"Skipped empty pages: {stats.pages_skipped_empty}")
    print(f"Skipped missing images: {stats.pages_skipped_missing_image}")
    print(f"Skipped unapproved pages: {stats.pages_skipped_unapproved}")
    print(f"Deduped facts: {stats.facts_deduped}")
    return 0


__all__ = ["DatasetBuildStats", "build_unsloth_chat_datasets", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
