from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import FinetuneConfig, load_finetune_config


@dataclass
class DatasetBuildStats:
    annotation_files: int = 0
    pages_seen: int = 0
    samples_written_train: int = 0
    samples_written_val: int = 0
    pages_skipped_empty: int = 0
    pages_skipped_missing_image: int = 0


def _resolve_prompt_template(cfg: FinetuneConfig) -> str:
    if cfg.prompt.use_custom_prompt:
        prompt_path = cfg.prompt.prompt_path
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. Set prompt.prompt_path or disable prompt.use_custom_prompt."
            )
        return prompt_path.read_text(encoding="utf-8")
    return cfg.prompt.fallback_template


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


def _doc_split_map(annotation_files: List[Path], val_ratio: float) -> Dict[str, bool]:
    doc_ids = [p.stem for p in annotation_files]
    if not doc_ids or val_ratio <= 0.0:
        return {doc_id: False for doc_id in doc_ids}

    if len(doc_ids) == 1:
        # Keep single-doc datasets in train by default. Validation requirement is
        # enforced later in training when needed.
        return {doc_ids[0]: False}

    in_val = {doc_id for doc_id in doc_ids if _doc_in_val_split(doc_id, val_ratio)}
    if not in_val:
        in_val = {sorted(doc_ids)[0]}
    if len(in_val) == len(doc_ids):
        in_val.remove(sorted(doc_ids)[0])

    return {doc_id: (doc_id in in_val) for doc_id in doc_ids}


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


def _transform_page_for_target(cfg: FinetuneConfig, page: Dict[str, Any]) -> Dict[str, Any]:
    facts = page.get("facts") if isinstance(page.get("facts"), list) else []
    out_facts: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        item = dict(fact)
        if cfg.data.bbox_policy == "drop_all":
            item.pop("bbox", None)
        out_facts.append(item)

    return {
        "meta": page.get("meta") if isinstance(page.get("meta"), dict) else {},
        "facts": out_facts,
    }


def build_unsloth_chat_datasets(cfg: FinetuneConfig) -> DatasetBuildStats:
    stats = DatasetBuildStats()

    prompt_template = _resolve_prompt_template(cfg)
    annotation_files = list(_iter_annotation_files(cfg.data.annotations_glob))
    split_map = _doc_split_map(annotation_files, cfg.data.val_ratio)
    train_path = cfg.data.output_train_jsonl
    val_path = cfg.data.output_val_jsonl
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for annotation_path in annotation_files:
            stats.annotation_files += 1
            doc_id = annotation_path.stem
            is_val_doc = bool(split_map.get(doc_id, False))
            out_f = val_f if is_val_doc else train_f

            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                stats.pages_seen += 1

                facts = page.get("facts") if isinstance(page.get("facts"), list) else []
                if not cfg.data.include_empty_pages and not facts:
                    stats.pages_skipped_empty += 1
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

                target_obj = _transform_page_for_target(cfg, page)
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
                    },
                }

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if is_val_doc:
                    stats.samples_written_val += 1
                else:
                    stats.samples_written_train += 1

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Unsloth chat datasets from FineTree annotations.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    stats = build_unsloth_chat_datasets(cfg)
    print(f"Annotation files: {stats.annotation_files}")
    print(f"Pages seen: {stats.pages_seen}")
    print(f"Train samples: {stats.samples_written_train}")
    print(f"Val samples: {stats.samples_written_val}")
    print(f"Skipped empty pages: {stats.pages_skipped_empty}")
    print(f"Skipped missing images: {stats.pages_skipped_missing_image}")
    return 0


__all__ = ["DatasetBuildStats", "build_unsloth_chat_datasets", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
