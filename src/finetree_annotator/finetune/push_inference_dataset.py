from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi

from ..schema_contract import (
    PROMPT_FACT_KEYS,
    PROMPT_PAGE_META_KEYS,
    build_custom_extraction_schema_preview,
)
from ..workspace import page_image_paths
from .push_dataset_hub import (
    _copy_or_resize_image,
    _dataset_for_push,
    _parse_selected_keys,
    _resolve_resize_bounds,
    _resolve_system_prompt,
    resolve_hf_token,
)

DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_PAGE_META_KEYS: tuple[str, ...] = tuple(PROMPT_PAGE_META_KEYS)
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


def inference_dataset_short_id(doc_id: str) -> str:
    normalized = str(doc_id or "").strip() or "document"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:4]


def inference_dataset_name(doc_id: str, max_pixels: int) -> str:
    return f"pdf_{inference_dataset_short_id(doc_id)}_for_inference_{int(max_pixels)}"


def build_inference_instruction(
    *,
    page_meta_keys: tuple[str, ...] | None = None,
    fact_keys: tuple[str, ...] | None = None,
) -> str:
    selected_page_meta_keys = tuple(DEFAULT_PAGE_META_KEYS if page_meta_keys is None else page_meta_keys)
    selected_fact_keys = tuple(DEFAULT_FACT_KEYS if fact_keys is None else fact_keys)
    schema_preview = build_custom_extraction_schema_preview(
        page_meta_keys=selected_page_meta_keys,
        fact_keys=selected_fact_keys,
        include_bbox=False,
    )
    selected_page_meta = ", ".join(selected_page_meta_keys) if selected_page_meta_keys else "(none)"
    selected_fact = ", ".join(selected_fact_keys) if selected_fact_keys else "(none)"
    lines = [
        "You are extracting financial-statement annotations from a single page image.",
        "",
        "Return ONLY valid JSON.",
        "Do NOT return markdown, code fences, comments, prose, or extra keys.",
        "",
        "Return the exact page-level object shown below. Include only the selected page-meta and fact keys.",
        "",
        "Selected page meta keys:",
        f"- {selected_page_meta}",
        "",
        "Selected fact keys:",
        f"- {selected_fact}",
        "",
        "Exact schema:",
        schema_preview,
        "",
        "Rules:",
        "1. Return only a single page-level object with `meta` and `facts`.",
        "2. Extract only visible numeric/table facts. Do not emit standalone labels or headings as facts.",
        "3. Preserve value text exactly as printed, including `%`, commas, parentheses, and dash placeholders.",
        "4. Use JSON `null` for missing optional values. Do not emit the string `\"null\"`.",
        "5. Keep UTF-8 Hebrew directly; do not escape it to unicode sequences.",
        "6. Order facts top-to-bottom; within each row use right-to-left for Hebrew pages and left-to-right for English pages.",
    ]
    next_rule_num = 7
    if "fact_num" in selected_fact_keys:
        lines.append(f"{next_rule_num}. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.")
        next_rule_num += 1
    if "path" in selected_fact_keys:
        lines.append(f"{next_rule_num}. Keep `path` as a list of visible hierarchy labels; use `[]` when unknown.")
        next_rule_num += 1
    if "page_type" in selected_page_meta_keys or "statement_type" in selected_page_meta_keys:
        lines.append(f"{next_rule_num}. Classify page type and statement type from visible page context only.")
        next_rule_num += 1
    lines.extend(
        [
            "",
            "Output formatting:",
            "1. Return the final JSON as a single compact line.",
            "2. Do not pretty-print, indent, add line breaks, or add extra spaces between JSON tokens.",
            "3. Do not add any prefix, suffix, explanation, or surrounding text.",
        ]
    )
    return "\n".join(lines).strip()


def _default_repo_id(api: HfApi, *, doc_id: str, max_pixels: int) -> str:
    owner = str(api.whoami().get("name") or "").strip() or "user"
    return f"{owner}/{inference_dataset_name(doc_id, max_pixels)}"


def _rows_from_export_jsonl(export_path: Path, export_dir: Path) -> list[dict[str, str]]:
    if not export_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    for line in export_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if not isinstance(sample, dict):
            continue
        image_path = sample.get("image")
        system_prompt = sample.get("system")
        instruction = sample.get("instruction")
        doc_id = sample.get("doc_id")
        page_name = sample.get("page_image")
        if not all(isinstance(value, str) and value.strip() for value in (image_path, system_prompt, instruction, doc_id, page_name)):
            continue
        rows.append(
            {
                "image": str((export_dir / str(image_path)).resolve()),
                "system": str(system_prompt),
                "instruction": str(instruction),
                "doc_id": str(doc_id),
                "page_image": str(page_name),
            }
        )
    return rows


def build_hf_inference_dataset_from_export(export_dir: Path) -> tuple[DatasetDict, int]:
    rows = _rows_from_export_jsonl(export_dir / "train.jsonl", export_dir)
    features = Features(
        {
            "image": Image(),
            "system": Value("string"),
            "instruction": Value("string"),
            "doc_id": Value("string"),
            "page_image": Value("string"),
        }
    )
    empty_payload = {"image": [], "system": [], "instruction": [], "doc_id": [], "page_image": []}
    train_ds = Dataset.from_list(rows, features=features) if rows else Dataset.from_dict(empty_payload, features=features)
    return DatasetDict({"train": train_ds}), len(rows)


def export_pdf_for_inference(
    root: Path,
    *,
    doc_id: str,
    images_dir: Path,
    export_dir: Path,
    max_pixels: int,
    page_meta_keys: tuple[str, ...] | None = None,
    fact_keys: tuple[str, ...] | None = None,
) -> int:
    resize_bounds = _resolve_resize_bounds(None, max_pixels)
    resolved_max = resize_bounds[1] if resize_bounds else None
    if resolved_max is None:
        raise RuntimeError("Inference export requires --max-pixels.")

    image_paths = page_image_paths(images_dir)
    if not image_paths:
        raise RuntimeError(f"No page images found in {images_dir}")

    selected_page_meta_keys = tuple(DEFAULT_PAGE_META_KEYS if page_meta_keys is None else page_meta_keys)
    selected_fact_keys = tuple(DEFAULT_FACT_KEYS if fact_keys is None else fact_keys)
    system_prompt = _resolve_system_prompt(root)
    instruction = build_inference_instruction(
        page_meta_keys=selected_page_meta_keys,
        fact_keys=selected_fact_keys,
    )

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    resized_images = 0
    unchanged_images = 0
    for image_path in image_paths:
        rel_img = Path("images") / str(doc_id).strip() / image_path.name
        dst_img = export_dir / rel_img
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        resize_stats = _copy_or_resize_image(
            image_path,
            dst_img,
            min_pixels=None,
            max_pixels=resolved_max,
        )
        if resize_stats["resized"]:
            resized_images += 1
        else:
            unchanged_images += 1
        rows.append(
            json.dumps(
                {
                    "image": rel_img.as_posix(),
                    "system": system_prompt,
                    "instruction": instruction,
                    "doc_id": str(doc_id).strip(),
                    "page_image": image_path.name,
                },
                ensure_ascii=False,
            )
        )

    (export_dir / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (export_dir / "README.md").write_text(
        "# FineTree PDF Inference Dataset\n\n"
        f"- Dataset name: {inference_dataset_name(doc_id, resolved_max)}\n"
        f"- Source doc id: {doc_id}\n"
        f"- Max pixels: {resolved_max}\n"
        f"- Page meta keys: {list(selected_page_meta_keys)}\n"
        f"- Fact keys: {['bbox', *selected_fact_keys]}\n"
        f"- Rows: {len(rows)}\n"
        f"- Resized images: {resized_images}\n"
        f"- Unchanged images: {unchanged_images}\n",
        encoding="utf-8",
    )
    print(
        "EXPORT_STATS:",
        json.dumps(
            {
                "doc_id": str(doc_id).strip(),
                "dataset_name": inference_dataset_name(doc_id, resolved_max),
                "max_pixels": resolved_max,
                "rows": len(rows),
                "resized_images": resized_images,
                "unchanged_images": unchanged_images,
                "page_meta_keys": list(selected_page_meta_keys),
                "fact_keys": list(selected_fact_keys),
            },
            ensure_ascii=False,
        ),
    )
    return len(rows)


def push_inference_dataset(
    dataset: DatasetDict,
    *,
    token: str,
    doc_id: str,
    max_pixels: int,
    repo_id: str | None = None,
    private: bool = False,
) -> str:
    api = HfApi(token=token)
    resolved_repo_id = str(repo_id).strip() if repo_id else _default_repo_id(api, doc_id=doc_id, max_pixels=max_pixels)
    dataset_to_push, kept_splits, dropped_splits = _dataset_for_push(dataset)
    api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)
    for split in kept_splits:
        dataset_to_push[split].push_to_hub(
            repo_id=resolved_repo_id,
            token=token,
            private=private,
            split=split,
        )
    print(f"PUSH_SPLITS: kept={kept_splits} dropped={dropped_splits}")
    return resolved_repo_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push a per-PDF FineTree inference dataset to Hugging Face.")
    parser.add_argument("--doc-id", required=True, help="Workspace document id.")
    parser.add_argument("--images-dir", required=True, help="Directory containing page images for one PDF.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Maximum image pixel budget.")
    parser.add_argument("--repo-id", default=None, help="Optional explicit HF dataset repo id.")
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN/HF_TOKEN/Doppler).")
    parser.add_argument("--export-dir", default=None, help="Optional export directory. Defaults under artifacts/ using the dataset name.")
    parser.add_argument("--page-meta-keys", default=None, help="Comma-separated page meta keys to include in the schema.")
    parser.add_argument("--fact-keys", default=None, help="Comma-separated fact keys to include in the schema. `bbox` stays included.")
    parser.set_defaults(public=True)
    parser.add_argument("--public", dest="public", action="store_true", help="Create/push dataset as public (default).")
    parser.add_argument("--private", dest="public", action="store_false", help="Create/push dataset as private.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(".").resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token, export FINETREE_HF_TOKEN/HF_TOKEN, or configure HF_TOKEN in Doppler."
        )

    doc_id = str(args.doc_id or "").strip()
    images_dir = Path(args.images_dir).expanduser()
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    if not images_dir.is_dir():
        raise RuntimeError(f"Images directory not found: {images_dir}")

    max_pixels = int(args.max_pixels)
    if max_pixels <= 0:
        raise RuntimeError("--max-pixels must be > 0.")

    selected_page_meta_keys = _parse_selected_keys(
        args.page_meta_keys,
        allowed=PROMPT_PAGE_META_KEYS,
        flag_name="--page-meta-keys",
    )
    selected_fact_keys = _parse_selected_keys(
        args.fact_keys,
        allowed=PROMPT_FACT_KEYS,
        flag_name="--fact-keys",
    )
    page_meta_keys = DEFAULT_PAGE_META_KEYS if selected_page_meta_keys is None else selected_page_meta_keys
    fact_keys = DEFAULT_FACT_KEYS if selected_fact_keys is None else selected_fact_keys
    dataset_name = inference_dataset_name(doc_id, max_pixels)
    export_dir = (
        Path(args.export_dir).expanduser()
        if args.export_dir
        else root / "artifacts" / "hf_inference_export" / dataset_name
    )
    if not export_dir.is_absolute():
        export_dir = (root / export_dir).resolve()

    row_count = export_pdf_for_inference(
        root,
        doc_id=doc_id,
        images_dir=images_dir,
        export_dir=export_dir,
        max_pixels=max_pixels,
        page_meta_keys=page_meta_keys,
        fact_keys=fact_keys,
    )
    dataset, dataset_rows = build_hf_inference_dataset_from_export(export_dir)
    pushed_repo_id = push_inference_dataset(
        dataset,
        token=token,
        doc_id=doc_id,
        max_pixels=max_pixels,
        repo_id=args.repo_id,
        private=not args.public,
    )

    print(f"DATASET_NAME: {dataset_name}")
    print(f"DOC_ID: {doc_id}")
    print(f"IMAGES_DIR: {images_dir}")
    print(f"EXPORT_DIR: {export_dir}")
    print(f"ROWS: {row_count}")
    print(f"DATASET_ROWS: {dataset_rows}")
    print(f"MAX_PIXELS: {max_pixels}")
    print(f"PAGE_META_KEYS: {list(page_meta_keys)}")
    print(f"FACT_KEYS: {list(fact_keys)}")
    print(f"PUSHED: {pushed_repo_id}")
    return 0


__all__ = [
    "DEFAULT_FACT_KEYS",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_PAGE_META_KEYS",
    "build_hf_inference_dataset_from_export",
    "export_pdf_for_inference",
    "inference_dataset_name",
    "inference_dataset_short_id",
    "main",
    "parse_args",
    "push_inference_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
