from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

from ..schema_contract import PROMPT_PAGE_META_KEYS, build_custom_extraction_prompt_template
from .push_dataset_hub import build_dataset, resolve_hf_token
from .push_dataset_hub_no_bbox import (
    build_hf_dataset_no_bbox_from_export,
    export_for_hf_no_bbox,
    push_train_validation_separately_no_bbox,
)

DEFAULT_CONFIG_PATH = Path("configs/finetune_qwen35a3_vl.yaml")
DEFAULT_EXPORT_DIR = Path("artifacts/hf_public_dataset_no_bbox")
DEFAULT_VALIDATION_DOC_IDS: tuple[str, ...] = ("test",)
DEFAULT_REPO_BASENAME = "finetree-v2.5-not-approved"
DEFAULT_MAX_PIXELS = 1_400_000
MAX_PIXELS_ENV_VAR = "FINETREE_PUBLIC_DATASET_MAX_PIXELS"
PUBLIC_DATASET_FACT_KEYS: tuple[str, ...] = (
    "value",
    "fact_num",
    "comment_ref",
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
)
PUBLIC_DATASET_EXCLUDED_VALUE_CONTEXTS: tuple[str, ...] = ("mixed", "textual")


def _default_public_repo_base(api: HfApi) -> str:
    owner = str(api.whoami().get("name") or "").strip() or "user"
    return f"{owner}/{DEFAULT_REPO_BASENAME}"


def _resolve_max_pixels() -> int:
    raw_value = str(os.getenv(MAX_PIXELS_ENV_VAR) or "").strip()
    if not raw_value:
        return DEFAULT_MAX_PIXELS
    try:
        parsed = int(raw_value)
    except Exception as exc:
        raise RuntimeError(f"{MAX_PIXELS_ENV_VAR} must be an integer when set.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{MAX_PIXELS_ENV_VAR} must be > 0 when set.")
    return parsed


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise RuntimeError("This command takes no flags. Export HF_TOKEN or FINETREE_HF_TOKEN, then run it directly.")

    root = Path(".").resolve()
    token = resolve_hf_token(None)
    if not token:
        raise RuntimeError(
            "Missing HF token. Export FINETREE_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or HUGGINGFACEHUB_API_TOKEN."
        )

    config_path = (root / DEFAULT_CONFIG_PATH).resolve()
    export_dir = (root / DEFAULT_EXPORT_DIR).resolve()
    max_pixels = _resolve_max_pixels()
    prompt_template = build_custom_extraction_prompt_template(
        page_meta_keys=PROMPT_PAGE_META_KEYS,
        fact_keys=PUBLIC_DATASET_FACT_KEYS,
        include_bbox=False,
    )

    build_dataset(
        config_path,
        validation_doc_ids=set(DEFAULT_VALIDATION_DOC_IDS),
        approved_pages_only=False,
        prompt_template_override=prompt_template,
        selected_page_meta_keys=PROMPT_PAGE_META_KEYS,
        selected_fact_keys=PUBLIC_DATASET_FACT_KEYS,
        page_only_wrapper=True,
        excluded_value_contexts=PUBLIC_DATASET_EXCLUDED_VALUE_CONTEXTS,
        include_empty_pages_override=True,
        dedupe_exact_facts=True,
    )
    train_rows, val_rows = export_for_hf_no_bbox(
        root,
        export_dir,
        instruction_mode="source",
        max_pixels=max_pixels,
    )
    dataset, _, _ = build_hf_dataset_no_bbox_from_export(export_dir, instruction_mode="source")

    api = HfApi(token=token)
    base_repo_id = _default_public_repo_base(api)
    pushed = push_train_validation_separately_no_bbox(
        dataset,
        token=token,
        base_repo_id=str(base_repo_id),
        private=False,
    )

    print(f"CONFIG: {config_path}")
    print(f"EXPORT_DIR: {export_dir}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"PAGE_META_KEYS: {list(PROMPT_PAGE_META_KEYS)}")
    print(f"FACT_KEYS: {list(PUBLIC_DATASET_FACT_KEYS)}")
    print(f"EXCLUDED_VALUE_CONTEXTS: {list(PUBLIC_DATASET_EXCLUDED_VALUE_CONTEXTS)}")
    print(f"VALIDATION_DOC_IDS: {list(DEFAULT_VALIDATION_DOC_IDS)}")
    print(f"BASE_REPO_ID: {base_repo_id}")
    print(f"MAX_PIXELS: {max_pixels}")
    print("DEDUPE_EXACT_FACTS: True")
    print("APPROVED_PAGES_ONLY: False")
    print("INCLUDE_EMPTY_PAGES: True")
    print(f"PUSHED_TRAIN_REPO: {pushed.get('train')}")
    print(f"PUSHED_VALIDATION_REPO: {pushed.get('validation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
