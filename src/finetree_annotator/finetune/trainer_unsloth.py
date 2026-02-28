from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
from pathlib import Path
from typing import Any

from .config import FinetuneConfig, load_finetune_config, resolve_hf_token
from .dataset_builder import build_unsloth_chat_datasets

_MIN_VISION_MAX_SEQ_LENGTH = 512
_RECOMMENDED_VISION_MAX_SEQ_LENGTH = 768


def _ensure_cuda_available() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run fine-tuning on a CUDA-enabled GPU VM. "
            "No CPU fallback is enabled by default."
        )


def _verify_training_dependencies() -> None:
    required_modules = (
        "datasets",
        "trl",
        "transformers",
        "unsloth",
        "huggingface_hub",
        "peft",
    )
    missing: list[str] = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Training dependencies are missing. Install required modules: "
            + ", ".join(sorted(missing))
        )


def _dtype_from_config(cfg: FinetuneConfig) -> Any:
    import torch

    if cfg.model.dtype == "bfloat16":
        return torch.bfloat16
    if cfg.model.dtype == "float16":
        return torch.float16
    if cfg.model.dtype == "float32":
        return torch.float32
    return None


def _prepare_datasets(cfg: FinetuneConfig) -> None:
    if not cfg.data.output_train_jsonl.is_file() or not cfg.data.output_val_jsonl.is_file():
        build_unsloth_chat_datasets(cfg)


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)  # validate line format early
            count += 1
    return count


def _push_adapter_folder(cfg: FinetuneConfig, adapter_dir: Path) -> None:
    if not cfg.push_to_hub.enabled:
        return
    if cfg.push_to_hub.default_artifact not in {"adapters_only", "both"}:
        return

    from huggingface_hub import HfApi

    token = resolve_hf_token(cfg.push_to_hub)
    if not token:
        raise RuntimeError("Hugging Face token missing for push_to_hub.")
    api = HfApi(token=token)
    api.create_repo(repo_id=str(cfg.push_to_hub.repo_id), private=True, exist_ok=True)
    api.upload_folder(
        repo_id=str(cfg.push_to_hub.repo_id),
        folder_path=str(adapter_dir),
        commit_message="Upload FineTree LoRA adapters",
    )


def _build_vision_data_collator(collator_cls: Any, model: Any, tokenizer: Any) -> Any:
    try:
        params = inspect.signature(collator_cls).parameters
    except Exception:
        params = {}

    kwargs: dict[str, Any] = {}
    if "model" in params:
        kwargs["model"] = model
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processor" in params:
        kwargs["processor"] = tokenizer

    if kwargs:
        return collator_cls(**kwargs)

    for fallback_kwargs in (
        {"model": model, "tokenizer": tokenizer},
        {"model": model},
        {"processor": tokenizer},
        {"tokenizer": tokenizer},
        {},
    ):
        try:
            return collator_cls(**fallback_kwargs)
        except TypeError:
            continue

    raise RuntimeError("Failed to initialize UnslothVisionDataCollator with current Unsloth version.")


def _adapter_gc_value(raw: str) -> str | bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    return raw


def _build_peft_kwargs(fast_vision_model: Any, cfg: FinetuneConfig) -> dict[str, Any]:
    target_modules: Any = cfg.adapter.target_modules
    if target_modules == "auto":
        target_modules = None

    requested_kwargs: dict[str, Any] = {
        "r": int(cfg.adapter.r),
        "lora_alpha": int(cfg.adapter.alpha),
        "lora_dropout": float(cfg.adapter.dropout),
        "bias": str(cfg.adapter.bias),
        "use_rslora": bool(cfg.adapter.use_rslora),
        "target_modules": target_modules,
        # Unsloth supports "unsloth" here for maximum memory savings.
        "use_gradient_checkpointing": _adapter_gc_value(str(cfg.adapter.gradient_checkpointing)),
        "random_state": int(cfg.run.seed),
        "finetune_vision_layers": bool(cfg.adapter.finetune_vision_layers),
        "finetune_language_layers": bool(cfg.adapter.finetune_language_layers),
        "finetune_attention_modules": bool(cfg.adapter.finetune_attention_modules),
        "finetune_mlp_modules": bool(cfg.adapter.finetune_mlp_modules),
    }

    try:
        params = inspect.signature(fast_vision_model.get_peft_model).parameters
    except Exception:
        return requested_kwargs

    return {key: value for key, value in requested_kwargs.items() if key in params}


def _effective_vision_max_seq_length(configured: int) -> int:
    if configured >= _MIN_VISION_MAX_SEQ_LENGTH:
        return configured
    print(
        "[WARN] model.max_seq_length="
        f"{configured} is too small for Qwen3.5-VL image tokenization. "
        f"Using {_MIN_VISION_MAX_SEQ_LENGTH} instead."
    )
    return _MIN_VISION_MAX_SEQ_LENGTH


def _extract_mm_token_count(error_message: str, key: str) -> int | None:
    match = re.search(rf"{key}=\[(\d+)\]", error_message)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def run_training(cfg: FinetuneConfig, dry_run: bool = False) -> Path:
    _ensure_cuda_available()
    _verify_training_dependencies()
    _prepare_datasets(cfg)

    train_rows = _count_jsonl_rows(cfg.data.output_train_jsonl)
    if train_rows == 0:
        raise RuntimeError(
            f"Training dataset is empty: {cfg.data.output_train_jsonl}. "
            "Check annotations_glob/images/prompt settings and rerun dataset build."
        )
    eval_rows = _count_jsonl_rows(cfg.data.output_val_jsonl)

    if dry_run:
        return cfg.run.output_dir / "adapter"

    try:
        from unsloth import FastVisionModel, UnslothVisionDataCollator

        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Training dependencies are missing. Install unsloth + trl + datasets + transformers."
        ) from exc

    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.run.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    effective_max_seq_length = _effective_vision_max_seq_length(int(cfg.model.max_seq_length))

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=effective_max_seq_length,
        dtype=_dtype_from_config(cfg),
        load_in_4bit=bool(cfg.model.load_in_4bit),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )

    # QLoRA is exposed by config, but MoE-style models often perform better with standard LoRA.
    model = FastVisionModel.get_peft_model(model, **_build_peft_kwargs(FastVisionModel, cfg))

    train_ds = load_dataset("json", data_files=str(cfg.data.output_train_jsonl), split="train")
    eval_ds = load_dataset("json", data_files=str(cfg.data.output_val_jsonl), split="train") if eval_rows > 0 else None

    args = TrainingArguments(
        output_dir=str(cfg.run.output_dir),
        seed=int(cfg.run.seed),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        learning_rate=float(cfg.training.learning_rate),
        num_train_epochs=float(cfg.training.num_train_epochs),
        warmup_ratio=float(cfg.training.warmup_ratio),
        weight_decay=float(cfg.training.weight_decay),
        lr_scheduler_type=str(cfg.training.lr_scheduler_type),
        optim=str(cfg.training.optim),
        logging_steps=int(cfg.training.logging_steps),
        save_strategy=str(cfg.training.save_strategy),
        save_steps=int(cfg.training.save_steps),
        bf16=bool(cfg.training.bf16),
        fp16=bool(cfg.training.fp16),
        report_to=str(cfg.training.report_to),
        ddp_find_unused_parameters=bool(cfg.training.ddp_find_unused_parameters),
        dataloader_num_workers=int(cfg.infra.dataloader_num_workers),
        remove_unused_columns=False,
    )

    data_collator = _build_vision_data_collator(UnslothVisionDataCollator, model, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    FastVisionModel.for_training(model)
    try:
        trainer.train()
    except ValueError as exc:
        msg = str(exc)
        if "Mismatch in `image` token count" in msg and "truncation='max_length'" in msg:
            text_tokens = _extract_mm_token_count(msg, "text")
            recommended = _RECOMMENDED_VISION_MAX_SEQ_LENGTH
            if text_tokens is not None:
                # Keep headroom for prompt + answer tokens on top of image tokens.
                recommended = max(recommended, text_tokens + 128)
            raise RuntimeError(
                "Multimodal tokenization failed due to sequence truncation. "
                f"Configured model.max_seq_length={cfg.model.max_seq_length}; "
                f"effective={effective_max_seq_length}. "
                f"Increase model.max_seq_length to at least {recommended} and retry."
            ) from exc
        raise

    if trainer.is_world_process_zero():
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        _push_adapter_folder(cfg, adapter_dir)
    return adapter_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal Qwen model with Unsloth from FineTree datasets.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate environment and config without training.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    adapter_dir = run_training(cfg, dry_run=bool(args.dry_run))
    if args.dry_run:
        print(f"Dry-run successful. Adapter output path: {adapter_dir}")
    else:
        print(f"Training complete. Adapter saved to: {adapter_dir}")
    return 0


__all__ = ["run_training", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
