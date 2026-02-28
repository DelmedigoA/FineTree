from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from .config import FinetuneConfig, load_finetune_config, resolve_hf_token
from .dataset_builder import build_unsloth_chat_datasets


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

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        dtype=_dtype_from_config(cfg),
        load_in_4bit=bool(cfg.model.load_in_4bit),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )

    # QLoRA is exposed by config, but MoE-style models often perform better with standard LoRA.
    model = FastVisionModel.get_peft_model(
        model,
        r=int(cfg.adapter.r),
        lora_alpha=int(cfg.adapter.alpha),
        lora_dropout=float(cfg.adapter.dropout),
        bias=str(cfg.adapter.bias),
        use_rslora=bool(cfg.adapter.use_rslora),
        target_modules=None if cfg.adapter.target_modules == "auto" else cfg.adapter.target_modules,
    )

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
        dataloader_num_workers=int(cfg.infra.dataloader_num_workers),
        remove_unused_columns=False,
    )

    data_collator = UnslothVisionDataCollator(model=model, tokenizer=tokenizer)

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
    trainer.train()

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
