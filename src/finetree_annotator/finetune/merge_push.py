from __future__ import annotations

import argparse
from pathlib import Path

from .config import FinetuneConfig, load_finetune_config, resolve_hf_token


def _resolve_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return None


def merge_and_optionally_push(cfg: FinetuneConfig, push: bool = False) -> Path:
    adapter_dir = cfg.run.output_dir / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. Run finetree-ft-train first."
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Merge dependencies missing. Install transformers, peft, and torch."
        ) from exc

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        torch_dtype=_resolve_dtype(cfg.model.dtype),
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    merged_dir = cfg.run.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))

    should_push = push and cfg.push_to_hub.enabled
    if should_push:
        from huggingface_hub import HfApi

        repo_id = str(cfg.push_to_hub.repo_id)
        token = resolve_hf_token(cfg.push_to_hub)
        if not token:
            raise RuntimeError("Hugging Face token missing for push_to_hub.")
        api = HfApi(token=token)
        merged_repo_id = f"{repo_id}-merged"
        api.create_repo(repo_id=merged_repo_id, private=True, exist_ok=True)
        api.upload_folder(
            repo_id=merged_repo_id,
            folder_path=str(merged_dir),
            commit_message="Upload merged FineTree model",
        )

    return merged_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and optionally push merged weights.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    parser.add_argument("--push", action="store_true", help="Push merged model to Hugging Face Hub.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    merged_dir = merge_and_optionally_push(cfg, push=bool(args.push))
    print(f"Merged model saved to: {merged_dir}")
    return 0


__all__ = ["merge_and_optionally_push", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
