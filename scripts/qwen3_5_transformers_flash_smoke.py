from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swift-parity memory smoke for Qwen3.5 with Transformers.")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    p.add_argument("--dataset-id", default="asafd60/FineTree-annotated-pages")
    p.add_argument("--output-dir", default="artifacts/qwen35_transformers_flash_smoke")
    p.add_argument("--target-image-pixels", type=int, default=100_000)
    p.add_argument("--processor-min-pixels", type=int, default=100_000)
    p.add_argument("--processor-max-pixels", type=int, default=100_000)
    p.add_argument("--max-train-steps", type=int, default=3)
    p.add_argument("--max-train-samples", type=int, default=32)
    p.add_argument("--max-eval-samples", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--disable-eval", action="store_true")
    p.add_argument("--push-merged", action="store_true")
    return p.parse_args()


def cuda_mem(profile: list[dict[str, Any]], tag: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any]
    if not torch.cuda.is_available():
        payload = {"tag": tag, "cuda": False}
    else:
        payload = {
            "tag": tag,
            "cuda": True,
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        }
    if extra:
        payload.update(extra)
    profile.append(payload)
    print(payload)


def _is_vision_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["vision", "visual", "vit", "image_tower", "vision_tower"])


def _is_aligner_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["align", "projector", "mm_projector", "multi_modal_projector", "connector"])


def apply_freeze_policy(model_obj: Any, freeze_vision: bool = True, freeze_aligner: bool = True) -> None:
    frozen_v = 0
    frozen_a = 0
    for n, p in model_obj.named_parameters():
        if freeze_vision and _is_vision_name(n):
            if p.requires_grad:
                p.requires_grad = False
                frozen_v += 1
        if freeze_aligner and _is_aligner_name(n):
            if p.requires_grad:
                p.requires_grad = False
                frozen_a += 1
    print(f"Frozen vision params: {frozen_v}, aligner params: {frozen_a}")


def trainability_audit(model_obj: Any) -> dict[str, dict[str, int]]:
    groups = {
        "llm": {"total": 0, "trainable": 0},
        "vision": {"total": 0, "trainable": 0},
        "aligner": {"total": 0, "trainable": 0},
        "lora": {"total": 0, "trainable": 0},
    }
    for n, p in model_obj.named_parameters():
        c = p.numel()
        n_low = n.lower()
        if "lora_" in n_low:
            key = "lora"
        elif _is_vision_name(n):
            key = "vision"
        elif _is_aligner_name(n):
            key = "aligner"
        else:
            key = "llm"
        groups[key]["total"] += c
        if p.requires_grad:
            groups[key]["trainable"] += c
    for k, v in groups.items():
        ratio = (v["trainable"] / max(1, v["total"])) * 100.0
        print(f"{k:8s} total={v['total']:,} trainable={v['trainable']:,} ({ratio:.4f}%)")
    return groups


def resize_image_to_pixel_budget(image: Image.Image, pixel_budget: int):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    width, height = image.size
    original_pixels = width * height
    if original_pixels <= pixel_budget:
        return image

    scale = math.sqrt(pixel_budget / float(original_pixels))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    if new_w * new_h > pixel_budget:
        shrink = math.sqrt(pixel_budget / float(new_w * new_h))
        new_w = max(1, int(new_w * shrink))
        new_h = max(1, int(new_h * shrink))
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return resized


def build_conversation(system_message: str, sample: dict[str, Any], include_assistant: bool = True):
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["instruction"]},
            ],
        },
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]})
    return messages


def resolve_hf_token() -> str | None:
    for name in ("HF_TOKEN", "FINETREE_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out = Path(args.output_dir)
    adapter_dir = out / "adapter"
    merged_dir = out / "merged"
    memory_path = out / "memory_profile.json"
    parity_path = out / "parity_report.json"
    out.mkdir(parents=True, exist_ok=True)

    system_message = "You are a careful JSON extraction assistant. Return only valid JSON."
    memory_profile: list[dict[str, Any]] = []

    try:
        import flash_attn  # noqa: F401
        requested_attn_impl = "flash_attention_2"
    except Exception:
        requested_attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=requested_attn_impl,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=int(args.processor_min_pixels),
        max_pixels=int(args.processor_max_pixels),
    )

    model.gradient_checkpointing_enable()
    attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
    attn_impl_in_use = str(attn_impl) if attn_impl is not None else requested_attn_impl
    cuda_mem(memory_profile, "after_model_load", {"attn_impl": attn_impl_in_use})

    raw_ds = load_dataset(args.dataset_id)
    train_split = raw_ds["train"]
    eval_split = raw_ds["validation"] if "validation" in raw_ds else train_split.select(range(min(8, len(train_split))))
    train_smoke = train_split.select(range(min(args.max_train_samples, len(train_split))))
    eval_smoke = eval_split.select(range(min(args.max_eval_samples, len(eval_split))))

    apply_freeze_policy(model, freeze_vision=True, freeze_aligner=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    audit_stats = trainability_audit(model)

    vision_audit_once = {"printed": False}

    def collate_fn(examples):
        formatted = [build_conversation(system_message, ex, include_assistant=True) for ex in examples]
        texts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            for conv in formatted
        ]
        resized_images = [resize_image_to_pixel_budget(ex["image"], args.target_image_pixels) for ex in examples]

        batch = processor(text=texts, images=resized_images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels

        if not vision_audit_once["printed"]:
            pv = batch.get("pixel_values")
            shape = list(pv.shape) if hasattr(pv, "shape") else None
            est = None
            if hasattr(pv, "ndim") and pv.ndim == 4:
                _, _, h, w = pv.shape
                est = int((h * w) / (14 * 14))
            print({"pixel_values_shape": shape, "visual_tokens_estimate": est})
            vision_audit_once["printed"] = True
        return batch

    cuda_mem(memory_profile, "before_forward_probe")
    probe_batch = collate_fn([train_smoke[0]])
    probe_batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in probe_batch.items()}
    probe_out = model(**probe_batch)
    cuda_mem(memory_profile, "after_forward", {"probe_loss": float(probe_out.loss.detach().float().cpu().item())})
    probe_out.loss.backward()
    cuda_mem(memory_profile, "after_backward")
    probe_opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-9)
    probe_opt.step()
    probe_opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    cuda_mem(memory_profile, "after_optimizer_step")
    del probe_opt, probe_out, probe_batch
    torch.cuda.empty_cache()

    sft_params = set(inspect.signature(SFTConfig).parameters.keys())
    do_eval_flag = (not args.disable_eval) and len(eval_smoke) > 0
    cfg_kwargs = {
        "output_dir": str(out),
        "max_steps": int(args.max_train_steps),
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 6,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 5,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "bf16": True,
        "report_to": "none",
        "warmup_ratio": 0.05,
        "max_length": int(args.max_length),
        "group_by_length": True,
        "do_eval": bool(do_eval_flag),
        "gradient_checkpointing": True,
    }
    if "eval_strategy" in sft_params:
        cfg_kwargs["eval_strategy"] = "steps" if do_eval_flag else "no"
    elif "evaluation_strategy" in sft_params:
        cfg_kwargs["evaluation_strategy"] = "steps" if do_eval_flag else "no"
    if do_eval_flag and "eval_steps" in sft_params:
        cfg_kwargs["eval_steps"] = 5
    sft_config = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in sft_params})

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_smoke,
        "eval_dataset": eval_smoke if do_eval_flag else None,
        "data_collator": collate_fn,
    }
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = ""
    if "dataset_kwargs" in trainer_params:
        trainer_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    cuda_mem(memory_profile, "after_train")
    metrics: dict[str, Any] = {}
    if do_eval_flag:
        metrics = trainer.evaluate()
        cuda_mem(memory_profile, "after_eval")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    memory_path.write_text(json.dumps({"model_id": args.model_id, "dataset_id": args.dataset_id, "entries": memory_profile}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    parity_payload = {
        "swift_parity_config": {
            "enable_gradient_checkpointing": True,
            "freeze_vision": True,
            "freeze_aligner": True,
            "processor_min_pixels": args.processor_min_pixels,
            "processor_max_pixels": args.processor_max_pixels,
            "smoke_disable_eval": args.disable_eval,
            "max_length": args.max_length,
            "gradient_accumulation_steps": 6,
            "learning_rate": 1e-4,
        },
        "audit_stats": audit_stats,
        "sweep_plan": [
            {"name": "A_parity_fixed_pixels", "processor_max_pixels": args.processor_max_pixels, "do_eval": False},
            {"name": "B_pixel_sweep", "processor_max_pixels": [50_000, 100_000, 200_000], "do_eval": False},
            {"name": "C_eval_cadence", "processor_max_pixels": args.processor_max_pixels, "do_eval": True},
        ],
        "metrics": metrics,
    }
    parity_path.write_text(json.dumps(parity_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    del trainer
    torch.cuda.empty_cache()

    merge_base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl_in_use,
    )
    merge_model = PeftModel.from_pretrained(merge_base, str(adapter_dir))
    merged_model = merge_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))

    if args.push_merged:
        token = resolve_hf_token()
        if not token:
            raise RuntimeError("Missing HF token env var")
        base_repo_id = (os.getenv("HF_REPO_ID") or "").strip()
        if not base_repo_id:
            raise RuntimeError("HF_REPO_ID is required")
        merged_repo_id = f"{base_repo_id}-merged"
        api = HfApi(token=token)
        api.create_repo(repo_id=merged_repo_id, private=False, exist_ok=True)
        api.upload_folder(repo_id=merged_repo_id, folder_path=str(merged_dir), commit_message=f"Upload merged {args.model_id} smoke model")
        print(f"Pushed merged model to: {merged_repo_id}")

    print(f"Done. Artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
