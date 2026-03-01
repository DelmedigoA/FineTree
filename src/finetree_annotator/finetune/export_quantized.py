from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import FinetuneConfig, load_finetune_config


def _resolve_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return None


def _effective_quantization_mode(cfg: FinetuneConfig, explicit_mode: Optional[str]) -> str:
    if explicit_mode:
        return explicit_mode
    if cfg.inference.load_in_4bit:
        return "bnb_4bit"
    return str(cfg.inference.quantization_mode)


def _output_dir_for_mode(base_dir: Path, mode: str) -> Path:
    if mode == "bnb_8bit":
        return base_dir / "merged-8bit"
    if mode == "bnb_4bit":
        return base_dir / "merged-4bit"
    return base_dir / "merged-fp"


def export_quantized_model(
    cfg: FinetuneConfig,
    *,
    source_model: Optional[str] = None,
    quantization_mode: Optional[str] = None,
    output_dir: Optional[Path] = None,
    validate_reload: bool = True,
) -> Path:
    mode = _effective_quantization_mode(cfg, quantization_mode)
    if mode not in {"none", "bnb_8bit", "bnb_4bit"}:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    source_ref = str(source_model or (cfg.run.output_dir / "merged")).strip()
    if not source_ref:
        raise ValueError("source_model must be a valid local path or Hugging Face model id.")

    target_dir = output_dir or _output_dir_for_mode(cfg.run.output_dir, mode)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Export requires transformers.") from exc

    model_kwargs = {
        "trust_remote_code": bool(cfg.inference.trust_remote_code),
        "device_map": str(cfg.inference.device_map),
        "torch_dtype": _resolve_dtype(cfg.inference.torch_dtype),
        "attn_implementation": str(cfg.inference.attn_implementation),
    }
    if mode == "bnb_8bit":
        model_kwargs["load_in_8bit"] = True
    elif mode == "bnb_4bit":
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForImageTextToText.from_pretrained(source_ref, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        source_ref,
        trust_remote_code=bool(cfg.inference.trust_remote_code),
    )

    model.save_pretrained(str(target_dir), safe_serialization=True)
    processor.save_pretrained(str(target_dir))

    manifest = {
        "source_model": source_ref,
        "quantization_mode": mode,
        "attn_implementation": str(cfg.inference.attn_implementation),
        "require_flash_attention": bool(cfg.inference.require_flash_attention),
    }
    (target_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if validate_reload:
        reload_kwargs = dict(model_kwargs)
        AutoModelForImageTextToText.from_pretrained(str(target_dir), **reload_kwargs)
        AutoProcessor.from_pretrained(str(target_dir), trust_remote_code=bool(cfg.inference.trust_remote_code))

    return target_dir


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged Qwen model into quantized artifact format.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    parser.add_argument(
        "--source-model",
        default=None,
        help="Source model path/id (default: <run.output_dir>/merged).",
    )
    parser.add_argument(
        "--quantization-mode",
        default="bnb_8bit",
        choices=["none", "bnb_8bit", "bnb_4bit"],
        help="Quantization mode (default: bnb_8bit).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <run.output_dir>/merged-8bit|merged-4bit|merged-fp).",
    )
    parser.add_argument(
        "--no-validate-reload",
        action="store_true",
        help="Skip reload validation after export.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    exported = export_quantized_model(
        cfg,
        source_model=args.source_model,
        quantization_mode=args.quantization_mode,
        output_dir=output_dir,
        validate_reload=not bool(args.no_validate_reload),
    )
    print(f"Quantized export saved to: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
