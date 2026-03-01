from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from ..finetune.config import FinetuneConfig, load_finetune_config
from ..inference.model_family import is_qwen35_a3_model


def _resolve_model_for_serving(cfg: FinetuneConfig) -> str:
    model_name = str(cfg.inference.model_path or cfg.model.base_model).strip()
    if not model_name:
        raise ValueError("Cannot resolve model id for endpoint from config.")
    return model_name


def _resolve_max_model_len(cfg: FinetuneConfig, explicit_max_model_len: Optional[int]) -> int:
    if explicit_max_model_len is not None:
        return max(int(explicit_max_model_len), 1)
    return max(int(cfg.model.max_seq_length), 512)


def build_vllm_endpoint_env(
    cfg: FinetuneConfig,
    *,
    served_model_name: Optional[str] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.92,
) -> Dict[str, str]:
    if gpu_memory_utilization <= 0.0 or gpu_memory_utilization > 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")

    model_name = _resolve_model_for_serving(cfg)
    env: Dict[str, str] = {
        "MODEL_NAME": model_name,
        "MAX_MODEL_LEN": str(_resolve_max_model_len(cfg, max_model_len)),
        "GPU_MEMORY_UTILIZATION": f"{gpu_memory_utilization:.2f}",
    }

    if served_model_name:
        env["OPENAI_SERVED_MODEL_NAME_OVERRIDE"] = served_model_name.strip()

    adapter_ref = str(cfg.inference.adapter_path or "").strip()
    if adapter_ref:
        # RunPod can use this in custom handlers, and it is useful metadata even when
        # the endpoint uses merged weights.
        env["FINETREE_ADAPTER_REF"] = adapter_ref

    if is_qwen35_a3_model(model_name) or is_qwen35_a3_model(cfg.model.base_model):
        env["FINETREE_MODEL_FAMILY"] = "qwen3_5_a3"
    else:
        env["FINETREE_MODEL_FAMILY"] = "generic"

    return env


def write_env_file(env_values: Dict[str, str], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(env_values.items())]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RunPod endpoint env values from FineTree config.")
    parser.add_argument("--config", required=True, help="FineTree YAML config path.")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="OpenAI-compatible model alias exposed by the endpoint (optional).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override MAX_MODEL_LEN env value (default comes from config model.max_seq_length).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Set GPU_MEMORY_UTILIZATION (default: 0.92).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/runpod/endpoint.env",
        help="Output .env file path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    env_values = build_vllm_endpoint_env(
        cfg,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    output_path = write_env_file(env_values, Path(args.output))
    print(f"Wrote endpoint env file: {output_path}")
    for key in sorted(env_values.keys()):
        print(f"{key}={env_values[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
