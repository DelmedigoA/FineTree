from __future__ import annotations

import argparse
import base64
import io
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from ..inference.auth import resolve_hf_token_from_env

DEFAULT_MODEL_ID = "asafd60/Qwen3.5-27B_finetuned"
DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_IMAGE_FACTOR = 28


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Simple API config must be a mapping object: {path}")
    return raw


@dataclass
class RuntimeConfig:
    model_id: str = DEFAULT_MODEL_ID
    torch_dtype: str = "bfloat16"
    device: str = "auto"  # auto|cuda|cpu
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = "flash_attention_2"
    max_pixels: int = DEFAULT_MAX_PIXELS
    image_factor: int = DEFAULT_IMAGE_FACTOR
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    enable_thinking: bool = False


def load_runtime_config(config_path: Optional[str] = None) -> RuntimeConfig:
    cfg = RuntimeConfig()

    path = config_path or os.getenv("FINETREE_SIMPLE_API_CONFIG")
    if path:
        config_file = Path(path).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Simple API config not found: {config_file}")
        raw = _load_yaml(config_file)
        if "model_id" in raw:
            cfg.model_id = str(raw["model_id"]).strip() or cfg.model_id
        if "torch_dtype" in raw:
            cfg.torch_dtype = str(raw["torch_dtype"]).strip() or cfg.torch_dtype
        if "device" in raw:
            cfg.device = str(raw["device"]).strip() or cfg.device
        if "device_map" in raw:
            cfg.device_map = None if raw["device_map"] is None else str(raw["device_map"]).strip()
        if "trust_remote_code" in raw:
            cfg.trust_remote_code = bool(raw["trust_remote_code"])
        if "attn_implementation" in raw:
            cfg.attn_implementation = (
                None if raw["attn_implementation"] is None else str(raw["attn_implementation"]).strip()
            )
        if "max_pixels" in raw:
            cfg.max_pixels = int(raw["max_pixels"])
        if "image_factor" in raw:
            cfg.image_factor = int(raw["image_factor"])
        if "max_new_tokens" in raw:
            cfg.max_new_tokens = int(raw["max_new_tokens"])
        if "temperature" in raw:
            cfg.temperature = float(raw["temperature"])
        if "top_p" in raw:
            cfg.top_p = float(raw["top_p"])
        if "do_sample" in raw:
            cfg.do_sample = bool(raw["do_sample"])
        if "enable_thinking" in raw:
            cfg.enable_thinking = bool(raw["enable_thinking"])

    # Env overrides
    cfg.model_id = str(
        os.getenv("FINETREE_SIMPLE_API_MODEL_ID")
        or os.getenv("MODEL_ID")
        or cfg.model_id
    ).strip()
    cfg.torch_dtype = str(os.getenv("FINETREE_SIMPLE_API_TORCH_DTYPE") or cfg.torch_dtype).strip().lower()
    cfg.device = str(os.getenv("FINETREE_SIMPLE_API_DEVICE") or cfg.device).strip().lower()
    if "FINETREE_SIMPLE_API_DEVICE_MAP" in os.environ:
        raw_device_map = str(os.getenv("FINETREE_SIMPLE_API_DEVICE_MAP") or "").strip()
        cfg.device_map = raw_device_map or None
    cfg.trust_remote_code = _env_bool("FINETREE_SIMPLE_API_TRUST_REMOTE_CODE", cfg.trust_remote_code)
    if "FINETREE_SIMPLE_API_ATTN_IMPLEMENTATION" in os.environ:
        raw_attn = str(os.getenv("FINETREE_SIMPLE_API_ATTN_IMPLEMENTATION") or "").strip()
        cfg.attn_implementation = raw_attn or None
    if "FINETREE_SIMPLE_API_MAX_PIXELS" in os.environ:
        cfg.max_pixels = int(str(os.getenv("FINETREE_SIMPLE_API_MAX_PIXELS")).strip())
    if "FINETREE_SIMPLE_API_IMAGE_FACTOR" in os.environ:
        cfg.image_factor = int(str(os.getenv("FINETREE_SIMPLE_API_IMAGE_FACTOR")).strip())
    if "FINETREE_SIMPLE_API_MAX_NEW_TOKENS" in os.environ:
        cfg.max_new_tokens = int(str(os.getenv("FINETREE_SIMPLE_API_MAX_NEW_TOKENS")).strip())
    if "FINETREE_SIMPLE_API_TEMPERATURE" in os.environ:
        cfg.temperature = float(str(os.getenv("FINETREE_SIMPLE_API_TEMPERATURE")).strip())
    if "FINETREE_SIMPLE_API_TOP_P" in os.environ:
        cfg.top_p = float(str(os.getenv("FINETREE_SIMPLE_API_TOP_P")).strip())
    cfg.do_sample = _env_bool("FINETREE_SIMPLE_API_DO_SAMPLE", cfg.do_sample)
    cfg.enable_thinking = _env_bool("FINETREE_SIMPLE_API_ENABLE_THINKING", cfg.enable_thinking)

    if cfg.max_pixels <= 0:
        raise ValueError("max_pixels must be > 0")
    if cfg.image_factor <= 0:
        raise ValueError("image_factor must be > 0")
    if cfg.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if cfg.top_p <= 0 or cfg.top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if cfg.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return cfg


def _resolve_torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": None,
        "none": None,
    }
    return mapping.get(name.lower())


def _resolve_device(mode: str) -> str:
    import torch

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FINETREE_SIMPLE_API_DEVICE=cuda but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resize_to_qwen_max_pixels(image: Image.Image, *, max_pixels: int, factor: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    pixels = width * height
    scale = 1.0 if pixels <= max_pixels else (max_pixels / float(pixels)) ** 0.5
    target_w = max(1, int(width * scale))
    target_h = max(1, int(height * scale))

    # Keep Qwen vision-grid alignment.
    target_w = max(factor, (target_w // factor) * factor)
    target_h = max(factor, (target_h // factor) * factor)

    # Ensure final size is within pixel budget after factor snapping.
    while target_w * target_h > max_pixels and (target_w > factor or target_h > factor):
        if target_w >= target_h and target_w > factor:
            target_w -= factor
        elif target_h > factor:
            target_h -= factor
        else:
            break

    if (target_w, target_h) == (width, height):
        return image
    return image.resize((target_w, target_h), Image.Resampling.BICUBIC)


def decode_image(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc


def _model_input_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _load_model(cfg: RuntimeConfig) -> tuple[Any, Any, str]:
    import torch
    import transformers

    device = _resolve_device(cfg.device)
    dtype = _resolve_torch_dtype(cfg.torch_dtype)
    if device == "cpu":
        dtype = None

    token = resolve_hf_token_from_env()
    common_kwargs: dict[str, Any] = {"trust_remote_code": bool(cfg.trust_remote_code)}
    if token:
        common_kwargs["token"] = token

    processor = transformers.AutoProcessor.from_pretrained(cfg.model_id, **common_kwargs)

    model_kwargs: dict[str, Any] = dict(common_kwargs)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    if device == "cuda":
        model_kwargs["device_map"] = cfg.device_map or "auto"

    model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_cls is None:
        model_cls = transformers.AutoModelForImageTextToText

    try:
        model = model_cls.from_pretrained(cfg.model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = model_cls.from_pretrained(cfg.model_id, **model_kwargs)
    except Exception:
        if model_cls is not transformers.AutoModelForImageTextToText:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            model = transformers.AutoModelForImageTextToText.from_pretrained(cfg.model_id, **fallback_kwargs)
        else:
            raise

    if device == "cpu":
        model.to("cpu")
    elif device == "cuda" and model_kwargs.get("device_map") is None:
        model.to("cuda")
    model.eval()
    return model, processor, device


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]], enable_thinking: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": enable_thinking,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


class InferRequest(BaseModel):
    image_b64: str
    prompt: str
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    do_sample: Optional[bool] = None
    max_pixels: Optional[int] = Field(default=None, ge=1)


def create_app(
    *,
    config_path: Optional[str] = None,
    preloaded: Optional[tuple[Any, Any, str]] = None,
    load_on_startup: bool = True,
) -> FastAPI:
    cfg = load_runtime_config(config_path)
    initial_model = preloaded[0] if preloaded is not None else None
    initial_processor = preloaded[1] if preloaded is not None else None
    initial_device = preloaded[2] if preloaded is not None else None

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if load_on_startup and (app.state.model is None or app.state.processor is None):
            model, processor, device = _load_model(cfg)
            app.state.model = model
            app.state.processor = processor
            app.state.device = device
        yield

    app = FastAPI(title="FineTree Simple Qwen Inference API", version="1.0.0", lifespan=_lifespan)
    app.state.cfg = cfg
    app.state.model = initial_model
    app.state.processor = initial_processor
    app.state.device = initial_device

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": cfg.model_id,
            "device": app.state.device or "unloaded",
            "loaded": bool(app.state.model is not None and app.state.processor is not None),
        }

    @app.post("/infer")
    def infer(req: InferRequest) -> dict[str, Any]:
        model = app.state.model
        processor = app.state.processor
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        image = decode_image(req.image_b64)
        image = resize_to_qwen_max_pixels(
            image,
            max_pixels=int(req.max_pixels or cfg.max_pixels),
            factor=int(cfg.image_factor),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": req.prompt},
                ],
            }
        ]

        inputs = _apply_chat_template(processor, messages, enable_thinking=bool(cfg.enable_thinking))
        device = _model_input_device(model)
        if device is not None:
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        import torch

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(req.max_new_tokens or cfg.max_new_tokens),
                temperature=float(cfg.temperature if req.temperature is None else req.temperature),
                top_p=float(cfg.top_p if req.top_p is None else req.top_p),
                do_sample=bool(cfg.do_sample if req.do_sample is None else req.do_sample),
            )

        input_ids = inputs.get("input_ids")
        if input_ids is not None and generated is not None and len(generated) > 0:
            new_tokens = generated[0][input_ids.shape[-1] :]
        else:
            new_tokens = generated[0]
        text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"text": text}

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineTree simple Qwen image->text inference API.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("FINETREE_SIMPLE_API_PORT", "8000"))),
        help="Bind port (default: 8000 or $PORT)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Simple API requires uvicorn. Install with `pip install uvicorn`.") from exc

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
