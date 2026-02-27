from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import Thread
from typing import Any, Iterator, Optional

from .finetune.config import FinetuneConfig, load_finetune_config
from .gemini_vlm import parse_page_extraction_text

_DEFAULT_CONFIG_PATH = Path("configs/finetune_qwen35a3_vl.yaml")

_MODEL_CACHE: dict[str, Any] = {}


def _resolve_config_path(explicit_config: Optional[str]) -> Path:
    candidates: list[Path] = []
    if explicit_config:
        candidates.append(Path(explicit_config).expanduser())
    env_cfg = os.getenv("FINETREE_QWEN_CONFIG")
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser())
    candidates.append(_DEFAULT_CONFIG_PATH)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Qwen config file not found. Set FINETREE_QWEN_CONFIG or pass --config. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _dtype_from_name(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return None


def _ensure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Qwen GT local inference. "
            "Run on a CUDA-enabled machine or disable Qwen GT."
        )


def _cache_key(model_name: str, adapter_path: Optional[str], cfg: FinetuneConfig) -> str:
    return "|".join(
        [
            model_name,
            str(adapter_path or ""),
            str(cfg.inference.torch_dtype),
            str(cfg.inference.device_map),
            str(cfg.inference.load_in_4bit),
        ]
    )


def _load_model_bundle(cfg: FinetuneConfig, model_override: Optional[str] = None) -> tuple[Any, Any]:
    _ensure_cuda()

    model_name = str(model_override or cfg.inference.model_path or cfg.model.base_model)
    adapter_path = cfg.inference.adapter_path
    key = _cache_key(model_name, adapter_path, cfg)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for Qwen GT local inference.") from exc

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg.inference.trust_remote_code),
        "device_map": str(cfg.inference.device_map),
        "torch_dtype": _dtype_from_name(cfg.inference.torch_dtype),
    }
    if cfg.inference.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=bool(cfg.inference.trust_remote_code))

    if adapter_path:
        adapter = Path(adapter_path).expanduser()
        if not adapter.exists():
            raise FileNotFoundError(f"Configured inference.adapter_path not found: {adapter}")
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("peft is required to load LoRA adapters for Qwen GT.") from exc
        model = PeftModel.from_pretrained(model, str(adapter))

    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _prepare_inputs(processor: Any, image_path: Path, prompt: str) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )
    return inputs


def stream_content_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Iterator[str]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = load_finetune_config(_resolve_config_path(config_path))
    model_obj, processor = _load_model_bundle(cfg, model_override=model)

    try:
        from transformers import TextIteratorStreamer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers TextIteratorStreamer is required for streaming.") from exc

    inputs = _prepare_inputs(processor, image_path=image_path, prompt=prompt)

    device = getattr(model_obj, "device", None)
    if device is not None:
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)

    streamer = TextIteratorStreamer(
        tokenizer=getattr(processor, "tokenizer", processor),
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(cfg.inference.max_new_tokens),
        do_sample=bool(cfg.inference.do_sample),
        temperature=float(cfg.inference.temperature),
        top_p=float(cfg.inference.top_p),
    )

    worker = Thread(target=model_obj.generate, kwargs=generate_kwargs, daemon=True)
    worker.start()
    for token in streamer:
        if token:
            yield token
    worker.join(timeout=1.0)


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    return "".join(stream_content_from_image(image_path=image_path, prompt=prompt, model=model, config_path=config_path))


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
):
    text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=model,
        config_path=config_path,
    )
    return parse_page_extraction_text(text)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen VLM generation on an image.")
    parser.add_argument("--config", default=None, help="Fine-tune config YAML path.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--prompt", default=None, help="Prompt text. If omitted, reads --prompt-path.")
    parser.add_argument("--prompt-path", default="prompt.txt", help="Prompt template/text file path.")
    parser.add_argument("--model", default=None, help="Override inference model path/id.")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens to stdout.")
    return parser.parse_args(argv)


def _load_prompt(args: argparse.Namespace, image_path: Path) -> str:
    if isinstance(args.prompt, str) and args.prompt.strip():
        return args.prompt
    prompt_path = Path(args.prompt_path).expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt not provided and prompt file not found: {prompt_path}. Provide --prompt or --prompt-path."
        )
    template = prompt_path.read_text(encoding="utf-8")
    text = template.replace("{{PAGE_IMAGE}}", str(image_path))
    text = text.replace("{{IMAGE_NAME}}", image_path.name)
    return text


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    image_path = Path(args.image)
    prompt = _load_prompt(args, image_path)

    if args.stream:
        for chunk in stream_content_from_image(
            image_path=image_path,
            prompt=prompt,
            model=args.model,
            config_path=args.config,
        ):
            print(chunk, end="", flush=True)
        print()
        return 0

    text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=args.model,
        config_path=args.config,
    )
    print(text)
    return 0


__all__ = [
    "generate_content_from_image",
    "generate_page_extraction_from_image",
    "main",
    "stream_content_from_image",
]


if __name__ == "__main__":
    raise SystemExit(main())
