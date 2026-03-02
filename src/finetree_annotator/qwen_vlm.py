from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import warnings
from pathlib import Path
from threading import Thread
from typing import Any, Iterator, Optional, Tuple

from .finetune.config import FinetuneConfig, load_finetune_config
from .inference.auth import resolve_hf_token_from_env

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


def _cache_key(model_name: str, adapter_path: Optional[str], quantization_mode: str, cfg: FinetuneConfig) -> str:
    return "|".join(
        [
            model_name,
            str(adapter_path or ""),
            str(quantization_mode),
            str(cfg.inference.torch_dtype),
            str(cfg.inference.attn_implementation),
            str(cfg.inference.require_flash_attention),
            str(cfg.inference.device_map),
            str(cfg.inference.load_in_4bit),
            str(cfg.inference.fallback_model_path or ""),
            str(cfg.inference.fallback_disable_adapter),
            str(cfg.inference.max_memory_per_gpu_gb),
            str(cfg.inference.gpu_memory_utilization),
            str(os.getenv("FINETREE_QWEN_FALLBACK_MODEL") or ""),
            str(os.getenv("FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB") or ""),
            str(os.getenv("FINETREE_QWEN_GPU_MEMORY_UTILIZATION") or ""),
        ]
    )


def _resolve_adapter_reference(adapter_path: Optional[str]) -> Tuple[Optional[str], bool]:
    if not adapter_path:
        return None, False

    raw = str(adapter_path).strip()
    if not raw:
        return None, False

    expanded = Path(raw).expanduser()
    if expanded.exists():
        return str(expanded.resolve()), True

    if raw.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Configured inference.adapter_path not found: {expanded}")

    return raw, False


def _load_with_optional_token(load_fn: Any, ref: str, token: Optional[str], **kwargs: Any) -> Any:
    try:
        if token:
            return load_fn(ref, token=token, **kwargs)
        return load_fn(ref, **kwargs)
    except TypeError as exc:
        if token:
            message = str(exc)
            if "token" in message or "use_auth_token" in message:
                return load_fn(ref, use_auth_token=token, **kwargs)
        raise


def _is_missing_flash_attn_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if not message:
        return False
    candidates = (
        "flash_attn",
        "flashattention2",
        "flash attention 2",
        "flash attention",
    )
    return any(token in message for token in candidates)


def _env_flag(name: str) -> Optional[bool]:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _effective_model_name(cfg: FinetuneConfig, model_override: Optional[str]) -> str:
    override = str(model_override or "").strip()
    if override:
        return override
    env_model = str(os.getenv("FINETREE_QWEN_MODEL") or "").strip()
    if env_model:
        return env_model
    configured = str(cfg.inference.model_path or cfg.model.base_model or "").strip()
    if configured:
        return configured
    raise RuntimeError("No model configured for local Qwen inference.")


def _effective_fallback_model_name(cfg: FinetuneConfig, primary_model_name: str) -> Optional[str]:
    env_fallback = str(os.getenv("FINETREE_QWEN_FALLBACK_MODEL") or "").strip()
    if env_fallback:
        fallback = env_fallback
    else:
        fallback = str(cfg.inference.fallback_model_path or "").strip()
    if not fallback:
        return None
    if fallback == primary_model_name:
        return None
    return fallback


def _effective_adapter_path(cfg: FinetuneConfig) -> Optional[str]:
    for name in ("FINETREE_ADAPTER_REF", "FINETREE_QWEN_ADAPTER_PATH"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    configured = str(cfg.inference.adapter_path or "").strip()
    if configured:
        return configured
    return None


def _effective_quantization_mode(cfg: FinetuneConfig) -> str:
    env_mode = str(os.getenv("FINETREE_QWEN_QUANTIZATION") or "").strip().lower()
    if env_mode in {"none", "bnb_8bit", "bnb_4bit"}:
        return env_mode
    env_load_in_4bit = _env_flag("FINETREE_QWEN_LOAD_IN_4BIT")
    if env_load_in_4bit is True:
        return "bnb_4bit"
    if cfg.inference.load_in_4bit:
        return "bnb_4bit"
    return str(cfg.inference.quantization_mode)


def _effective_max_memory_per_gpu_gb(cfg: FinetuneConfig) -> Optional[int]:
    env_value = str(os.getenv("FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB") or "").strip()
    if env_value:
        try:
            parsed = int(env_value)
        except ValueError:
            warnings.warn(
                f"Ignoring invalid FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB={env_value!r}; expected integer.",
                RuntimeWarning,
            )
        else:
            if parsed > 0:
                return parsed
            warnings.warn(
                f"Ignoring non-positive FINETREE_QWEN_MAX_MEMORY_PER_GPU_GB={env_value!r}.",
                RuntimeWarning,
            )

    configured = cfg.inference.max_memory_per_gpu_gb
    if configured is None:
        return None
    parsed = int(configured)
    if parsed <= 0:
        return None
    return parsed


def _effective_gpu_memory_utilization(cfg: FinetuneConfig) -> float:
    env_value = str(os.getenv("FINETREE_QWEN_GPU_MEMORY_UTILIZATION") or "").strip()
    if env_value:
        try:
            parsed = float(env_value)
        except ValueError:
            warnings.warn(
                f"Ignoring invalid FINETREE_QWEN_GPU_MEMORY_UTILIZATION={env_value!r}; expected float.",
                RuntimeWarning,
            )
        else:
            if 0.0 < parsed <= 1.0:
                return parsed
            warnings.warn(
                f"Ignoring out-of-range FINETREE_QWEN_GPU_MEMORY_UTILIZATION={env_value!r}; expected (0, 1].",
                RuntimeWarning,
            )
    return float(cfg.inference.gpu_memory_utilization)


def _build_max_memory_map(cfg: FinetuneConfig) -> Optional[dict[Any, str]]:
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    gpu_count = int(torch.cuda.device_count())
    if gpu_count <= 0:
        return None

    fixed_gb = _effective_max_memory_per_gpu_gb(cfg)
    utilization = _effective_gpu_memory_utilization(cfg)
    max_memory: dict[Any, str] = {}
    for gpu_idx in range(gpu_count):
        total_mem_bytes = int(torch.cuda.get_device_properties(gpu_idx).total_memory)
        total_mem_gib = max(1, int(total_mem_bytes // (1024**3)))
        if fixed_gb is not None:
            target_gib = min(total_mem_gib, int(fixed_gb))
        else:
            target_gib = max(1, int(total_mem_gib * utilization))
        max_memory[gpu_idx] = f"{target_gib}GiB"
    return max_memory


def _uses_flash_attention(model_obj: Any) -> bool:
    for candidate in (model_obj, getattr(model_obj, "config", None)):
        if candidate is None:
            continue
        for attr in ("attn_implementation", "_attn_implementation"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and "flash" in value.lower():
                return True
    return False


def _ensure_flash_attention(cfg: FinetuneConfig, model_obj: Any) -> None:
    if not bool(cfg.inference.require_flash_attention):
        return
    if _uses_flash_attention(model_obj):
        return
    raise RuntimeError(
        "Flash attention is required for inference but is not active. "
        "Set inference.attn_implementation=flash_attention_2 and ensure flash-attn is installed."
    )


def _build_quantization_kwargs(cfg: FinetuneConfig, quant_mode: str, transformers_module: Any) -> dict[str, Any]:
    if quant_mode not in {"bnb_4bit", "bnb_8bit"}:
        return {}

    bits_and_bytes_config_cls = getattr(transformers_module, "BitsAndBytesConfig", None)
    if bits_and_bytes_config_cls is None:
        if quant_mode == "bnb_4bit":
            return {"load_in_4bit": True}
        return {"load_in_8bit": True}

    bnb_kwargs: dict[str, Any] = {}
    if quant_mode == "bnb_4bit":
        bnb_kwargs["load_in_4bit"] = True
        compute_dtype = _dtype_from_name(cfg.inference.torch_dtype)
        if compute_dtype is not None:
            bnb_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
    else:
        bnb_kwargs["load_in_8bit"] = True
    return {"quantization_config": bits_and_bytes_config_cls(**bnb_kwargs)}


def _load_model_bundle_once(
    cfg: FinetuneConfig,
    *,
    model_name: str,
    adapter_ref: Optional[str],
    hf_token: Optional[str],
    quant_mode: str,
    transformers_module: Any,
) -> tuple[Any, Any]:
    auto_model_cls = getattr(transformers_module, "AutoModelForImageTextToText")
    auto_processor_cls = getattr(transformers_module, "AutoProcessor")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg.inference.trust_remote_code),
        "device_map": str(cfg.inference.device_map),
        "torch_dtype": _dtype_from_name(cfg.inference.torch_dtype),
        "attn_implementation": str(cfg.inference.attn_implementation),
    }
    max_memory = _build_max_memory_map(cfg)
    if max_memory:
        model_kwargs["max_memory"] = max_memory
    model_kwargs.update(_build_quantization_kwargs(cfg, quant_mode, transformers_module))

    try:
        model = _load_with_optional_token(auto_model_cls.from_pretrained, model_name, hf_token, **model_kwargs)
    except ImportError as exc:
        requested_attn = str(model_kwargs.get("attn_implementation") or "").strip().lower()
        if requested_attn != "flash_attention_2" or bool(cfg.inference.require_flash_attention):
            raise
        if not _is_missing_flash_attn_error(exc):
            raise
        fallback_model_kwargs = dict(model_kwargs)
        fallback_model_kwargs["attn_implementation"] = "sdpa"
        model = _load_with_optional_token(
            auto_model_cls.from_pretrained,
            model_name,
            hf_token,
            **fallback_model_kwargs,
        )

    processor = _load_with_optional_token(
        auto_processor_cls.from_pretrained,
        model_name,
        hf_token,
        trust_remote_code=bool(cfg.inference.trust_remote_code),
    )

    if adapter_ref:
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("peft is required to load LoRA adapters for Qwen GT.") from exc
        try:
            if hf_token:
                model = PeftModel.from_pretrained(model, adapter_ref, token=hf_token)
            else:
                model = PeftModel.from_pretrained(model, adapter_ref)
        except TypeError:
            if hf_token:
                model = PeftModel.from_pretrained(model, adapter_ref, use_auth_token=hf_token)
            else:
                raise

    _ensure_flash_attention(cfg, model)
    return model, processor


def _load_model_bundle(cfg: FinetuneConfig, model_override: Optional[str] = None) -> tuple[Any, Any]:
    _ensure_cuda()

    model_name = _effective_model_name(cfg, model_override)
    adapter_ref, _ = _resolve_adapter_reference(_effective_adapter_path(cfg))
    fallback_model_name = _effective_fallback_model_name(cfg, model_name)
    quant_mode = _effective_quantization_mode(cfg)
    key = _cache_key(model_name, adapter_ref, quant_mode, cfg)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    hf_token = resolve_hf_token_from_env()

    try:
        import transformers
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for Qwen GT local inference.") from exc

    try:
        model, processor = _load_model_bundle_once(
            cfg,
            model_name=model_name,
            adapter_ref=adapter_ref,
            hf_token=hf_token,
            quant_mode=quant_mode,
            transformers_module=transformers,
        )
    except Exception as primary_exc:
        if not fallback_model_name:
            raise

        fallback_adapter_ref = None if bool(cfg.inference.fallback_disable_adapter) else adapter_ref
        warnings.warn(
            "Primary model load failed. Falling back to original model "
            f"{fallback_model_name!r} with adapter={fallback_adapter_ref or 'none'}: {primary_exc!r}",
            RuntimeWarning,
        )
        fallback_key = _cache_key(fallback_model_name, fallback_adapter_ref, quant_mode, cfg)
        fallback_cached = _MODEL_CACHE.get(fallback_key)
        if fallback_cached is not None:
            _MODEL_CACHE[key] = fallback_cached
            return fallback_cached
        try:
            model, processor = _load_model_bundle_once(
                cfg,
                model_name=fallback_model_name,
                adapter_ref=fallback_adapter_ref,
                hf_token=hf_token,
                quant_mode=quant_mode,
                transformers_module=transformers,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Primary model load failed for {model_name!r}, and fallback model load failed "
                f"for {fallback_model_name!r}. primary={primary_exc!r}, fallback={fallback_exc!r}"
            ) from fallback_exc
        _MODEL_CACHE[fallback_key] = (model, processor)

    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _apply_chat_template_with_thinking_control(processor: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    base_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    try:
        return processor.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **base_kwargs,
        )
    except TypeError:
        pass

    try:
        return processor.apply_chat_template(
            messages,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            **base_kwargs,
        )
    except TypeError:
        if not enable_thinking:
            raise RuntimeError(
                "Configured inference.enable_thinking=false but this tokenizer/processor "
                "does not support thinking control in apply_chat_template."
            )
        return processor.apply_chat_template(messages, **base_kwargs)


def _prepare_inputs(processor: Any, image_path: Path, prompt: str, enable_thinking: bool) -> dict[str, Any]:
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

    chat_text = _apply_chat_template_with_thinking_control(
        processor=processor,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )
    return inputs


def _endpoint_api_key(cfg: FinetuneConfig) -> Optional[str]:
    if isinstance(cfg.inference.endpoint_api_key, str) and cfg.inference.endpoint_api_key.strip():
        return cfg.inference.endpoint_api_key.strip()
    return resolve_hf_token_from_env(preferred_env=cfg.inference.endpoint_api_key_env)


def _encode_image_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _resolve_endpoint_model(cfg: FinetuneConfig, model_override: Optional[str]) -> str:
    model_name = str(model_override or cfg.inference.endpoint_model or cfg.inference.model_path or cfg.model.base_model).strip()
    if not model_name:
        raise RuntimeError("No endpoint model configured. Set inference.endpoint_model or inference.model_path.")
    return model_name


def _stream_content_from_runpod_endpoint(
    cfg: FinetuneConfig,
    image_path: Path,
    prompt: str,
    model_override: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    base_url = str(cfg.inference.endpoint_base_url or os.getenv("FINETREE_QWEN_ENDPOINT_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Qwen endpoint backend is enabled but no endpoint base URL is configured. "
            "Set inference.endpoint_base_url or FINETREE_QWEN_ENDPOINT_BASE_URL."
        )
    api_key = _endpoint_api_key(cfg)
    if not api_key:
        raise RuntimeError(
            "Qwen endpoint backend is enabled but API key is missing. "
            f"Set inference.endpoint_api_key or env var {cfg.inference.endpoint_api_key_env}."
        )

    model_name = _resolve_endpoint_model(cfg, model_override)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    image_data_uri = _encode_image_data_uri(image_path)

    resolved_max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else int(cfg.inference.max_new_tokens)
    effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
    effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
    effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
    effective_enable_thinking = bool(cfg.inference.enable_thinking)

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        "max_tokens": resolved_max_new_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": effective_enable_thinking},
    }
    if effective_do_sample:
        payload["temperature"] = effective_temperature
        payload["top_p"] = effective_top_p

    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("httpx is required for endpoint-based Qwen inference.") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=float(cfg.inference.endpoint_timeout_sec)) as client:
        with client.stream("POST", endpoint, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Endpoint request failed ({response.status_code}): {body[:500]}")

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text:
                    yield text


def _resolve_runpod_queue_urls(cfg: FinetuneConfig) -> tuple[str, str]:
    base_url = str(cfg.inference.endpoint_base_url or os.getenv("FINETREE_QWEN_ENDPOINT_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Qwen endpoint backend is enabled but no endpoint base URL is configured. "
            "Set inference.endpoint_base_url or FINETREE_QWEN_ENDPOINT_BASE_URL."
        )

    normalized = base_url.rstrip("/")
    if normalized.endswith("/openai/v1") or normalized.endswith("/chat/completions"):
        raise RuntimeError(
            "runpod_queue backend expects a queue endpoint URL (for /run and /status), "
            "not an OpenAI-compatible URL. Example: https://api.runpod.ai/v2/<ENDPOINT_ID>"
        )

    if normalized.endswith("/run"):
        run_url = normalized
    elif normalized.endswith("/status"):
        run_url = normalized[: -len("/status")] + "/run"
    else:
        run_url = normalized + "/run"

    status_base_url = str(cfg.inference.endpoint_status_base_url or "").strip().rstrip("/")
    if not status_base_url:
        status_base_url = run_url[: -len("/run")] + "/status"

    return run_url, status_base_url


def _runpod_stream_base_url(run_url: str) -> str:
    base = run_url
    if base.endswith("/run"):
        base = base[: -len("/run")]
    return base.rstrip("/") + "/stream"


def _decode_runpod_error(error_value: Any) -> Optional[str]:
    if error_value is None:
        return None
    if isinstance(error_value, dict):
        for key in ("error_message", "message", "detail", "error"):
            value = error_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(error_value, ensure_ascii=False)
    if isinstance(error_value, str):
        stripped = error_value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return _decode_runpod_error(parsed)
    return str(error_value)


def _extract_runpod_text_output(job_payload: Any) -> Optional[str]:
    if not isinstance(job_payload, dict):
        return None

    output = job_payload.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            parts = [str(item) for item in text if isinstance(item, (str, int, float))]
            if parts:
                return "".join(parts)
        result = output.get("result")
        if result is not None:
            return json.dumps(result, ensure_ascii=False)
        return json.dumps(output, ensure_ascii=False)
    if output is not None:
        return json.dumps(output, ensure_ascii=False)

    text = job_payload.get("text")
    if isinstance(text, str):
        return text
    return None


def _runpod_status(status_value: Any) -> str:
    return str(status_value or "").strip().upper()


def _iter_text_from_runpod_stream_payload(payload: Any) -> Iterator[str]:
    if payload is None:
        return

    if isinstance(payload, str):
        if payload:
            yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            yield from _iter_text_from_runpod_stream_payload(item)
        return

    if isinstance(payload, dict):
        for key in ("chunk", "text", "output"):
            if key in payload:
                yield from _iter_text_from_runpod_stream_payload(payload.get(key))
        return


def _stream_content_from_runpod_queue(
    cfg: FinetuneConfig,
    image_path: Path,
    prompt: str,
    model_override: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    run_url, status_base_url = _resolve_runpod_queue_urls(cfg)
    api_key = _endpoint_api_key(cfg)
    if not api_key:
        raise RuntimeError(
            "Qwen endpoint backend is enabled but API key is missing. "
            f"Set inference.endpoint_api_key or env var {cfg.inference.endpoint_api_key_env}."
        )

    payload_input: dict[str, Any] = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "image_mime_type": mimetypes.guess_type(str(image_path))[0] or "image/png",
        "response_mode": "text",
        "prompt": prompt,
    }
    if max_new_tokens is not None:
        payload_input["max_tokens"] = int(max_new_tokens)

    model_name = _resolve_endpoint_model(cfg, model_override)
    if model_name:
        payload_input["model"] = model_name

    effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
    effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
    effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
    effective_enable_thinking = bool(cfg.inference.enable_thinking)
    if effective_do_sample:
        payload_input["temperature"] = effective_temperature
        payload_input["top_p"] = effective_top_p
    payload_input["chat_template_kwargs"] = {"enable_thinking": effective_enable_thinking}

    payload = {"input": payload_input}

    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("httpx is required for endpoint-based Qwen inference.") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout_sec = float(cfg.inference.endpoint_timeout_sec)
    deadline = time.monotonic() + timeout_sec

    stream_base_url = _runpod_stream_base_url(run_url)

    with httpx.Client(timeout=timeout_sec) as client:
        run_response = client.post(run_url, headers=headers, json=payload)
        if run_response.status_code >= 400:
            body = run_response.text if hasattr(run_response, "text") else str(run_response.read())
            raise RuntimeError(f"Queue endpoint request failed ({run_response.status_code}): {body[:500]}")

        try:
            job_payload = run_response.json()
        except Exception as exc:
            body = run_response.text if hasattr(run_response, "text") else str(run_response.read())
            raise RuntimeError(f"Queue endpoint returned non-JSON response: {body[:500]}") from exc

        job_id = str(job_payload.get("id") or "").strip()
        if not job_id:
            raise RuntimeError("RunPod queue job did not return an id for status polling.")

        # First try real stream endpoint for incremental tokens.
        if hasattr(client, "stream"):
            stream_url = f"{stream_base_url}/{job_id}"
            try:
                streamed_any = False
                # Some RunPod deployments keep /stream open without emitting events.
                # Use a short read timeout so we can quickly fall back to /status polling.
                stream_timeout = httpx.Timeout(
                    connect=min(10.0, timeout_sec),
                    read=min(8.0, timeout_sec),
                    write=min(10.0, timeout_sec),
                    pool=min(10.0, timeout_sec),
                )
                with client.stream("GET", stream_url, headers=headers, timeout=stream_timeout) as stream_response:
                    if stream_response.status_code < 400:
                        for raw_line in stream_response.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.strip()
                            if line.startswith("data:"):
                                line = line[len("data:") :].strip()
                            if not line:
                                continue
                            if line == "[DONE]":
                                break
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            for piece in _iter_text_from_runpod_stream_payload(event):
                                if piece:
                                    streamed_any = True
                                    yield piece
                if streamed_any:
                    return
            except Exception:
                # Fall back to status polling below when stream is unavailable.
                pass

        while True:
            status = _runpod_status(job_payload.get("status"))
            output_text = _extract_runpod_text_output(job_payload)

            if status in {"COMPLETED", "SUCCESS", "SUCCEEDED"} or (not status and output_text is not None):
                if output_text is None:
                    raise RuntimeError("RunPod queue job completed but returned no output.")
                yield output_text
                return

            if status in {"FAILED", "CANCELLED", "TIMED_OUT", "TIMEOUT"}:
                error_text = _decode_runpod_error(job_payload.get("error")) or f"RunPod queue job failed ({status})."
                raise RuntimeError(error_text)

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"RunPod queue job timed out after {int(timeout_sec)}s while waiting for completion (job id: {job_id})."
                )

            time.sleep(1.0)
            status_response = client.get(f"{status_base_url}/{job_id}", headers=headers)
            if status_response.status_code >= 400:
                body = status_response.text if hasattr(status_response, "text") else str(status_response.read())
                raise RuntimeError(f"Queue status request failed ({status_response.status_code}): {body[:500]}")
            try:
                job_payload = status_response.json()
            except Exception as exc:
                body = status_response.text if hasattr(status_response, "text") else str(status_response.read())
                raise RuntimeError(f"Queue status returned non-JSON response: {body[:500]}") from exc


def _resolve_input_device(model_obj: Any) -> Optional[Any]:
    hf_device_map = getattr(model_obj, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return f"cuda:{mapped_device}"
            if isinstance(mapped_device, str):
                lowered = mapped_device.lower()
                if lowered not in {"cpu", "disk"}:
                    return mapped_device
    return getattr(model_obj, "device", None)


def _stream_content_local(
    cfg: FinetuneConfig,
    image_path: Path,
    prompt: str,
    model_override: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    model_obj, processor = _load_model_bundle(cfg, model_override=model_override)

    try:
        from transformers import TextIteratorStreamer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers TextIteratorStreamer is required for streaming.") from exc

    inputs = _prepare_inputs(
        processor,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=bool(cfg.inference.enable_thinking),
    )

    device = _resolve_input_device(model_obj)
    if device is not None:
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)

    streamer = TextIteratorStreamer(
        tokenizer=getattr(processor, "tokenizer", processor),
        skip_prompt=True,
        skip_special_tokens=True,
    )

    resolved_max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else int(cfg.inference.max_new_tokens)
    effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
    effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
    effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=resolved_max_new_tokens,
        do_sample=effective_do_sample,
    )
    if effective_do_sample:
        generate_kwargs["temperature"] = effective_temperature
        generate_kwargs["top_p"] = effective_top_p

    worker = Thread(target=model_obj.generate, kwargs=generate_kwargs, daemon=True)
    worker.start()
    for token in streamer:
        if token:
            yield token
    worker.join(timeout=1.0)


def stream_content_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = load_finetune_config(_resolve_config_path(config_path))
    if cfg.inference.backend == "runpod_openai":
        yield from _stream_content_from_runpod_endpoint(
            cfg=cfg,
            image_path=image_path,
            prompt=prompt,
            model_override=model,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return
    if cfg.inference.backend == "runpod_queue":
        yield from _stream_content_from_runpod_queue(
            cfg=cfg,
            image_path=image_path,
            prompt=prompt,
            model_override=model,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return

    yield from _stream_content_local(
        cfg=cfg,
        image_path=image_path,
        prompt=prompt,
        model_override=model,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    return "".join(
        stream_content_from_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            config_path=config_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    )


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
):
    from .gemini_vlm import parse_page_extraction_text

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
