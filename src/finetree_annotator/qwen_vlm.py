from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Iterator, Optional, Tuple

from .finetune.config import FinetuneConfig, load_finetune_config
from .inference.auth import resolve_hf_token_from_env
from .schema_contract import default_extraction_prompt_template
from .vision_resize import prepared_dimensions_for_max_pixels

_DEFAULT_CONFIG_PATH = Path("configs/finetune_qwen35a3_vl.yaml")
_DEFAULT_QWEN_FLASH_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEFAULT_QWEN_FLASH_MODEL = "qwen3.5-flash"
_DEFAULT_QWEN_MAX_NEW_TOKENS = 10_000
DEFAULT_QWEN_BBOX_MAX_PIXELS = 1_400_000
_DEFAULT_EXTRACTION_PROMPT_PATH = Path("prompts/extraction_prompt.txt")
_HOSTED_QWEN_GT_MODELS: tuple[str, ...] = (
    "qwen3.5-flash",
    "qwen3.5-flash-2026-02-23",
    "qwen3.5-plus",
    "qwen3.5-plus-2026-02-15",
)

_MODEL_CACHE: dict[str, Any] = {}


def _qwen_logs_dir() -> Path:
    path = Path.cwd() / "qwen_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename_fragment(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def _serialize_qwen_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_qwen_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_qwen_log_value(item) for item in value]
    if hasattr(value, "__dict__"):
        try:
            return _serialize_qwen_log_value(vars(value))
        except Exception:
            pass
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _serialize_qwen_log_value(method())
            except Exception:
                pass
    return str(value)


def _copy_qwen_log_image(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _save_qwen_log_image(
    source_path: Path,
    target_path: Path,
    *,
    prepared_image: Optional[Any] = None,
    prepared_size: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    from PIL import Image

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(prepared_image, Path):
        image = Image.open(prepared_image).convert("RGB")
    elif prepared_image is not None:
        if not hasattr(prepared_image, "copy") or not hasattr(prepared_image, "save"):
            raise TypeError("prepared_image must be a PIL image or filesystem path.")
        image = prepared_image.copy()
    else:
        image = Image.open(source_path).convert("RGB")

    if prepared_size is not None and image.size != tuple(prepared_size):
        image = image.resize(tuple(prepared_size), Image.Resampling.BICUBIC)
    image.save(target_path, format="PNG")
    return int(image.size[0]), int(image.size[1])


def _image_dimensions_from_path(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _try_image_dimensions_from_path(image_path: Path) -> tuple[int, int] | None:
    try:
        return _image_dimensions_from_path(image_path)
    except Exception:
        return None


def _append_qwen_log_event(session_dir: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with (session_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")


def _read_qwen_log_request(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "request.json").read_text(encoding="utf-8"))


def _update_qwen_log_request(session_dir: Path, updates: dict[str, Any]) -> None:
    payload = _read_qwen_log_request(session_dir)
    payload.update(updates)
    (session_dir / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _find_qwen_thinking_signal(payload: Any, *, _path: str = "") -> Optional[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).strip().lower()
            next_path = f"{_path}.{key}" if _path else str(key)
            if key_text in {"thinking", "thought", "thoughts", "reasoning"} and value not in ("", None, [], {}):
                return next_path
            found = _find_qwen_thinking_signal(value, _path=next_path)
            if found:
                return found
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found = _find_qwen_thinking_signal(item, _path=f"{_path}[{index}]" if _path else f"[{index}]")
            if found:
                return found
    return None


def _qwen_request_summary(
    *,
    model: Optional[str],
    backend: str,
    enable_thinking: Optional[bool],
    exact_request: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    thinking_transport = None
    effective_enable_thinking = enable_thinking
    if isinstance(exact_request, dict):
        payload = exact_request.get("payload")
        if isinstance(payload, dict):
            if "enable_thinking" in payload:
                thinking_transport = "enable_thinking"
                effective_enable_thinking = bool(payload.get("enable_thinking"))
            else:
                nested_payload = payload.get("input")
                if isinstance(nested_payload, dict):
                    payload = nested_payload
                chat_template_kwargs = payload.get("chat_template_kwargs")
                if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
                    thinking_transport = "chat_template_kwargs.enable_thinking"
                    effective_enable_thinking = bool(chat_template_kwargs.get("enable_thinking"))
    if enable_thinking is None:
        requested_mode = "default"
    else:
        requested_mode = "thinking" if bool(enable_thinking) else "non-thinking"
    return {
        "resolved_model": str(model or ""),
        "backend": backend,
        "requested_enable_thinking": enable_thinking,
        "effective_enable_thinking": effective_enable_thinking,
        "requested_mode": requested_mode,
        "thinking_transport": thinking_transport,
        "resolved_request": _serialize_qwen_log_value(exact_request),
    }


def _qwen_response_summary(*, observed_thinking_signal_path: Optional[str], backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "observed_thinking_signal": observed_thinking_signal_path is not None,
        "observed_thinking_signal_path": observed_thinking_signal_path,
        "streamed": True,
    }


def _build_logged_qwen_messages(session_dir: Path, prompt: str, *, preserve_legacy_single_turn: bool) -> list[dict[str, Any]]:
    request_payload = _read_qwen_log_request(session_dir)
    target_content = (
        [
            {"type": "text", "text": prompt},
            {"type": "image_file", "file": request_payload["logged_image_path"]},
        ]
        if preserve_legacy_single_turn
        else [
            {"type": "image_file", "file": request_payload["logged_image_path"]},
            {"type": "text", "text": prompt},
        ]
    )
    examples = request_payload.get("few_shot_examples") or []
    if not examples:
        return [{"role": "user", "content": target_content}]

    messages: list[dict[str, Any]] = []
    for example in examples:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_file", "file": example["logged_image_path"]},
                    {"type": "text", "text": "Example input page."},
                ],
            }
        )
        messages.append({"role": "assistant", "content": example["expected_json"]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_file", "file": request_payload["logged_image_path"]},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def _write_qwen_log_output(session_dir: Path, *, text: str, raw: Any = None) -> None:
    (session_dir / "output.txt").write_text(str(text or ""), encoding="utf-8")
    (session_dir / "response.json").write_text(
        json.dumps(_serialize_qwen_log_value(raw if raw is not None else {"text": text}), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _create_qwen_log_session(
    *,
    operation: str,
    model: Optional[str],
    image_path: Path,
    prompt: str,
    enable_thinking: Optional[bool],
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    extra_request: Optional[dict[str, Any]] = None,
    prepared_image: Optional[Any] = None,
    prepared_size: Optional[tuple[int, int]] = None,
    original_size: Optional[tuple[float, float]] = None,
    bbox_max_pixels: Optional[int] = None,
    require_prepared_image: bool = False,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    session_dir = _qwen_logs_dir() / (
        f"{timestamp}_{_safe_filename_fragment(operation)}_{_safe_filename_fragment(model or 'default')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    original_image_size = (
        (int(round(float(original_size[0]))), int(round(float(original_size[1]))))
        if original_size is not None
        else _try_image_dimensions_from_path(image_path)
    )
    uses_prepared_image = bool(require_prepared_image or prepared_image is not None or prepared_size is not None)

    if uses_prepared_image:
        original_copy_name = f"input_original{image_path.suffix.lower() or '.bin'}"
        _copy_qwen_log_image(image_path, session_dir / original_copy_name)
        target_copy_name = "input_target_prepared.png"
        prepared_width, prepared_height = _save_qwen_log_image(
            image_path,
            session_dir / target_copy_name,
            prepared_image=prepared_image,
            prepared_size=prepared_size,
        )
        logged_image_mime_type = "image/png"
    else:
        target_copy_name = f"input_target{image_path.suffix.lower() or '.bin'}"
        _copy_qwen_log_image(image_path, session_dir / target_copy_name)
        if original_image_size is None:
            prepared_width, prepared_height = (0, 0)
        else:
            prepared_width, prepared_height = original_image_size
        logged_image_mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"

    logged_examples: list[dict[str, Any]] = []
    for index, raw_example in enumerate(few_shot_examples or [], start=1):
        if not isinstance(raw_example, dict):
            continue
        raw_image_path = raw_example.get("image_path")
        example_image_path = raw_image_path if isinstance(raw_image_path, Path) else Path(str(raw_image_path or "")).expanduser()
        if not example_image_path.is_file():
            continue
        copy_name = f"few_shot_{index:02d}_{example_image_path.name}"
        _copy_qwen_log_image(example_image_path, session_dir / copy_name)
        logged_examples.append(
            {
                "source_image_path": str(example_image_path),
                "logged_image_path": copy_name,
                "expected_json": str(raw_example.get("expected_json") or ""),
            }
        )

    request_payload = {
        "operation": operation,
        "model": str(model or ""),
        "prompt": prompt,
        "image_path": str(image_path),
        "original_image_size": (
            {
                "width": int(original_image_size[0]),
                "height": int(original_image_size[1]),
            }
            if original_image_size is not None
            else None
        ),
        "prepared_image_size": {
            "width": int(prepared_width),
            "height": int(prepared_height),
        },
        "bbox_max_pixels": int(bbox_max_pixels) if bbox_max_pixels is not None else None,
        "uses_prepared_image": uses_prepared_image,
        "logged_image_path": target_copy_name,
        "logged_image_mime_type": logged_image_mime_type,
        "enable_thinking": enable_thinking,
        "few_shot_examples": logged_examples,
    }
    if extra_request:
        request_payload.update(extra_request)
    (session_dir / "request.json").write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _append_qwen_log_event(
        session_dir,
        "session_started",
        {
            "operation": operation,
            "model": str(model or ""),
            "image_path": str(image_path),
            "enable_thinking": enable_thinking,
        },
    )
    return session_dir


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


def _normalize_model_selector(model_name: Optional[str]) -> str:
    raw = str(model_name or "").strip().lower()
    if not raw:
        return ""
    return re.sub(r"-+", "-", re.sub(r"[\s_]+", "-", raw))


def is_qwen_flash_model_requested(model_name: Optional[str]) -> bool:
    normalized = _normalize_model_selector(model_name)
    if not normalized:
        return False
    if normalized in {"qwen-flash-gt", "qwen-flash"}:
        return True
    if normalized.startswith("qwen3.5-flash"):
        return True
    if normalized.startswith("qwen3.5-plus"):
        return True
    return False


def _resolve_qwen_flash_model_name(model_name: Optional[str]) -> str:
    alias_default = str(os.getenv("FINETREE_QWEN_FLASH_MODEL") or _DEFAULT_QWEN_FLASH_MODEL).strip()
    alias_default = alias_default or _DEFAULT_QWEN_FLASH_MODEL
    if is_qwen_flash_model_requested(model_name):
        normalized = _normalize_model_selector(model_name)
        if normalized in {"qwen-flash-gt", "qwen-flash"}:
            return alias_default
    resolved = str(model_name or "").strip()
    if not resolved:
        return alias_default
    return resolved


def current_qwen_gt_model_choices(config_path: Optional[str] = None) -> tuple[str, ...]:
    choices: list[str] = ["qwen-flash-gt", *_HOSTED_QWEN_GT_MODELS]
    configured_values: list[str] = [
        str(os.getenv("FINETREE_QWEN_FLASH_MODEL") or "").strip(),
        str(os.getenv("FINETREE_QWEN_MODEL") or "").strip(),
    ]

    try:
        cfg = load_finetune_config(_resolve_config_path(config_path))
    except Exception:
        cfg = None

    if cfg is not None:
        configured_values.extend(
            [
                str(cfg.inference.endpoint_model or "").strip(),
                str(cfg.inference.model_path or "").strip(),
                str(cfg.inference.fallback_model_path or "").strip(),
                str(cfg.model.base_model or "").strip(),
            ]
        )

    seen: set[str] = set()
    out: list[str] = []
    for raw_value in [*choices, *configured_values]:
        value = str(raw_value or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return tuple(out)


def _resolve_qwen_flash_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    doppler_value = _qwen_flash_api_key_from_doppler()
    candidates = (
        explicit_api_key,
        os.getenv("FINETREE_QWEN_FLASH_API_KEY"),
        os.getenv("FINETREE_QWEN_API_KEY"),
        os.getenv("QWEN_API_KEY"),
        os.getenv("DASHSCOPE_API_KEY"),
        doppler_value,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def resolve_qwen_flash_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    return _resolve_qwen_flash_api_key(explicit_api_key)


@lru_cache(maxsize=1)
def _qwen_flash_api_key_from_doppler() -> Optional[str]:
    if shutil.which("doppler") is None:
        return None

    project = str(os.getenv("DOPPLER_PROJECT") or "").strip()
    config = str(os.getenv("DOPPLER_CONFIG") or "").strip()
    scope_args: list[str] = []
    if project:
        scope_args.extend(["--project", project])
    if config:
        scope_args.extend(["--config", config])

    for secret_name in ("DASHSCOPE_API_KEY", "FINETREE_QWEN_FLASH_API_KEY", "FINETREE_QWEN_API_KEY", "QWEN_API_KEY"):
        cmd = ["doppler", "secrets", "get", secret_name, "--plain", *scope_args]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue
        value = proc.stdout.strip()
        if value:
            return value
    return None


def _resolve_qwen_flash_base_url(explicit_base_url: Optional[str] = None) -> str:
    raw = str(explicit_base_url or os.getenv("FINETREE_QWEN_FLASH_BASE_URL") or _DEFAULT_QWEN_FLASH_BASE_URL).strip()
    return raw.rstrip("/")


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
    configured = str(cfg.inference.adapter_path or "").strip()
    if configured:
        return configured

    allow_env_override = _env_flag("FINETREE_QWEN_ALLOW_ENV_ADAPTER_OVERRIDE")
    if allow_env_override is not True:
        return None

    for name in ("FINETREE_ADAPTER_REF", "FINETREE_QWEN_ADAPTER_PATH"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
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


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None


def _is_cuda_oom_error(exc: BaseException) -> bool:
    for candidate in _iter_exception_chain(exc):
        class_name = candidate.__class__.__name__.lower()
        if "outofmemory" in class_name:
            return True
        message = str(candidate).lower()
        if "out of memory" in message and ("cuda" in message or "gpu" in message):
            return True
    return False


def _best_effort_cuda_gc() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch
    except Exception:
        return

    try:
        if not torch.cuda.is_available():
            return
    except Exception:
        return

    for fn_name in ("empty_cache", "ipc_collect"):
        fn = getattr(torch.cuda, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _load_adapter_with_optional_token(
    peft_model_cls: Any,
    model_obj: Any,
    adapter_ref: str,
    hf_token: Optional[str],
) -> Any:
    load_fn = lambda ref, **kwargs: peft_model_cls.from_pretrained(model_obj, ref, **kwargs)

    try:
        if hf_token:
            return _load_with_optional_token(load_fn, adapter_ref, hf_token, low_cpu_mem_usage=True)
        return peft_model_cls.from_pretrained(model_obj, adapter_ref, low_cpu_mem_usage=True)
    except TypeError as exc:
        if "low_cpu_mem_usage" not in str(exc):
            raise
        if hf_token:
            return _load_with_optional_token(load_fn, adapter_ref, hf_token)
        return peft_model_cls.from_pretrained(model_obj, adapter_ref)


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
            model = _load_adapter_with_optional_token(PeftModel, model, adapter_ref, hf_token)
        except Exception as exc:
            if _is_cuda_oom_error(exc) and bool(cfg.inference.fallback_disable_adapter):
                _best_effort_cuda_gc()
                warnings.warn(
                    "Adapter load failed due CUDA OOM. Continuing without adapter. "
                    "To force adapter loading, set inference.fallback_disable_adapter=false.",
                    RuntimeWarning,
                )
            else:
                if _is_cuda_oom_error(exc):
                    raise RuntimeError(
                        "Adapter load failed due CUDA OOM. "
                        "Set inference.fallback_disable_adapter=true to continue without adapter, "
                        "or reduce memory usage (bnb_4bit / smaller model)."
                    ) from exc
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


def _prepare_inputs(
    processor: Any,
    image_path: Path,
    prompt: str,
    enable_thinking: bool,
    *,
    require_prepared_resize: bool = False,
) -> dict[str, Any]:
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
    processor_kwargs: dict[str, Any] = {
        "text": [chat_text],
        "images": [image],
        "return_tensors": "pt",
    }
    if require_prepared_resize:
        try:
            return processor(
                **processor_kwargs,
                do_resize=False,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Qwen bbox-only requires processor(..., do_resize=False) support "
                "so prepared-image pixel coordinates remain stable."
            ) from exc
    inputs = processor(**processor_kwargs)
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


def _build_openai_vision_messages(
    *,
    image_path: Path,
    prompt: str,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    preserve_legacy_single_turn: bool = True,
) -> list[dict[str, Any]]:
    image_data_uri = _encode_image_data_uri(image_path)
    default_content = (
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
        if preserve_legacy_single_turn
        else [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": prompt},
        ]
    )
    if not few_shot_examples:
        return [{"role": "user", "content": default_content}]

    messages: list[dict[str, Any]] = []
    valid_examples = 0
    for raw_example in few_shot_examples:
        if not isinstance(raw_example, dict):
            continue
        raw_example_path = raw_example.get("image_path")
        if isinstance(raw_example_path, Path):
            example_image_path = raw_example_path.expanduser()
        else:
            example_image_path = Path(str(raw_example_path or "")).expanduser()
        if not example_image_path.is_file():
            continue

        expected_json = str(raw_example.get("expected_json") or "").strip()
        if not expected_json:
            continue

        example_data_uri = _encode_image_data_uri(example_image_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": example_data_uri}},
                    {"type": "text", "text": "Example input page."},
                ],
            }
        )
        messages.append({"role": "assistant", "content": expected_json})
        valid_examples += 1

    if valid_examples == 0:
        return [{"role": "user", "content": default_content}]

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


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
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    log_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
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

    resolved_max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else int(cfg.inference.max_new_tokens)
    effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
    effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
    effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
    effective_enable_thinking = bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking)

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": _build_openai_vision_messages(
            image_path=image_path,
            prompt=prompt,
            few_shot_examples=few_shot_examples,
            preserve_legacy_single_turn=True,
        ),
        "max_tokens": resolved_max_new_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": effective_enable_thinking},
    }
    if effective_do_sample:
        payload["temperature"] = effective_temperature
        payload["top_p"] = effective_top_p
    if log_event is not None:
        log_event(
            "backend_request",
            {
                "backend": "runpod_openai",
                "endpoint": endpoint,
                "payload": _serialize_qwen_log_value(payload),
            },
        )

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
                line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if log_event is not None:
                    log_event("backend_stream_event", {"backend": "runpod_openai", "event_payload": event})

                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text:
                    yield text


def _stream_content_from_qwen_flash_endpoint(
    image_path: Path,
    prompt: str,
    *,
    model_override: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout_sec: Optional[float] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    log_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> Iterator[str]:
    resolved_api_key = _resolve_qwen_flash_api_key(api_key)
    if not resolved_api_key:
        raise RuntimeError(
            "Qwen Flash API key not found. Set FINETREE_QWEN_FLASH_API_KEY, "
            "FINETREE_QWEN_API_KEY, QWEN_API_KEY, or DASHSCOPE_API_KEY."
        )

    resolved_model_name = _resolve_qwen_flash_model_name(model_override)
    resolved_base_url = _resolve_qwen_flash_base_url(base_url)
    endpoint = resolved_base_url
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"

    resolved_max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else _DEFAULT_QWEN_MAX_NEW_TOKENS
    effective_do_sample = True if do_sample is None else bool(do_sample)
    effective_temperature = 0.7 if temperature is None else float(temperature)
    effective_top_p = 0.8 if top_p is None else float(top_p)
    effective_timeout_sec = 180.0 if timeout_sec is None else float(timeout_sec)

    payload: dict[str, Any] = {
        "model": resolved_model_name,
        "messages": _build_openai_vision_messages(
            image_path=image_path,
            prompt=prompt,
            few_shot_examples=few_shot_examples,
            preserve_legacy_single_turn=False,
        ),
        "max_tokens": resolved_max_new_tokens,
        "stream": True,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = bool(enable_thinking)
    if effective_do_sample:
        payload["temperature"] = effective_temperature
        payload["top_p"] = effective_top_p
    if log_event is not None:
        log_event(
            "backend_request",
            {
                "backend": "qwen_flash",
                "endpoint": endpoint,
                "payload": _serialize_qwen_log_value(payload),
            },
        )

    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("httpx is required for Qwen Flash endpoint inference.") from exc

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=effective_timeout_sec) as client:
        with client.stream("POST", endpoint, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Qwen Flash request failed ({response.status_code}): {body[:500]}")

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if log_event is not None:
                    log_event("backend_stream_event", {"backend": "qwen_flash", "event_payload": event})

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
    enable_thinking: Optional[bool] = None,
    log_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
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
    effective_enable_thinking = bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking)
    if effective_do_sample:
        payload_input["temperature"] = effective_temperature
        payload_input["top_p"] = effective_top_p
    payload_input["chat_template_kwargs"] = {"enable_thinking": effective_enable_thinking}

    payload = {"input": payload_input}
    if log_event is not None:
        log_event(
            "backend_request",
            {
                "backend": "runpod_queue",
                "run_url": run_url,
                "payload": _serialize_qwen_log_value(payload),
            },
        )

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
        if log_event is not None:
            log_event("backend_job_created", {"backend": "runpod_queue", "job_payload": job_payload})

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
                            if log_event is not None:
                                log_event("backend_stream_event", {"backend": "runpod_queue", "event_payload": event})
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
            if log_event is not None:
                log_event(
                    "backend_status",
                    {"backend": "runpod_queue", "status": status, "job_payload": _serialize_qwen_log_value(job_payload)},
                )

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
    enable_thinking: Optional[bool] = None,
    require_prepared_resize: bool = False,
    log_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
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
        enable_thinking=bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking),
        require_prepared_resize=require_prepared_resize,
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
    if log_event is not None:
        log_event(
            "backend_request",
            {
                "backend": "local",
                "model": _effective_model_name(cfg, model_override),
                "generate_kwargs": {
                    "max_new_tokens": resolved_max_new_tokens,
                    "do_sample": effective_do_sample,
                    "temperature": effective_temperature if effective_do_sample else None,
                    "top_p": effective_top_p if effective_do_sample else None,
                    "enable_thinking": bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking),
                },
            },
        )

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
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    prepared_image: Optional[Any] = None,
    prepared_size: Optional[tuple[int, int]] = None,
    original_size: Optional[tuple[float, float]] = None,
    bbox_max_pixels: Optional[int] = None,
    require_prepared_resize: bool = False,
) -> Iterator[str]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_original_size = (
        (float(original_size[0]), float(original_size[1]))
        if original_size is not None
        else (
            tuple(float(value) for value in detected_size)
            if (detected_size := _try_image_dimensions_from_path(image_path)) is not None
            else None
        )
    )
    resolved_bbox_max_pixels = int(bbox_max_pixels) if bbox_max_pixels is not None else None
    resolved_prepared_size = prepared_size
    if require_prepared_resize and resolved_prepared_size is None:
        if resolved_original_size is None:
            raise RuntimeError("Qwen bbox-only requires a valid image so the prepared resize dimensions can be computed.")
        prepared_h, prepared_w = prepared_dimensions_for_max_pixels(
            original_width=resolved_original_size[0],
            original_height=resolved_original_size[1],
            max_pixels=resolved_bbox_max_pixels or DEFAULT_QWEN_BBOX_MAX_PIXELS,
        )
        resolved_prepared_size = (int(prepared_w), int(prepared_h))

    session_dir = _create_qwen_log_session(
        operation="stream_content",
        model=model,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=enable_thinking,
        few_shot_examples=few_shot_examples,
        extra_request={
            "config_path": config_path,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
        prepared_image=prepared_image,
        prepared_size=resolved_prepared_size,
        original_size=resolved_original_size,
        bbox_max_pixels=resolved_bbox_max_pixels,
        require_prepared_image=require_prepared_resize,
    )
    request_payload = _read_qwen_log_request(session_dir)
    runtime_image_path = session_dir / str(request_payload["logged_image_path"])
    runtime_image_mime_type = str(request_payload.get("logged_image_mime_type") or mimetypes.guess_type(str(runtime_image_path))[0] or "image/png")
    observed_thinking_signal_path: Optional[str] = None
    active_backend = "unknown"

    def _log_event(event: str, payload: dict[str, Any]) -> None:
        nonlocal observed_thinking_signal_path, active_backend
        signal_path = _find_qwen_thinking_signal(payload)
        if observed_thinking_signal_path is None and signal_path is not None:
            observed_thinking_signal_path = signal_path
        backend_value = payload.get("backend")
        if isinstance(backend_value, str) and backend_value.strip():
            active_backend = backend_value.strip()
        _append_qwen_log_event(session_dir, event, payload)

    _append_qwen_log_event(
        session_dir,
        "request",
        {
            "prompt": prompt,
            "model": str(model or ""),
            "few_shot_count": len(few_shot_examples or []),
        },
    )

    collected_text: list[str] = []
    try:
        if is_qwen_flash_model_requested(model):
            resolved_model_name = _resolve_qwen_flash_model_name(model)
            active_backend = "qwen_flash"
            effective_do_sample = True if do_sample is None else bool(do_sample)
            exact_request = {
                "backend": "qwen_flash",
                "endpoint": (_resolve_qwen_flash_base_url(None) + "/chat/completions"),
                "payload": {
                    "model": resolved_model_name,
                    "messages": _build_logged_qwen_messages(session_dir, prompt, preserve_legacy_single_turn=False),
                    "max_tokens": int(max_new_tokens) if max_new_tokens is not None else _DEFAULT_QWEN_MAX_NEW_TOKENS,
                    "stream": True,
                },
            }
            if enable_thinking is not None:
                exact_request["payload"]["enable_thinking"] = bool(enable_thinking)
            if effective_do_sample:
                exact_request["payload"]["temperature"] = 0.7 if temperature is None else float(temperature)
                exact_request["payload"]["top_p"] = 0.8 if top_p is None else float(top_p)
            request_summary = _qwen_request_summary(
                model=resolved_model_name,
                backend=active_backend,
                enable_thinking=enable_thinking,
                exact_request=exact_request,
            )
            _update_qwen_log_request(session_dir, {"exact_request": exact_request, "request_summary": request_summary})
            _append_qwen_log_event(session_dir, "thinking_request_summary", request_summary)
            generator = _stream_content_from_qwen_flash_endpoint(
                image_path=runtime_image_path,
                prompt=prompt,
                model_override=model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                few_shot_examples=few_shot_examples,
                enable_thinking=enable_thinking,
                log_event=_log_event,
            )
        else:
            cfg = load_finetune_config(_resolve_config_path(config_path))
            _append_qwen_log_event(session_dir, "backend_selected", {"backend": cfg.inference.backend})
            if cfg.inference.backend == "runpod_openai":
                active_backend = "runpod_openai"
                effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
                effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
                effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
                effective_enable_thinking = bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking)
                exact_request = {
                    "backend": "runpod_openai",
                    "endpoint": str(cfg.inference.endpoint_base_url or os.getenv("FINETREE_QWEN_ENDPOINT_BASE_URL") or "").strip().rstrip("/") + "/chat/completions",
                    "payload": {
                        "model": _resolve_endpoint_model(cfg, model),
                        "messages": _build_logged_qwen_messages(session_dir, prompt, preserve_legacy_single_turn=True),
                        "max_tokens": int(max_new_tokens) if max_new_tokens is not None else int(cfg.inference.max_new_tokens),
                        "stream": True,
                        "chat_template_kwargs": {"enable_thinking": effective_enable_thinking},
                    },
                }
                if effective_do_sample:
                    exact_request["payload"]["temperature"] = effective_temperature
                    exact_request["payload"]["top_p"] = effective_top_p
                request_summary = _qwen_request_summary(
                    model=_resolve_endpoint_model(cfg, model),
                    backend=active_backend,
                    enable_thinking=effective_enable_thinking,
                    exact_request=exact_request,
                )
                _update_qwen_log_request(session_dir, {"exact_request": exact_request, "request_summary": request_summary})
                _append_qwen_log_event(session_dir, "thinking_request_summary", request_summary)
                generator = _stream_content_from_runpod_endpoint(
                    cfg=cfg,
                    image_path=runtime_image_path,
                    prompt=prompt,
                    model_override=model,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    few_shot_examples=few_shot_examples,
                    enable_thinking=enable_thinking,
                    log_event=_log_event,
                )
            elif cfg.inference.backend == "runpod_queue":
                active_backend = "runpod_queue"
                effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
                effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
                effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
                effective_enable_thinking = bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking)
                queue_input: dict[str, Any] = {
                    "image_file": request_payload["logged_image_path"],
                    "image_mime_type": runtime_image_mime_type,
                    "response_mode": "text",
                    "prompt": prompt,
                    "chat_template_kwargs": {"enable_thinking": effective_enable_thinking},
                }
                if max_new_tokens is not None:
                    queue_input["max_tokens"] = int(max_new_tokens)
                model_name = _resolve_endpoint_model(cfg, model)
                if model_name:
                    queue_input["model"] = model_name
                if effective_do_sample:
                    queue_input["temperature"] = effective_temperature
                    queue_input["top_p"] = effective_top_p
                exact_request = {
                    "backend": "runpod_queue",
                    "payload": {"input": queue_input},
                }
                request_summary = _qwen_request_summary(
                    model=model_name,
                    backend=active_backend,
                    enable_thinking=effective_enable_thinking,
                    exact_request=exact_request,
                )
                _update_qwen_log_request(session_dir, {"exact_request": exact_request, "request_summary": request_summary})
                _append_qwen_log_event(session_dir, "thinking_request_summary", request_summary)
                generator = _stream_content_from_runpod_queue(
                    cfg=cfg,
                    image_path=runtime_image_path,
                    prompt=prompt,
                    model_override=model,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    log_event=_log_event,
                )
            else:
                active_backend = "local"
                effective_do_sample = bool(cfg.inference.do_sample) if do_sample is None else bool(do_sample)
                effective_temperature = float(cfg.inference.temperature) if temperature is None else float(temperature)
                effective_top_p = float(cfg.inference.top_p) if top_p is None else float(top_p)
                effective_enable_thinking = bool(cfg.inference.enable_thinking) if enable_thinking is None else bool(enable_thinking)
                exact_request = {
                    "backend": "local",
                    "input": {
                        "image_file": request_payload["logged_image_path"],
                        "prompt": prompt,
                    },
                    "generate_kwargs": {
                        "model": _effective_model_name(cfg, model),
                        "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else int(cfg.inference.max_new_tokens),
                        "do_sample": effective_do_sample,
                        "temperature": effective_temperature if effective_do_sample else None,
                        "top_p": effective_top_p if effective_do_sample else None,
                        "enable_thinking": effective_enable_thinking,
                    },
                }
                request_summary = _qwen_request_summary(
                    model=_effective_model_name(cfg, model),
                    backend=active_backend,
                    enable_thinking=effective_enable_thinking,
                    exact_request=exact_request,
                )
                _update_qwen_log_request(session_dir, {"exact_request": exact_request, "request_summary": request_summary})
                _append_qwen_log_event(session_dir, "thinking_request_summary", request_summary)
                generator = _stream_content_local(
                    cfg=cfg,
                    image_path=runtime_image_path,
                    prompt=prompt,
                    model_override=model,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    require_prepared_resize=require_prepared_resize,
                    log_event=_log_event,
                )

        for index, piece in enumerate(generator):
            if piece:
                collected_text.append(piece)
            _append_qwen_log_event(session_dir, "output_chunk", {"index": index, "text": piece or ""})
            if piece:
                yield piece
    except Exception as exc:
        _append_qwen_log_event(session_dir, "error", {"error": str(exc)})
        _write_qwen_log_output(session_dir, text="".join(collected_text), raw={"error": str(exc), "text": "".join(collected_text)})
        raise
    _append_qwen_log_event(
        session_dir,
        "thinking_response_summary",
        _qwen_response_summary(
            observed_thinking_signal_path=observed_thinking_signal_path,
            backend=active_backend,
        ),
    )
    _write_qwen_log_output(session_dir, text="".join(collected_text), raw={"streamed": True, "text": "".join(collected_text)})


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    prepared_image: Optional[Any] = None,
    prepared_size: Optional[tuple[int, int]] = None,
    original_size: Optional[tuple[float, float]] = None,
    bbox_max_pixels: Optional[int] = None,
    require_prepared_resize: bool = False,
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
            few_shot_examples=few_shot_examples,
            enable_thinking=enable_thinking,
            prepared_image=prepared_image,
            prepared_size=prepared_size,
            original_size=original_size,
            bbox_max_pixels=bbox_max_pixels,
            require_prepared_resize=require_prepared_resize,
        )
    )


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
):
    from .gemini_vlm import parse_page_extraction_text

    text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=model,
        config_path=config_path,
        enable_thinking=enable_thinking,
    )
    return parse_page_extraction_text(text)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen VLM generation on an image.")
    parser.add_argument("--config", default=None, help="Fine-tune config YAML path.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--prompt", default=None, help="Prompt text. If omitted, reads --prompt-path.")
    parser.add_argument(
        "--prompt-path",
        default=str(_DEFAULT_EXTRACTION_PROMPT_PATH),
        help="Prompt template/text file path.",
    )
    parser.add_argument("--model", default=None, help="Override inference model path/id.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_QWEN_MAX_NEW_TOKENS,
        help=f"Maximum generation tokens (default: {_DEFAULT_QWEN_MAX_NEW_TOKENS}).",
    )
    parser.add_argument("--stream", action="store_true", help="Stream output tokens to stdout.")
    return parser.parse_args(argv)


def _load_prompt(args: argparse.Namespace, image_path: Path) -> str:
    if isinstance(args.prompt, str) and args.prompt.strip():
        return args.prompt
    prompt_path = Path(args.prompt_path).expanduser()
    if not prompt_path.is_file() and prompt_path == _DEFAULT_EXTRACTION_PROMPT_PATH:
        legacy_path = Path("prompt.txt")
        if legacy_path.is_file():
            prompt_path = legacy_path
    if not prompt_path.is_file():
        template = default_extraction_prompt_template()
    else:
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
            max_new_tokens=args.max_new_tokens,
        ):
            print(chunk, end="", flush=True)
        print()
        return 0

    text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=args.model,
        config_path=args.config,
        max_new_tokens=args.max_new_tokens,
    )
    print(text)
    return 0


__all__ = [
    "DEFAULT_QWEN_BBOX_MAX_PIXELS",
    "current_qwen_gt_model_choices",
    "generate_content_from_image",
    "generate_page_extraction_from_image",
    "is_qwen_flash_model_requested",
    "main",
    "resolve_qwen_flash_api_key",
    "stream_content_from_image",
]


if __name__ == "__main__":
    raise SystemExit(main())
