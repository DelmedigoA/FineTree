from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from functools import lru_cache
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - import presence depends on runtime environment
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

from .fact_normalization import normalize_fact_payload, normalize_note_num
from .fact_ordering import normalize_document_meta
from .schema_registry import SchemaRegistry
from .schemas import Fact, PageExtraction, PageMeta, split_legacy_page_type


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SUPPORTED_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)
_SUPPORTED_GEMINI_MODEL_MAP: dict[str, str] = {
    model.lower(): model
    for model in SUPPORTED_GEMINI_MODELS
}
_ALLOWED_RESPONSE_JSON_SCHEMA_KEYS = {
    "$id",
    "$defs",
    "$ref",
    "$anchor",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "additionalProperties",
    "required",
    "propertyOrdering",
}
_EXTRACT_CONTRACT = SchemaRegistry.get_prompt_contract("extraction")
_PATCH_CONTRACT = SchemaRegistry.get_prompt_contract("gemini_fill")
_VALID_PAGE_TYPES = set(_EXTRACT_CONTRACT["enums"]["page_types"])
_VALID_STATEMENT_TYPES = set(_EXTRACT_CONTRACT["enums"]["statement_types"])
_VALID_CURRENCIES = set(_EXTRACT_CONTRACT["enums"]["currencies"])
_VALID_SCALES = set(_EXTRACT_CONTRACT["enums"]["scales"])
_VALID_VALUE_TYPES = set(_EXTRACT_CONTRACT["enums"]["value_types"])
_VALID_PATCH_FACT_FIELDS = set(_PATCH_CONTRACT["fact_patch_fields"])
_VALID_PATCH_TOP_LEVEL_KEYS = {"meta_updates", "fact_updates"}
_LEGACY_THINKING_BUDGET_AUTO = -1
_LEGACY_THINKING_BUDGET_DISABLED = 0
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_GEMINI_3_THINKING_LEVEL_ATTRS = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


def _normalize_gemini_model_name(model_name: Optional[str]) -> str:
    text = str(model_name or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"^gemini[-]?3", "gemini-3", text)
    return text


def resolve_supported_gemini_model_name(model_name: Optional[str]) -> str:
    normalized = _normalize_gemini_model_name(model_name)
    if not normalized:
        return DEFAULT_GEMINI_MODEL
    direct = _SUPPORTED_GEMINI_MODEL_MAP.get(normalized)
    if direct is not None:
        return direct
    if "pro" in normalized and "flash" not in normalized:
        if "2.5" in normalized:
            return "gemini-2.5-pro"
        return "gemini-3.1-pro-preview"
    if "flash" in normalized:
        if "2.5" in normalized:
            return "gemini-2.5-flash"
        if "lite" in normalized or "3.1" in normalized:
            return "gemini-3.1-flash-lite"
        return "gemini-3-flash-preview"
    return str(model_name or "").strip() or DEFAULT_GEMINI_MODEL


def _is_gemini_3_model(model_name: Optional[str]) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    return normalized.startswith("gemini-3")


def _is_gemini_3_pro_model(model_name: Optional[str]) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    if not normalized.startswith("gemini-3"):
        return False
    return "pro" in normalized and "flash" not in normalized


def _normalize_thinking_level(thinking_level: Optional[str]) -> str | None:
    if thinking_level is None:
        return None
    text = str(thinking_level).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "minimal",
        "off": "minimal",
        "no_thinking": "minimal",
    }
    normalized = aliases.get(text, text)
    if normalized not in _VALID_THINKING_LEVELS:
        raise ValueError("thinking_level must be one of: minimal, low, medium, high.")
    return normalized


def _thinking_level_for_gemini_3_model(
    model_name: Optional[str],
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
) -> Any:
    if types is None:
        return None

    normalized_level = _normalize_thinking_level(thinking_level)
    if normalized_level is None:
        if enable_thinking is None:
            return None
        normalized_level = "high" if enable_thinking else "minimal"

    if _is_gemini_3_pro_model(model_name) and normalized_level == "minimal":
        # Gemini 3.1 Pro does not expose the minimal level, so fall back to low.
        normalized_level = "low"

    attr_name = _GEMINI_3_THINKING_LEVEL_ATTRS.get(normalized_level)
    if attr_name is not None:
        level = getattr(types.ThinkingLevel, attr_name, None)
        if level is not None:
            return level

    level_name = normalized_level.upper()
    level = getattr(types.ThinkingLevel, level_name, None)
    if level is not None:
        return level

    # Compatibility fallback for SDK builds that expose only HIGH/MINIMAL.
    if normalized_level in {"low", "medium"}:
        return getattr(types.ThinkingLevel, "HIGH", getattr(types.ThinkingLevel, "MINIMAL", None))
    if normalized_level == "minimal":
        return getattr(types.ThinkingLevel, "MINIMAL", getattr(types.ThinkingLevel, "LOW", None))
    if normalized_level == "high":
        return getattr(types.ThinkingLevel, "HIGH", None)

    # Gemini 3.1 Pro does not support minimal.
    return None


def _thinking_config_for_model(
    model_name: Optional[str],
    enable_thinking: Optional[bool],
    thinking_level: Optional[str] = None,
) -> Any:
    if enable_thinking is None and thinking_level is None:
        return None
    if types is None:
        return None

    if _is_gemini_3_model(model_name):
        level = _thinking_level_for_gemini_3_model(model_name, enable_thinking, thinking_level)
        if level is None:
            return None
        return types.ThinkingConfig(thinking_level=level)

    normalized_level = _normalize_thinking_level(thinking_level)
    if normalized_level is None:
        if enable_thinking is None:
            return None
        normalized_level = "high" if enable_thinking else "minimal"
    budget = _LEGACY_THINKING_BUDGET_DISABLED if normalized_level == "minimal" else _LEGACY_THINKING_BUDGET_AUTO
    return types.ThinkingConfig(thinking_budget=budget)


def _generation_config(
    model_name: Optional[str],
    *,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    response_json_schema: Any = None,
) -> Any:
    _require_google_genai()

    config_kwargs: dict[str, Any] = {}
    thinking_config = _thinking_config_for_model(model_name, enable_thinking, thinking_level=thinking_level)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    if response_mime_type is not None:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_json_schema is not None:
        config_kwargs["response_json_schema"] = response_json_schema
    if not config_kwargs:
        return None
    return types.GenerateContentConfig(**config_kwargs)


def _require_google_genai() -> None:
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is required for Gemini calls. "
            "Install it with: python -m pip install google-genai"
        )


def _infer_mime_type(image_path: Path, explicit_mime_type: Optional[str]) -> str:
    if explicit_mime_type:
        return explicit_mime_type
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "application/octet-stream"


def _part_from_text(text: str) -> Any:
    from_text = getattr(types.Part, "from_text", None) if types is not None else None
    if callable(from_text):
        return from_text(text=text)
    return {"text": text}


def _gemini_logs_dir() -> Path:
    path = Path.cwd() / "gemini_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename_fragment(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def _serialize_gemini_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_gemini_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_gemini_log_value(item) for item in value]

    for method_name in ("model_dump", "to_json_dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                if method_name == "model_dump":
                    return _serialize_gemini_log_value(method(mode="json"))
                return _serialize_gemini_log_value(method())
            except TypeError:
                try:
                    return _serialize_gemini_log_value(method())
                except Exception:
                    pass
            except Exception:
                pass

    if hasattr(value, "__dict__"):
        try:
            return _serialize_gemini_log_value(vars(value))
        except Exception:
            pass

    return str(value)


def _copy_gemini_log_image(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _create_gemini_log_session(
    *,
    operation: str,
    model: str,
    image_path: Path,
    prompt: str,
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    extra_request: Optional[dict[str, Any]] = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dirname = (
        f"{timestamp}_{_safe_filename_fragment(operation)}_{_safe_filename_fragment(model)}"
    )
    session_dir = _gemini_logs_dir() / dirname
    session_dir.mkdir(parents=True, exist_ok=True)

    target_copy_name = f"input_target{image_path.suffix.lower() or '.bin'}"
    _copy_gemini_log_image(image_path, session_dir / target_copy_name)

    logged_examples: list[dict[str, Any]] = []
    for index, raw_example in enumerate(few_shot_examples or [], start=1):
        if not isinstance(raw_example, dict):
            continue
        raw_image_path = raw_example.get("image_path")
        example_image_path = raw_image_path if isinstance(raw_image_path, Path) else Path(str(raw_image_path or "")).expanduser()
        if not example_image_path.is_file():
            continue
        copy_name = f"few_shot_{index:02d}_{example_image_path.name}"
        _copy_gemini_log_image(example_image_path, session_dir / copy_name)
        logged_examples.append(
            {
                "source_image_path": str(example_image_path),
                "logged_image_path": copy_name,
                "expected_json": str(raw_example.get("expected_json") or ""),
            }
        )

    request_payload = {
        "operation": operation,
        "model": model,
        "prompt": prompt,
        "image_path": str(image_path),
        "logged_image_path": target_copy_name,
        "enable_thinking": enable_thinking,
        "thinking_level": thinking_level,
        "few_shot_examples": logged_examples,
    }
    if extra_request:
        request_payload.update(extra_request)
    (session_dir / "request.json").write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _append_gemini_log_event(
        session_dir,
        "session_started",
        {
            "operation": operation,
            "model": model,
            "image_path": str(image_path),
            "enable_thinking": enable_thinking,
            "thinking_level": thinking_level,
        },
    )
    return session_dir


def _append_gemini_log_event(session_dir: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with (session_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")


def _read_gemini_log_request(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "request.json").read_text(encoding="utf-8"))


def _update_gemini_log_request(session_dir: Path, updates: dict[str, Any]) -> None:
    payload = _read_gemini_log_request(session_dir)
    payload.update(updates)
    (session_dir / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _model_family_label(model_name: Optional[str]) -> str:
    return "gemini_3" if _is_gemini_3_model(model_name) else "gemini_legacy"


def _extract_serialized_thinking_config(config_payload: Any) -> dict[str, Any] | None:
    if isinstance(config_payload, dict):
        direct = config_payload.get("thinking_config")
        if isinstance(direct, dict):
            return direct
        kwargs = config_payload.get("kwargs")
        if isinstance(kwargs, dict):
            return _extract_serialized_thinking_config(kwargs)
    return None


def _thinking_request_summary(
    model_name: Optional[str],
    *,
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
    config: Any,
) -> dict[str, Any]:
    config_payload = _serialize_gemini_log_value(config)
    thinking_config = _extract_serialized_thinking_config(config_payload)
    normalized_level = _normalize_thinking_level(thinking_level)
    if normalized_level is None and enable_thinking is not None:
        normalized_level = "high" if enable_thinking else "minimal"
    semantics = "none"
    effective_value: Any = None
    if isinstance(thinking_config, dict):
        if "thinking_level" in thinking_config:
            semantics = "thinking_level"
            effective_value = thinking_config.get("thinking_level")
        elif "thinking_budget" in thinking_config:
            semantics = "thinking_budget"
            effective_value = thinking_config.get("thinking_budget")
        else:
            kwargs = thinking_config.get("kwargs")
            if isinstance(kwargs, dict):
                if "thinking_level" in kwargs:
                    semantics = "thinking_level"
                    effective_value = kwargs.get("thinking_level")
                elif "thinking_budget" in kwargs:
                    semantics = "thinking_budget"
                    effective_value = kwargs.get("thinking_budget")
    return {
        "resolved_model": str(model_name or ""),
        "model_family": _model_family_label(model_name),
        "requested_enable_thinking": enable_thinking,
        "requested_thinking_level": thinking_level,
        "normalized_requested_thinking_level": normalized_level,
        "requested_mode": (
            "default"
            if enable_thinking is None and thinking_level is None
            else ("thinking" if enable_thinking is not False else "non-thinking")
        ),
        "thinking_semantics": semantics,
        "effective_thinking_value": effective_value,
        "resolved_generation_config": config_payload,
    }


def _find_thinking_signal(payload: Any, *, _path: str = "") -> Optional[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).strip().lower()
            next_path = f"{_path}.{key}" if _path else str(key)
            if key_text in {"thinking", "thought", "thoughts", "thought_signature", "thought_summary"} and value not in ("", None, [], {}):
                return next_path
            found = _find_thinking_signal(value, _path=next_path)
            if found:
                return found
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found = _find_thinking_signal(value, _path=f"{_path}[{index}]")
            if found:
                return found
    return None


def _response_thinking_summary(raw_payload: Any) -> dict[str, Any]:
    signal_path = _find_thinking_signal(_serialize_gemini_log_value(raw_payload))
    return {
        "observed_thinking_signal": signal_path is not None,
        "observed_thinking_signal_path": signal_path,
    }


def _build_logged_gemini_contents(session_dir: Path, prompt: str) -> list[Any]:
    request_payload = _read_gemini_log_request(session_dir)
    target_ref = {"type": "image_file", "file": request_payload["logged_image_path"]}
    examples = request_payload.get("few_shot_examples") or []
    if not examples:
        return [target_ref, prompt]

    contents: list[Any] = []
    for example in examples:
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"type": "image_file", "file": example["logged_image_path"]},
                    {"type": "text", "text": "Example input page."},
                ],
            }
        )
        contents.append(
            {
                "role": "model",
                "parts": [{"type": "text", "text": example["expected_json"]}],
            }
        )
    contents.append(
        {
            "role": "user",
            "parts": [target_ref, {"type": "text", "text": prompt}],
        }
    )
    return contents


def _write_gemini_log_output(session_dir: Path, *, text: str, raw: Any) -> None:
    (session_dir / "output.txt").write_text(str(text or ""), encoding="utf-8")
    (session_dir / "response.json").write_text(
        json.dumps(_serialize_gemini_log_value(raw), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _build_generation_contents(
    *,
    image_path: Path,
    prompt: str,
    mime_type: Optional[str],
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
) -> list[Any]:
    inferred_mime_type = _infer_mime_type(image_path, mime_type)
    image_bytes = image_path.read_bytes()
    target_part = types.Part.from_bytes(data=image_bytes, mime_type=inferred_mime_type)

    # Preserve exact legacy format when few-shot is disabled.
    if not few_shot_examples:
        return [target_part, prompt]

    contents: list[Any] = []
    valid_examples = 0
    for raw_example in few_shot_examples:
        if not isinstance(raw_example, dict):
            continue

        raw_image_path = raw_example.get("image_path")
        if isinstance(raw_image_path, Path):
            example_image_path = raw_image_path
        else:
            example_image_path = Path(str(raw_image_path or "")).expanduser()
        if not example_image_path.is_file():
            continue

        expected_json = str(raw_example.get("expected_json") or "").strip()
        if not expected_json:
            continue

        example_mime = _infer_mime_type(example_image_path, None)
        example_bytes = example_image_path.read_bytes()
        contents.append(
            {
                "role": "user",
                "parts": [
                    types.Part.from_bytes(data=example_bytes, mime_type=example_mime),
                    _part_from_text("Example input page."),
                ],
            }
        )
        contents.append(
            {
                "role": "model",
                "parts": [_part_from_text(expected_json)],
            }
        )
        valid_examples += 1

    # If all examples were invalid/unavailable, keep the original single-image behavior.
    if valid_examples == 0:
        return [target_part, prompt]

    contents.append(
        {
            "role": "user",
            "parts": [
                target_part,
                _part_from_text(prompt),
            ],
        }
    )
    return contents


def _resolve_config_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path.cwd() / "finetree_config.toml")
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "finetree_config.toml")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _api_key_from_config() -> Optional[str]:
    config_path = _resolve_config_path()
    if not config_path:
        return None
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    value = data.get("api_key")
    if isinstance(value, str) and value.strip():
        return value.strip()

    gemini_section = data.get("gemini")
    if isinstance(gemini_section, dict):
        value = gemini_section.get("api_key")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=1)
def _api_key_from_doppler() -> Optional[str]:
    if shutil.which("doppler") is None:
        return None

    project = str(os.getenv("DOPPLER_PROJECT") or "").strip()
    config = str(os.getenv("DOPPLER_CONFIG") or "").strip()
    scope_args: list[str] = []
    if project:
        scope_args.extend(["--project", project])
    if config:
        scope_args.extend(["--config", config])

    for secret_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "FINETREE_GEMINI_API_KEY"):
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


def _resolve_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    return (
        explicit_api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("FINETREE_GEMINI_API_KEY")
        or _api_key_from_doppler()
        or _api_key_from_config()
    )


def resolve_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    return _resolve_api_key(explicit_api_key)


def _sanitize_response_json_schema(node: Any, parent_key: Optional[str] = None) -> Any:
    if isinstance(node, dict):
        if parent_key in {"$defs", "properties"}:
            return {k: _sanitize_response_json_schema(v) for k, v in node.items()}
        # Per Gemini docs, if a schema object includes $ref, only $-prefixed keys are allowed.
        if "$ref" in node:
            return {k: _sanitize_response_json_schema(v) for k, v in node.items() if k.startswith("$")}
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _ALLOWED_RESPONSE_JSON_SCHEMA_KEYS:
                cleaned[key] = _sanitize_response_json_schema(value, parent_key=key)
        return cleaned
    if isinstance(node, list):
        return [_sanitize_response_json_schema(x, parent_key=parent_key) for x in node]
    return node


def _extract_balanced_json_block(text: str) -> Optional[str]:
    start_idx = -1
    open_char = ""
    close_char = ""
    for i, ch in enumerate(text):
        if ch == "{":
            start_idx = i
            open_char = "{"
            close_char = "}"
            break
        if ch == "[":
            start_idx = i
            open_char = "["
            close_char = "]"
            break
    if start_idx < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _extract_balanced_block_from_index(text: str, start_idx: int, open_char: str, close_char: str) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != open_char:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _clean_json_candidate(candidate: str) -> str:
    fixed = candidate.strip()
    # Some models prefix fenced payloads with a language hint on the first line.
    if "\n" in fixed:
        first_line, rest = fixed.split("\n", 1)
        if first_line.strip().lower() in {"json", "javascript", "js"}:
            fixed = rest.strip()
    fixed = fixed.replace("“", "\"").replace("”", "\"").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _parse_llm_json(text: str) -> Any:
    candidates: list[str] = []
    raw = text.strip()
    if raw:
        candidates.append(raw)

    # Accept 3+ backticks because some providers emit ````json fences.
    fence_matches = re.findall(r"`{3,}\s*(?:json)?\s*([\s\S]*?)`{3,}", text, flags=re.IGNORECASE)
    candidates.extend([m.strip() for m in fence_matches if m.strip()])
    # Also handle an opening fence without a closing fence.
    open_fence_match = re.match(r"^\s*`{3,}\s*(?:json)?\s*([\s\S]*)$", raw, flags=re.IGNORECASE)
    if open_fence_match:
        candidates.append(open_fence_match.group(1).strip())

    balanced = _extract_balanced_json_block(text)
    if balanced:
        candidates.append(balanced.strip())

    seen: set[str] = set()
    uniq_candidates: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq_candidates.append(c)

    parse_errors: list[str] = []
    for candidate in uniq_candidates:
        for variant in (candidate, _clean_json_candidate(candidate)):
            try:
                return json.loads(variant)
            except Exception as exc:
                parse_errors.append(str(exc))
            try:
                # Fallback for Python-like dict output: single quotes, None/True/False.
                parsed = ast.literal_eval(variant)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception as exc:
                parse_errors.append(str(exc))

    raise ValueError(
        "Could not parse Gemini output as JSON. "
        f"Sample: {raw[:240]!r}. Last errors: {parse_errors[-2:]}"
    )


def _to_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_bbox(raw_bbox: Any) -> Optional[dict[str, float]]:
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        x = _to_float(raw_bbox[0])
        y = _to_float(raw_bbox[1])
        w = _to_float(raw_bbox[2])
        h = _to_float(raw_bbox[3])
        if None not in (x, y, w, h):
            return {"x": x, "y": y, "w": w, "h": h}
        return None

    if not isinstance(raw_bbox, dict):
        return None

    x = _to_float(raw_bbox.get("x"))
    y = _to_float(raw_bbox.get("y"))
    w = _to_float(raw_bbox.get("w"))
    h = _to_float(raw_bbox.get("h"))
    if None not in (x, y, w, h):
        return {"x": x, "y": y, "w": w, "h": h}

    x1 = _to_float(raw_bbox.get("x1"))
    y1 = _to_float(raw_bbox.get("y1"))
    x2 = _to_float(raw_bbox.get("x2"))
    y2 = _to_float(raw_bbox.get("y2"))
    if None not in (x1, y1, x2, y2):
        return {"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)}
    return None


def _normalize_page_extraction_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = {"facts": payload}
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object for extraction payload.")

    images_dir = _to_optional_str(payload.get("images_dir"))
    metadata = normalize_document_meta(payload.get("metadata", payload.get("document_meta")))

    page_in = payload
    if isinstance(payload.get("pages"), list) and payload["pages"]:
        first_page = next((p for p in payload["pages"] if isinstance(p, dict)), None)
        if first_page is not None:
            page_in = first_page

    meta_in = page_in.get("meta")
    if not isinstance(meta_in, dict):
        meta_in = {}

    raw_page_type = meta_in.get("page_type")
    raw_statement_type = meta_in.get("statement_type")
    legacy_type = meta_in.get("type")
    if raw_page_type in ("", None) and raw_statement_type in ("", None):
        page_type, statement_type = split_legacy_page_type(legacy_type)
    else:
        page_type = _to_optional_str(raw_page_type) or "statements"
        statement_type = _to_optional_str(raw_statement_type)
        if page_type not in _VALID_PAGE_TYPES:
            page_type, inferred_statement_type = split_legacy_page_type(page_type)
            if statement_type is None:
                statement_type = inferred_statement_type
        if statement_type is not None:
            statement_type = statement_type.strip()
        if statement_type not in _VALID_STATEMENT_TYPES:
            statement_type = None

    if page_type not in _VALID_PAGE_TYPES:
        page_type = "other"
    meta_out = {
        "entity_name": _to_optional_str(meta_in.get("entity_name") or meta_in.get("entity")),
        "page_num": _to_optional_str(meta_in.get("page_num")),
        "page_type": page_type,
        "statement_type": statement_type,
        "title": _to_optional_str(meta_in.get("title")),
    }

    facts_in = page_in.get("facts")
    if not isinstance(facts_in, list):
        facts_in = []
    facts_out: list[dict[str, Any]] = []
    for raw_fact in facts_in:
        if not isinstance(raw_fact, dict):
            continue
        bbox = _normalize_bbox(raw_fact.get("bbox") or raw_fact.get("box") or raw_fact.get("bounding_box"))
        value = raw_fact.get("value")
        if value is None:
            value = raw_fact.get("amount", raw_fact.get("number"))
        if bbox is None or value is None:
            continue

        normalized_fact, _warnings = normalize_fact_payload({**raw_fact, "bbox": bbox, "value": str(value)}, include_bbox=True)
        if normalized_fact.get("bbox") is None:
            continue
        facts_out.append(normalized_fact)

    return {
        "images_dir": images_dir,
        "metadata": metadata,
        "pages": [
            {
                "image": _to_optional_str(page_in.get("image", payload.get("image"))),
                "meta": meta_out,
                "facts": facts_out,
            }
        ],
    }


class StreamingPageExtractionParser:
    def __init__(self) -> None:
        self.buffer = ""
        self._meta_emitted = False
        self._facts_array_start: Optional[int] = None
        self._facts_scan_pos: Optional[int] = None
        self._facts_done = False
        self._latest_meta: Optional[dict[str, Any]] = None
        self._all_facts: list[dict[str, Any]] = []

    def feed(self, text_chunk: str) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        if text_chunk:
            self.buffer += text_chunk
        meta = self._extract_meta_once()
        facts = self._extract_new_facts()
        return meta, facts

    def _extract_meta_once(self) -> Optional[dict[str, Any]]:
        if self._meta_emitted:
            return None
        for token in ('"meta"', "'meta'"):
            idx = self.buffer.find(token)
            if idx < 0:
                continue
            colon = self.buffer.find(":", idx + len(token))
            if colon < 0:
                continue
            brace = self.buffer.find("{", colon + 1)
            if brace < 0:
                continue
            obj = _extract_balanced_block_from_index(self.buffer, brace, "{", "}")
            if not obj:
                continue
            try:
                parsed_meta = _parse_llm_json(obj)
                normalized = _normalize_page_extraction_payload({"meta": parsed_meta, "facts": []})
                first_page = normalized["pages"][0]
                self._latest_meta = first_page["meta"]
                self._meta_emitted = True
                return first_page["meta"]
            except Exception:
                continue
        return None

    def _ensure_facts_array_start(self) -> None:
        if self._facts_array_start is not None or self._facts_done:
            return
        for token in ('"facts"', "'facts'"):
            idx = self.buffer.find(token)
            if idx < 0:
                continue
            colon = self.buffer.find(":", idx + len(token))
            if colon < 0:
                continue
            array_start = self.buffer.find("[", colon + 1)
            if array_start < 0:
                continue
            self._facts_array_start = array_start
            self._facts_scan_pos = array_start + 1
            return

    def _extract_new_facts(self) -> list[dict[str, Any]]:
        self._ensure_facts_array_start()
        out: list[dict[str, Any]] = []
        if self._facts_scan_pos is None or self._facts_done:
            return out

        pos = self._facts_scan_pos
        while pos < len(self.buffer):
            while pos < len(self.buffer) and self.buffer[pos] in " \r\n\t,":
                pos += 1
            if pos >= len(self.buffer):
                break
            if self.buffer[pos] == "]":
                self._facts_done = True
                pos += 1
                break
            if self.buffer[pos] != "{":
                pos += 1
                continue
            obj = _extract_balanced_block_from_index(self.buffer, pos, "{", "}")
            if not obj:
                break
            try:
                parsed_fact = _parse_llm_json(obj)
                normalized = _normalize_page_extraction_payload({"meta": {}, "facts": [parsed_fact]})
                page_facts = normalized["pages"][0]["facts"]
                if page_facts:
                    out.append(page_facts[0])
                    self._all_facts.append(page_facts[0])
            except Exception:
                pass
            pos += len(obj)

        self._facts_scan_pos = pos
        return out

    def finalize(self) -> PageExtraction:
        try:
            parsed = _parse_llm_json(self.buffer)
            normalized = _normalize_page_extraction_payload(parsed)
            return PageExtraction.model_validate(normalized)
        except Exception:
            if not self._latest_meta and not self._all_facts:
                raise
            fallback_payload = {
                "meta": self._latest_meta
                or {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "other",
                    "statement_type": None,
                    "title": None,
                },
                "facts": self._all_facts,
            }
            normalized = _normalize_page_extraction_payload(fallback_payload)
            return PageExtraction.model_validate(normalized)


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> str:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_model = resolve_supported_gemini_model_name(model)
    session_dir = _create_gemini_log_session(
        operation="generate_content",
        model=resolved_model,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        few_shot_examples=few_shot_examples,
    )
    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    contents = _build_generation_contents(
        image_path=image_path,
        prompt=prompt,
        mime_type=mime_type,
        few_shot_examples=few_shot_examples,
    )
    _append_gemini_log_event(
        session_dir,
        "request",
        {
            "few_shot_count": len(few_shot_examples or []),
            "contents": _serialize_gemini_log_value(contents),
        },
    )
    request_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "contents": contents,
    }
    config = _generation_config(resolved_model, enable_thinking=enable_thinking, thinking_level=thinking_level)
    if config is not None:
        request_kwargs["config"] = config
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "exact_request": {
                "model": resolved_model,
                "contents": _build_logged_gemini_contents(session_dir, prompt),
                "config": _serialize_gemini_log_value(config),
            }
        },
    )
    _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
    response = client.models.generate_content(**request_kwargs)
    text = response.text or ""
    response_summary = _response_thinking_summary(response)
    _append_gemini_log_event(
        session_dir,
        "response",
        {
            "text": text,
            "raw": _serialize_gemini_log_value(response),
        },
    )
    _append_gemini_log_event(session_dir, "thinking_response_summary", response_summary)
    _write_gemini_log_output(session_dir, text=text, raw=response)
    return text


def stream_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> Iterator[str]:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    resolved_model = resolve_supported_gemini_model_name(model)
    session_dir = _create_gemini_log_session(
        operation="stream_content",
        model=resolved_model,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        few_shot_examples=few_shot_examples,
    )
    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    contents = _build_generation_contents(
        image_path=image_path,
        prompt=prompt,
        mime_type=mime_type,
        few_shot_examples=few_shot_examples,
    )
    _append_gemini_log_event(
        session_dir,
        "request",
        {
            "few_shot_count": len(few_shot_examples or []),
            "contents": _serialize_gemini_log_value(contents),
        },
    )

    stream_fn = getattr(client.models, "generate_content_stream", None)
    if callable(stream_fn):
        collected_text: list[str] = []
        stream_status = "completed"
        stream_error: str | None = None
        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "contents": contents,
        }
        config = _generation_config(resolved_model, enable_thinking=enable_thinking, thinking_level=thinking_level)
        if config is not None:
            request_kwargs["config"] = config
        request_summary = _thinking_request_summary(
            resolved_model,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            config=config,
        )
        _update_gemini_log_request(
            session_dir,
            {
                "request_summary": request_summary,
                "exact_request": {
                    "model": resolved_model,
                    "contents": _build_logged_gemini_contents(session_dir, prompt),
                    "config": _serialize_gemini_log_value(config),
                }
            },
        )
        _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
        stream = stream_fn(**request_kwargs)
        observed_thinking_signal_path: Optional[str] = None
        try:
            for index, chunk in enumerate(stream):
                text = getattr(chunk, "text", None)
                chunk_summary = _response_thinking_summary(chunk)
                if observed_thinking_signal_path is None and chunk_summary["observed_thinking_signal"]:
                    observed_thinking_signal_path = str(chunk_summary["observed_thinking_signal_path"])
                _append_gemini_log_event(
                    session_dir,
                    "stream_chunk",
                    {
                        "index": index,
                        "text": text or "",
                        "raw": _serialize_gemini_log_value(chunk),
                        **chunk_summary,
                    },
                )
                if text:
                    collected_text.append(text)
                    yield text
        except GeneratorExit:
            stream_status = "aborted"
            _append_gemini_log_event(
                session_dir,
                "stream_aborted",
                {"reason": "consumer_closed", "collected_text_chars": len("".join(collected_text))},
            )
            raise
        except Exception as exc:
            stream_status = "error"
            stream_error = str(exc)
            _append_gemini_log_event(
                session_dir,
                "stream_error",
                {"error": stream_error, "collected_text_chars": len("".join(collected_text))},
            )
            raise
        finally:
            _append_gemini_log_event(
                session_dir,
                "thinking_response_summary",
                {
                    "observed_thinking_signal": observed_thinking_signal_path is not None,
                    "observed_thinking_signal_path": observed_thinking_signal_path,
                    "streamed": True,
                    "status": stream_status,
                },
            )
            _write_gemini_log_output(
                session_dir,
                text="".join(collected_text),
                raw={
                    "streamed": True,
                    "status": stream_status,
                    "error": stream_error,
                },
            )
        return

    _append_gemini_log_event(session_dir, "stream_fallback", {"reason": "generate_content_stream unavailable"})
    request_kwargs = {
        "model": resolved_model,
        "contents": contents,
    }
    config = _generation_config(resolved_model, enable_thinking=enable_thinking, thinking_level=thinking_level)
    if config is not None:
        request_kwargs["config"] = config
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "exact_request": {
                "model": resolved_model,
                "contents": _build_logged_gemini_contents(session_dir, prompt),
                "config": _serialize_gemini_log_value(config),
            }
        },
    )
    _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
    response = client.models.generate_content(**request_kwargs)
    text = response.text or ""
    response_summary = _response_thinking_summary(response)
    _append_gemini_log_event(
        session_dir,
        "response",
        {
            "text": text,
            "raw": _serialize_gemini_log_value(response),
        },
    )
    _append_gemini_log_event(session_dir, "thinking_response_summary", response_summary)
    _write_gemini_log_output(session_dir, text=text, raw=response)
    if text:
        yield text


def generate_structured_json_from_image(
    image_path: Path,
    prompt: str,
    schema: dict[str, Any],
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> str:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_model = resolve_supported_gemini_model_name(model)
    session_dir = _create_gemini_log_session(
        operation="generate_structured_json",
        model=resolved_model,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        extra_request={"schema": _serialize_gemini_log_value(schema)},
    )

    image_bytes = image_path.read_bytes()
    inferred_mime_type = _infer_mime_type(image_path, mime_type)

    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=inferred_mime_type,
        ),
        prompt,
    ]
    _append_gemini_log_event(
        session_dir,
        "request",
        {
            "contents": _serialize_gemini_log_value(contents),
            "schema": _serialize_gemini_log_value(schema),
        },
    )
    config = _generation_config(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "exact_request": {
                "model": resolved_model,
                "contents": [{"type": "image_file", "file": _read_gemini_log_request(session_dir)["logged_image_path"]}, prompt],
                "config": _serialize_gemini_log_value(config),
            }
        },
    )
    _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
    response = client.models.generate_content(
        model=resolved_model,
        contents=contents,
        config=config,
    )
    text = response.text or ""
    response_summary = _response_thinking_summary(response)
    _append_gemini_log_event(
        session_dir,
        "response",
        {
            "text": text,
            "raw": _serialize_gemini_log_value(response),
        },
    )
    _append_gemini_log_event(session_dir, "thinking_response_summary", response_summary)
    _write_gemini_log_output(session_dir, text=text, raw=response)
    if not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> PageExtraction:
    # Temporarily avoid Gemini-side schema enforcement to prevent INVALID_ARGUMENT
    # errors from response_json_schema. We still enforce strict local validation.
    raw_text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        api_key=api_key,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
    )
    return parse_page_extraction_text(raw_text)


def parse_page_extraction_text(raw_text: str) -> PageExtraction:
    if not str(raw_text).strip():
        raise ValueError("Gemini returned an empty response.")
    parsed = _parse_llm_json(raw_text)
    normalized = _normalize_page_extraction_payload(parsed)
    return PageExtraction.model_validate(normalized)


def _validate_patch_fact_updates(
    updates_payload: Any,
    *,
    allowed_fact_fields: set[str],
) -> dict[str, Any]:
    if not isinstance(updates_payload, dict):
        raise ValueError("Each patch updates object must be a JSON object.")

    normalized_updates = dict(updates_payload)
    if "equations" in allowed_fact_fields:
        legacy_equation_payload = any(
            key in normalized_updates for key in ("equation", "fact_equation", "equation_children")
        )
        if legacy_equation_payload:
            normalized_equation_payload, _warnings = normalize_fact_payload(
                {
                    "value": "0",
                    "path": [],
                    **normalized_updates,
                },
                include_bbox=False,
            )
            normalized_updates.pop("equation", None)
            normalized_updates.pop("fact_equation", None)
            normalized_updates.pop("equation_children", None)
            if "equations" not in normalized_updates and normalized_equation_payload.get("equations") is not None:
                normalized_updates["equations"] = normalized_equation_payload.get("equations")

    unknown_update_keys = sorted(str(key) for key in normalized_updates.keys() if str(key) not in allowed_fact_fields)
    if unknown_update_keys:
        raise ValueError(
            f"Patch updates contain non-requested keys: {', '.join(unknown_update_keys)}."
        )

    baseline_row_role = "total" if "equations" in normalized_updates else "detail"
    baseline = {
        "value": "0",
        "equations": None,
        "comment_ref": None,
        "note_flag": False,
        "note_name": None,
        "note_num": None,
        "note_ref": None,
        "date": None,
        "period_type": None,
        "period_start": None,
        "period_end": None,
        "duration_type": None,
        "recurring_period": None,
        "path": [],
        "path_source": None,
        "currency": None,
        "scale": None,
        "value_type": None,
        "value_context": None,
        "natural_sign": None,
        "row_role": baseline_row_role,
    }
    validated = Fact.model_validate({**baseline, **normalized_updates}).model_dump(mode="json")
    return {key: validated[key] for key in normalized_updates.keys()}


def parse_selected_field_patch_text(
    raw_text: str,
    *,
    allowed_fact_fields: set[str] | list[str] | tuple[str, ...],
    allow_statement_type: bool,
) -> dict[str, Any]:
    if not str(raw_text).strip():
        raise ValueError("Gemini returned an empty patch response.")

    parsed = _parse_llm_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Patch response must be a JSON object.")

    requested_fact_fields = {str(key).strip() for key in allowed_fact_fields if str(key).strip()}
    invalid_requested = sorted(requested_fact_fields - _VALID_PATCH_FACT_FIELDS)
    if invalid_requested:
        raise ValueError(
            f"Invalid requested patch fields: {', '.join(invalid_requested)}."
        )

    unknown_top_level = sorted(str(key) for key in parsed.keys() if str(key) not in _VALID_PATCH_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(f"Patch response contains unknown top-level keys: {', '.join(unknown_top_level)}.")

    raw_meta_updates = parsed.get("meta_updates")
    meta_updates: dict[str, Any] = {}
    if raw_meta_updates is not None:
        if not isinstance(raw_meta_updates, dict):
            raise ValueError("meta_updates must be a JSON object when provided.")
        unknown_meta_keys = sorted(str(key) for key in raw_meta_updates.keys() if str(key) != "statement_type")
        if unknown_meta_keys:
            raise ValueError(f"meta_updates contains unknown keys: {', '.join(unknown_meta_keys)}.")
        if allow_statement_type and "statement_type" in raw_meta_updates:
            normalized_meta = PageMeta.model_validate(
                {
                    "entity_name": None,
                    "page_num": None,
                    "page_type": "statements",
                    "statement_type": raw_meta_updates.get("statement_type"),
                    "title": None,
                }
            ).model_dump(mode="json")
            meta_updates["statement_type"] = normalized_meta.get("statement_type")

    raw_fact_updates = parsed.get("fact_updates")
    if not isinstance(raw_fact_updates, list):
        raise ValueError("fact_updates must be a JSON array.")

    seen_fact_nums: set[int] = set()
    normalized_fact_updates: list[dict[str, Any]] = []
    for entry in raw_fact_updates:
        if not isinstance(entry, dict):
            raise ValueError("Each fact_updates entry must be a JSON object.")
        unknown_entry_keys = sorted(str(key) for key in entry.keys() if str(key) not in {"fact_num", "updates"})
        if unknown_entry_keys:
            raise ValueError(f"fact_updates entry contains unknown keys: {', '.join(unknown_entry_keys)}.")
        fact_num = entry.get("fact_num")
        if isinstance(fact_num, bool) or not isinstance(fact_num, int) or fact_num < 1:
            raise ValueError("fact_num must be an integer >= 1.")
        if fact_num in seen_fact_nums:
            raise ValueError(f"Duplicate fact_num in patch response: {fact_num}.")
        seen_fact_nums.add(fact_num)

        updates = _validate_patch_fact_updates(
            entry.get("updates"),
            allowed_fact_fields=requested_fact_fields,
        )
        normalized_fact_updates.append({"fact_num": fact_num, "updates": updates})

    return {
        "meta_updates": meta_updates,
        "fact_updates": normalized_fact_updates,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini image+text generation on a local image.")
    parser.add_argument("image", help="Path to image file.")
    parser.add_argument("prompt", help="Prompt to send with the image.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument("--mime-type", default=None, help="Override MIME type (default: infer from extension).")
    parser.add_argument("--api-key", default=None, help="Google API key (default: env GOOGLE_API_KEY/GEMINI_API_KEY).")
    parser.add_argument(
        "--thinking-level",
        default=None,
        choices=sorted(_VALID_THINKING_LEVELS),
        help="Optional thinking level override: minimal|low|medium|high.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = generate_content_from_image(
            image_path=Path(args.image),
            prompt=args.prompt,
            model=args.model,
            mime_type=args.mime_type,
            api_key=args.api_key,
            thinking_level=args.thinking_level,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
