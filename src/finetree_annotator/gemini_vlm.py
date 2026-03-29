from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image, ImageDraw
from pydantic import ValidationError

try:  # pragma: no cover - import presence depends on runtime environment
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

from .bbox_utils import bbox_to_list, denormalize_bbox_from_1000
from .fact_normalization import normalize_fact_payload, normalize_note_num
from .fact_ordering import normalize_document_meta
from .schema_registry import SchemaRegistry
from .schemas import ExtractedFact, Fact, PageExtraction, PageMeta, split_legacy_page_type
from .vision_resize import (
    DEFAULT_QWEN_VISION_FACTOR,
    fallback_smart_resize_dimensions as _shared_fallback_smart_resize_dimensions,
    smart_resize_dimensions as _shared_smart_resize_dimensions,
)


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
VERTEX_TUNED_GEMINI_MODEL = "gemini-flash-hf-tuned"
DEFAULT_VERTEX_PROJECT_ID = "gen-lang-client-0533315636"
DEFAULT_VERTEX_ENDPOINT_ID = "4766539037060104192"
DEFAULT_VERTEX_REGION = "europe-west4"
DEFAULT_VERTEX_TUNED_MAX_PIXELS = 1_200_000
DEFAULT_GEMINI_GT_RESPONSE_MIME_TYPE = "application/json"
DEFAULT_GEMINI_GT_MEDIA_RESOLUTION = "high"
_BBOX_MODE_PIXEL_AS_IS = "pixel_as_is"
_BBOX_MODE_NORMALIZED_1000_TO_PIXEL = "normalized_1000_to_pixel"
_BBOX_MODE_SWITCH_MARGIN = 0.08
_VERTEX_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
SUPPORTED_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    VERTEX_TUNED_GEMINI_MODEL,
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
_BBOX_ARRAY_LITERAL_RE = re.compile(
    r'("bbox"\s*:\s*\[\s*)(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(\s*\])'
)
_PROMPT_IMAGE_SIZE_RE = re.compile(
    r"Current image size:\s*(\d+)\s*x\s*(\d+)\s*pixels",
    flags=re.IGNORECASE,
)
_FACT_TRACE_FILE_NAME = "fact_trace.jsonl"
_ISSUE_SUMMARY_FILE_NAME = "issue_summary.json"


def _normalize_gemini_model_name(model_name: Optional[str]) -> str:
    text = str(model_name or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    if "gemini" in text and "flash" in text and "hf" in text and "tuned" in text:
        return VERTEX_TUNED_GEMINI_MODEL
    text = re.sub(r"^gemini[-]?3", "gemini-3", text)
    return text


def is_vertex_gemini_model_requested(model_name: Optional[str]) -> bool:
    return _normalize_gemini_model_name(model_name) == VERTEX_TUNED_GEMINI_MODEL


def resolve_supported_gemini_model_name(model_name: Optional[str]) -> str:
    normalized = _normalize_gemini_model_name(model_name)
    if not normalized:
        return DEFAULT_GEMINI_MODEL
    if normalized == VERTEX_TUNED_GEMINI_MODEL:
        return VERTEX_TUNED_GEMINI_MODEL
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
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    response_mime_type: Optional[str] = None,
    response_json_schema: Any = None,
    media_resolution: Optional[str] = None,
) -> Any:
    _require_google_genai()

    config_kwargs: dict[str, Any] = {}
    thinking_config = _thinking_config_for_model(model_name, enable_thinking, thinking_level=thinking_level)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    normalized_system_prompt = _normalize_system_prompt(system_prompt)
    if normalized_system_prompt is not None:
        config_kwargs["system_instruction"] = normalized_system_prompt
    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        config_kwargs["temperature"] = normalized_temperature
    if response_mime_type is not None:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_json_schema is not None:
        config_kwargs["response_json_schema"] = response_json_schema
    resolved_media_resolution = _resolve_media_resolution(media_resolution)
    if resolved_media_resolution is not None:
        config_kwargs["media_resolution"] = resolved_media_resolution
    if not config_kwargs:
        return None
    return types.GenerateContentConfig(**config_kwargs)


def _normalize_system_prompt(system_prompt: Optional[str]) -> str | None:
    text = str(system_prompt or "").strip()
    return text or None


def _normalize_temperature(temperature: Optional[float]) -> float | None:
    if temperature is None:
        return None
    try:
        normalized = float(temperature)
    except Exception as exc:
        raise ValueError("temperature must be numeric.") from exc
    if math.isnan(normalized) or math.isinf(normalized):
        raise ValueError("temperature must be finite.")
    if normalized < 0.0 or normalized > 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")
    return normalized


def _resolve_media_resolution(media_resolution: Optional[str]) -> Any:
    if media_resolution is None:
        return None
    if not isinstance(media_resolution, str):
        return media_resolution
    normalized = str(media_resolution).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if types is None:
        return normalized
    enum_cls = getattr(types, "MediaResolution", None)
    if enum_cls is None:
        return normalized
    attr_map = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
    }
    attr_name = attr_map.get(normalized)
    if not attr_name:
        return normalized
    return getattr(enum_cls, attr_name, normalized)

def _create_genai_client_for_model(model_name: str, api_key: Optional[str]) -> Any:
    _require_google_genai()
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _require_google_genai() -> None:
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is required for Gemini calls. "
            "Install it with: python -m pip install google-genai"
        )


def _resolve_vertex_project_id(explicit_project_id: Optional[str] = None) -> Optional[str]:
    candidates = (
        explicit_project_id,
        os.getenv("FINETREE_VERTEX_PROJECT_ID"),
        os.getenv("GOOGLE_CLOUD_PROJECT"),
        os.getenv("GCLOUD_PROJECT"),
        os.getenv("GCP_PROJECT"),
        DEFAULT_VERTEX_PROJECT_ID,
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text

    if shutil.which("gcloud") is None:
        return None
    try:
        proc = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    if not value or value == "(unset)":
        return None
    return value


def resolve_vertex_project_id(explicit_project_id: Optional[str] = None) -> Optional[str]:
    return _resolve_vertex_project_id(explicit_project_id)


def _resolve_vertex_endpoint_id(explicit_endpoint_id: Optional[str] = None) -> str:
    return str(
        explicit_endpoint_id
        or os.getenv("FINETREE_VERTEX_ENDPOINT_ID")
        or DEFAULT_VERTEX_ENDPOINT_ID
    ).strip()


def _resolve_vertex_region(explicit_region: Optional[str] = None) -> str:
    return str(
        explicit_region
        or os.getenv("FINETREE_VERTEX_REGION")
        or DEFAULT_VERTEX_REGION
    ).strip()


def _resolve_vertex_endpoint_url(
    *,
    explicit_url: Optional[str] = None,
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> str:
    direct_url = str(explicit_url or os.getenv("FINETREE_VERTEX_ENDPOINT_URL") or "").strip()
    if direct_url:
        return direct_url
    resolved_project_id = _resolve_vertex_project_id(project_id)
    if not resolved_project_id:
        raise RuntimeError(
            "Vertex AI project id is missing. Set FINETREE_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
        )
    resolved_region = _resolve_vertex_region(region)
    resolved_endpoint_id = _resolve_vertex_endpoint_id(endpoint_id)
    return (
        f"https://{resolved_region}-aiplatform.googleapis.com/v1/projects/{resolved_project_id}"
        f"/locations/{resolved_region}/endpoints/{resolved_endpoint_id}:predict"
    )


def resolve_vertex_endpoint_url(
    *,
    explicit_url: Optional[str] = None,
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> str:
    return _resolve_vertex_endpoint_url(
        explicit_url=explicit_url,
        project_id=project_id,
        region=region,
        endpoint_id=endpoint_id,
    )


def _vertex_access_token_from_adc() -> Optional[str]:
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception:
        return None

    try:
        credentials, _project_id = google.auth.default(scopes=[_VERTEX_OAUTH_SCOPE])
    except Exception:
        return None
    if credentials is None:
        return None
    token = str(getattr(credentials, "token", "") or "").strip()
    needs_refresh = not token or bool(getattr(credentials, "expired", False))
    if needs_refresh:
        try:
            credentials.refresh(Request())
        except Exception:
            return None
        token = str(getattr(credentials, "token", "") or "").strip()
    return token or None


@lru_cache(maxsize=1)
def _vertex_access_token_from_gcloud() -> Optional[str]:
    if shutil.which("gcloud") is None:
        return None
    try:
        proc = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value or None


def _resolve_vertex_access_token(explicit_access_token: Optional[str] = None) -> Optional[str]:
    return (
        explicit_access_token
        or os.getenv("FINETREE_VERTEX_ACCESS_TOKEN")
        or os.getenv("VERTEX_AI_ACCESS_TOKEN")
        or os.getenv("GOOGLE_CLOUD_ACCESS_TOKEN")
        or _vertex_access_token_from_adc()
        or _vertex_access_token_from_gcloud()
    )


def resolve_vertex_access_token(explicit_access_token: Optional[str] = None) -> Optional[str]:
    return _resolve_vertex_access_token(explicit_access_token)


def resolve_gemini_api_key_for_model(
    model_name: Optional[str],
    explicit_api_key: Optional[str] = None,
) -> Optional[str]:
    if is_vertex_gemini_model_requested(model_name):
        return None
    return _resolve_api_key(explicit_api_key)


def ensure_gemini_backend_credentials(
    model_name: Optional[str],
    explicit_api_key: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    resolved_model = resolve_supported_gemini_model_name(model_name)
    if is_vertex_gemini_model_requested(resolved_model):
        project_id = _resolve_vertex_project_id()
        if not project_id:
            return None, (
                "Vertex AI project id not found.\n\n"
                "Set FINETREE_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
            )
        token = _resolve_vertex_access_token()
        if not token:
            return None, (
                "Vertex AI access token not found.\n\n"
                "Authenticate with Application Default Credentials, run "
                "`gcloud auth print-access-token`, or set FINETREE_VERTEX_ACCESS_TOKEN."
            )
        return None, None

    api_key = _resolve_api_key(explicit_api_key)
    if api_key:
        return api_key, None
    return None, (
        "Gemini API key not found.\n\n"
        "Set GOOGLE_API_KEY or GEMINI_API_KEY, or run through Doppler "
        "with a secret named GOOGLE_API_KEY / GEMINI_API_KEY."
    )


def _infer_mime_type(image_path: Path, explicit_mime_type: Optional[str]) -> str:
    if explicit_mime_type:
        return explicit_mime_type
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "application/octet-stream"


@dataclass(frozen=True)
class _PreparedGeminiImage:
    image_bytes: bytes
    mime_type: str
    original_width: int
    original_height: int
    prepared_width: int
    prepared_height: int

    @property
    def resized(self) -> bool:
        return (
            self.original_width != self.prepared_width
            or self.original_height != self.prepared_height
        )


def _fallback_smart_resize_dimensions(
    height: int,
    width: int,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
    factor: int = DEFAULT_QWEN_VISION_FACTOR,
) -> tuple[int, int]:
    return _shared_fallback_smart_resize_dimensions(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        factor=factor,
    )


def _smart_resize_dimensions(height: int, width: int, *, min_pixels: int | None, max_pixels: int | None) -> tuple[int, int]:
    return _shared_smart_resize_dimensions(height, width, min_pixels=min_pixels, max_pixels=max_pixels)


def _prepared_image_format(image: Image.Image, mime_type: str) -> str:
    if mime_type == "image/png":
        return "PNG"
    if mime_type == "image/webp":
        return "WEBP"
    if mime_type == "image/jpeg":
        return "JPEG"
    return str(image.format or "PNG").upper()


def _prepare_vertex_tuned_gemini_image(
    image_path: Path,
    *,
    mime_type: Optional[str],
    max_pixels: int = DEFAULT_VERTEX_TUNED_MAX_PIXELS,
) -> _PreparedGeminiImage:
    inferred_mime_type = _infer_mime_type(image_path, mime_type)
    with Image.open(image_path) as image:
        image.load()
        original_width, original_height = image.size
        prepared_width, prepared_height = original_width, original_height
        if original_width * original_height > int(max_pixels):
            prepared_height, prepared_width = _smart_resize_dimensions(
                original_height,
                original_width,
                min_pixels=None,
                max_pixels=int(max_pixels),
            )
        if prepared_width == original_width and prepared_height == original_height:
            return _PreparedGeminiImage(
                image_bytes=image_path.read_bytes(),
                mime_type=inferred_mime_type,
                original_width=original_width,
                original_height=original_height,
                prepared_width=prepared_width,
                prepared_height=prepared_height,
            )

        resized = image.resize((prepared_width, prepared_height), Image.Resampling.BICUBIC)
        if inferred_mime_type not in {"image/png", "image/webp", "image/jpeg"}:
            inferred_mime_type = "image/png"
        image_format = _prepared_image_format(resized, inferred_mime_type)
        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        if image_format == "JPEG":
            resized = resized.convert("RGB")
            save_kwargs["quality"] = 95
        resized.save(buffer, format=image_format, **save_kwargs)
        return _PreparedGeminiImage(
            image_bytes=buffer.getvalue(),
            mime_type=inferred_mime_type,
            original_width=original_width,
            original_height=original_height,
            prepared_width=prepared_width,
            prepared_height=prepared_height,
        )


def _vertex_tuned_effective_prompt(prompt: str, prepared: _PreparedGeminiImage) -> str:
    if not prepared.resized:
        return prompt
    note = (
        f"\n\nRuntime note: the image sent to you has been resized to "
        f"{prepared.prepared_width} x {prepared.prepared_height} pixels before inference. "
        "Any `bbox` you emit must use pixel coordinates of this resized image, not the original source image. "
        "Runtime will map your bbox coordinates back to the original image after inference."
    )
    adjusted = prompt
    adjusted = adjusted.replace("original-image pixel coordinates", "resized-image pixel coordinates")
    adjusted = adjusted.replace("original image pixel coordinates", "resized image pixel coordinates")
    adjusted = adjusted.replace("pixel coordinates of the original image", "pixel coordinates of the resized image")
    adjusted = adjusted.replace("in pixel coordinates of the original image", "in pixel coordinates of the resized image")
    return adjusted + note


def _iter_fact_dicts(payload: Any) -> list[dict[str, Any]]:
    facts_out: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return facts_out

    facts = payload.get("facts")
    if isinstance(facts, list):
        facts_out.extend([fact for fact in facts if isinstance(fact, dict)])

    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_facts = page.get("facts")
            if isinstance(page_facts, list):
                facts_out.extend([fact for fact in page_facts if isinstance(fact, dict)])
    return facts_out


def _rescale_bbox_to_original_space(bbox: Any, prepared: _PreparedGeminiImage) -> Any:
    normalized = _normalize_bbox(bbox)
    if normalized is None:
        return bbox
    scale_x = float(prepared.original_width) / float(prepared.prepared_width)
    scale_y = float(prepared.original_height) / float(prepared.prepared_height)
    return [
        round(normalized["x"] * scale_x, 2),
        round(normalized["y"] * scale_y, 2),
        round(normalized["w"] * scale_x, 2),
        round(normalized["h"] * scale_y, 2),
    ]


def _restore_resized_bbox_chunk_text(raw_text: str, prepared: _PreparedGeminiImage) -> str:
    if not prepared.resized or not str(raw_text).strip():
        return raw_text

    scale_x = float(prepared.original_width) / float(prepared.prepared_width)
    scale_y = float(prepared.original_height) / float(prepared.prepared_height)

    def _replace(match: re.Match[str]) -> str:
        x = round(float(match.group(2)) * scale_x, 2)
        y = round(float(match.group(3)) * scale_y, 2)
        w = round(float(match.group(4)) * scale_x, 2)
        h = round(float(match.group(5)) * scale_y, 2)
        return f'{match.group(1)}{x}, {y}, {w}, {h}{match.group(6)}'

    return _BBOX_ARRAY_LITERAL_RE.sub(_replace, raw_text)


def _split_streaming_bbox_restore_buffer(raw_text: str) -> tuple[str, str]:
    if not raw_text:
        return "", ""

    last_bbox_idx = raw_text.rfind('"bbox"')
    if last_bbox_idx < 0:
        return raw_text, ""

    colon_idx = raw_text.find(":", last_bbox_idx + len('"bbox"'))
    if colon_idx < 0:
        return raw_text[:last_bbox_idx], raw_text[last_bbox_idx:]

    array_start = raw_text.find("[", colon_idx + 1)
    if array_start < 0:
        return raw_text[:last_bbox_idx], raw_text[last_bbox_idx:]

    array_end = raw_text.find("]", array_start + 1)
    if array_end < 0:
        return raw_text[:last_bbox_idx], raw_text[last_bbox_idx:]

    return raw_text, ""


@dataclass
class _StreamingResizedBBoxRestorer:
    prepared: _PreparedGeminiImage
    buffer: str = ""

    def push(self, raw_text: str) -> str:
        if not self.prepared.resized or not raw_text:
            return raw_text

        self.buffer += raw_text
        ready_text, pending_text = _split_streaming_bbox_restore_buffer(self.buffer)
        self.buffer = pending_text
        if not ready_text:
            return ""
        return _restore_resized_bbox_chunk_text(ready_text, self.prepared)

    def flush(self) -> str:
        if not self.buffer:
            return ""
        remaining = self.buffer
        self.buffer = ""
        return _restore_resized_bbox_chunk_text(remaining, self.prepared)


def _restore_resized_bbox_text(raw_text: str, prepared: _PreparedGeminiImage) -> str:
    if not prepared.resized or not str(raw_text).strip():
        return raw_text
    try:
        payload = _parse_llm_json(raw_text)
    except Exception:
        return raw_text
    if not isinstance(payload, dict):
        return raw_text

    fact_dicts = _iter_fact_dicts(payload)
    if not fact_dicts:
        return raw_text

    for fact in fact_dicts:
        if isinstance(fact, dict) and fact.get("bbox") is not None:
            fact["bbox"] = _rescale_bbox_to_original_space(fact.get("bbox"), prepared)
    return json.dumps(payload, ensure_ascii=False)


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


def _image_dimensions_for_path(image_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except Exception:
        return None
    return int(width), int(height)


def _image_summary_for_log(image_path: Path, *, logged_image_path: str, mime_type: Optional[str]) -> dict[str, Any]:
    dims = _image_dimensions_for_path(image_path)
    summary: dict[str, Any] = {
        "source_image_path": str(image_path),
        "logged_image_path": logged_image_path,
        "mime_type": _infer_mime_type(image_path, mime_type),
        "file_size_bytes": image_path.stat().st_size if image_path.exists() else None,
        "image_width": None,
        "image_height": None,
    }
    if dims is not None:
        summary["image_width"], summary["image_height"] = dims
    return summary


def _prompt_image_size_summary(prompt: str, *, image_summary: dict[str, Any]) -> dict[str, Any]:
    match = _PROMPT_IMAGE_SIZE_RE.search(str(prompt or ""))
    summary: dict[str, Any] = {
        "mentions_current_image_size": match is not None,
        "prompt_image_size_text": None,
        "prompt_image_width": None,
        "prompt_image_height": None,
        "matches_source_image": None,
    }
    if match is None:
        return summary
    width = int(match.group(1))
    height = int(match.group(2))
    source_width = image_summary.get("image_width")
    source_height = image_summary.get("image_height")
    summary.update(
        {
            "prompt_image_size_text": f"{width} x {height} pixels",
            "prompt_image_width": width,
            "prompt_image_height": height,
            "matches_source_image": (
                source_width == width and source_height == height
                if source_width is not None and source_height is not None
                else None
            ),
        }
    )
    return summary


def _schema_signature_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "top_level_keys": [],
            "page_meta_keys": [],
            "fact_keys": [],
            "page_count": 0,
            "fact_count": 0,
        }
    top_level_keys = list(payload.keys())
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return {
            "top_level_keys": top_level_keys,
            "page_meta_keys": [],
            "fact_keys": [],
            "page_count": 0,
            "fact_count": 0,
        }
    first_page = pages[0] if pages and isinstance(pages[0], dict) else {}
    facts = first_page.get("facts") if isinstance(first_page, dict) else []
    first_fact = facts[0] if isinstance(facts, list) and facts and isinstance(facts[0], dict) else {}
    return {
        "top_level_keys": top_level_keys,
        "page_meta_keys": list((first_page.get("meta") or {}).keys()) if isinstance(first_page.get("meta"), dict) else [],
        "fact_keys": list(first_fact.keys()) if isinstance(first_fact, dict) else [],
        "page_count": len(pages),
        "fact_count": len(facts) if isinstance(facts, list) else 0,
    }


def _schema_signature_from_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or ""))
    except Exception as exc:
        return {
            "top_level_keys": [],
            "page_meta_keys": [],
            "fact_keys": [],
            "page_count": 0,
            "fact_count": 0,
            "parse_error": str(exc),
        }
    return _schema_signature_from_payload(payload)


def _create_gemini_log_session(
    *,
    operation: str,
    model: str,
    image_path: Path,
    prompt: str,
    mime_type: Optional[str],
    system_prompt: Optional[str],
    temperature: Optional[float],
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
    image_summary = _image_summary_for_log(image_path, logged_image_path=target_copy_name, mime_type=mime_type)

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
        expected_json = str(raw_example.get("expected_json") or "")
        logged_examples.append(
            {
                "source_image_path": str(example_image_path),
                "logged_image_path": copy_name,
                "expected_json": expected_json,
                "image_summary": _image_summary_for_log(example_image_path, logged_image_path=copy_name, mime_type=None),
                "schema_signature": _schema_signature_from_text(expected_json),
            }
        )

    request_payload = {
        "session_id": session_dir.name,
        "operation": operation,
        "model": model,
        "prompt": prompt,
        "system_prompt": _normalize_system_prompt(system_prompt),
        "image_path": str(image_path),
        "logged_image_path": target_copy_name,
        "image_summary": image_summary,
        "prompt_image_size": _prompt_image_size_summary(prompt, image_summary=image_summary),
        "temperature": _normalize_temperature(temperature),
        "enable_thinking": enable_thinking,
        "thinking_level": thinking_level,
        "few_shot_examples": logged_examples,
        "issue_summary_path": _ISSUE_SUMMARY_FILE_NAME,
        "fact_trace_path": _FACT_TRACE_FILE_NAME,
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
            "image_summary": image_summary,
            "system_prompt": _normalize_system_prompt(system_prompt),
            "temperature": _normalize_temperature(temperature),
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


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_serialize_gemini_log_value(payload), ensure_ascii=False, default=str))
        fh.write("\n")


def _fact_trace_path(session_dir: Path) -> Path:
    return session_dir / _FACT_TRACE_FILE_NAME


def _issue_summary_path(session_dir: Path) -> Path:
    return session_dir / _ISSUE_SUMMARY_FILE_NAME


def _append_fact_trace(session_dir: Optional[Path], payload: dict[str, Any]) -> None:
    if session_dir is None:
        return
    _append_jsonl_record(_fact_trace_path(session_dir), payload)


def _read_issue_summary(session_dir: Path) -> dict[str, Any] | None:
    path = _issue_summary_path(session_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_issue_summary(session_dir: Optional[Path], payload: dict[str, Any]) -> None:
    if session_dir is None:
        return
    serialized = _serialize_gemini_log_value(payload)
    _issue_summary_path(session_dir).write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _update_gemini_log_request(
        session_dir,
        {
            "issue_summary_path": _ISSUE_SUMMARY_FILE_NAME,
            "fact_trace_path": _FACT_TRACE_FILE_NAME,
            "issue_summary": serialized,
        },
    )


def _stable_signature_from_payload(payload: Any) -> str | None:
    if payload in ("", None, [], {}, ()):
        return None
    try:
        encoded = json.dumps(_serialize_gemini_log_value(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    except Exception:
        encoded = str(payload).encode("utf-8", errors="ignore")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _read_gemini_log_events(session_dir: Path) -> list[dict[str, Any]]:
    events_path = session_dir / "events.jsonl"
    if not events_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _usage_tokens_from_events(events: list[dict[str, Any]]) -> dict[str, int | None]:
    prompt_token_count: int | None = None
    candidate_token_count: int | None = None
    cached_content_token_count: int | None = None
    for event in events:
        raw_payload = event.get("raw")
        if not isinstance(raw_payload, dict):
            continue
        usage = raw_payload.get("usage_metadata")
        if not isinstance(usage, dict):
            continue
        prompt_value = usage.get("prompt_token_count")
        candidate_value = usage.get("candidates_token_count")
        cached_value = usage.get("cached_content_token_count")
        if isinstance(prompt_value, int):
            prompt_token_count = max(prompt_token_count or 0, prompt_value)
        if isinstance(candidate_value, int):
            candidate_token_count = max(candidate_token_count or 0, candidate_value)
        if isinstance(cached_value, int):
            cached_content_token_count = max(cached_content_token_count or 0, cached_value)
    return {
        "prompt_token_count": prompt_token_count,
        "candidate_token_count": candidate_token_count,
        "cached_content_token_count": cached_content_token_count,
    }


def _raw_page_components_from_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}, []
    page_in = payload
    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        first_page = next((page for page in pages if isinstance(page, dict)), None)
        if first_page is not None:
            page_in = first_page
    meta = page_in.get("meta")
    facts = page_in.get("facts")
    return (meta if isinstance(meta, dict) else {}), [fact for fact in facts or [] if isinstance(fact, dict)] if isinstance(facts, list) else []


def _normalize_issue_code(message: str, *, fallback: str) -> str:
    text = str(message or "").strip().lower()
    if "note_num requires note_flag=true" in text:
        return "note_num_without_note_flag"
    if "note_ref requires note_flag=true" in text:
        return "note_ref_without_note_flag"
    if "duration facts require both period_start and period_end" in text:
        return "duration_missing_period_boundary"
    if "page meta" in text:
        return "meta_validation_error"
    return fallback


def _normalize_validation_errors(exc: Exception, *, fallback_code: str) -> list[dict[str, Any]]:
    if isinstance(exc, ValidationError):
        out: list[dict[str, Any]] = []
        for error in exc.errors():
            raw_loc = error.get("loc") or ()
            if isinstance(raw_loc, tuple):
                loc = [str(part) for part in raw_loc]
            elif isinstance(raw_loc, list):
                loc = [str(part) for part in raw_loc]
            else:
                loc = [str(raw_loc)] if raw_loc not in ("", None) else []
            message = str(error.get("msg") or str(exc))
            out.append(
                {
                    "code": _normalize_issue_code(message, fallback=fallback_code),
                    "field_path": ".".join(loc),
                    "message": message,
                }
            )
        if out:
            return out
    message = str(exc)
    return [
        {
            "code": _normalize_issue_code(message, fallback=fallback_code),
            "field_path": "",
            "message": message,
        }
    ]


def _group_issue_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        code = str(entry.get("code") or "unknown_issue").strip() or "unknown_issue"
        group = grouped.setdefault(
            code,
            {
                "code": code,
                "count": 0,
                "sample_fact_indexes": [],
                "sample_messages": [],
            },
        )
        group["count"] += 1
        fact_index = entry.get("fact_index")
        if isinstance(fact_index, int) and fact_index not in group["sample_fact_indexes"] and len(group["sample_fact_indexes"]) < 3:
            group["sample_fact_indexes"].append(fact_index)
        message = str(entry.get("message") or "").strip()
        if message and message not in group["sample_messages"] and len(group["sample_messages"]) < 2:
            group["sample_messages"].append(message)
    return sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item["code"])))


def _issue_category_for_code(code: str) -> str:
    if code.startswith("bbox_"):
        return "bbox"
    if code in {"stream_parse_error", "final_parse_error"}:
        return "parse"
    if code == "meta_validation_error":
        return "meta"
    return "schema"


def _build_issue_signature(summary_payload: dict[str, Any]) -> str:
    grouped = summary_payload.get("validation_failure_groups") or []
    normalized_groups = [
        {"code": str(group.get("code") or ""), "count": int(group.get("count") or 0)}
        for group in grouped
        if str(group.get("code") or "").strip()
    ]
    signature_source = {
        "model": summary_payload.get("model"),
        "operation": summary_payload.get("operation"),
        "statement_type": summary_payload.get("statement_type"),
        "bbox_resolution_policy": summary_payload.get("bbox_resolution_policy"),
        "groups": normalized_groups,
    }
    return _stable_signature_from_payload(signature_source) or "no_issues"


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
    temperature: Optional[float],
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
        "requested_temperature": _normalize_temperature(temperature),
        "effective_temperature": (
            config_payload.get("kwargs", {}).get("temperature")
            if isinstance(config_payload, dict) and isinstance(config_payload.get("kwargs"), dict)
            else (config_payload.get("temperature") if isinstance(config_payload, dict) else None)
        ),
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


def _vertex_system_instruction_payload(system_prompt: Optional[str]) -> dict[str, Any] | None:
    normalized_system_prompt = _normalize_system_prompt(system_prompt)
    if normalized_system_prompt is None:
        return None
    return {"parts": [{"text": normalized_system_prompt}]}


def _write_gemini_log_output(session_dir: Path, *, text: str, raw: Any) -> None:
    (session_dir / "output.txt").write_text(str(text or ""), encoding="utf-8")
    (session_dir / "response.json").write_text(
        json.dumps(_serialize_gemini_log_value(raw), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bbox_ink_ratio(image: Image.Image, bbox: dict[str, float]) -> float:
    x1 = max(0, int(round(bbox["x"])))
    y1 = max(0, int(round(bbox["y"])))
    x2 = min(image.width, int(round(bbox["x"] + bbox["w"])))
    y2 = min(image.height, int(round(bbox["y"] + bbox["h"])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    histogram = image.crop((x1, y1, x2, y2)).convert("L").histogram()
    dark_pixels = sum(histogram[:235])
    total_pixels = sum(histogram) or 1
    return float(dark_pixels) / float(total_pixels)


def _write_bbox_overlay(image_path: Path, boxes: list[dict[str, float]], overlay_path: Path) -> None:
    with Image.open(image_path) as image:
        image.load()
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for bbox in boxes[:40]:
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = bbox["x"] + bbox["w"]
        y2 = bbox["y"] + bbox["h"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    canvas.save(overlay_path)


def _bbox_issue_codes_for_box(
    bbox: dict[str, float],
    *,
    image_width: int,
    image_height: int,
) -> list[str]:
    issues: list[str] = []
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    if w <= 0.0 or h <= 0.0:
        issues.append("bbox_zero_area")
    if x < 0.0 or y < 0.0 or (x + w) > float(image_width) + 1e-6 or (y + h) > float(image_height) + 1e-6:
        issues.append("bbox_out_of_bounds")
    if x <= (0.15 * float(image_width)) and y <= (0.15 * float(image_height)):
        issues.append("bbox_near_origin_cluster")
    image_area = max(float(image_width * image_height), 1.0)
    if (w * h) >= (0.20 * image_area):
        issues.append("bbox_huge_area")
    if h > 0.0 and (w / max(h, 1e-6)) >= 10.0 and h <= 20.0:
        issues.append("bbox_repeated_strip_pattern")
    return issues


def analyze_bbox_payloads(
    image_path: Path,
    fact_payloads: list[dict[str, Any]],
    *,
    overlay_path: Optional[Path] = None,
    bbox_mode_scores: Optional[dict[str, float]] = None,
    preferred_bbox_mode: Optional[str] = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "image_path": str(image_path),
        "image_width": None,
        "image_height": None,
        "bbox_count": 0,
        "x_range": None,
        "y_range": None,
        "w_range": None,
        "h_range": None,
        "out_of_bounds_count": 0,
        "near_origin_count": 0,
        "repeated_strip_count": 0,
        "repeated_strip_ratio": None,
        "median_aspect_ratio": None,
        "max_aspect_ratio": None,
        "median_ink_ratio": None,
        "min_ink_ratio": None,
        "elongated_strip_count": 0,
        "zero_area_count": 0,
        "huge_area_count": 0,
        "suspicious": False,
        "suspicion_reasons": [],
        "bbox_mode_scores": bbox_mode_scores,
        "preferred_bbox_mode": preferred_bbox_mode,
        "overlay_path": None,
    }
    dims = _image_dimensions_for_path(image_path)
    if dims is None:
        return diagnostics
    diagnostics["image_width"], diagnostics["image_height"] = dims
    boxes: list[dict[str, float]] = []
    for fact in fact_payloads:
        if not isinstance(fact, dict):
            continue
        bbox = _normalize_bbox(fact.get("bbox") or fact.get("box") or fact.get("bounding_box"))
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        return diagnostics

    xs = [box["x"] for box in boxes]
    ys = [box["y"] for box in boxes]
    ws = [box["w"] for box in boxes]
    hs = [box["h"] for box in boxes]
    aspect_ratios = [box["w"] / max(box["h"], 1e-6) for box in boxes]
    repeated_keys = [(round(box["x"], 1), round(box["w"], 1), round(box["h"], 1)) for box in boxes]
    repeated_strip_count = len(repeated_keys) - len(set(repeated_keys))
    issue_codes_by_box = [
        _bbox_issue_codes_for_box(box, image_width=int(dims[0]), image_height=int(dims[1]))
        for box in boxes
    ]
    out_of_bounds_count = sum(1 for codes in issue_codes_by_box if "bbox_out_of_bounds" in codes)
    near_origin_count = sum(1 for codes in issue_codes_by_box if "bbox_near_origin_cluster" in codes)
    zero_area_count = sum(1 for codes in issue_codes_by_box if "bbox_zero_area" in codes)
    huge_area_count = sum(1 for codes in issue_codes_by_box if "bbox_huge_area" in codes)
    with Image.open(image_path) as image:
        image.load()
        grayscale = image.convert("L")
        ink_ratios = [_bbox_ink_ratio(grayscale, box) for box in boxes[: min(len(boxes), 24)]]
    elongated_strip_count = sum(
        1 for box in boxes if box["h"] <= 20.0 and (box["w"] / max(box["h"], 1e-6)) >= 10.0
    )
    suspicion_reasons: list[str] = []
    if out_of_bounds_count > 0:
        suspicion_reasons.append("bbox_out_of_bounds")
    if near_origin_count >= max(6, len(boxes) // 2):
        suspicion_reasons.append("bbox_near_origin_cluster")
    if repeated_strip_count >= max(6, len(boxes) // 2):
        suspicion_reasons.append("bbox_repeated_strip_pattern")
    if boxes and elongated_strip_count >= max(6, len(boxes) // 2) and (_median(ink_ratios) or 0.0) < 0.14:
        if "bbox_repeated_strip_pattern" not in suspicion_reasons:
            suspicion_reasons.append("bbox_repeated_strip_pattern")

    diagnostics.update(
        {
            "bbox_count": len(boxes),
            "x_range": [round(min(xs), 2), round(max(xs), 2)],
            "y_range": [round(min(ys), 2), round(max(ys), 2)],
            "w_range": [round(min(ws), 2), round(max(ws), 2)],
            "h_range": [round(min(hs), 2), round(max(hs), 2)],
            "out_of_bounds_count": out_of_bounds_count,
            "near_origin_count": near_origin_count,
            "repeated_strip_count": repeated_strip_count,
            "repeated_strip_ratio": round(repeated_strip_count / float(len(boxes)), 4) if boxes else None,
            "median_aspect_ratio": round(_median(aspect_ratios) or 0.0, 4),
            "max_aspect_ratio": round(max(aspect_ratios), 4),
            "median_ink_ratio": round(_median(ink_ratios) or 0.0, 4),
            "min_ink_ratio": round(min(ink_ratios), 4) if ink_ratios else None,
            "elongated_strip_count": elongated_strip_count,
            "zero_area_count": zero_area_count,
            "huge_area_count": huge_area_count,
            "suspicious": bool(suspicion_reasons),
            "suspicion_reasons": suspicion_reasons,
        }
    )
    if overlay_path is not None and suspicion_reasons:
        _write_bbox_overlay(image_path, boxes, overlay_path)
        diagnostics["overlay_path"] = str(overlay_path)
    return diagnostics


def analyze_bbox_response(
    image_path: Path,
    raw_text: str,
    *,
    overlay_path: Optional[Path] = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "image_path": str(image_path),
        "image_width": None,
        "image_height": None,
        "bbox_count": 0,
        "x_range": None,
        "y_range": None,
        "w_range": None,
        "h_range": None,
        "out_of_bounds_count": 0,
        "repeated_strip_count": 0,
        "repeated_strip_ratio": None,
        "median_aspect_ratio": None,
        "max_aspect_ratio": None,
        "median_ink_ratio": None,
        "min_ink_ratio": None,
        "elongated_strip_count": 0,
        "suspicious": False,
        "suspicion_reasons": [],
        "bbox_mode_scores": None,
        "preferred_bbox_mode": None,
        "overlay_path": None,
    }
    dims = _image_dimensions_for_path(image_path)
    if dims is None or not str(raw_text or "").strip():
        return diagnostics
    diagnostics["image_width"], diagnostics["image_height"] = dims
    try:
        payload = _parse_llm_json(raw_text)
    except Exception as exc:
        diagnostics["parse_error"] = str(exc)
        return diagnostics
    if not isinstance(payload, dict):
        return diagnostics

    fact_payloads = [fact for fact in _iter_fact_dicts(payload) if isinstance(fact, dict)]
    if not fact_payloads:
        return diagnostics

    bbox_mode_scores: dict[str, float] | None = None
    preferred_bbox_mode: str | None = None
    try:
        from .ai.bbox import (
            BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
            BBOX_MODE_PIXEL_AS_IS,
            BBOX_MODE_SWITCH_MARGIN,
            payloads_for_bbox_mode,
            score_bbox_candidate_payloads,
        )

        pixel_payloads = payloads_for_bbox_mode(
            fact_payloads,
            mode=BBOX_MODE_PIXEL_AS_IS,
            image_width=float(dims[0]),
            image_height=float(dims[1]),
        )
        normalized_payloads = payloads_for_bbox_mode(
            fact_payloads,
            mode=BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
            image_width=float(dims[0]),
            image_height=float(dims[1]),
        )
        pixel_mode_score = score_bbox_candidate_payloads(image_path, pixel_payloads)
        normalized_mode_score = score_bbox_candidate_payloads(image_path, normalized_payloads)
        preferred_bbox_mode = (
            BBOX_MODE_NORMALIZED_1000_TO_PIXEL
            if normalized_mode_score > (pixel_mode_score + BBOX_MODE_SWITCH_MARGIN)
            else BBOX_MODE_PIXEL_AS_IS
        )
        bbox_mode_scores = {
            BBOX_MODE_PIXEL_AS_IS: round(pixel_mode_score, 4),
            BBOX_MODE_NORMALIZED_1000_TO_PIXEL: round(normalized_mode_score, 4),
        }
    except Exception:
        pass
    analyzed = analyze_bbox_payloads(
        image_path,
        fact_payloads,
        overlay_path=overlay_path,
        bbox_mode_scores=bbox_mode_scores,
        preferred_bbox_mode=preferred_bbox_mode,
    )
    diagnostics.update(analyzed)
    if diagnostics.get("suspicion_reasons") == ["bbox_repeated_strip_pattern"]:
        diagnostics["suspicion_reasons"] = ["wide_low_ink_strips"]
    return diagnostics


def analyze_logged_bbox_session(session_dir: Path) -> dict[str, Any]:
    request_payload = _read_gemini_log_request(session_dir)
    logged_image_path = str(request_payload.get("logged_image_path") or "").strip()
    if not logged_image_path:
        diagnostics = {
            "bbox_count": 0,
            "suspicious": False,
            "suspicion_reasons": [],
            "overlay_path": None,
            "error": "logged_image_path_missing",
        }
        _append_gemini_log_event(session_dir, "bbox_diagnostics", diagnostics)
        _update_gemini_log_request(session_dir, {"bbox_diagnostics": diagnostics, "bbox_diagnostics_raw": diagnostics})
        return diagnostics
    image_path = session_dir / logged_image_path
    raw_text = (session_dir / "output.txt").read_text(encoding="utf-8") if (session_dir / "output.txt").is_file() else ""
    overlay_path = session_dir / "bbox_overlay.png"
    diagnostics = analyze_bbox_response(
        image_path,
        raw_text,
        overlay_path=overlay_path,
    )
    if diagnostics.get("overlay_path") is not None:
        diagnostics["overlay_path"] = overlay_path.name
    elif overlay_path.exists():
        overlay_path.unlink()
    _append_gemini_log_event(session_dir, "bbox_diagnostics", diagnostics)
    _update_gemini_log_request(session_dir, {"bbox_diagnostics": diagnostics, "bbox_diagnostics_raw": diagnostics})
    return diagnostics


def _finalize_bbox_diagnostics(session_dir: Path) -> None:
    try:
        analyze_logged_bbox_session(session_dir)
    except Exception as exc:
        _append_gemini_log_event(session_dir, "bbox_diagnostics_error", {"error": str(exc)})


def _payloads_for_bbox_mode_local(
    fact_payloads: list[dict[str, Any]],
    *,
    mode: str,
    image_width: float,
    image_height: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fact_payload in fact_payloads:
        if not isinstance(fact_payload, dict):
            continue
        bbox = _normalize_bbox(fact_payload.get("bbox"))
        if bbox is None:
            continue
        resolved_bbox = (
            denormalize_bbox_from_1000(bbox, image_width, image_height)
            if mode == _BBOX_MODE_NORMALIZED_1000_TO_PIXEL
            else bbox
        )
        out.append({**deepcopy(fact_payload), "bbox": bbox_to_list(resolved_bbox)})
    return out


def _score_bbox_payloads_with_pil(image_path: Path, fact_payloads: list[dict[str, Any]]) -> float:
    if not fact_payloads or not image_path.is_file():
        return 0.0

    with Image.open(image_path) as image:
        image.load()
        grayscale = image.convert("L")
        image_width = grayscale.width
        image_height = grayscale.height

        total_score = 0.0
        total_coverage = 0.0
        scored_count = 0
        for payload in fact_payloads:
            bbox = _normalize_bbox(payload.get("bbox"))
            if bbox is None:
                continue
            area = max(float(bbox["w"]) * float(bbox["h"]), 1.0)
            left = float(bbox["x"])
            top = float(bbox["y"])
            right = left + float(bbox["w"])
            bottom = top + float(bbox["h"])

            sample_left = max(0, min(image_width - 1, int(math.floor(left))))
            sample_top = max(0, min(image_height - 1, int(math.floor(top))))
            sample_right = max(0, min(image_width, int(math.ceil(right))))
            sample_bottom = max(0, min(image_height, int(math.ceil(bottom))))
            if sample_right <= sample_left or sample_bottom <= sample_top:
                continue

            clipped_area = float((sample_right - sample_left) * (sample_bottom - sample_top))
            coverage = max(0.0, min(1.0, clipped_area / area))
            ink_ratio = _bbox_ink_ratio(grayscale, bbox)
            ink_score = max(0.0, min(1.0, ink_ratio / 0.12))
            score = (0.75 * ink_score) + (0.25 * coverage)
            if coverage < 0.35:
                score *= coverage / 0.35
            total_score += score
            total_coverage += coverage
            scored_count += 1

    if scored_count <= 0:
        return 0.0
    avg_score = total_score / float(scored_count)
    avg_coverage = total_coverage / float(scored_count)
    return max(0.0, avg_score - ((1.0 - avg_coverage) * 0.35))


def _resolve_page_extraction_bbox_mode(
    normalized_payload: dict[str, Any],
    *,
    image_path: Optional[Path] = None,
) -> tuple[dict[str, Any], str, dict[str, float], list[str], str]:
    pages = normalized_payload.get("pages")
    first_page = pages[0] if isinstance(pages, list) and pages and isinstance(pages[0], dict) else None
    facts = first_page.get("facts") if isinstance(first_page, dict) and isinstance(first_page.get("facts"), list) else []
    empty_scores = {
        _BBOX_MODE_PIXEL_AS_IS: 0.0,
        _BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
    }
    if image_path is None or not image_path.is_file() or not facts:
        return normalized_payload, _BBOX_MODE_PIXEL_AS_IS, empty_scores, [], "page_locked"

    image_dimensions = _image_dimensions_for_path(image_path)
    if image_dimensions is None:
        return normalized_payload, _BBOX_MODE_PIXEL_AS_IS, empty_scores, [], "page_locked"

    try:
        from .ai.bbox import resolve_bbox_mode as resolve_bbox_mode_qt

        resolved = resolve_bbox_mode_qt(
            image_path=image_path,
            image_dimensions=image_dimensions,
            fact_payloads=facts,
        )
        out_payload = deepcopy(normalized_payload)
        out_payload["pages"][0]["facts"] = resolved.payloads
        return out_payload, resolved.mode, {
            str(key): float(value)
            for key, value in dict(resolved.scores).items()
        }, list(resolved.fact_modes), str(resolved.policy)
    except Exception:
        image_width, image_height = image_dimensions
        pixel_payloads = _payloads_for_bbox_mode_local(
            facts,
            mode=_BBOX_MODE_PIXEL_AS_IS,
            image_width=float(image_width),
            image_height=float(image_height),
        )
        normalized_payloads = _payloads_for_bbox_mode_local(
            facts,
            mode=_BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
            image_width=float(image_width),
            image_height=float(image_height),
        )
        pixel_score = _score_bbox_payloads_with_pil(image_path, pixel_payloads)
        normalized_score = _score_bbox_payloads_with_pil(image_path, normalized_payloads)
        scores = {
            _BBOX_MODE_PIXEL_AS_IS: float(pixel_score),
            _BBOX_MODE_NORMALIZED_1000_TO_PIXEL: float(normalized_score),
        }
        if normalized_score > (pixel_score + _BBOX_MODE_SWITCH_MARGIN):
            out_payload = deepcopy(normalized_payload)
            out_payload["pages"][0]["facts"] = normalized_payloads
            return out_payload, _BBOX_MODE_NORMALIZED_1000_TO_PIXEL, scores, [
                _BBOX_MODE_NORMALIZED_1000_TO_PIXEL for _ in normalized_payloads
            ], "page_locked"
        out_payload = deepcopy(normalized_payload)
        out_payload["pages"][0]["facts"] = pixel_payloads
        return out_payload, _BBOX_MODE_PIXEL_AS_IS, scores, [
            _BBOX_MODE_PIXEL_AS_IS for _ in pixel_payloads
        ], "page_locked"


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
        input_json = str(raw_example.get("input_json") or "").strip()
        example_user_text = (
            f"Example input page.\n\n{input_json}" if input_json else "Example input page."
        )
        contents.append(
            {
                "role": "user",
                "parts": [
                    types.Part.from_bytes(data=example_bytes, mime_type=example_mime),
                    _part_from_text(example_user_text),
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


def _resolve_vertex_generate_content_url(
    *,
    explicit_url: Optional[str] = None,
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> str:
    direct_url = str(explicit_url or os.getenv("FINETREE_VERTEX_GENERATE_CONTENT_URL") or "").strip()
    if direct_url:
        return direct_url
    return _resolve_vertex_endpoint_url(
        explicit_url=None,
        project_id=project_id,
        region=region,
        endpoint_id=endpoint_id,
    ).removesuffix(":predict") + ":generateContent"


def _resolve_vertex_stream_generate_content_url(
    *,
    explicit_url: Optional[str] = None,
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> str:
    direct_url = str(explicit_url or os.getenv("FINETREE_VERTEX_STREAM_GENERATE_CONTENT_URL") or "").strip()
    if direct_url:
        return direct_url
    return _resolve_vertex_endpoint_url(
        explicit_url=None,
        project_id=project_id,
        region=region,
        endpoint_id=endpoint_id,
    ).removesuffix(":predict") + ":streamGenerateContent"


def _build_vertex_generate_content_contents(
    *,
    prepared_image: _PreparedGeminiImage,
    prompt: str,
) -> list[dict[str, Any]]:
    image_b64 = base64.b64encode(prepared_image.image_bytes).decode("ascii")
    return [
        {
            "role": "user",
            "parts": [
                {
                    "inlineData": {
                        "data": image_b64,
                        "mimeType": prepared_image.mime_type,
                    }
                },
                {"text": prompt},
            ],
        }
    ]


def _build_logged_vertex_generate_content_request(
    *,
    prompt: str,
    session_dir: Path,
    system_prompt: Optional[str] = None,
    generation_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    request_payload = _read_gemini_log_request(session_dir)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "file": request_payload["logged_image_path"],
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
    }
    system_instruction = _vertex_system_instruction_payload(system_prompt)
    if system_instruction is not None:
        payload["systemInstruction"] = system_instruction
    if generation_config:
        payload["generationConfig"] = deepcopy(generation_config)
    return payload


def _iter_json_objects_from_stream(chunks: Iterator[str]) -> Iterator[dict[str, Any]]:
    buffer = ""
    scan_pos = 0
    decoder = json.JSONDecoder()

    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        while True:
            while scan_pos < len(buffer) and buffer[scan_pos] in " \r\n\t,[]":
                scan_pos += 1
            if scan_pos >= len(buffer):
                if len(buffer) > 4096:
                    buffer = ""
                    scan_pos = 0
                break
            try:
                payload, end_idx = decoder.raw_decode(buffer, scan_pos)
            except json.JSONDecodeError:
                if scan_pos > 0:
                    buffer = buffer[scan_pos:]
                    scan_pos = 0
                break
            scan_pos = end_idx
            if isinstance(payload, dict):
                yield payload
        if scan_pos > 0 and scan_pos >= len(buffer):
            buffer = ""
            scan_pos = 0


def _find_text_in_vertex_prediction(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, dict):
        text_responses = payload.get("textResponses")
        if isinstance(text_responses, list):
            for item in text_responses:
                found = _find_text_in_vertex_prediction(item)
                if found:
                    return found
        content = payload.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                texts = [
                    str(part.get("text")).strip()
                    for part in parts
                    if isinstance(part, dict) and str(part.get("text") or "").strip()
                ]
                if texts:
                    return "".join(texts)
        for key in ("text", "generated_text", "prediction", "output", "response", "content"):
            found = _find_text_in_vertex_prediction(payload.get(key))
            if found:
                return found
        for value in payload.values():
            found = _find_text_in_vertex_prediction(value)
            if found:
                return found
        return None
    if isinstance(payload, list):
        for item in payload:
            found = _find_text_in_vertex_prediction(item)
            if found:
                return found
    return None


def _extract_vertex_prediction_text(response_payload: Any) -> str:
    if isinstance(response_payload, dict):
        predictions = response_payload.get("predictions")
        if isinstance(predictions, list) and predictions:
            for prediction in predictions:
                found = _find_text_in_vertex_prediction(prediction)
                if found:
                    return found
        found = _find_text_in_vertex_prediction(response_payload)
        if found:
            return found
    raise ValueError("Vertex AI endpoint returned no text prediction.")


def _extract_vertex_generate_content_text(response_payload: Any) -> str:
    found = _find_text_in_vertex_prediction(response_payload)
    if found:
        return found
    raise ValueError("Vertex AI generateContent response returned no text.")


def _stream_content_from_vertex_endpoint(
    *,
    image_path: Path,
    prompt: str,
    model: str,
    mime_type: Optional[str],
    system_prompt: Optional[str],
    temperature: Optional[float],
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
    session_dir: Path,
) -> Iterator[str]:
    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for Vertex AI endpoint inference.") from exc

    endpoint = _resolve_vertex_stream_generate_content_url()
    access_token = _resolve_vertex_access_token()
    if not access_token:
        raise RuntimeError(
            "Vertex AI access token not found. Authenticate with ADC/gcloud or set FINETREE_VERTEX_ACCESS_TOKEN."
        )

    prepared_image = _prepare_vertex_tuned_gemini_image(
        image_path,
        mime_type=mime_type,
    )
    effective_prompt = _vertex_tuned_effective_prompt(prompt, prepared_image)
    contents = _build_vertex_generate_content_contents(
        prepared_image=prepared_image,
        prompt=effective_prompt,
    )
    generation_config: dict[str, Any] = {}
    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        generation_config["temperature"] = normalized_temperature
    payload: dict[str, Any] = {"contents": contents}
    system_instruction = _vertex_system_instruction_payload(system_prompt)
    if system_instruction is not None:
        payload["systemInstruction"] = deepcopy(system_instruction)
    if generation_config:
        payload["generationConfig"] = generation_config
    logged_payload = _build_logged_vertex_generate_content_request(
        prompt=effective_prompt,
        session_dir=session_dir,
        system_prompt=system_prompt,
        generation_config=generation_config or None,
    )
    request_summary = _vertex_request_summary(
        endpoint,
        model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        temperature=temperature,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "backend": "vertex_stream_generate_content",
            "system_prompt": _normalize_system_prompt(system_prompt),
            "request_summary": request_summary,
            "image_resize": {
                "max_pixels": DEFAULT_VERTEX_TUNED_MAX_PIXELS,
                "original_width": prepared_image.original_width,
                "original_height": prepared_image.original_height,
                "prepared_width": prepared_image.prepared_width,
                "prepared_height": prepared_image.prepared_height,
                "resized": prepared_image.resized,
            },
            "effective_prompt": effective_prompt,
            "exact_request": {
                "endpoint": endpoint,
                "method": "POST",
                "json": logged_payload,
            },
        },
    )
    _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
    _append_gemini_log_event(
        session_dir,
        "request",
        {
            "backend": "vertex_stream_generate_content",
            "endpoint": endpoint,
            "payload": logged_payload,
        },
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    collected_text: list[str] = []
    raw_chunks: list[Any] = []
    observed_thinking_signal_path: Optional[str] = None
    stream_status = "completed"
    stream_error: Optional[str] = None
    bbox_restorer = _StreamingResizedBBoxRestorer(prepared_image) if prepared_image.resized else None

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except Exception as exc:
                    body = getattr(response, "text", "")
                    raise RuntimeError(f"Vertex AI endpoint request failed ({response.status_code}): {body[:500]}") from exc

                stream_iter = _iter_json_objects_from_stream(response.iter_text())
                for index, response_payload in enumerate(stream_iter):
                    raw_chunks.append(response_payload)
                    text = _extract_vertex_generate_content_text(response_payload)
                    emitted_text = bbox_restorer.push(text) if bbox_restorer is not None and text else text
                    chunk_summary = _response_thinking_summary(response_payload)
                    if observed_thinking_signal_path is None and chunk_summary["observed_thinking_signal"]:
                        observed_thinking_signal_path = str(chunk_summary["observed_thinking_signal_path"])
                    _append_gemini_log_event(
                        session_dir,
                        "stream_chunk",
                        {
                            "index": index,
                            "backend": "vertex_stream_generate_content",
                            "text": emitted_text,
                            "raw": response_payload,
                            **chunk_summary,
                        },
                    )
                    if emitted_text:
                        collected_text.append(emitted_text)
                        yield emitted_text

                if bbox_restorer is not None:
                    flushed_text = bbox_restorer.flush()
                    if flushed_text:
                        _append_gemini_log_event(
                            session_dir,
                            "stream_resized_bbox_flush",
                            {
                                "backend": "vertex_stream_generate_content",
                                "text": flushed_text,
                            },
                        )
                        collected_text.append(flushed_text)
                        yield flushed_text
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
                "backend": "vertex_stream_generate_content",
                "observed_thinking_signal": observed_thinking_signal_path is not None,
                "observed_thinking_signal_path": observed_thinking_signal_path,
                "streamed": True,
                "status": stream_status,
                "resized_input": prepared_image.resized,
            },
        )
        _write_gemini_log_output(
            session_dir,
            text="".join(collected_text),
            raw={
                "backend": "vertex_stream_generate_content",
                "streamed": True,
                "status": stream_status,
                "error": stream_error,
                "chunks": raw_chunks,
            },
        )
        _finalize_bbox_diagnostics(session_dir)


def _vertex_request_summary(
    endpoint: str,
    model_name: str,
    *,
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
    temperature: Optional[float],
) -> dict[str, Any]:
    return {
        "backend": "vertex_generate_content",
        "endpoint": endpoint,
        "resolved_model": model_name,
        "requested_enable_thinking": enable_thinking,
        "requested_thinking_level": thinking_level,
        "requested_temperature": _normalize_temperature(temperature),
        "thinking_supported": True,
    }


def _generate_content_from_vertex_endpoint(
    *,
    image_path: Path,
    prompt: str,
    model: str,
    mime_type: Optional[str],
    system_prompt: Optional[str],
    temperature: Optional[float],
    enable_thinking: Optional[bool],
    thinking_level: Optional[str],
    session_dir: Path,
) -> str:
    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for Vertex AI endpoint inference.") from exc

    endpoint = _resolve_vertex_generate_content_url()
    access_token = _resolve_vertex_access_token()
    if not access_token:
        raise RuntimeError(
            "Vertex AI access token not found. Authenticate with ADC/gcloud or set FINETREE_VERTEX_ACCESS_TOKEN."
        )

    prepared_image = _prepare_vertex_tuned_gemini_image(
        image_path,
        mime_type=mime_type,
    )
    effective_prompt = _vertex_tuned_effective_prompt(prompt, prepared_image)
    contents = _build_vertex_generate_content_contents(
        prepared_image=prepared_image,
        prompt=effective_prompt,
    )
    generation_config: dict[str, Any] = {}
    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        generation_config["temperature"] = normalized_temperature
    payload: dict[str, Any] = {"contents": contents}
    system_instruction = _vertex_system_instruction_payload(system_prompt)
    if system_instruction is not None:
        payload["systemInstruction"] = deepcopy(system_instruction)
    if generation_config:
        payload["generationConfig"] = generation_config
    logged_payload = _build_logged_vertex_generate_content_request(
        prompt=effective_prompt,
        session_dir=session_dir,
        system_prompt=system_prompt,
        generation_config=generation_config or None,
    )
    request_summary = _vertex_request_summary(
        endpoint,
        model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        temperature=temperature,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "backend": "vertex_generate_content",
            "system_prompt": _normalize_system_prompt(system_prompt),
            "request_summary": request_summary,
            "image_resize": {
                "max_pixels": DEFAULT_VERTEX_TUNED_MAX_PIXELS,
                "original_width": prepared_image.original_width,
                "original_height": prepared_image.original_height,
                "prepared_width": prepared_image.prepared_width,
                "prepared_height": prepared_image.prepared_height,
                "resized": prepared_image.resized,
            },
            "effective_prompt": effective_prompt,
            "exact_request": {
                "endpoint": endpoint,
                "method": "POST",
                "json": logged_payload,
            },
        },
    )
    _append_gemini_log_event(session_dir, "thinking_request_summary", request_summary)
    _append_gemini_log_event(
        session_dir,
        "request",
        {
            "backend": "vertex_generate_content",
            "endpoint": endpoint,
            "payload": logged_payload,
        },
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(endpoint, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except Exception as exc:
        body = getattr(response, "text", "")
        raise RuntimeError(f"Vertex AI endpoint request failed ({response.status_code}): {body[:500]}") from exc

    try:
        response_payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Vertex AI endpoint returned non-JSON response: {response.text[:500]}") from exc

    raw_text = _extract_vertex_prediction_text(response_payload)
    text = _restore_resized_bbox_text(raw_text, prepared_image)
    response_summary = {
        "backend": "vertex_generate_content",
        "streamed": False,
        "resized_input": prepared_image.resized,
        **_response_thinking_summary(response_payload),
    }
    _append_gemini_log_event(
        session_dir,
        "response",
        {
            "backend": "vertex_generate_content",
            "text": text,
            "raw_text": raw_text,
            "raw": response_payload,
        },
    )
    _append_gemini_log_event(session_dir, "thinking_response_summary", response_summary)
    _write_gemini_log_output(session_dir, text=text, raw=response_payload)
    _finalize_bbox_diagnostics(session_dir)
    return text


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


def _coerce_fact_note_fields_for_statement_type(
    fact_payload: dict[str, Any],
    *,
    statement_type: Optional[str],
) -> dict[str, Any]:
    if statement_type == "notes_to_financial_statements":
        return fact_payload
    if not bool(fact_payload.get("note_flag")) and fact_payload.get("note_num") in ("", None):
        return fact_payload
    normalized = dict(fact_payload)
    normalized["note_flag"] = False
    normalized["note_num"] = None
    return normalized


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
        normalized_fact = _coerce_fact_note_fields_for_statement_type(
            normalized_fact,
            statement_type=statement_type,
        )
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


def _first_page_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                return page
    return {}


def _build_page_extraction_issue_summary(
    *,
    session_dir: Optional[Path],
    meta_payload: dict[str, Any],
    bbox_mode: str,
    bbox_scores: dict[str, float],
    bbox_fact_modes: list[str],
    bbox_resolution_policy: str,
    bbox_diagnostics_raw: dict[str, Any] | None,
    bbox_diagnostics_resolved: dict[str, Any] | None,
    streamed_fact_count: int,
    final_raw_fact_count: int,
    kept_fact_count: int,
    dropped_fact_count: int,
    issue_entries: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    request_payload = _read_gemini_log_request(session_dir) if session_dir is not None else {}
    image_summary = request_payload.get("image_summary") if isinstance(request_payload.get("image_summary"), dict) else {}
    prompt_image_size = request_payload.get("prompt_image_size") if isinstance(request_payload.get("prompt_image_size"), dict) else {}
    few_shot_examples = request_payload.get("few_shot_examples") if isinstance(request_payload.get("few_shot_examples"), list) else []
    events = _read_gemini_log_events(session_dir) if session_dir is not None else []
    token_usage = _usage_tokens_from_events(events)
    validation_failure_groups = _group_issue_entries(issue_entries)
    issue_category_counts: dict[str, int] = {}
    for entry in issue_entries:
        category = _issue_category_for_code(str(entry.get("code") or ""))
        issue_category_counts[category] = issue_category_counts.get(category, 0) + 1
    summary = {
        "session_id": session_dir.name if session_dir is not None else None,
        "status": status,
        "model": request_payload.get("model"),
        "operation": request_payload.get("operation"),
        "temperature": request_payload.get("temperature"),
        "page_image_path": request_payload.get("image_path"),
        "logged_image_path": request_payload.get("logged_image_path"),
        "original_image_width": image_summary.get("image_width"),
        "original_image_height": image_summary.get("image_height"),
        "prompt_image_size": prompt_image_size,
        "prompt_signature": _stable_signature_from_payload(request_payload.get("prompt")),
        "few_shot_count": len(few_shot_examples),
        "few_shot_example_paths": [
            example.get("source_image_path") or example.get("logged_image_path")
            for example in few_shot_examples
            if isinstance(example, dict)
        ],
        "few_shot_signature": _stable_signature_from_payload(
            [
                {
                    "source_image_path": example.get("source_image_path"),
                    "logged_image_path": example.get("logged_image_path"),
                    "schema_signature": example.get("schema_signature"),
                }
                for example in few_shot_examples
                if isinstance(example, dict)
            ]
        ),
        "few_shot_schema_signature": [
            example.get("schema_signature")
            for example in few_shot_examples
            if isinstance(example, dict)
        ],
        **token_usage,
        "statement_type": meta_payload.get("statement_type"),
        "page_type": meta_payload.get("page_type"),
        "bbox_mode_scores": {str(key): round(float(value), 4) for key, value in bbox_scores.items()},
        "preferred_bbox_mode": bbox_mode,
        "bbox_fact_modes": list(bbox_fact_modes),
        "bbox_resolution_policy": bbox_resolution_policy,
        "bbox_diagnostics_raw": bbox_diagnostics_raw,
        "bbox_diagnostics_resolved": bbox_diagnostics_resolved,
        "bbox_issue_counts": {
            "out_of_bounds": int((bbox_diagnostics_resolved or {}).get("out_of_bounds_count") or 0),
            "near_origin_cluster": int((bbox_diagnostics_resolved or {}).get("near_origin_count") or 0),
            "repeated_strip_pattern": int((bbox_diagnostics_resolved or {}).get("repeated_strip_count") or 0),
            "zero_area": int((bbox_diagnostics_resolved or {}).get("zero_area_count") or 0),
            "huge_area": int((bbox_diagnostics_resolved or {}).get("huge_area_count") or 0),
        },
        "streamed_fact_count": int(streamed_fact_count),
        "final_raw_fact_count": int(final_raw_fact_count),
        "kept_fact_count": int(kept_fact_count),
        "dropped_fact_count": int(dropped_fact_count),
        "validation_failure_groups": validation_failure_groups,
        "issue_category_counts": issue_category_counts,
        "issue_count": len(issue_entries),
    }
    summary["issue_signature"] = _build_issue_signature(summary)
    return summary


def format_issue_summary_brief(
    summary: dict[str, Any] | None,
    *,
    session_dir: Optional[Path] = None,
    streamed_live_facts_remain: bool = False,
    no_changes_applied: bool = False,
) -> str:
    if not summary:
        if no_changes_applied:
            return "No changes were applied."
        return "No issue summary available."
    kept = int(summary.get("kept_fact_count") or 0)
    dropped = int(summary.get("dropped_fact_count") or 0)
    groups = summary.get("validation_failure_groups") if isinstance(summary.get("validation_failure_groups"), list) else []
    lines = [f"Kept {kept} valid fact(s), dropped {dropped} invalid fact(s)."]
    if groups:
        top_groups = ", ".join(
            f"{group.get('code')} ({int(group.get('count') or 0)})"
            for group in groups[:3]
            if isinstance(group, dict)
        )
        if top_groups:
            lines.append(f"Top issues: {top_groups}.")
    if streamed_live_facts_remain:
        lines.append("Some streamed facts were rendered live and remain on the page.")
    elif no_changes_applied:
        lines.append("No changes were applied.")
    if session_dir is not None:
        lines.append(f"Session log: {session_dir}")
    return "\n".join(lines)


def load_issue_summary(session_dir: Path) -> dict[str, Any] | None:
    return _read_issue_summary(session_dir)


def _append_bbox_issue_entries(
    issue_entries: list[dict[str, Any]],
    *,
    diagnostics: dict[str, Any] | None,
) -> None:
    if not isinstance(diagnostics, dict):
        return
    if int(diagnostics.get("out_of_bounds_count") or 0) > 0:
        issue_entries.append(
            {
                "code": "bbox_out_of_bounds",
                "message": "Resolved bbox output contains out-of-bounds boxes.",
            }
        )
    if "bbox_near_origin_cluster" in set(diagnostics.get("suspicion_reasons") or []):
        issue_entries.append(
            {
                "code": "bbox_near_origin_cluster",
                "message": "Resolved bbox output is clustered near the top-left origin.",
            }
        )
    if (
        "bbox_repeated_strip_pattern" in set(diagnostics.get("suspicion_reasons") or [])
        or float(diagnostics.get("repeated_strip_ratio") or 0.0) >= 0.5
    ):
        issue_entries.append(
            {
                "code": "bbox_repeated_strip_pattern",
                "message": "Resolved bbox output repeats strip-like geometry.",
            }
        )


def _finalize_page_extraction_payload(
    payload: Any,
    *,
    raw_text: str,
    image_path: Optional[Path] = None,
    session_dir: Optional[Path] = None,
    streamed_fact_count: int = 0,
) -> tuple[PageExtraction, dict[str, Any]]:
    raw_meta, raw_facts = _raw_page_components_from_payload(payload)
    normalized = _normalize_page_extraction_payload(payload)
    resolved_payload, bbox_mode, bbox_scores, bbox_fact_modes, bbox_resolution_policy = _resolve_page_extraction_bbox_mode(
        normalized,
        image_path=image_path,
    )
    first_page = _first_page_from_payload(resolved_payload)
    resolved_facts = [fact for fact in first_page.get("facts") or [] if isinstance(fact, dict)]
    meta_payload = first_page.get("meta") if isinstance(first_page.get("meta"), dict) else {}

    bbox_diagnostics_raw = None
    bbox_diagnostics_resolved = None
    if session_dir is not None:
        request_payload = _read_gemini_log_request(session_dir)
        bbox_diagnostics_raw = request_payload.get("bbox_diagnostics_raw") or request_payload.get("bbox_diagnostics")
    if image_path is not None and image_path.is_file():
        bbox_diagnostics_resolved = analyze_bbox_payloads(
            image_path,
            resolved_facts,
            overlay_path=(session_dir / "bbox_overlay_resolved.png") if session_dir is not None else None,
            bbox_mode_scores={str(key): round(float(value), 4) for key, value in bbox_scores.items()},
            preferred_bbox_mode=bbox_mode,
        )
        if session_dir is not None:
            raw_suspicious = bool((bbox_diagnostics_raw or {}).get("suspicious"))
            resolved_suspicious = bool((bbox_diagnostics_resolved or {}).get("suspicious"))
            if raw_suspicious and not resolved_suspicious and resolved_facts:
                overlay_path = session_dir / "bbox_overlay_resolved.png"
                resolved_boxes = [
                    bbox
                    for bbox in (_normalize_bbox(fact.get("bbox")) for fact in resolved_facts)
                    if bbox is not None
                ]
                if resolved_boxes:
                    _write_bbox_overlay(image_path, resolved_boxes, overlay_path)
                    bbox_diagnostics_resolved["overlay_path"] = "bbox_overlay_resolved.png"
            if bbox_diagnostics_resolved.get("overlay_path") is not None:
                bbox_diagnostics_resolved["overlay_path"] = "bbox_overlay_resolved.png"
            overlay_path = session_dir / "bbox_overlay_resolved.png"
            if bbox_diagnostics_resolved.get("overlay_path") is None and overlay_path.exists():
                overlay_path.unlink()

    issue_entries: list[dict[str, Any]] = []
    _append_bbox_issue_entries(issue_entries, diagnostics=bbox_diagnostics_resolved)

    try:
        validated_meta = PageMeta.model_validate(meta_payload).model_dump(mode="json")
    except Exception as exc:
        issue_entries.extend(_normalize_validation_errors(exc, fallback_code="meta_validation_error"))
        summary = _build_page_extraction_issue_summary(
            session_dir=session_dir,
            meta_payload=meta_payload,
            bbox_mode=bbox_mode,
            bbox_scores=bbox_scores,
            bbox_fact_modes=bbox_fact_modes,
            bbox_resolution_policy=bbox_resolution_policy,
            bbox_diagnostics_raw=bbox_diagnostics_raw,
            bbox_diagnostics_resolved=bbox_diagnostics_resolved,
            streamed_fact_count=streamed_fact_count,
            final_raw_fact_count=len(raw_facts),
            kept_fact_count=0,
            dropped_fact_count=len(raw_facts),
            issue_entries=issue_entries,
            status="error",
        )
        _write_issue_summary(session_dir, summary)
        raise ValueError(format_issue_summary_brief(summary, session_dir=session_dir))

    valid_facts: list[dict[str, Any]] = []
    kept_count = 0
    dropped_count = 0
    max_trace_count = max(len(raw_facts), len(resolved_facts))
    for fact_index in range(max_trace_count):
        raw_fact = raw_facts[fact_index] if fact_index < len(raw_facts) else None
        resolved_fact = resolved_facts[fact_index] if fact_index < len(resolved_facts) else None
        fact_seq = fact_index + 1
        if raw_fact is not None:
            _append_fact_trace(
                session_dir,
                {
                    "fact_seq": fact_seq,
                    "source_stage": "final_parse",
                    "raw_payload": raw_fact,
                    "raw_bbox": _normalize_bbox(raw_fact.get("bbox") or raw_fact.get("box") or raw_fact.get("bounding_box")),
                    "validation_result": None,
                },
            )
        if resolved_fact is None:
            dropped_count += 1
            issue_entries.append(
                {
                    "code": "final_parse_error",
                    "message": "Fact was dropped during normalization before bbox resolution.",
                    "fact_index": fact_index,
                }
            )
            _append_fact_trace(
                session_dir,
                {
                    "fact_seq": fact_seq,
                    "source_stage": "final_dropped",
                    "raw_payload": raw_fact,
                    "normalized_payload": None,
                    "validation_result": "invalid",
                    "normalized_errors": [
                        {
                            "code": "final_parse_error",
                            "field_path": "",
                            "message": "Fact was dropped during normalization before bbox resolution.",
                        }
                    ],
                },
            )
            continue

        resolved_bbox = _normalize_bbox(resolved_fact.get("bbox"))
        _append_fact_trace(
            session_dir,
            {
                "fact_seq": fact_seq,
                "source_stage": "final_resolved",
                "raw_payload": raw_fact,
                "normalized_payload": resolved_fact,
                "raw_bbox": _normalize_bbox(raw_fact.get("bbox") or raw_fact.get("box") or raw_fact.get("bounding_box")) if isinstance(raw_fact, dict) else None,
                "resolved_bbox": resolved_bbox,
                "bbox_mode": bbox_fact_modes[fact_index] if fact_index < len(bbox_fact_modes) else bbox_mode,
                "bbox_issue_codes": (
                    _bbox_issue_codes_for_box(
                        resolved_bbox,
                        image_width=int(bbox_diagnostics_resolved.get("image_width") or 0),
                        image_height=int(bbox_diagnostics_resolved.get("image_height") or 0),
                    )
                    if resolved_bbox is not None
                    and isinstance(bbox_diagnostics_resolved, dict)
                    and int(bbox_diagnostics_resolved.get("image_width") or 0) > 0
                    and int(bbox_diagnostics_resolved.get("image_height") or 0) > 0
                    else []
                ),
                "validation_result": None,
            },
        )
        try:
            validated_fact = ExtractedFact.model_validate(resolved_fact).model_dump(mode="json")
        except Exception as exc:
            errors = _normalize_validation_errors(exc, fallback_code="schema_validation_error")
            for error in errors:
                error["fact_index"] = fact_index
            issue_entries.extend(errors)
            dropped_count += 1
            _append_fact_trace(
                session_dir,
                {
                    "fact_seq": fact_seq,
                    "source_stage": "final_dropped",
                    "raw_payload": raw_fact,
                    "normalized_payload": resolved_fact,
                    "resolved_bbox": resolved_bbox,
                    "bbox_mode": bbox_fact_modes[fact_index] if fact_index < len(bbox_fact_modes) else bbox_mode,
                    "validation_result": "invalid",
                    "normalized_errors": errors,
                },
            )
            continue

        kept_count += 1
        valid_facts.append(validated_fact)
        _append_fact_trace(
            session_dir,
            {
                "fact_seq": fact_seq,
                "source_stage": "final_validated",
                "raw_payload": raw_fact,
                "normalized_payload": resolved_fact,
                "resolved_bbox": resolved_bbox,
                "bbox_mode": bbox_fact_modes[fact_index] if fact_index < len(bbox_fact_modes) else bbox_mode,
                "validation_result": "valid",
                "validated_payload": validated_fact,
            },
        )

    final_payload = {
        "images_dir": resolved_payload.get("images_dir"),
        "metadata": resolved_payload.get("metadata"),
        "pages": [
            {
                "image": first_page.get("image"),
                "meta": validated_meta,
                "facts": valid_facts,
            }
        ],
    }
    status = "warning" if issue_entries else "success"
    summary = _build_page_extraction_issue_summary(
        session_dir=session_dir,
        meta_payload=validated_meta,
        bbox_mode=bbox_mode,
        bbox_scores=bbox_scores,
        bbox_fact_modes=bbox_fact_modes,
        bbox_resolution_policy=bbox_resolution_policy,
        bbox_diagnostics_raw=bbox_diagnostics_raw,
        bbox_diagnostics_resolved=bbox_diagnostics_resolved,
        streamed_fact_count=streamed_fact_count,
        final_raw_fact_count=len(raw_facts),
        kept_fact_count=kept_count,
        dropped_fact_count=dropped_count,
        issue_entries=issue_entries,
        status=status,
    )
    _write_issue_summary(session_dir, summary)
    if raw_facts and not valid_facts:
        summary = _build_page_extraction_issue_summary(
            session_dir=session_dir,
            meta_payload=validated_meta,
            bbox_mode=bbox_mode,
            bbox_scores=bbox_scores,
            bbox_fact_modes=bbox_fact_modes,
            bbox_resolution_policy=bbox_resolution_policy,
            bbox_diagnostics_raw=bbox_diagnostics_raw,
            bbox_diagnostics_resolved=bbox_diagnostics_resolved,
            streamed_fact_count=streamed_fact_count,
            final_raw_fact_count=len(raw_facts),
            kept_fact_count=kept_count,
            dropped_fact_count=dropped_count,
            issue_entries=issue_entries,
            status="error",
        )
        _write_issue_summary(session_dir, summary)
        raise ValueError(format_issue_summary_brief(summary, session_dir=session_dir))
    try:
        extraction = PageExtraction.model_validate(final_payload)
    except Exception as exc:
        page_level_entries = _normalize_validation_errors(exc, fallback_code="meta_validation_error")
        for entry in page_level_entries:
            entry["fact_index"] = None
        issue_entries.extend(page_level_entries)
        summary = _build_page_extraction_issue_summary(
            session_dir=session_dir,
            meta_payload=validated_meta,
            bbox_mode=bbox_mode,
            bbox_scores=bbox_scores,
            bbox_fact_modes=bbox_fact_modes,
            bbox_resolution_policy=bbox_resolution_policy,
            bbox_diagnostics_raw=bbox_diagnostics_raw,
            bbox_diagnostics_resolved=bbox_diagnostics_resolved,
            streamed_fact_count=streamed_fact_count,
            final_raw_fact_count=len(raw_facts),
            kept_fact_count=kept_count,
            dropped_fact_count=dropped_count,
            issue_entries=issue_entries,
            status="error",
        )
        _write_issue_summary(session_dir, summary)
        raise ValueError(format_issue_summary_brief(summary, session_dir=session_dir))
    return extraction, summary


class StreamingPageExtractionParser:
    def __init__(self, *, image_path: Optional[Path] = None, session_dir: Optional[Path] = None) -> None:
        self.image_path = image_path
        self.session_dir = session_dir
        self.buffer = ""
        self._meta_emitted = False
        self._facts_array_start: Optional[int] = None
        self._facts_scan_pos: Optional[int] = None
        self._facts_done = False
        self._latest_meta: Optional[dict[str, Any]] = None
        self._all_facts: list[dict[str, Any]] = []
        self._fact_seq = 0
        self.last_issue_summary: dict[str, Any] | None = None

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
            self._fact_seq += 1
            fact_seq = self._fact_seq
            try:
                parsed_fact = _parse_llm_json(obj)
                _append_fact_trace(
                    self.session_dir,
                    {
                        "fact_seq": fact_seq,
                        "source_stage": "stream_partial",
                        "raw_snippet": obj,
                        "raw_payload": parsed_fact,
                        "raw_bbox": _normalize_bbox(parsed_fact.get("bbox") or parsed_fact.get("box") or parsed_fact.get("bounding_box"))
                        if isinstance(parsed_fact, dict)
                        else None,
                        "validation_result": None,
                    },
                )
                normalized = _normalize_page_extraction_payload(
                    {"meta": self._latest_meta or {}, "facts": [parsed_fact]}
                )
                page_facts = normalized["pages"][0]["facts"]
                if page_facts:
                    out.append(page_facts[0])
                    self._all_facts.append(page_facts[0])
                    _append_fact_trace(
                        self.session_dir,
                        {
                            "fact_seq": fact_seq,
                            "source_stage": "stream_normalized",
                            "raw_payload": parsed_fact,
                            "normalized_payload": page_facts[0],
                            "raw_bbox": _normalize_bbox(parsed_fact.get("bbox") or parsed_fact.get("box") or parsed_fact.get("bounding_box"))
                            if isinstance(parsed_fact, dict)
                            else None,
                            "resolved_bbox": _normalize_bbox(page_facts[0].get("bbox")),
                            "validation_result": None,
                        },
                    )
                else:
                    _append_fact_trace(
                        self.session_dir,
                        {
                            "fact_seq": fact_seq,
                            "source_stage": "final_dropped",
                            "raw_payload": parsed_fact,
                            "normalized_payload": None,
                            "validation_result": "invalid",
                            "normalized_errors": [
                                {
                                    "code": "stream_parse_error",
                                    "field_path": "",
                                    "message": "Fact was dropped during streaming normalization.",
                                }
                            ],
                        },
                    )
            except Exception as exc:
                _append_fact_trace(
                    self.session_dir,
                    {
                        "fact_seq": fact_seq,
                        "source_stage": "stream_partial",
                        "raw_snippet": obj,
                        "validation_result": "invalid",
                        "normalized_errors": _normalize_validation_errors(exc, fallback_code="stream_parse_error"),
                    },
                )
            pos += len(obj)

        self._facts_scan_pos = pos
        return out

    def finalize(self) -> PageExtraction:
        try:
            extraction = parse_page_extraction_text(
                self.buffer,
                image_path=self.image_path,
                session_dir=self.session_dir,
                streamed_fact_count=len(self._all_facts),
            )
            self.last_issue_summary = _read_issue_summary(self.session_dir) if self.session_dir is not None else None
            return extraction
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
            extraction, summary = _finalize_page_extraction_payload(
                fallback_payload,
                raw_text=self.buffer,
                image_path=self.image_path,
                session_dir=self.session_dir,
                streamed_fact_count=len(self._all_facts),
            )
            self.last_issue_summary = summary
            return extraction


def _project_bbox_value_fact(raw_fact: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw_fact, dict):
        return None
    bbox = _normalize_bbox(raw_fact.get("bbox") or raw_fact.get("box") or raw_fact.get("bounding_box"))
    if bbox is None:
        return None
    return {
        "bbox": bbox_to_list(bbox),
        "value": _to_optional_str(raw_fact.get("value")),
    }


def _project_bbox_value_extraction(extraction: Any) -> Any:
    from types import SimpleNamespace

    extraction_meta = getattr(extraction, "meta", None)
    if hasattr(extraction_meta, "model_dump"):
        meta_payload = extraction_meta.model_dump(mode="json")
    elif isinstance(extraction_meta, dict):
        meta_payload = extraction_meta
    else:
        meta_payload = {}
    extraction_facts = getattr(extraction, "facts", [])
    if not isinstance(extraction_facts, list):
        extraction_facts = []
    facts_out = [
        projected
        for fact in extraction_facts
        for projected in [_project_bbox_value_fact(fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact)]
        if projected is not None
    ]
    return SimpleNamespace(meta=meta_payload, facts=facts_out)


def parse_bbox_only_text(raw_text: str) -> list[dict[str, Any]]:
    extraction = parse_page_extraction_text(raw_text)
    return _project_bbox_value_extraction(extraction).facts


class StreamingBBoxOnlyParser:
    def __init__(self, *, image_path: Optional[Path] = None, session_dir: Optional[Path] = None) -> None:
        self._delegate = StreamingPageExtractionParser(image_path=image_path, session_dir=session_dir)

    @property
    def buffer(self) -> str:
        return self._delegate.buffer

    def feed(self, text_chunk: str) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        meta, facts = self._delegate.feed(text_chunk)
        projected_facts = [projected for fact in facts for projected in [_project_bbox_value_fact(fact)] if projected is not None]
        return meta, projected_facts

    def finalize(self) -> Any:
        return _project_bbox_value_extraction(self._delegate.finalize())


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    media_resolution: Optional[str] = None,
    session_dir: Optional[Path] = None,
) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_model = resolve_supported_gemini_model_name(model)
    if session_dir is None:
        session_dir = _create_gemini_log_session(
            operation="generate_content",
            model=resolved_model,
            image_path=image_path,
            prompt=prompt,
            mime_type=mime_type,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            few_shot_examples=few_shot_examples,
        )
    if is_vertex_gemini_model_requested(resolved_model):
        return _generate_content_from_vertex_endpoint(
            image_path=image_path,
            prompt=prompt,
            model=resolved_model,
            mime_type=mime_type,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            session_dir=session_dir,
        )

    key = resolve_gemini_api_key_for_model(resolved_model, explicit_api_key=api_key)
    client = _create_genai_client_for_model(resolved_model, key)
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
    config = _generation_config(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        system_prompt=system_prompt,
        temperature=temperature,
        response_mime_type=response_mime_type,
        media_resolution=media_resolution,
    )
    if config is not None:
        request_kwargs["config"] = config
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        temperature=temperature,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "system_prompt": _normalize_system_prompt(system_prompt),
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
    _finalize_bbox_diagnostics(session_dir)
    return text


def stream_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    few_shot_examples: Optional[list[dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    media_resolution: Optional[str] = None,
    session_dir: Optional[Path] = None,
) -> Iterator[str]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    resolved_model = resolve_supported_gemini_model_name(model)
    if session_dir is None:
        session_dir = _create_gemini_log_session(
            operation="stream_content",
            model=resolved_model,
            image_path=image_path,
            prompt=prompt,
            mime_type=mime_type,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            few_shot_examples=few_shot_examples,
        )
    if is_vertex_gemini_model_requested(resolved_model):
        yield from _stream_content_from_vertex_endpoint(
            image_path=image_path,
            prompt=prompt,
            model=resolved_model,
            mime_type=mime_type,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            session_dir=session_dir,
        )
        return

    key = resolve_gemini_api_key_for_model(resolved_model, explicit_api_key=api_key)
    client = _create_genai_client_for_model(resolved_model, key)
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
        config = _generation_config(
            resolved_model,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            temperature=temperature,
            response_mime_type=response_mime_type,
            media_resolution=media_resolution,
        )
        if config is not None:
            request_kwargs["config"] = config
        request_summary = _thinking_request_summary(
            resolved_model,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            temperature=temperature,
            config=config,
        )
        _update_gemini_log_request(
            session_dir,
            {
                "request_summary": request_summary,
                "system_prompt": _normalize_system_prompt(system_prompt),
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
            _finalize_bbox_diagnostics(session_dir)
        return

    _append_gemini_log_event(session_dir, "stream_fallback", {"reason": "generate_content_stream unavailable"})
    request_kwargs = {
        "model": resolved_model,
        "contents": contents,
    }
    config = _generation_config(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        system_prompt=system_prompt,
        temperature=temperature,
        response_mime_type=response_mime_type,
        media_resolution=media_resolution,
    )
    if config is not None:
        request_kwargs["config"] = config
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        temperature=temperature,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "system_prompt": _normalize_system_prompt(system_prompt),
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
    _finalize_bbox_diagnostics(session_dir)
    if text:
        yield text


def generate_structured_json_from_image(
    image_path: Path,
    prompt: str,
    schema: dict[str, Any],
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_model = resolve_supported_gemini_model_name(model)
    if is_vertex_gemini_model_requested(resolved_model):
        text = generate_content_from_image(
            image_path=image_path,
            prompt=prompt,
            model=resolved_model,
            mime_type=mime_type,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
        )
        if not text.strip():
            raise ValueError("Gemini returned an empty response.")
        return text

    session_dir = _create_gemini_log_session(
        operation="generate_structured_json",
        model=resolved_model,
        image_path=image_path,
        prompt=prompt,
        mime_type=mime_type,
        system_prompt=system_prompt,
        temperature=temperature,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        extra_request={"schema": _serialize_gemini_log_value(schema)},
    )

    image_bytes = image_path.read_bytes()
    inferred_mime_type = _infer_mime_type(image_path, mime_type)

    key = resolve_gemini_api_key_for_model(resolved_model, explicit_api_key=api_key)
    client = _create_genai_client_for_model(resolved_model, key)
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
        system_prompt=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    request_summary = _thinking_request_summary(
        resolved_model,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        temperature=temperature,
        config=config,
    )
    _update_gemini_log_request(
        session_dir,
        {
            "request_summary": request_summary,
            "system_prompt": _normalize_system_prompt(system_prompt),
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
    _finalize_bbox_diagnostics(session_dir)
    if not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    thinking_level: Optional[str] = None,
) -> PageExtraction:
    resolved_model = resolve_supported_gemini_model_name(model)
    session_dir = _create_gemini_log_session(
        operation="generate_content",
        model=resolved_model,
        image_path=image_path,
        prompt=prompt,
        mime_type=mime_type,
        system_prompt=system_prompt,
        temperature=temperature,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        few_shot_examples=None,
    )
    # Temporarily avoid Gemini-side schema enforcement to prevent INVALID_ARGUMENT
    # errors from response_json_schema. We still enforce strict local validation.
    raw_text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=resolved_model,
        mime_type=mime_type,
        api_key=api_key,
        system_prompt=system_prompt,
        temperature=temperature,
        enable_thinking=enable_thinking,
        thinking_level=thinking_level,
        response_mime_type=DEFAULT_GEMINI_GT_RESPONSE_MIME_TYPE,
        media_resolution=DEFAULT_GEMINI_GT_MEDIA_RESOLUTION,
        session_dir=session_dir,
    )
    return parse_page_extraction_text(raw_text, image_path=image_path, session_dir=session_dir)


def parse_page_extraction_text(
    raw_text: str,
    *,
    image_path: Optional[Path] = None,
    session_dir: Optional[Path] = None,
    streamed_fact_count: int = 0,
) -> PageExtraction:
    if not str(raw_text).strip():
        if session_dir is not None:
            summary = _build_page_extraction_issue_summary(
                session_dir=session_dir,
                meta_payload={},
                bbox_mode=_BBOX_MODE_PIXEL_AS_IS,
                bbox_scores={
                    _BBOX_MODE_PIXEL_AS_IS: 0.0,
                    _BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
                },
                bbox_fact_modes=[],
                bbox_resolution_policy="page_locked",
                bbox_diagnostics_raw=None,
                bbox_diagnostics_resolved=None,
                streamed_fact_count=streamed_fact_count,
                final_raw_fact_count=0,
                kept_fact_count=0,
                dropped_fact_count=0,
                issue_entries=[{"code": "final_parse_error", "message": "Gemini returned an empty response."}],
                status="error",
            )
            _write_issue_summary(session_dir, summary)
        raise ValueError("Gemini returned an empty response.")
    try:
        parsed = _parse_llm_json(raw_text)
    except Exception as exc:
        if session_dir is not None:
            summary = _build_page_extraction_issue_summary(
                session_dir=session_dir,
                meta_payload={},
                bbox_mode=_BBOX_MODE_PIXEL_AS_IS,
                bbox_scores={
                    _BBOX_MODE_PIXEL_AS_IS: 0.0,
                    _BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
                },
                bbox_fact_modes=[],
                bbox_resolution_policy="page_locked",
                bbox_diagnostics_raw=_read_gemini_log_request(session_dir).get("bbox_diagnostics_raw"),
                bbox_diagnostics_resolved=None,
                streamed_fact_count=streamed_fact_count,
                final_raw_fact_count=0,
                kept_fact_count=0,
                dropped_fact_count=0,
                issue_entries=_normalize_validation_errors(exc, fallback_code="final_parse_error"),
                status="error",
            )
            _write_issue_summary(session_dir, summary)
        raise
    extraction, _summary = _finalize_page_extraction_payload(
        parsed,
        raw_text=raw_text,
        image_path=image_path,
        session_dir=session_dir,
        streamed_fact_count=streamed_fact_count,
    )
    return extraction


def _validate_patch_fact_updates(
    updates_payload: Any,
    *,
    allowed_fact_fields: set[str],
) -> dict[str, Any]:
    if not isinstance(updates_payload, dict):
        raise ValueError("Each patch updates object must be a JSON object.")

    normalized_updates = {
        key: value
        for key, value in dict(updates_payload).items()
        if not (isinstance(value, str) and not value.strip())
    }
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
