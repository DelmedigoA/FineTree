from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import mimetypes
import os
import shutil
import subprocess
import sys
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - import presence depends on runtime environment
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

from .schemas import PageExtraction, PageType


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
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
_VALID_PAGE_TYPES = {t.value for t in PageType}
_VALID_CURRENCIES = {"ILS", "USD", "EUR", "GBP"}
_VALID_SCALES = {1, 1000, 1000000}
_VALID_VALUE_TYPES = {"amount", "%"}


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
    fixed = fixed.replace("“", "\"").replace("”", "\"").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _parse_llm_json(text: str) -> Any:
    candidates: list[str] = []
    raw = text.strip()
    if raw:
        candidates.append(raw)

    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend([m.strip() for m in fence_matches if m.strip()])

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

    if isinstance(payload.get("pages"), list) and payload["pages"]:
        first_page = next((p for p in payload["pages"] if isinstance(p, dict)), None)
        if first_page is not None:
            payload = first_page

    meta_in = payload.get("meta")
    if not isinstance(meta_in, dict):
        meta_in = {}
    page_type = str(meta_in.get("type") or meta_in.get("page_type") or "other").strip() or "other"
    if page_type not in _VALID_PAGE_TYPES:
        page_type = "other"
    meta_out = {
        "entity_name": _to_optional_str(meta_in.get("entity_name") or meta_in.get("entity")),
        "page_num": _to_optional_str(meta_in.get("page_num")),
        "type": page_type,
        "title": _to_optional_str(meta_in.get("title")),
    }

    facts_in = payload.get("facts")
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

        raw_path = raw_fact.get("path")
        if isinstance(raw_path, str):
            path = [raw_path.strip()] if raw_path.strip() else []
        elif isinstance(raw_path, list):
            path = [str(p).strip() for p in raw_path if str(p).strip()]
        else:
            path = []

        currency = _to_optional_str(raw_fact.get("currency"))
        if currency is not None:
            currency = currency.upper()
            if currency not in _VALID_CURRENCIES:
                currency = None

        scale_value = raw_fact.get("scale")
        try:
            scale = int(scale_value) if scale_value is not None and str(scale_value).strip() else None
        except Exception:
            scale = None
        if scale not in _VALID_SCALES:
            scale = None

        value_type_raw = _to_optional_str(raw_fact.get("value_type"))
        if value_type_raw:
            low = value_type_raw.lower()
            if low in {"percent", "percentage", "%"}:
                value_type = "%"
            elif low in {"amount", "regular"}:
                value_type = "amount"
            else:
                value_type = value_type_raw
        else:
            value_type = None
        if value_type not in _VALID_VALUE_TYPES:
            value_type = None

        facts_out.append(
            {
                "bbox": bbox,
                "value": str(value),
                "note": _to_optional_str(raw_fact.get("note", raw_fact.get("footnote"))),
                "is_beur": _to_optional_bool(raw_fact.get("is_beur", raw_fact.get("beur"))),
                "beur_num": _to_optional_str(raw_fact.get("beur_num", raw_fact.get("beur_number"))),
                "refference": _to_optional_str(
                    raw_fact.get("refference", raw_fact.get("reference", raw_fact.get("ref")))
                )
                or "",
                "date": _to_optional_str(raw_fact.get("date")),
                "path": path,
                "currency": currency,
                "scale": scale,
                "value_type": value_type,
            }
        )

    return {"meta": meta_out, "facts": facts_out}


class StreamingPageExtractionParser:
    def __init__(self) -> None:
        self.buffer = ""
        self._meta_emitted = False
        self._facts_array_start: Optional[int] = None
        self._facts_scan_pos: Optional[int] = None
        self._facts_done = False

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
                self._meta_emitted = True
                return normalized["meta"]
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
                if normalized["facts"]:
                    out.append(normalized["facts"][0])
            except Exception:
                pass
            pos += len(obj)

        self._facts_scan_pos = pos
        return out

    def finalize(self) -> PageExtraction:
        parsed = _parse_llm_json(self.buffer)
        normalized = _normalize_page_extraction_payload(parsed)
        return PageExtraction.model_validate(normalized)


def generate_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    inferred_mime_type = _infer_mime_type(image_path, mime_type)

    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=inferred_mime_type,
            ),
            prompt,
        ],
    )
    return response.text or ""


def stream_content_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Iterator[str]:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    inferred_mime_type = _infer_mime_type(image_path, mime_type)
    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()

    stream_fn = getattr(client.models, "generate_content_stream", None)
    if callable(stream_fn):
        stream = stream_fn(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=inferred_mime_type,
                ),
                prompt,
            ],
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
        return

    text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        api_key=api_key,
    )
    if text:
        yield text


def generate_structured_json_from_image(
    image_path: Path,
    prompt: str,
    schema: dict[str, Any],
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    _require_google_genai()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    inferred_mime_type = _infer_mime_type(image_path, mime_type)

    key = _resolve_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=inferred_mime_type,
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    text = response.text or ""
    if not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text


def generate_page_extraction_from_image(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> PageExtraction:
    # Temporarily avoid Gemini-side schema enforcement to prevent INVALID_ARGUMENT
    # errors from response_json_schema. We still enforce strict local validation.
    raw_text = generate_content_from_image(
        image_path=image_path,
        prompt=prompt,
        model=model,
        mime_type=mime_type,
        api_key=api_key,
    )
    return parse_page_extraction_text(raw_text)


def parse_page_extraction_text(raw_text: str) -> PageExtraction:
    if not str(raw_text).strip():
        raise ValueError("Gemini returned an empty response.")
    parsed = _parse_llm_json(raw_text)
    normalized = _normalize_page_extraction_payload(parsed)
    return PageExtraction.model_validate(normalized)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini image+text generation on a local image.")
    parser.add_argument("image", help="Path to image file.")
    parser.add_argument("prompt", help="Prompt to send with the image.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument("--mime-type", default=None, help="Override MIME type (default: infer from extension).")
    parser.add_argument("--api-key", default=None, help="Google API key (default: env GOOGLE_API_KEY/GEMINI_API_KEY).")
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
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
