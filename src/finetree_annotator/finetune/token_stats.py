"""Token statistics using Qwen tokenizer — callable library."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_sample_fields(sample: dict[str, Any]) -> tuple[str, str, str]:
    system = ""
    instruction = ""
    response = ""

    if isinstance(sample.get("system"), str):
        system = str(sample["system"]).strip()
    elif isinstance(sample.get("system_prompt"), str):
        system = str(sample["system_prompt"]).strip()

    if isinstance(sample.get("instruction"), str):
        instruction = str(sample["instruction"]).strip()
    elif isinstance(sample.get("query"), str):
        instruction = str(sample["query"]).strip()

    if isinstance(sample.get("text"), str):
        response = str(sample["text"]).strip()
    elif isinstance(sample.get("response"), str):
        response = str(sample["response"]).strip()

    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        user_texts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            text = _text_from_content(message.get("content"))
            if role == "system" and text and not system:
                system = text
            elif role == "user" and text:
                user_texts.append(text)
            elif role == "assistant" and text and not response:
                response = text
        if user_texts:
            instruction = "\n".join(user_texts).strip()

    return system, instruction, response


def _token_len_plain(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    return len(ids) if isinstance(ids, list) else 0


def _token_len_chat(tokenizer: Any, *, system: str, instruction: str, response: str) -> int:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if instruction:
        messages.append({"role": "user", "content": instruction})
    if response:
        messages.append({"role": "assistant", "content": response})
    if not messages:
        return 0

    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(ids, list):
            return len(ids)
        if isinstance(ids, dict):
            input_ids = ids.get("input_ids", [])
            return len(input_ids) if isinstance(input_ids, list) else 0
    except Exception:
        pass

    # Fallback if chat template is unavailable.
    joined = "\n".join([m["content"] for m in messages if m.get("content")])
    return _token_len_plain(tokenizer, joined)


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(float(statistics.mean(values)), 3),
        "median": round(float(statistics.median(values)), 3),
    }


def _load_processor(tokenizer_name_or_path: str) -> Any | None:
    try:
        from transformers import AutoProcessor  # type: ignore
        return AutoProcessor.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, min_pixels=1)
    except Exception:
        return None


def _resolve_image_path(sample: dict[str, Any], *, input_path: Path) -> Path | None:
    image_value = sample.get("image")
    if not isinstance(image_value, str) or not image_value.strip():
        return None
    image_path = Path(image_value).expanduser()
    if image_path.is_absolute():
        return image_path if image_path.is_file() else None
    candidate = (input_path.parent / image_path).resolve()
    if candidate.is_file():
        return candidate
    fallback = image_path.resolve()
    return fallback if fallback.is_file() else None


def _image_token_len(processor: Any, image_path: Path) -> int:
    from PIL import Image  # type: ignore

    with Image.open(image_path) as image:
        prepared = image.convert("RGB")
        batch = processor(text=[""], images=[prepared], return_tensors="pt", padding=True)
    getter = getattr(batch, "get", None)
    if callable(getter):
        grid = getter("image_grid_thw")
        if grid is not None and hasattr(grid, "tolist"):
            values = grid.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list) and len(values[0]) == 3:
                t, h, w = (int(values[0][0]), int(values[0][1]), int(values[0][2]))
                return t * h * w
        pixel_values = getter("pixel_values")
        if pixel_values is not None and hasattr(pixel_values, "shape") and len(pixel_values.shape) >= 1:
            return int(pixel_values.shape[0])
    return 0


def compute_jsonl_token_stats(
    jsonl_paths: list[Path],
    *,
    tokenizer_name: str = "Qwen/Qwen3.5-27B",
    limit: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute token statistics over one or more JSONL files.

    Returns a summary dict with per-page text/image/full-sequence token stats.

    Parameters
    ----------
    jsonl_paths:
        Paths to JSONL files to process.
    tokenizer_name:
        HuggingFace tokenizer/model repo id.
    limit:
        Maximum number of samples to process (0 = all).
    progress_callback:
        Optional callable(current, total) called after each sample.
        ``total`` is 0 when the total count is not yet known.

    Returns
    -------
    {
      "tokenizer": str,
      "total_samples": int,
      "per_page_text_tokens": {"min", "max", "mean", "median", "count"},
      "per_page_image_tokens": {"min", "max", "mean", "median", "count"},
      "per_page_full_sequence_tokens": {"min", "max", "mean", "median", "count"},
      "total_text_tokens": int,
    }
    """
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    processor = _load_processor(tokenizer_name)

    # Pre-count total samples for progress reporting when limit is not set.
    total_samples = 0
    if progress_callback is not None:
        for path in jsonl_paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    total_samples += 1
        if limit > 0:
            total_samples = min(total_samples, limit)

    text_tokens_list: list[int] = []
    image_tokens_list: list[int] = []
    full_sequence_list: list[int] = []
    processed = 0

    for path in jsonl_paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                continue
            system, instruction, response = _extract_sample_fields(sample)
            if not response:
                continue

            text_tokens = _token_len_chat(
                tokenizer,
                system=system,
                instruction=instruction,
                response=response,
            )
            text_tokens_list.append(text_tokens)

            image_tokens = 0
            if processor is not None:
                image_path = _resolve_image_path(sample, input_path=path)
                if image_path is not None:
                    image_tokens = _image_token_len(processor, image_path)
            image_tokens_list.append(image_tokens)
            full_sequence_list.append(text_tokens + image_tokens)

            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total_samples)
            if limit > 0 and processed >= limit:
                break
        if limit > 0 and processed >= limit:
            break

    return {
        "tokenizer": tokenizer_name,
        "total_samples": processed,
        "per_page_text_tokens": _summary(text_tokens_list),
        "per_page_image_tokens": _summary(image_tokens_list),
        "per_page_full_sequence_tokens": _summary(full_sequence_list),
        "total_text_tokens": sum(text_tokens_list),
    }


__all__ = [
    "compute_jsonl_token_stats",
    "_extract_sample_fields",
    "_token_len_plain",
    "_token_len_chat",
    "_summary",
    "_load_processor",
    "_resolve_image_path",
    "_image_token_len",
    "_text_from_content",
]
