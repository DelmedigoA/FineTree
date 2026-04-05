#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token stats on samples with Qwen/Qwen3.5-27B tokenizer."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Input JSONL files. Defaults to data/finetune/train.jsonl and val.jsonl when present.",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B", help="HF tokenizer repo id.")
    parser.add_argument(
        "--system-prompt-file",
        default="system_prompt.txt",
        help=(
            "Fallback system prompt file for with_prompt_and_system mode when samples have no system field. "
            "Set empty string to disable."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples to process (0 = all).")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    return parser.parse_args(argv)


def _default_inputs(root: Path) -> list[Path]:
    out: list[Path] = []
    train = root / "data" / "finetune" / "train.jsonl"
    val = root / "data" / "finetune" / "val.jsonl"
    if train.is_file():
        out.append(train)
    if val.is_file():
        out.append(val)
    return out


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


def _load_fallback_system_prompt(root: Path, arg_value: str) -> tuple[str, str | None]:
    candidate = str(arg_value or "").strip()
    if not candidate:
        return "", None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        return "", str(path)
    text = path.read_text(encoding="utf-8").strip()
    return text, str(path)


def _load_processor(tokenizer_name_or_path: str) -> Any | None:
    try:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(".").resolve()
    inputs = [Path(p).expanduser().resolve() for p in (args.inputs or [])]
    if not inputs:
        inputs = _default_inputs(root)
    inputs = [p for p in inputs if p.is_file()]
    if not inputs:
        raise FileNotFoundError("No input JSONL files found.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    processor = _load_processor(args.tokenizer)
    fallback_system_prompt, fallback_system_prompt_path = _load_fallback_system_prompt(root, args.system_prompt_file)

    response_only: list[int] = []
    prompt_only_no_system: list[int] = []
    prompt_only_with_system: list[int] = []
    with_prompt_no_system: list[int] = []
    with_prompt_and_system: list[int] = []
    image_only: list[int] = []
    with_prompt_and_system_and_image: list[int] = []
    processed = 0
    fallback_system_used = 0
    image_samples_processed = 0

    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                continue
            system, instruction, response = _extract_sample_fields(sample)
            if not response:
                continue
            processed += 1
            response_only.append(_token_len_plain(tokenizer, response))
            prompt_only_no_system.append(
                _token_len_chat(tokenizer, system="", instruction=instruction, response="")
            )
            with_prompt_no_system.append(
                _token_len_chat(tokenizer, system="", instruction=instruction, response=response)
            )
            effective_system = system
            if not effective_system and fallback_system_prompt:
                effective_system = fallback_system_prompt
                fallback_system_used += 1
            prompt_only_with_system.append(
                _token_len_chat(tokenizer, system=effective_system, instruction=instruction, response="")
            )
            total_text_tokens = _token_len_chat(
                tokenizer,
                system=effective_system,
                instruction=instruction,
                response=response,
            )
            with_prompt_and_system.append(total_text_tokens)
            if processor is not None:
                image_path = _resolve_image_path(sample, input_path=path)
                if image_path is not None:
                    image_tokens = _image_token_len(processor, image_path)
                    image_only.append(image_tokens)
                    with_prompt_and_system_and_image.append(total_text_tokens + image_tokens)
                    image_samples_processed += 1
            if args.limit > 0 and processed >= args.limit:
                break
        if args.limit > 0 and processed >= args.limit:
            break

    report = {
        "tokenizer": args.tokenizer,
        "inputs": [str(p) for p in inputs],
        "samples_processed": processed,
        "fallback_system_prompt_path": fallback_system_prompt_path,
        "fallback_system_prompt_used_for_samples": fallback_system_used,
        "processor": args.tokenizer if processor is not None else None,
        "image_samples_processed": image_samples_processed,
        "response_only": _summary(response_only),
        "prompt_only_no_system": _summary(prompt_only_no_system),
        "prompt_only_with_system": _summary(prompt_only_with_system),
        "with_prompt_no_system": _summary(with_prompt_no_system),
        "with_prompt_and_system": _summary(with_prompt_and_system),
    }
    if image_only:
        report["image_only"] = _summary(image_only)
        report["with_prompt_and_system_and_image"] = _summary(with_prompt_and_system_and_image)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"TOKEN_STATS tokenizer={report['tokenizer']}")
        print(f"inputs={report['inputs']}")
        print(f"samples_processed={report['samples_processed']}")
        print(
            "fallback_system_prompt="
            f"{report['fallback_system_prompt_path']} "
            f"used_for_samples={report['fallback_system_prompt_used_for_samples']}"
        )
        print(f"processor={report['processor']} image_samples_processed={report['image_samples_processed']}")
        for key in (
            "response_only",
            "prompt_only_no_system",
            "prompt_only_with_system",
            "with_prompt_no_system",
            "with_prompt_and_system",
            "image_only",
            "with_prompt_and_system_and_image",
        ):
            if key not in report:
                continue
            stats = report[key]
            print(
                f"{key}: count={stats['count']} min={stats['min']} max={stats['max']} "
                f"mean={stats['mean']} median={stats['median']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
