#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


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
    fallback_system_prompt, fallback_system_prompt_path = _load_fallback_system_prompt(root, args.system_prompt_file)

    response_only: list[int] = []
    prompt_only_no_system: list[int] = []
    prompt_only_with_system: list[int] = []
    with_prompt_no_system: list[int] = []
    with_prompt_and_system: list[int] = []
    processed = 0
    fallback_system_used = 0

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
            with_prompt_and_system.append(
                _token_len_chat(tokenizer, system=effective_system, instruction=instruction, response=response)
            )
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
        "response_only": _summary(response_only),
        "prompt_only_no_system": _summary(prompt_only_no_system),
        "prompt_only_with_system": _summary(prompt_only_with_system),
        "with_prompt_no_system": _summary(with_prompt_no_system),
        "with_prompt_and_system": _summary(with_prompt_and_system),
    }

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
        for key in (
            "response_only",
            "prompt_only_no_system",
            "prompt_only_with_system",
            "with_prompt_no_system",
            "with_prompt_and_system",
        ):
            stats = report[key]
            print(
                f"{key}: count={stats['count']} min={stats['min']} max={stats['max']} "
                f"mean={stats['mean']} median={stats['median']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
