#!/usr/bin/env python3
"""CLI wrapper around finetree_annotator.finetune.token_stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetree_annotator.finetune.token_stats import (
    _extract_sample_fields,
    _image_token_len,
    _load_processor,
    _resolve_image_path,
    _summary,
    _text_from_content,
    _token_len_chat,
    _token_len_plain,
)


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


def _default_inputs(root: Path) -> list[Path]:
    out: list[Path] = []
    train = root / "data" / "finetune" / "train.jsonl"
    val = root / "data" / "finetune" / "val.jsonl"
    if train.is_file():
        out.append(train)
    if val.is_file():
        out.append(val)
    return out


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


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer  # type: ignore

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
