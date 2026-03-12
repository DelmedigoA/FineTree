#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finetree_annotator.model_prompt_serialization import MODEL_PROMPT_MODE, build_single_page_payload


def _latest_log_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    return candidates[0] if candidates else None


def _sample_prompt_payload(image_name: str) -> dict[str, Any]:
    return build_single_page_payload(
        page_name=image_name,
        page_meta={
            "entity_name": None,
            "page_num": None,
            "page_type": "other",
            "statement_type": None,
            "title": None,
        },
        facts=[],
        mode=MODEL_PROMPT_MODE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Gemini/Qwen request payloads and optionally run a live smoke call.")
    parser.add_argument("--provider", choices=("gemini", "qwen"), required=True)
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--prompt", default="Return empty page JSON.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None, help="Qwen config path when --provider=qwen.")
    parser.add_argument("--enable-thinking", action="store_true", help="Request provider thinking mode when supported.")
    parser.add_argument(
        "--thinking-level",
        default=None,
        help="Gemini thinking level override: minimal|low|medium|high.",
    )
    parser.add_argument("--live", action="store_true", help="Actually call the provider if credentials/config are available.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    sample_payload = _sample_prompt_payload(image_path.name)
    print("SAMPLE_PROMPT_PAYLOAD:")
    print(json.dumps(sample_payload, ensure_ascii=False, indent=2))

    if not args.live:
        return 0

    if args.provider == "gemini":
        from finetree_annotator import gemini_vlm

        text = gemini_vlm.generate_content_from_image(
            image_path=image_path,
            prompt=args.prompt,
            model=args.model or gemini_vlm.DEFAULT_GEMINI_MODEL,
            enable_thinking=args.enable_thinking,
            thinking_level=args.thinking_level,
        )
        logs_root = Path.cwd() / "gemini_logs"
    else:
        from finetree_annotator import qwen_vlm

        text = qwen_vlm.generate_content_from_image(
            image_path=image_path,
            prompt=args.prompt,
            model=args.model,
            config_path=args.config,
            enable_thinking=args.enable_thinking,
            max_new_tokens=256,
        )
        logs_root = Path.cwd() / "qwen_logs"

    print("LIVE_RESPONSE_TEXT:")
    print(text)

    latest = _latest_log_dir(logs_root)
    if latest is None:
        return 0
    request_path = latest / "request.json"
    if request_path.is_file():
        print("LATEST_REQUEST_SUMMARY:")
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        print(json.dumps(request_payload.get("request_summary", request_payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
