#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from first_smoke_test import (
    DEFAULT_MAX_PIXELS,
    _default_pdf_path,
    _default_system_and_instruction,
    _extract_delta_text,
    _image_to_data_uri,
    _normalize_base_url,
    _pdf_page_count,
    _prepare_image,
    _render_pdf_page,
    _resolve_model,
)


def _default_output_dir(pdf_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = pdf_path.stem.replace("/", "_")
    return SCRIPT_DIR / "outputs" / f"{safe_name}_{stamp}"


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full-PDF OpenAI-compatible batch inference page by page."
    )
    parser.add_argument("base_url", help="OpenAI-compatible base URL or chat completions URL")
    parser.add_argument("--pdf", help="PDF path. Defaults to the first file in vllm_api_tests/pdfs")
    parser.add_argument("--model", help="Model id. If omitted, the script uses /v1/models")
    parser.add_argument("--system", help="System prompt override")
    parser.add_argument("--instruction", help="Instruction override")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF pages")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Resize image to this maximum pixel count")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--page-start", type=int, default=1, help="First page number to process")
    parser.add_argument("--page-end", type=int, default=None, help="Last page number to process")
    parser.add_argument("--output-dir", help="Directory for manifest and results")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: openai") from exc

    pdf_path = Path(args.pdf).expanduser() if args.pdf else _default_pdf_path()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    total_pages = _pdf_page_count(pdf_path)
    if total_pages <= 0:
        raise SystemExit(f"PDF has no pages: {pdf_path}")

    page_start = max(int(args.page_start), 1)
    page_end = total_pages if args.page_end is None else min(int(args.page_end), total_pages)
    if page_start > page_end:
        raise SystemExit(
            f"Invalid page range: start={page_start} end={page_end} total_pages={total_pages}"
        )

    default_system, default_instruction = _default_system_and_instruction()
    system_prompt = str(args.system or default_system).strip()
    instruction = str(args.instruction or default_instruction).strip()

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(pdf_path)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"

    base_url = _normalize_base_url(args.base_url)
    client = OpenAI(
        base_url=base_url,
        api_key="unused",
        timeout=300.0,
    )
    model = _resolve_model(client, args.model)

    manifest = {
        "base_url": base_url,
        "model": model,
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "page_start": page_start,
        "page_end": page_end,
        "dpi": int(args.dpi),
        "max_pixels": int(args.max_pixels),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "enable_thinking": False,
        "output_dir": str(output_dir),
        "prompt_source": "artifacts/hf_finetree_2_9/train.jsonl",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with results_path.open("w", encoding="utf-8") as handle:
        for page_num in range(page_start, page_end + 1):
            print(f"[page {page_num}/{total_pages}] rendering", file=sys.stderr)
            page_image = _render_pdf_page(pdf_path, page_num=page_num, dpi=args.dpi)
            prepared_image, original_size, prepared_size = _prepare_image(
                page_image,
                max_pixels=args.max_pixels,
            )
            image_data_uri = _image_to_data_uri(prepared_image)

            print(f"[page {page_num}/{total_pages}] streaming", file=sys.stderr)
            print(f"\n===== page {page_num}/{total_pages} =====", flush=True)
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    },
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            chunks: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                text = _extract_delta_text(getattr(delta, "content", None))
                if text:
                    chunks.append(text)
                    print(text, end="", flush=True)
            assistant_text = "".join(chunks).strip()
            parsed = _try_parse_json(assistant_text)
            if chunks:
                print(flush=True)

            row = {
                "page": page_num,
                "image_name": f"page_{page_num:04d}.png",
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "prepared_size": {"width": prepared_size[0], "height": prepared_size[1]},
                "assistant_text": assistant_text,
                "parsed_json": parsed,
                "json_valid": parsed is not None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[page {page_num}/{total_pages}] done json_valid={parsed is not None}",
                file=sys.stderr,
            )

    print(f"manifest={manifest_path}")
    print(f"results={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
