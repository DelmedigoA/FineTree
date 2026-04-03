#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
import threading


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from first_smoke_test import (
    DEFAULT_MAX_PIXELS,
    _default_system_and_instruction,
    _extract_delta_text,
    _image_to_data_uri,
    _normalize_base_url,
    _pdf_page_count,
    _prepare_image,
    _render_pdf_page,
    _resolve_model,
)


_STDOUT_LOCK = threading.Lock()
_STDERR_LOCK = threading.Lock()


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _default_pdfs() -> list[Path]:
    pdf_dir = SCRIPT_DIR / "pdfs"
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def _safe_name(value: str) -> str:
    text = str(value).strip().replace("/", "_").replace("\n", " ")
    return "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in text).strip() or "pdf"


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "outputs_parallel" / stamp


def _emit_stdout(text: str, *, end: str = "\n", flush: bool = True) -> None:
    with _STDOUT_LOCK:
        print(text, end=end, flush=flush)


def _emit_stderr(text: str) -> None:
    with _STDERR_LOCK:
        print(text, file=sys.stderr, flush=True)


def _run_one_pdf(
    *,
    base_url: str,
    pdf_path: Path,
    model_name: str | None,
    system_prompt: str,
    instruction: str,
    dpi: int,
    max_pixels: int,
    max_tokens: int,
    temperature: float,
    output_root: Path,
) -> dict[str, str]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: openai") from exc

    pdf_path = pdf_path.resolve()
    label = _safe_name(pdf_path.stem)
    pdf_output_dir = output_root / label
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pdf_output_dir / "manifest.json"
    results_path = pdf_output_dir / "results.jsonl"
    stream_log_path = pdf_output_dir / "stream.log"

    total_pages = _pdf_page_count(pdf_path)
    if total_pages <= 0:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    client = OpenAI(
        base_url=base_url,
        api_key="unused",
        timeout=300.0,
    )
    resolved_model = _resolve_model(client, model_name)

    manifest = {
        "base_url": base_url,
        "model": resolved_model,
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "dpi": int(dpi),
        "max_pixels": int(max_pixels),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "enable_thinking": False,
        "output_dir": str(pdf_output_dir),
        "results": str(results_path),
        "stream_log": str(stream_log_path),
        "prompt_source": "artifacts/hf_finetree_2_9/train.jsonl",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _emit_stderr(f"[{label}] start pages=1..{total_pages}")
    with results_path.open("w", encoding="utf-8") as results_handle, stream_log_path.open("w", encoding="utf-8") as stream_handle:
        for page_num in range(1, total_pages + 1):
            _emit_stderr(f"[{label}] page {page_num}/{total_pages} rendering")
            page_image = _render_pdf_page(pdf_path, page_num=page_num, dpi=dpi)
            prepared_image, original_size, prepared_size = _prepare_image(
                page_image,
                max_pixels=max_pixels,
            )
            image_data_uri = _image_to_data_uri(prepared_image)

            _emit_stdout(f"\n===== {label} page {page_num}/{total_pages} =====")
            _emit_stderr(f"[{label}] page {page_num}/{total_pages} streaming")

            stream = client.chat.completions.create(
                model=resolved_model,
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
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            chunks: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                text = _extract_delta_text(getattr(delta, "content", None))
                if not text:
                    continue
                chunks.append(text)
                stream_handle.write(text)
                stream_handle.flush()
                _emit_stdout(f"[{label} p{page_num}] {text}", end="")

            assistant_text = "".join(chunks).strip()
            parsed = _try_parse_json(assistant_text)
            if chunks:
                _emit_stdout("")
                stream_handle.write("\n")
                stream_handle.flush()

            row = {
                "page": page_num,
                "image_name": f"page_{page_num:04d}.png",
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "prepared_size": {"width": prepared_size[0], "height": prepared_size[1]},
                "assistant_text": assistant_text,
                "parsed_json": parsed,
                "json_valid": parsed is not None,
            }
            results_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            results_handle.flush()
            _emit_stderr(f"[{label}] page {page_num}/{total_pages} done json_valid={parsed is not None}")

    _emit_stderr(f"[{label}] completed results={results_path}")
    return {
        "pdf": str(pdf_path),
        "label": label,
        "manifest": str(manifest_path),
        "results": str(results_path),
        "stream_log": str(stream_log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full-PDF OpenAI-compatible batch inference on multiple PDFs in parallel."
    )
    parser.add_argument("base_url", help="OpenAI-compatible base URL or chat completions URL")
    parser.add_argument("--pdf", action="append", help="PDF path. Repeat to select specific PDFs. Defaults to all PDFs in vllm_api_tests/pdfs")
    parser.add_argument("--model", help="Model id. If omitted, the script uses /v1/models")
    parser.add_argument("--system", help="System prompt override")
    parser.add_argument("--instruction", help="Instruction override")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF pages")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Resize image to this maximum pixel count")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count. Defaults to number of selected PDFs.")
    parser.add_argument("--output-root", help="Root directory for all run outputs")
    args = parser.parse_args()

    selected_pdfs = [Path(value).expanduser() for value in args.pdf] if args.pdf else _default_pdfs()
    if not selected_pdfs:
        raise SystemExit(f"No PDFs found in {SCRIPT_DIR / 'pdfs'}")
    for pdf_path in selected_pdfs:
        if not pdf_path.is_file():
            raise SystemExit(f"PDF not found: {pdf_path}")

    output_root = Path(args.output_root).expanduser() if args.output_root else _default_output_root()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    default_system, default_instruction = _default_system_and_instruction()
    system_prompt = str(args.system or default_system).strip()
    instruction = str(args.instruction or default_instruction).strip()
    base_url = _normalize_base_url(args.base_url)
    workers = max(1, int(args.workers or len(selected_pdfs)))

    summary_path = output_root / "summary.json"
    summary = {
        "base_url": base_url,
        "model": args.model,
        "pdfs": [str(path.resolve()) for path in selected_pdfs],
        "workers": workers,
        "enable_thinking": False,
        "output_root": str(output_root),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _run_one_pdf,
                base_url=base_url,
                pdf_path=pdf_path,
                model_name=args.model,
                system_prompt=system_prompt,
                instruction=instruction,
                dpi=args.dpi,
                max_pixels=args.max_pixels,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_root=output_root,
            ): pdf_path
            for pdf_path in selected_pdfs
        }
        for future in as_completed(future_map):
            pdf_path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                raise SystemExit(f"Parallel inference failed for {pdf_path}: {exc}") from exc
            results.append(result)

    final_summary = dict(summary)
    final_summary["results"] = results
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
