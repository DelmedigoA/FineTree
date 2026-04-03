#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetree_annotator.vision_resize import prepared_dimensions_for_max_pixels


DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise financial statement extraction system. "
    "Return only valid JSON that matches the required schema."
)
DEFAULT_INSTRUCTION = """You are extracting financial-statement annotations from a single page image.

Return ONLY valid JSON.
Do NOT return markdown, code fences, comments, prose, or extra keys.

Return the exact page-level object shown below. Include only the selected page-meta and fact keys.

Selected page meta keys:
- entity_name, page_num, page_type, statement_type, title

Selected fact keys:
- value, fact_num, comment_ref, note_flag, note_name, note_num, note_ref, period_type, period_start, period_end, path, path_source, currency, scale, value_type, value_context

Exact schema:
{
    "meta": {
      "entity_name": "<string or null>",
  "page_num": "<string or null>",
  "page_type": "title|contents|declaration|statements|other",
  "statement_type": "balance_sheet|income_statement|cash_flow_statement|statement_of_changes_in_equity|notes_to_financial_statements|board_of_directors_report|auditors_report|statement_of_activities|other_declaration|null",
  "title": "<string or null>"
    },
    "facts": [
      {
        "value": "<string>",
"fact_num": <integer >= 1>,
"comment_ref": "<string or null>",
"note_flag": <true|false>,
"note_name": "<string or null>",
"note_num": "<string or null>",
"note_ref": "<string or null>",
"period_type": "instant|duration|expected|null",
"period_start": "<YYYY-MM-DD|null>",
"period_end": "<YYYY-MM-DD|null>",
"path": ["<string>", "..."],
"path_source": "observed|inferred|null",
"currency": "ILS|USD|EUR|GBP|null",
"scale": 1|1000|1000000|null,
"value_type": "amount|percent|ratio|count|points|null",
"value_context": "textual|tabular|mixed|null"
      }
    ]
  }

Rules:
1. Return only a single page-level object with `meta` and `facts`.
2. Extract only visible numeric/table facts. Do not emit standalone labels or headings as facts.
3. Preserve value text exactly as printed, including `%`, commas, parentheses, and dash placeholders.
4. Use JSON `null` for missing optional values. Do not emit the string `"null"`.
5. Keep UTF-8 Hebrew directly; do not escape it to unicode sequences.
6. Order facts top-to-bottom; within each row use right-to-left for Hebrew pages and left-to-right for English pages.
7. `fact_num` must be contiguous integers starting at 1 and must match the emitted fact order.
8. Keep `path` as a list of visible hierarchy labels; use `[]` when unknown.
9. Classify page type and statement type from visible page context only.

Output formatting:
1. Return the final JSON as a single compact line.
2. Do not pretty-print, indent, add line breaks, or add extra spaces between JSON tokens.
3. Do not add any prefix, suffix, explanation, or surrounding text."""


def _normalize_base_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _default_pdf_path() -> Path:
    pdf_dir = Path(__file__).resolve().parent / "pdfs"
    pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    return pdfs[0]


def _default_system_and_instruction() -> tuple[str, str]:
    preferred_export = ROOT / "artifacts" / "hf_finetree_2_9" / "train.jsonl"
    if preferred_export.is_file():
        try:
            first_line = preferred_export.read_text(encoding="utf-8").splitlines()[0]
            row = json.loads(first_line)
            system = str(row.get("system") or "").strip()
            instruction = str(row.get("instruction") or "").strip()
            if system and instruction:
                return system, instruction
        except Exception:
            pass

    system_path = ROOT / "prompts" / "system_prompt.txt"
    if system_path.is_file():
        system = system_path.read_text(encoding="utf-8").strip()
    else:
        system = DEFAULT_SYSTEM_PROMPT
    return system, DEFAULT_INSTRUCTION


def _resolve_model(client, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    models = list(client.models.list().data)
    if not models:
        raise SystemExit("No models returned by /v1/models. Pass --model explicitly.")

    model_id = str(getattr(models[0], "id", "") or "").strip()
    if not model_id:
        raise SystemExit("First /v1/models entry has no id. Pass --model explicitly.")
    return model_id


def _pdf_rendering_tools():
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pdf2image") from exc
    return convert_from_path, pdfinfo_from_path


def _pdf_page_count(pdf_path: Path) -> int:
    _convert_from_path, pdfinfo_from_path = _pdf_rendering_tools()
    try:
        info = pdfinfo_from_path(str(pdf_path))
        return int((info or {}).get("Pages") or 0)
    except Exception as exc:
        raise SystemExit(f"Failed to read PDF page count for {pdf_path}: {exc}") from exc


def _render_pdf_page(pdf_path: Path, page_num: int, dpi: int):
    convert_from_path, _pdfinfo_from_path = _pdf_rendering_tools()

    pages = convert_from_path(
        str(pdf_path),
        dpi=int(dpi),
        first_page=int(page_num),
        last_page=int(page_num),
        use_pdftocairo=True,
    )
    if not pages:
        raise SystemExit(f"Failed to render page {page_num} from {pdf_path}")
    return pages[0].convert("RGB")


def _render_first_page(pdf_path: Path, dpi: int):
    return _render_pdf_page(pdf_path, page_num=3, dpi=dpi)


def _prepare_image(image, max_pixels: int):
    from PIL import Image

    original_width, original_height = image.size
    target_height, target_width = prepared_dimensions_for_max_pixels(
        original_width=original_width,
        original_height=original_height,
        max_pixels=int(max_pixels),
    )
    target_size = (int(target_width), int(target_height))
    if image.size == target_size:
        return image, (int(original_width), int(original_height)), target_size

    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize(target_size, resampling.LANCZOS)
    return resized, (int(original_width), int(original_height)), target_size


def _image_to_data_uri(image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    mime_type = mimetypes.types_map.get(f".{image_format.lower()}", "image/png")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _extract_delta_text(delta_content) -> str:
    if isinstance(delta_content, str):
        return delta_content
    if isinstance(delta_content, list):
        pieces: list[str] = []
        for item in delta_content:
            if isinstance(item, str):
                pieces.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                pieces.append(text)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a minimal OpenAI-compatible vision request on the first page of a local PDF."
    )
    parser.add_argument("base_url", help="OpenAI-compatible base URL or chat completions URL")
    parser.add_argument("--pdf", help="PDF path. Defaults to the first file in vllm_api_tests/pdfs")
    parser.add_argument("--model", help="Model id. If omitted, the script uses /v1/models")
    parser.add_argument("--system", help="System prompt override")
    parser.add_argument("--instruction", help="Instruction override")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for the first PDF page")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Resize image to this maximum pixel count")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen thinking mode. Default is off.")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: openai") from exc

    pdf_path = Path(args.pdf).expanduser() if args.pdf else _default_pdf_path()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    default_system, default_instruction = _default_system_and_instruction()
    system_prompt = str(args.system or default_system).strip()
    instruction = str(args.instruction or default_instruction).strip()

    base_url = _normalize_base_url(args.base_url)
    client = OpenAI(
        base_url=base_url,
        api_key="unused",
        timeout=120.0,
    )
    model = _resolve_model(client, args.model)

    page_image = _render_first_page(pdf_path, dpi=args.dpi)
    prepared_image, original_size, prepared_size = _prepare_image(
        page_image,
        max_pixels=args.max_pixels,
    )
    image_data_uri = _image_to_data_uri(prepared_image)

    print(
        json.dumps(
            {
                "base_url": base_url,
                "model": model,
                "pdf": str(pdf_path),
                "page": 1,
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "prepared_size": {"width": prepared_size[0], "height": prepared_size[1]},
                "max_pixels": int(args.max_pixels),
                "enable_thinking": bool(args.enable_thinking),
                "prompt_source": "artifacts/hf_finetree_2_9/train.jsonl"
                if (ROOT / "artifacts" / "hf_finetree_2_9" / "train.jsonl").is_file()
                else "fallback",
            },
            ensure_ascii=False,
        )
    )
    print("assistant:")

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
            }
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}},
    )

    saw_text = False
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        text = _extract_delta_text(getattr(delta, "content", None))
        if not text:
            continue
        saw_text = True
        print(text, end="", flush=True)

    if saw_text:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
