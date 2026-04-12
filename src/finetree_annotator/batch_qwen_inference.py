from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Optional

from .annotation_core import normalize_import_payload_to_document, parse_import_json_text
from .finetune.config import load_finetune_config
from .vision_resize import prepared_dimensions_for_max_pixels
from .workspace import page_image_paths


DEFAULT_MAX_PIXELS = 1_400_000
DEFAULT_MAX_TOKENS = 24_000
DEFAULT_TEMPERATURE = 0.0
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_PROGRESS_EMIT_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class BatchQwenSettings:
    base_url: str
    model: str
    system_prompt: str
    instruction: str
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    enable_thinking: bool = False


@dataclass(frozen=True)
class BatchQwenDocumentJob:
    doc_id: str
    images_dir: Path
    page_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BatchQwenPageProgress:
    doc_id: str
    page_name: str
    page_index: int
    total_pages: int
    received_tokens: int
    completed_pages: int
    failed_pages: int
    fact_count: int


@dataclass(frozen=True)
class BatchQwenDocumentResult:
    doc_id: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    received_tokens: int
    fact_count: int
    imported_pages: tuple[dict[str, Any], ...]
    page_outputs: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, str], ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def estimate_text_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(str(text or "")))


def resolve_qwen_config_path() -> Optional[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("FINETREE_QWEN_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = _repo_root()
    candidates.append(root / "configs/qwen_ui_runpod_queue.yaml")
    candidates.append(root / "configs/finetune_qwen35a3_vl.yaml")
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "configs/qwen_ui_runpod_queue.yaml")
        candidates.append(parent / "configs/finetune_qwen35a3_vl.yaml")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def resolve_batch_qwen_base_url(
    *,
    config_path: Optional[Path | str] = None,
    base_url_override: Optional[str] = None,
) -> str:
    explicit_base_url = str(base_url_override or os.getenv("FINETREE_QWEN_ENDPOINT_BASE_URL") or "").strip()
    if explicit_base_url:
        return _normalize_base_url(explicit_base_url)

    cfg_path = Path(config_path).expanduser().resolve() if config_path is not None else resolve_qwen_config_path()
    if cfg_path is None or not cfg_path.is_file():
        raise RuntimeError(
            "Could not find Qwen config. Set FINETREE_QWEN_CONFIG or add configs/qwen_ui_runpod_queue.yaml."
        )

    cfg = load_finetune_config(cfg_path)
    config_base_url = str(cfg.inference.endpoint_base_url or "").strip()
    if not config_base_url:
        raise RuntimeError(
            "Missing Qwen endpoint base URL. Set inference.endpoint_base_url or FINETREE_QWEN_ENDPOINT_BASE_URL."
        )
    return _normalize_base_url(config_base_url)


def _normalize_base_url(raw_url: str) -> str:
    url = str(raw_url or "").strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def resolve_batch_qwen_model(
    *,
    base_url: str,
    explicit_model: Optional[str] = None,
) -> str:
    model = str(explicit_model or "").strip()
    if model:
        return model

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: openai") from exc

    client = OpenAI(
        base_url=_normalize_base_url(base_url),
        api_key="unused",
        timeout=60.0,
    )
    models = list(client.models.list().data)
    if not models:
        raise RuntimeError("No models returned by /v1/models. Enter the model explicitly.")
    model_id = str(getattr(models[0], "id", "") or "").strip()
    if not model_id:
        raise RuntimeError("First /v1/models entry has no id. Enter the model explicitly.")
    return model_id


def _default_system_prompt() -> str:
    return (
        "You are a precise financial statement extraction system. "
        "Return only valid JSON that matches the required schema."
    )


def _default_instruction() -> str:
    return """You are extracting financial-statement annotations from a single page image.

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


def _load_prompt_pair() -> tuple[str, str]:
    root = _repo_root()
    preferred_export = root / "artifacts" / "hf_finetree_2_9" / "train.jsonl"
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

    system_path = root / "prompts" / "system_prompt.txt"
    if system_path.is_file():
        system = system_path.read_text(encoding="utf-8").strip()
    else:
        system = _default_system_prompt()
    return system, _default_instruction()


def resolve_batch_qwen_settings(
    *,
    config_path: Optional[Path | str] = None,
    base_url_override: Optional[str] = None,
    model_override: Optional[str] = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BatchQwenSettings:
    base_url = resolve_batch_qwen_base_url(
        config_path=config_path,
        base_url_override=base_url_override,
    )
    model = resolve_batch_qwen_model(
        base_url=base_url,
        explicit_model=model_override or os.getenv("FINETREE_QWEN_MODEL"),
    )
    system_prompt, instruction = _load_prompt_pair()
    return BatchQwenSettings(
        base_url=_normalize_base_url(base_url),
        model=model,
        system_prompt=system_prompt,
        instruction=instruction,
        max_pixels=int(max_pixels),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        enable_thinking=False,
    )


def _prepare_image_bytes(image_path: Path, *, max_pixels: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        original_size = (int(rgb.size[0]), int(rgb.size[1]))
        prepared_h, prepared_w = prepared_dimensions_for_max_pixels(
            original_width=float(original_size[0]),
            original_height=float(original_size[1]),
            max_pixels=int(max_pixels),
        )
        prepared_size = (int(prepared_w), int(prepared_h))
        if rgb.size != prepared_size:
            resampling = getattr(Image, "Resampling", Image)
            rgb = rgb.resize(prepared_size, resampling.LANCZOS)
        buffer = BytesIO()
        rgb.save(buffer, format="PNG")

    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = mimetypes.types_map.get(".png", "image/png")
    return f"data:{mime_type};base64,{payload}", original_size, prepared_size


def _extract_delta_text(delta_content: Any) -> str:
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
                pieces.append(str(item.get("text")))
        return "".join(pieces)
    return ""


def _normalize_page_payload(assistant_text: str, *, page_name: str) -> tuple[dict[str, Any] | None, str | None]:
    if not str(assistant_text or "").strip():
        return None, "Empty model response."
    try:
        parse_result = parse_import_json_text(assistant_text)
        normalized = normalize_import_payload_to_document(parse_result.payload, [page_name], page_name)
    except Exception as exc:
        return None, str(exc)
    pages = normalized.get("pages") if isinstance(normalized, dict) else None
    if not isinstance(pages, list) or not pages:
        return None, "Model response did not contain an importable page payload."
    page_obj = pages[0]
    if not isinstance(page_obj, dict):
        return None, "Model response page payload was not an object."
    page_obj = dict(page_obj)
    page_obj["image"] = page_name
    return page_obj, None


def _page_fact_count(page_obj: dict[str, Any]) -> int:
    facts = page_obj.get("facts")
    if not isinstance(facts, list):
        return 0
    return len([fact for fact in facts if isinstance(fact, dict)])


def _write_raw_output_cache(cache_dir: Path, doc_id: str, page_name: str, assistant_text: str) -> None:
    """Persist raw VLM output immediately so nothing is lost if downstream save fails.

    Writes to: <cache_dir>/<doc_id>/<page_name>.txt
    Any exception is swallowed — caching must never break inference.
    """
    try:
        target_dir = Path(cache_dir) / str(doc_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{page_name}.txt").write_text(str(assistant_text or ""), encoding="utf-8")
    except Exception:
        pass


def run_batch_qwen_inference(
    jobs: list[BatchQwenDocumentJob],
    *,
    settings: BatchQwenSettings,
    progress_callback: Optional[Callable[[BatchQwenPageProgress], None]] = None,
    max_workers: Optional[int] = None,
    raw_output_cache_dir: Optional[Path] = None,
) -> dict[str, BatchQwenDocumentResult]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: openai") from exc

    normalized_jobs = [
        BatchQwenDocumentJob(
            doc_id=str(job.doc_id).strip(),
            images_dir=Path(job.images_dir),
            page_paths=tuple(Path(path) for path in job.page_paths),
        )
        for job in jobs
        if str(job.doc_id).strip() and tuple(job.page_paths)
    ]
    if not normalized_jobs:
        return {}

    doc_states: dict[str, dict[str, Any]] = {}
    for job in normalized_jobs:
        doc_states[job.doc_id] = {
            "total_pages": len(job.page_paths),
            "completed_pages": 0,
            "failed_pages": 0,
            "received_tokens": 0,
            "fact_count": 0,
            "page_tokens": {path.name: 0 for path in job.page_paths},
            "pages": [],
            "page_outputs": [],
            "failures": [],
        }

    total_tasks = sum(len(job.page_paths) for job in normalized_jobs)
    worker_count = max(1, int(max_workers or total_tasks))
    lock = threading.Lock()

    last_emit_state: dict[str, tuple[float, int, int, int, int, str]] = {}

    def _emit_progress(doc_id: str, page_name: str, page_index: int, *, force: bool = False) -> None:
        if progress_callback is None:
            return
        state = doc_states[doc_id]
        now = time.monotonic()
        current_signature = (
            int(state["completed_pages"]),
            int(state["failed_pages"]),
            int(state["received_tokens"]),
            int(state["fact_count"]),
            str(page_name),
        )
        previous = last_emit_state.get(doc_id)
        if not force and previous is not None:
            last_time, prev_completed, prev_failed, prev_tokens, prev_facts, prev_page = previous
            same_signature = current_signature == (prev_completed, prev_failed, prev_tokens, prev_facts, prev_page)
            if same_signature or (now - float(last_time)) < _PROGRESS_EMIT_INTERVAL_SECONDS:
                return
        last_emit_state[doc_id] = (now, *current_signature)
        progress_callback(
            BatchQwenPageProgress(
                doc_id=doc_id,
                page_name=page_name,
                page_index=page_index,
                total_pages=int(state["total_pages"]),
                received_tokens=int(state["received_tokens"]),
                completed_pages=int(state["completed_pages"]),
                failed_pages=int(state["failed_pages"]),
                fact_count=int(state["fact_count"]),
            )
        )

    def _run_page(job: BatchQwenDocumentJob, page_path: Path, page_index: int) -> tuple[str, int, str, dict[str, Any] | None, str | None]:
        client = OpenAI(
            base_url=settings.base_url,
            api_key="unused",
            timeout=300.0,
        )
        image_data_uri, _original_size, _prepared_size = _prepare_image_bytes(
            page_path,
            max_pixels=settings.max_pixels,
        )
        stream = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": settings.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": settings.instruction},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                },
            ],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(settings.enable_thinking)}},
        )

        chunks: list[str] = []
        page_name = page_path.name
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            text = _extract_delta_text(getattr(delta, "content", None))
            if not text:
                continue
            chunks.append(text)
            with lock:
                state = doc_states[job.doc_id]
                previous = int(state["page_tokens"].get(page_name, 0))
                current = estimate_text_tokens("".join(chunks))
                delta_tokens = max(0, current - previous)
                state["page_tokens"][page_name] = current
                state["received_tokens"] = int(state["received_tokens"]) + delta_tokens
                _emit_progress(job.doc_id, page_name, page_index)

        assistant_text = "".join(chunks).strip()
        page_obj, error = _normalize_page_payload(assistant_text, page_name=page_name)
        return page_name, page_index, assistant_text, page_obj, error

    futures = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for job in normalized_jobs:
            for page_index, page_path in enumerate(job.page_paths, start=1):
                futures[executor.submit(_run_page, job, page_path, page_index)] = (job, page_path, page_index)

        for future in as_completed(futures):
            job, page_path, page_index = futures[future]
            page_name = page_path.name
            try:
                _page_name, _page_index, assistant_text, page_obj, error = future.result()
            except Exception as exc:
                assistant_text = ""
                page_obj = None
                error = str(exc)
            # CRITICAL: persist raw VLM output to disk BEFORE any counter updates,
            # parsing, or annotation save. If anything downstream breaks, the raw
            # response survives and can be re-parsed without another inference call.
            if raw_output_cache_dir is not None and assistant_text:
                _write_raw_output_cache(raw_output_cache_dir, job.doc_id, page_name, assistant_text)
            with lock:
                state = doc_states[job.doc_id]
                state["completed_pages"] = int(state["completed_pages"]) + 1
                state["page_outputs"].append(
                    (
                        page_index,
                        {
                            "page_index": int(page_index),
                            "page_name": page_name,
                            "assistant_text": assistant_text,
                            "parsed_page": page_obj,
                            "error": str(error or "") if page_obj is None else None,
                        },
                    )
                )
                if page_obj is None:
                    state["failed_pages"] = int(state["failed_pages"]) + 1
                    state["failures"].append({"page": page_name, "error": str(error or "Unknown error")})
                else:
                    state["pages"].append((page_index, page_obj))
                    state["fact_count"] = int(state["fact_count"]) + _page_fact_count(page_obj)
                _emit_progress(job.doc_id, page_name, page_index, force=True)

    results: dict[str, BatchQwenDocumentResult] = {}
    for job in normalized_jobs:
        state = doc_states[job.doc_id]
        sorted_pages = tuple(page for _index, page in sorted(state["pages"], key=lambda item: item[0]))
        sorted_page_outputs = tuple(page for _index, page in sorted(state["page_outputs"], key=lambda item: item[0]))
        failures = tuple(
            {"page": str(item.get("page") or ""), "error": str(item.get("error") or "")}
            for item in state["failures"]
        )
        results[job.doc_id] = BatchQwenDocumentResult(
            doc_id=job.doc_id,
            total_pages=int(state["total_pages"]),
            completed_pages=int(state["completed_pages"]),
            failed_pages=int(state["failed_pages"]),
            received_tokens=int(state["received_tokens"]),
            fact_count=int(state["fact_count"]),
            imported_pages=sorted_pages,
            page_outputs=sorted_page_outputs,
            failures=failures,
        )
    return results


def build_batch_jobs_for_doc_ids(doc_summaries: list[tuple[str, Path]]) -> list[BatchQwenDocumentJob]:
    jobs: list[BatchQwenDocumentJob] = []
    for doc_id, images_dir in doc_summaries:
        page_paths = tuple(page_image_paths(images_dir))
        if not page_paths:
            continue
        jobs.append(
            BatchQwenDocumentJob(
                doc_id=str(doc_id).strip(),
                images_dir=Path(images_dir),
                page_paths=page_paths,
            )
        )
    return jobs


__all__ = [
    "BatchQwenDocumentJob",
    "BatchQwenDocumentResult",
    "BatchQwenPageProgress",
    "BatchQwenSettings",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "build_batch_jobs_for_doc_ids",
    "estimate_text_tokens",
    "resolve_batch_qwen_base_url",
    "resolve_batch_qwen_model",
    "resolve_batch_qwen_settings",
    "resolve_qwen_config_path",
    "run_batch_qwen_inference",
]
