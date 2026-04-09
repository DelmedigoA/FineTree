from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from finetree_annotator.batch_qwen_inference import estimate_text_tokens, _default_instruction, _default_system_prompt
from finetree_annotator.gemini_vlm import (
    _create_gemini_log_session,
    generate_content_from_image,
    resolve_supported_gemini_model_name,
)

from ..models import DocumentInput, ProgressSnapshot, ProviderDocumentOutput, ProviderPageOutput, ProviderRunOutput
from .base import ProviderOptions


def _normalize_page_payload(assistant_text: str, *, page_name: str) -> tuple[dict | None, str | None]:
    text = str(assistant_text or "").strip()
    if not text:
        return None, "Empty model response."
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return None, str(exc)
    if isinstance(parsed, dict) and isinstance(parsed.get("pages"), list) and parsed["pages"]:
        page = parsed["pages"][0]
    elif isinstance(parsed, dict) and ("meta" in parsed or "facts" in parsed):
        page = parsed
    else:
        return None, "Model response did not contain a page payload."
    if not isinstance(page, dict):
        return None, "Model response page payload was not an object."
    normalized = dict(page)
    normalized["image"] = page_name
    return normalized, None


def run_gemini_inference(
    documents: tuple[DocumentInput, ...],
    *,
    options: ProviderOptions,
    run_dir: Path,
    dataset_version_id: str | None,
    dataset_name: str | None,
    split: str,
    progress_callback=None,
) -> ProviderRunOutput:
    start_time = time.monotonic()
    provider_logs_root = run_dir / "provider_logs" / "gemini"
    provider_logs_root.mkdir(parents=True, exist_ok=True)
    system_prompt = options.system_prompt or _default_system_prompt()
    instruction = options.instruction or _default_instruction()
    resolved_model = resolve_supported_gemini_model_name(options.model)
    documents_state = {
        document.doc_id: {
            "doc_id": document.doc_id,
            "total_pages": len(document.pages),
            "completed_pages": 0,
            "failed_pages": 0,
            "fact_count": 0,
            "tokens_received": 0,
            "current_page": None,
        }
        for document in documents
    }
    provider_documents: list[ProviderDocumentOutput] = []

    for document in documents:
        page_outputs: list[ProviderPageOutput] = []
        failures: list[dict[str, str]] = []
        for page in document.pages:
            session_dir = _create_gemini_log_session(
                operation="benchmark_new_generate_content",
                model=resolved_model,
                image_path=page.image_path,
                prompt=instruction,
                mime_type=None,
                system_prompt=system_prompt,
                temperature=options.temperature,
                enable_thinking=options.enable_thinking,
                thinking_level=options.thinking_level,
                few_shot_examples=None,
            )
            error: str | None = None
            parsed_page: dict | None = None
            try:
                assistant_text = generate_content_from_image(
                    image_path=page.image_path,
                    prompt=instruction,
                    model=resolved_model,
                    api_key=options.api_key,
                    system_prompt=system_prompt,
                    temperature=options.temperature,
                    enable_thinking=options.enable_thinking,
                    thinking_level=options.thinking_level,
                    session_dir=session_dir,
                )
                parsed_page, error = _normalize_page_payload(assistant_text, page_name=page.image_name)
            except Exception as exc:
                assistant_text = ""
                error = str(exc)
            received_tokens = estimate_text_tokens(assistant_text)
            copied_log_dir = provider_logs_root / f"{document.doc_id}_{page.image_name.replace('.png', '')}"
            if copied_log_dir.exists():
                shutil.rmtree(copied_log_dir)
            shutil.copytree(session_dir, copied_log_dir)
            page_output = ProviderPageOutput(
                page_index=page.page_index,
                page_name=page.image_name,
                assistant_text=assistant_text,
                parsed_page=parsed_page,
                error=error,
                received_tokens=received_tokens,
                extra={"gemini_log_dir": str(copied_log_dir)},
            )
            page_outputs.append(page_output)
            state = documents_state[document.doc_id]
            state["completed_pages"] = int(state["completed_pages"]) + 1
            state["current_page"] = page.image_name
            state["tokens_received"] = int(state["tokens_received"]) + received_tokens
            if parsed_page is None:
                state["failed_pages"] = int(state["failed_pages"]) + 1
                failures.append({"page": page.image_name, "error": error or "Unknown error"})
            else:
                state["fact_count"] = int(state["fact_count"]) + len(parsed_page.get("facts") or [])
            if progress_callback is not None:
                total_pages = sum(int(item["total_pages"]) for item in documents_state.values())
                completed_pages = sum(int(item["completed_pages"]) for item in documents_state.values())
                failed_pages = sum(int(item["failed_pages"]) for item in documents_state.values())
                fact_count = sum(int(item["fact_count"]) for item in documents_state.values())
                tokens = sum(int(item["tokens_received"]) for item in documents_state.values())
                completed_docs = sum(1 for item in documents_state.values() if int(item["completed_pages"]) >= int(item["total_pages"]))
                elapsed = max(time.monotonic() - start_time, 1e-9)
                progress_callback(
                    ProgressSnapshot(
                        phase="infer",
                        provider="gemini",
                        dataset_version_id=dataset_version_id,
                        dataset_name=dataset_name,
                        split=split,
                        current_doc_id=document.doc_id,
                        total_documents=len(documents),
                        completed_documents=completed_docs,
                        total_pages=total_pages,
                        completed_pages=completed_pages,
                        failed_pages=failed_pages,
                        fact_count=fact_count,
                        total_tokens_received=tokens,
                        elapsed_seconds=elapsed,
                        tokens_per_second=float(tokens / elapsed),
                        documents=documents_state,
                    )
                )
        provider_documents.append(
            ProviderDocumentOutput(
                doc_id=document.doc_id,
                total_pages=len(document.pages),
                completed_pages=len(page_outputs),
                failed_pages=len(failures),
                received_tokens=sum(page_output.received_tokens for page_output in page_outputs),
                fact_count=sum(len((page_output.parsed_page or {}).get("facts") or []) for page_output in page_outputs if page_output.parsed_page),
                page_outputs=page_outputs,
                failures=failures,
            )
        )
    return ProviderRunOutput(provider="gemini", documents=provider_documents)
