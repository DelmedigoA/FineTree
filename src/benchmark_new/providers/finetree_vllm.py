from __future__ import annotations

import time
from pathlib import Path

from finetree_annotator.batch_qwen_inference import (
    BatchQwenDocumentJob,
    BatchQwenSettings,
    resolve_batch_qwen_settings,
    run_batch_qwen_inference,
)

from ..models import DocumentInput, ProgressSnapshot, ProviderDocumentOutput, ProviderPageOutput, ProviderRunOutput
from .base import ProviderOptions


def _build_settings(options: ProviderOptions) -> BatchQwenSettings:
    if options.system_prompt is None or options.instruction is None:
        resolved = resolve_batch_qwen_settings(
            base_url_override=options.base_url,
            model_override=options.model,
            max_pixels=options.max_pixels,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
        )
        if options.system_prompt is None and options.instruction is None:
            return resolved
        return BatchQwenSettings(
            base_url=resolved.base_url,
            model=options.model or resolved.model,
            system_prompt=options.system_prompt or resolved.system_prompt,
            instruction=options.instruction or resolved.instruction,
            max_pixels=options.max_pixels,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            enable_thinking=options.enable_thinking,
        )
    if not options.base_url:
        raise ValueError("finetree_vllm inference requires --base-url when prompt overrides are used.")
    return BatchQwenSettings(
        base_url=options.base_url,
        model=options.model,
        system_prompt=options.system_prompt,
        instruction=options.instruction,
        max_pixels=options.max_pixels,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        enable_thinking=options.enable_thinking,
    )


def run_finetree_vllm_inference(
    documents: tuple[DocumentInput, ...],
    *,
    options: ProviderOptions,
    run_dir: Path,
    dataset_version_id: str | None,
    dataset_name: str | None,
    split: str,
    progress_callback=None,
) -> ProviderRunOutput:
    settings = _build_settings(options)
    jobs = [
        BatchQwenDocumentJob(
            doc_id=document.doc_id,
            images_dir=document.images_dir,
            page_paths=tuple(page.image_path for page in document.pages),
        )
        for document in documents
    ]
    start_time = time.monotonic()
    per_doc = {
        document.doc_id: {
            "doc_id": document.doc_id,
            "total_pages": len(document.pages),
            "completed_pages": 0,
            "failed_pages": 0,
            "fact_count": 0,
            "tokens_received": 0,
        }
        for document in documents
    }

    def _progress_callback(progress) -> None:
        if progress_callback is None:
            return
        current = per_doc.setdefault(progress.doc_id, {})
        current.update(
            {
                "doc_id": progress.doc_id,
                "total_pages": progress.total_pages,
                "completed_pages": progress.completed_pages,
                "failed_pages": progress.failed_pages,
                "fact_count": progress.fact_count,
                "tokens_received": progress.received_tokens,
                "current_page": progress.page_name,
            }
        )
        total_pages = sum(int(state["total_pages"]) for state in per_doc.values())
        completed_pages = sum(int(state["completed_pages"]) for state in per_doc.values())
        failed_pages = sum(int(state["failed_pages"]) for state in per_doc.values())
        fact_count = sum(int(state["fact_count"]) for state in per_doc.values())
        tokens = sum(int(state["tokens_received"]) for state in per_doc.values())
        elapsed = max(time.monotonic() - start_time, 1e-9)
        completed_docs = sum(1 for state in per_doc.values() if int(state["completed_pages"]) >= int(state["total_pages"]))
        progress_callback(
            ProgressSnapshot(
                phase="infer",
                provider="finetree_vllm",
                dataset_version_id=dataset_version_id,
                dataset_name=dataset_name,
                split=split,
                current_doc_id=progress.doc_id,
                total_documents=len(documents),
                completed_documents=completed_docs,
                total_pages=total_pages,
                completed_pages=completed_pages,
                failed_pages=failed_pages,
                fact_count=fact_count,
                total_tokens_received=tokens,
                elapsed_seconds=elapsed,
                tokens_per_second=float(tokens / elapsed),
                documents=per_doc,
            )
        )

    results = run_batch_qwen_inference(jobs, settings=settings, progress_callback=_progress_callback)
    provider_documents: list[ProviderDocumentOutput] = []
    for document in documents:
        result = results[document.doc_id]
        provider_documents.append(
            ProviderDocumentOutput(
                doc_id=document.doc_id,
                total_pages=result.total_pages,
                completed_pages=result.completed_pages,
                failed_pages=result.failed_pages,
                received_tokens=result.received_tokens,
                fact_count=result.fact_count,
                page_outputs=[
                    ProviderPageOutput(
                        page_index=int(page_output.get("page_index") or 0),
                        page_name=str(page_output.get("page_name") or ""),
                        assistant_text=str(page_output.get("assistant_text") or ""),
                        parsed_page=page_output.get("parsed_page") if isinstance(page_output.get("parsed_page"), dict) else None,
                        error=str(page_output.get("error")) if page_output.get("error") else None,
                        received_tokens=0,
                    )
                    for page_output in result.page_outputs
                ],
                failures=[dict(item) for item in result.failures],
            )
        )
    return ProviderRunOutput(provider="finetree_vllm", documents=provider_documents)
