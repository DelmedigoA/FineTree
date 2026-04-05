"""Batch inference SSE endpoint — processes multiple documents via a custom OpenAI-compatible API."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Generator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .deps import get_data_root
from ..annotation_backups import atomic_write_text, create_annotation_backup
from ..annotation_core import serialize_annotations_json
from ..batch_qwen_inference import (
    BatchQwenDocumentJob,
    BatchQwenPageProgress,
    BatchQwenSettings,
    build_batch_jobs_for_doc_ids,
    run_batch_qwen_inference,
    _load_prompt_pair,
    _normalize_base_url,
)
from ..schema_io import canonicalize_with_findings
from ..workspace import annotations_root, pdf_images_root

router = APIRouter(prefix="/api/ai", tags=["ai-batch"])


class BatchInferRequest(BaseModel):
    doc_ids: list[str]
    base_url: str
    model_id: str
    action: str = "gt"          # "gt" = replace all, "autocomplete" = merge/append
    max_pixels: int = 1_400_000
    max_tokens: int = 24000
    temperature: float = 0.0
    enable_thinking: bool = False


def _save_batch_results(
    doc_id: str,
    imported_pages: tuple[dict[str, Any], ...],
    *,
    action: str,
    data_root: Path,
) -> int:
    """Merge imported_pages into the document annotations file. Returns saved fact count."""
    annotations_path = annotations_root(data_root) / f"{doc_id}.json"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    if action == "gt":
        payload: dict[str, Any] = {"pages": list(imported_pages)}
    else:
        # Append: overlay new pages onto existing ones by image name.
        if annotations_path.is_file():
            try:
                existing: dict[str, Any] = json.loads(annotations_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {"pages": []}
        else:
            existing = {"pages": []}

        existing_pages: dict[str, dict[str, Any]] = {
            p["image"]: p
            for p in existing.get("pages", [])
            if isinstance(p, dict) and p.get("image")
        }
        for new_page in imported_pages:
            image_name = new_page.get("image")
            if image_name:
                existing_pages[image_name] = new_page

        payload = {**existing, "pages": list(existing_pages.values())}

    try:
        canonical, _ = canonicalize_with_findings(payload, strict_equation_guards=False)
    except Exception:
        canonical = payload

    json_text = serialize_annotations_json(canonical)
    create_annotation_backup(annotations_path, reason="batch_infer")
    atomic_write_text(annotations_path, json_text)

    return sum(
        len(page.get("facts", []))
        for page in (canonical.get("pages") or [])
        if isinstance(page, dict)
    )


@router.post("/batch-infer")
def batch_infer_stream(request: BatchInferRequest) -> StreamingResponse:
    data_root = get_data_root()

    doc_summaries = [
        (doc_id, pdf_images_root(data_root) / doc_id)
        for doc_id in request.doc_ids
    ]
    jobs = build_batch_jobs_for_doc_ids(doc_summaries)

    if not jobs:
        def _empty() -> Generator[str, None, None]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No valid documents or page images found.'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _progress(prog: BatchQwenPageProgress) -> None:
        event_queue.put({
            "type": "progress",
            "doc_id": prog.doc_id,
            "page_name": prog.page_name,
            "page_index": prog.page_index,
            "total_pages": prog.total_pages,
            "completed_pages": prog.completed_pages,
            "failed_pages": prog.failed_pages,
            "received_tokens": prog.received_tokens,
            "fact_count": prog.fact_count,
        })

    def _run() -> None:
        # Announce all jobs upfront.
        for job in jobs:
            event_queue.put({
                "type": "start",
                "doc_id": job.doc_id,
                "total_pages": len(job.page_paths),
            })

        try:
            system_prompt, instruction = _load_prompt_pair()
            settings = BatchQwenSettings(
                base_url=_normalize_base_url(request.base_url),
                model=request.model_id,
                system_prompt=system_prompt,
                instruction=instruction,
                max_pixels=request.max_pixels,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                enable_thinking=request.enable_thinking,
            )
            results = run_batch_qwen_inference(
                jobs,
                settings=settings,
                progress_callback=_progress,
            )

            total_facts = 0
            for doc_id, result in results.items():
                saved_facts = 0
                save_error: str | None = None
                if result.imported_pages:
                    try:
                        saved_facts = _save_batch_results(
                            doc_id,
                            result.imported_pages,
                            action=request.action,
                            data_root=data_root,
                        )
                    except Exception as exc:
                        save_error = str(exc)
                        saved_facts = result.fact_count

                total_facts += saved_facts
                event_queue.put({
                    "type": "doc_done",
                    "doc_id": doc_id,
                    "completed_pages": result.completed_pages,
                    "failed_pages": result.failed_pages,
                    "fact_count": saved_facts,
                    "received_tokens": result.received_tokens,
                    "save_error": save_error,
                    "failures": [
                        {"page": str(f.get("page", "")), "error": str(f.get("error", ""))}
                        for f in result.failures[:10]
                    ],
                })

            event_queue.put({
                "type": "done",
                "total_docs": len(results),
                "total_facts": total_facts,
            })

        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    def _sse() -> Generator[str, None, None]:
        while True:
            try:
                event = event_queue.get(timeout=60.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
