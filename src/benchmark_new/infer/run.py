from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..eval import evaluate_predictions_bundle
from ..models import ProgressSnapshot, ProviderRunOutput, to_jsonable
from ..providers import ProviderOptions, get_provider_runner
from ..reports import (
    build_path_comparison_report,
    write_full_metrics_csv,
    write_mistakes_json,
    write_mistakes_values_json,
    write_path_comparison_report,
    write_path_mistakes_json,
    write_run_metrics_json,
    write_summary_csv,
    write_values_mistakes_json,
)


def create_run_dir(
    *,
    output_root: Path,
    provider: str,
    dataset_label: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = dataset_label.replace("/", "_").replace(" ", "_")
    run_dir = Path(output_root) / f"{timestamp}_{provider}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RunArtifactsWriter:
    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.progress_path = run_dir / "progress.json"
        self.requests_path = run_dir / "requests.jsonl"

    def append_event(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str))
            handle.write("\n")

    def write_progress(self, snapshot: ProgressSnapshot) -> None:
        self.progress_path.write_text(json.dumps(to_jsonable(snapshot), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_requests(self, provider_run: ProviderRunOutput) -> None:
        with self.requests_path.open("w", encoding="utf-8") as handle:
            for document in provider_run.documents:
                for page in document.page_outputs:
                    handle.write(
                        json.dumps(
                            {
                                "doc_id": document.doc_id,
                                "page_index": page.page_index,
                                "page_name": page.page_name,
                                "assistant_text": page.assistant_text,
                                "parsed_page": page.parsed_page,
                                "error": page.error,
                                "received_tokens": page.received_tokens,
                                **page.extra,
                            },
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")


def _print_progress(snapshot: ProgressSnapshot) -> None:
    print(
        (
            f"[{snapshot.phase}] provider={snapshot.provider} split={snapshot.split} "
            f"docs={snapshot.completed_documents}/{snapshot.total_documents} "
            f"pages={snapshot.completed_pages}/{snapshot.total_pages} "
            f"failed={snapshot.failed_pages} facts={snapshot.fact_count} "
            f"tokens={snapshot.total_tokens_received} tps={snapshot.tokens_per_second:.2f} "
            f"doc={snapshot.current_doc_id or '-'}"
        ),
        flush=True,
    )


def _write_inference_outputs(run_dir: Path, provider_run: ProviderRunOutput) -> None:
    documents_root = run_dir / "documents"
    documents_root.mkdir(parents=True, exist_ok=True)
    for document in provider_run.documents:
        doc_dir = documents_root / document.doc_id
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for page in document.page_outputs:
            page_path = pages_dir / f"page_{page.page_index:04d}.json"
            page_path.write_text(
                json.dumps(
                    {
                        "page_index": page.page_index,
                        "page_name": page.page_name,
                        "assistant_text": page.assistant_text,
                        "parsed_page": page.parsed_page,
                        "error": page.error,
                        "received_tokens": page.received_tokens,
                        **page.extra,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (doc_dir / "summary.json").write_text(json.dumps(to_jsonable(document), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bundle_from_provider_run(provider_run: ProviderRunOutput) -> dict[str, Any]:
    return {
        "documents": {
            document.doc_id: {
                "doc_id": document.doc_id,
                "pages": [
                    {
                        "page_index": page.page_index,
                        "page_name": page.page_name,
                        "assistant_text": page.assistant_text,
                        "parsed_page": page.parsed_page,
                        "error": page.error,
                        "received_tokens": page.received_tokens,
                    }
                    for page in document.page_outputs
                ],
            }
            for document in provider_run.documents
        }
    }


def run_inference_pipeline(
    *,
    documents,
    options: ProviderOptions,
    run_dir: Path,
    dataset_version_id: str | None,
    dataset_name: str | None,
    split: str,
    data_root: Path,
    evaluate_after_infer: bool,
) -> dict[str, Any]:
    writer = RunArtifactsWriter(run_dir=run_dir)
    started_at = time.time()
    manifest = {
        "provider": options.provider,
        "model": options.model,
        "dataset_version_id": dataset_version_id,
        "dataset_name": dataset_name,
        "split": split,
        "started_at": started_at,
        "run_dir": str(run_dir),
        "documents": [document.doc_id for document in documents],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    writer.append_event({"ts": datetime.now().isoformat(), "event": "started", **manifest})

    def _progress(snapshot: ProgressSnapshot) -> None:
        writer.write_progress(snapshot)
        writer.append_event({"ts": datetime.now().isoformat(), "event": "progress", **to_jsonable(snapshot)})
        _print_progress(snapshot)

    runner = get_provider_runner(options.provider)
    provider_run = runner(
        documents,
        options=options,
        run_dir=run_dir,
        dataset_version_id=dataset_version_id,
        dataset_name=dataset_name,
        split=split,
        progress_callback=_progress,
    )
    writer.write_requests(provider_run)
    _write_inference_outputs(run_dir, provider_run)
    writer.append_event({"ts": datetime.now().isoformat(), "event": "inference_completed", "provider": provider_run.provider})

    output: dict[str, Any] = {"provider_run": provider_run}
    if evaluate_after_infer:
        bundle = _bundle_from_provider_run(provider_run)
        run_result = evaluate_predictions_bundle(bundle, data_root=data_root)
        evaluation_dir = run_dir / "evaluation"
        write_run_metrics_json(evaluation_dir / "run_metrics.json", run_result)
        total_tokens = sum(document.received_tokens for document in provider_run.documents)
        total_failed_pages = sum(document.failed_pages for document in provider_run.documents)
        write_summary_csv(
            evaluation_dir / "summary.csv",
            run_result=run_result,
            provider=options.provider,
            dataset_version_id=dataset_version_id,
            dataset_name=dataset_name,
            split=split,
            total_tokens_received=total_tokens,
            total_failed_pages=total_failed_pages,
        )
        write_full_metrics_csv(
            evaluation_dir / "full_metrics.csv",
            run_result=run_result,
            provider=options.provider,
            dataset_version_id=dataset_version_id,
            dataset_name=dataset_name,
            split=split,
        )
        write_mistakes_json(
            evaluation_dir / "mistakes.json",
            run_result=run_result,
            bundle=bundle,
            data_root=data_root,
        )
        write_mistakes_values_json(
            evaluation_dir / "mistakes_values.json",
            run_result=run_result,
            bundle=bundle,
            data_root=data_root,
        )
        write_values_mistakes_json(
            evaluation_dir / "values_mistakes.json",
            run_result=run_result,
            bundle=bundle,
            data_root=data_root,
        )
        write_path_mistakes_json(
            evaluation_dir / "path_mistakes.json",
            run_result=run_result,
            bundle=bundle,
            data_root=data_root,
        )
        write_path_comparison_report(
            evaluation_dir / "path_comparison_report.json",
            build_path_comparison_report(run_result=run_result, bundle=bundle, data_root=data_root),
        )
        writer.append_event({"ts": datetime.now().isoformat(), "event": "evaluation_completed"})
        output["run_result"] = run_result
    return output
