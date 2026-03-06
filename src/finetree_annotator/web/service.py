from __future__ import annotations

import base64
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image
from fastapi import HTTPException
from pydantic import ValidationError

from ..annotation_core import (
    BoxRecord,
    PageState,
    apply_entity_name_to_pages,
    build_annotations_payload,
    denormalize_bbox_from_1000,
    extract_document_meta,
    load_page_states,
    normalize_bbox_data,
    normalize_fact_data,
    parse_import_payload,
    serialize_annotations_json,
)
from ..fact_normalization import normalize_annotation_payload
from ..fact_ordering import canonical_fact_order_indices, normalize_document_meta, resolve_reading_direction
from ..gemini_few_shot import (
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_FEW_SHOT_PAGES,
    build_repo_roots,
    load_complex_few_shot_examples,
    load_test_pdf_few_shot_examples,
)
from ..gemini_vlm import generate_content_from_image as generate_gemini_content
from ..gemini_vlm import parse_page_extraction_text
from ..page_issues import DocumentIssueSummary, PageIssueSummary, validate_document_issues
from ..qwen_vlm import generate_content_from_image as generate_qwen_content
from ..schema_contract import default_extraction_prompt_template
from ..schemas import PageExtraction, PageMeta, PageType
from ..workspace import (
    DEFAULT_DATA_ROOT,
    WorkspaceDocumentSummary,
    annotations_root,
    build_document_summary,
    discover_workspace_documents,
    ensure_pdf_images,
    import_pdf_to_workspace,
    load_annotation_payload,
    page_has_annotation,
    page_image_paths,
    pdf_images_root,
    sanitize_doc_id,
    set_document_checked,
    set_document_reviewed,
)
from .models import (
    ApiAppConfig,
    ApiBBox,
    ApiDocumentDetail,
    ApiDocumentIssueSummary,
    ApiDocumentMeta,
    ApiDocumentPage,
    ApiDocumentSaveRequest,
    ApiDocumentSaveResponse,
    ApiDocumentValidateResponse,
    ApiExtractionRequest,
    ApiExtractionResponse,
    ApiFact,
    ApiFormatFinding,
    ApiIssue,
    ApiPageExtraction,
    ApiPageIssueSummary,
    ApiPageInput,
    ApiWorkspaceDocumentSummary,
)

FEW_SHOT_PRESET_CLASSIC = "classic_4"
FEW_SHOT_PRESET_EXTENDED = "extended_7"


def _to_api_issue_summary(summary: PageIssueSummary) -> ApiPageIssueSummary:
    return ApiPageIssueSummary(
        page_image=summary.page_image,
        reg_flag_count=int(summary.reg_flag_count),
        warning_count=int(summary.warning_count),
        issues=[
            ApiIssue(
                severity=issue.severity,
                code=issue.code,
                message=issue.message,
                page_image=issue.page_image,
                fact_index=issue.fact_index,
                field_name=issue.field_name,
            )
            for issue in summary.issues
        ],
    )


def _to_api_document_issue_summary(summary: DocumentIssueSummary) -> ApiDocumentIssueSummary:
    return ApiDocumentIssueSummary(
        reg_flag_count=int(summary.reg_flag_count),
        warning_count=int(summary.warning_count),
        pages_with_reg_flags=int(summary.pages_with_reg_flags),
        pages_with_warnings=int(summary.pages_with_warnings),
        page_summaries={key: _to_api_issue_summary(value) for key, value in summary.page_summaries.items()},
    )


def _warning_findings(findings: Iterable[dict[str, Any]]) -> list[ApiFormatFinding]:
    warning_codes = {"noncanonical_date", "placeholder_value", "noncanonical_value"}
    out: list[ApiFormatFinding] = []
    for finding in findings:
        codes = [str(code) for code in finding.get("issue_codes", [])]
        if not any(code in warning_codes for code in codes):
            continue
        out.append(
            ApiFormatFinding(
                page=str(finding.get("page")) if finding.get("page") is not None else None,
                fact_index=int(finding.get("fact_index")) if finding.get("fact_index") is not None else None,
                issue_codes=codes,
                message=str(finding.get("message")) if finding.get("message") is not None else None,
            )
        )
    return out


class WorkspaceWebService:
    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT) -> None:
        self.data_root = Path(data_root)

    def app_config(self, *, startup_doc_id: str | None, frontend_dist: Path | None) -> ApiAppConfig:
        return ApiAppConfig.build(
            startup_doc_id=startup_doc_id,
            frontend_dist=frontend_dist,
            data_root=self.data_root,
        )

    def list_documents(self) -> list[ApiWorkspaceDocumentSummary]:
        return [self._to_api_summary(doc) for doc in discover_workspace_documents(self.data_root)]

    def get_document(self, doc_id: str) -> ApiDocumentDetail:
        summary = self._get_summary(doc_id)
        page_images = page_image_paths(summary.images_dir)
        if not page_images:
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' has no prepared page images.")

        payload = load_annotation_payload(summary.annotations_path)
        document_meta = normalize_document_meta(extract_document_meta(payload))
        page_states = load_page_states(payload, [page.name for page in page_images])
        ordered_states = self._ordered_page_states(page_images, page_states, document_meta)
        issues = validate_document_issues([(page.name, ordered_states[page.name]) for page in page_images])
        refreshed_summary = self._refresh_summary(summary, issues)

        pages: list[ApiDocumentPage] = []
        for index, page_path in enumerate(page_images):
            state = ordered_states.get(page_path.name, self._default_state(index))
            width, height = self._image_size(page_path)
            pages.append(
                ApiDocumentPage(
                    image=page_path.name,
                    image_path=str(page_path.resolve()),
                    width=width,
                    height=height,
                    meta=PageMeta(**{**self._default_state(index).meta, **(state.meta or {})}),
                    facts=[self._to_api_fact(record) for record in state.facts],
                    annotated=page_has_annotation(state, index),
                    issue_summary=_to_api_issue_summary(
                        issues.page_summaries.get(page_path.name, PageIssueSummary(page_image=page_path.name))
                    ),
                )
            )

        return ApiDocumentDetail(
            doc_id=refreshed_summary.doc_id,
            source_pdf=str(refreshed_summary.source_pdf.resolve()) if refreshed_summary.source_pdf else None,
            images_dir=str(refreshed_summary.images_dir.resolve()),
            annotations_path=str(refreshed_summary.annotations_path.resolve()),
            document_meta=ApiDocumentMeta.model_validate(document_meta),
            pages=pages,
            issue_summary=_to_api_document_issue_summary(issues),
            checked=bool(refreshed_summary.checked),
            reviewed=bool(refreshed_summary.reviewed),
            status=refreshed_summary.status,
            annotated_page_count=int(refreshed_summary.annotated_page_count),
            page_count=int(refreshed_summary.page_count),
            progress_pct=int(refreshed_summary.progress_pct),
            updated_at=refreshed_summary.updated_at,
        )

    def save_document(self, doc_id: str, request: ApiDocumentSaveRequest) -> ApiDocumentSaveResponse:
        summary = self._get_summary(doc_id)
        page_images = page_image_paths(summary.images_dir)
        if not page_images:
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' has no prepared page images.")

        payload, warning_findings = self._build_payload_from_request(summary.images_dir, page_images, request)
        serialized = serialize_annotations_json(payload)
        existing = ""
        if summary.annotations_path.is_file():
            try:
                existing = summary.annotations_path.read_text(encoding="utf-8")
            except OSError:
                existing = ""
        changed = existing != serialized
        summary.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        summary.annotations_path.write_text(serialized, encoding="utf-8")
        document = self.get_document(doc_id)
        document.save_warnings = warning_findings
        return ApiDocumentSaveResponse(document=document, changed=changed, save_warnings=warning_findings)

    def validate_document(self, doc_id: str, request: ApiDocumentSaveRequest) -> ApiDocumentValidateResponse:
        summary = self._get_summary(doc_id)
        page_images = page_image_paths(summary.images_dir)
        if not page_images:
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' has no prepared page images.")
        payload, warning_findings = self._build_payload_from_request(summary.images_dir, page_images, request)
        states = load_page_states(payload, [page.name for page in page_images])
        issues = validate_document_issues([(page.name, states.get(page.name, self._default_state(index))) for index, page in enumerate(page_images)])
        return ApiDocumentValidateResponse(
            issue_summary=_to_api_document_issue_summary(issues),
            save_warnings=warning_findings,
        )

    def import_json(self, doc_id: str, payload: Any, *, default_page_image: str | None, normalized_1000: bool) -> ApiDocumentDetail:
        summary = self._get_summary(doc_id)
        page_images = page_image_paths(summary.images_dir)
        if not page_images:
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' has no prepared page images.")
        current_payload = load_annotation_payload(summary.annotations_path)
        current_states = load_page_states(current_payload, [page.name for page in page_images])
        imported_states = parse_import_payload(
            payload,
            [page.name for page in page_images],
            default_page_image_name=default_page_image or (page_images[0].name if page_images else None),
        )
        if not imported_states:
            raise HTTPException(status_code=400, detail="Import payload did not match any document pages.")
        for page_name, state in imported_states.items():
            converted_facts: list[BoxRecord] = []
            width, height = self._image_size(summary.images_dir / page_name)
            for record in state.facts:
                bbox = normalize_bbox_data(record.bbox)
                if normalized_1000 and self._bbox_looks_normalized_1000(bbox):
                    bbox = denormalize_bbox_from_1000(bbox, width, height)
                converted_facts.append(BoxRecord(bbox=bbox, fact=normalize_fact_data(record.fact)))
            page_index = self._page_index(page_images, page_name)
            current_states[page_name] = PageState(
                meta={**self._default_state(page_index).meta, **(state.meta or {})},
                facts=converted_facts,
            )
        imported_document_meta = normalize_document_meta(extract_document_meta(payload))
        if any(value is not None for value in imported_document_meta.values()):
            document_meta = imported_document_meta
        else:
            document_meta = normalize_document_meta(extract_document_meta(current_payload))
        ordered_states = self._ordered_page_states(page_images, current_states, document_meta)
        final_payload = build_annotations_payload(
            summary.images_dir,
            page_images,
            ordered_states,
            document_meta=document_meta,
        )
        summary.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        summary.annotations_path.write_text(serialize_annotations_json(final_payload), encoding="utf-8")
        return self.get_document(doc_id)

    def apply_entity_name(self, doc_id: str, *, entity_name: str, overwrite_existing: bool) -> ApiDocumentDetail:
        summary = self._get_summary(doc_id)
        page_images = page_image_paths(summary.images_dir)
        if not page_images:
            raise HTTPException(status_code=409, detail=f"Document '{doc_id}' has no prepared page images.")
        normalized_entity = str(entity_name or "").strip()
        if not normalized_entity:
            raise HTTPException(status_code=400, detail="entity_name must be non-empty.")
        payload = load_annotation_payload(summary.annotations_path)
        document_meta = normalize_document_meta(extract_document_meta(payload))
        page_states = load_page_states(payload, [page.name for page in page_images])
        apply_entity_name_to_pages(
            page_states,
            page_images,
            normalized_entity,
            overwrite_existing=overwrite_existing,
        )
        ordered_states = self._ordered_page_states(page_images, page_states, document_meta)
        final_payload = build_annotations_payload(
            summary.images_dir,
            page_images,
            ordered_states,
            document_meta=document_meta,
        )
        summary.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        summary.annotations_path.write_text(serialize_annotations_json(final_payload), encoding="utf-8")
        return self.get_document(doc_id)

    def set_checked(self, doc_id: str, value: bool) -> ApiWorkspaceDocumentSummary:
        self._get_summary(doc_id)
        set_document_checked(doc_id, bool(value), data_root=self.data_root)
        return self._to_api_summary(build_document_summary(doc_id, self.data_root))

    def set_reviewed(self, doc_id: str, value: bool) -> ApiWorkspaceDocumentSummary:
        self._get_summary(doc_id)
        set_document_reviewed(doc_id, bool(value), data_root=self.data_root)
        return self._to_api_summary(build_document_summary(doc_id, self.data_root))

    def prepare_document(self, doc_id: str, *, dpi: int = 200) -> ApiDocumentDetail:
        summary = self._get_summary(doc_id)
        if summary.source_pdf is None or not summary.source_pdf.is_file():
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' does not have a managed PDF.")
        ensure_pdf_images(summary.source_pdf, summary.images_dir, dpi=dpi)
        return self.get_document(doc_id)

    def import_pdf_upload(self, *, filename: str, content_b64: str, dpi: int = 200) -> ApiDocumentDetail:
        try:
            pdf_bytes = base64.b64decode(content_b64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 upload: {exc}") from exc
        safe_name = Path(filename or "document.pdf").name or "document.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        with tempfile.TemporaryDirectory(prefix="finetree-upload-") as tmp_dir:
            temp_pdf = Path(tmp_dir) / safe_name
            temp_pdf.write_bytes(pdf_bytes)
            result = import_pdf_to_workspace(temp_pdf, data_root=self.data_root, dpi=dpi)
        return self.get_document(result.document.doc_id)

    def import_pdf_path(self, source_pdf: Path, *, dpi: int = 200) -> ApiDocumentDetail:
        result = import_pdf_to_workspace(Path(source_pdf).expanduser().resolve(), data_root=self.data_root, dpi=dpi)
        return self.get_document(result.document.doc_id)

    def import_image_directory(self, source_dir: Path) -> ApiDocumentDetail:
        source_dir = Path(source_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Image directory not found: {source_dir}")
        image_paths = page_image_paths(source_dir)
        if not image_paths:
            raise HTTPException(status_code=400, detail=f"No page images found in {source_dir}")
        doc_id = sanitize_doc_id(source_dir.name)
        target_dir = pdf_images_root(self.data_root) / doc_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            target_path = target_dir / image_path.name
            if not target_path.exists():
                target_path.write_bytes(image_path.read_bytes())
        annotations_root(self.data_root).mkdir(parents=True, exist_ok=True)
        return self.get_document(doc_id)

    def extract_page(self, doc_id: str, page_image: str, request: ApiExtractionRequest) -> ApiExtractionResponse:
        summary = self._get_summary(doc_id)
        page_path = summary.images_dir / page_image
        if not page_path.is_file():
            raise HTTPException(status_code=404, detail=f"Page image not found: {page_image}")
        prompt = self._prompt_text(page_path, request.prompt)
        model_name = self._resolve_model_name(request)
        few_shot_examples = self._load_few_shot_examples(request.few_shot_preset) if request.few_shot_enabled else None

        if request.provider == "gemini":
            raw_text = generate_gemini_content(
                image_path=page_path,
                prompt=prompt,
                model=model_name,
                few_shot_examples=few_shot_examples,
                enable_thinking=request.enable_thinking,
            )
        else:
            raw_text = generate_qwen_content(
                image_path=page_path,
                prompt=prompt,
                model=model_name,
                config_path=str(self._resolve_qwen_config_path()) if self._resolve_qwen_config_path() else None,
                few_shot_examples=few_shot_examples,
                enable_thinking=request.enable_thinking,
            )
        extraction = parse_page_extraction_text(raw_text)
        normalized = self._normalize_page_extraction(extraction, page_path)
        return ApiExtractionResponse(
            provider=request.provider,
            model=model_name,
            page_image=page_image,
            prompt=prompt,
            extraction=normalized,
        )

    def page_image_path(self, doc_id: str, page_image: str) -> Path:
        summary = self._get_summary(doc_id)
        path = summary.images_dir / page_image
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Page image not found: {page_image}")
        return path

    def _build_payload_from_request(
        self,
        images_dir: Path,
        page_images: list[Path],
        request: ApiDocumentSaveRequest,
    ) -> tuple[dict[str, Any], list[ApiFormatFinding]]:
        page_names = {page.name for page in page_images}
        states: dict[str, PageState] = {}
        for index, page in enumerate(request.pages):
            if page.image not in page_names:
                raise HTTPException(status_code=400, detail=f"Unknown page image '{page.image}'.")
            facts = [
                BoxRecord(
                    bbox=normalize_bbox_data(fact.bbox.model_dump(mode="json")),
                    fact=normalize_fact_data(fact.model_dump(mode="json", exclude={"bbox"})),
                )
                for fact in page.facts
            ]
            states[page.image] = PageState(
                meta=page.meta.model_dump(mode="json"),
                facts=facts,
            )
        ordered_states = self._ordered_page_states(page_images, states, request.document_meta.model_dump(mode="json"))
        try:
            payload = build_annotations_payload(
                images_dir,
                page_images,
                ordered_states,
                document_meta=request.document_meta.model_dump(mode="json"),
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _normalized_payload, findings = normalize_annotation_payload(payload)
        return payload, _warning_findings(findings)

    def _get_summary(self, doc_id: str) -> WorkspaceDocumentSummary:
        summary = build_document_summary(doc_id, self.data_root)
        if not summary.images_dir.exists() and not summary.annotations_path.exists() and not summary.source_pdf:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' was not found.")
        return summary

    def _refresh_summary(
        self,
        summary: WorkspaceDocumentSummary,
        issues: DocumentIssueSummary,
    ) -> WorkspaceDocumentSummary:
        reviewed = bool(summary.reviewed) and int(issues.reg_flag_count) == 0
        return replace(
            summary,
            reg_flag_count=int(issues.reg_flag_count),
            warning_count=int(issues.warning_count),
            pages_with_reg_flags=int(issues.pages_with_reg_flags),
            pages_with_warnings=int(issues.pages_with_warnings),
            reviewed=reviewed,
        )

    def _to_api_summary(self, summary: WorkspaceDocumentSummary) -> ApiWorkspaceDocumentSummary:
        return ApiWorkspaceDocumentSummary(
            doc_id=summary.doc_id,
            source_pdf=str(summary.source_pdf.resolve()) if summary.source_pdf else None,
            images_dir=str(summary.images_dir.resolve()),
            annotations_path=str(summary.annotations_path.resolve()),
            thumbnail_path=str(summary.thumbnail_path.resolve()) if summary.thumbnail_path else None,
            page_count=int(summary.page_count),
            annotated_page_count=int(summary.annotated_page_count),
            progress_pct=int(summary.progress_pct),
            status=summary.status,
            updated_at=summary.updated_at,
            annotated_token_count=int(summary.annotated_token_count),
            reg_flag_count=int(summary.reg_flag_count),
            warning_count=int(summary.warning_count),
            pages_with_reg_flags=int(summary.pages_with_reg_flags),
            pages_with_warnings=int(summary.pages_with_warnings),
            checked=bool(summary.checked),
            reviewed=bool(summary.reviewed),
        )

    def _to_api_fact(self, record: BoxRecord) -> ApiFact:
        return ApiFact.model_validate(
            {
                "bbox": record.bbox,
                **normalize_fact_data(record.fact),
            }
        )

    def _default_state(self, index: int) -> PageState:
        return PageState(
            meta=PageMeta(type=PageType.other).model_dump(mode="json") | {"entity_name": None, "page_num": None, "title": None},
            facts=[],
        )

    def _page_index(self, page_images: list[Path], page_name: str) -> int:
        for index, page_image in enumerate(page_images):
            if page_image.name == page_name:
                return index
        return -1

    def _ordered_page_states(
        self,
        page_images: list[Path],
        states: dict[str, PageState],
        document_meta: dict[str, Any],
    ) -> dict[str, PageState]:
        ordered: dict[str, PageState] = {}
        normalized_meta = normalize_document_meta(document_meta)
        for index, page_path in enumerate(page_images):
            state = states.get(page_path.name, self._default_state(index))
            ordered[page_path.name] = self._ordered_state_for_page(page_path.name, state, normalized_meta)
        return ordered

    def _ordered_state_for_page(
        self,
        page_name: str,
        state: PageState,
        document_meta: dict[str, Any],
    ) -> PageState:
        facts = list(state.facts or [])
        if len(facts) <= 1:
            return PageState(meta=dict(state.meta or {}), facts=facts)
        direction = resolve_reading_direction(
            document_meta,
            payload={
                "pages": [
                    {
                        "image": page_name,
                        "meta": dict(state.meta or {}),
                        "facts": [
                            {
                                "bbox": record.bbox,
                                **normalize_fact_data(record.fact),
                            }
                            for record in facts
                        ],
                    }
                ]
            },
            default_direction="rtl",
        )["direction"]
        order = canonical_fact_order_indices(
            [{"bbox": record.bbox} for record in facts],
            direction="rtl" if direction == "rtl" else "ltr",
            row_tolerance_ratio=0.35,
            row_tolerance_min_px=6.0,
        )
        return PageState(meta=dict(state.meta or {}), facts=[facts[idx] for idx in order if 0 <= idx < len(facts)])

    def _image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return int(image.width), int(image.height)

    @staticmethod
    def _bbox_looks_normalized_1000(bbox: dict[str, Any]) -> bool:
        try:
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            w = float(bbox.get("w", 0.0))
            h = float(bbox.get("h", 0.0))
        except Exception:
            return False
        limit = 1000.0 + 1e-6
        return (
            0.0 <= x <= limit
            and 0.0 <= y <= limit
            and 0.0 <= w <= limit
            and 0.0 <= h <= limit
            and (x + w) <= limit
            and (y + h) <= limit
        )

    def _normalize_page_extraction(self, extraction: PageExtraction, page_path: Path) -> ApiPageExtraction:
        width, height = self._image_size(page_path)
        facts: list[ApiFact] = []
        for fact in extraction.facts:
            bbox = normalize_bbox_data(fact.bbox.model_dump(mode="json"))
            if self._bbox_looks_normalized_1000(bbox):
                bbox = denormalize_bbox_from_1000(bbox, width, height)
            facts.append(
                ApiFact.model_validate(
                    {
                        "bbox": bbox,
                        **fact.model_dump(mode="json", exclude={"bbox"}),
                    }
                )
            )
        return ApiPageExtraction(meta=extraction.meta, facts=facts)

    def _prompt_text(self, page_path: Path, prompt_override: str | None) -> str:
        raw_text = str(prompt_override or "").strip()
        if not raw_text:
            prompt_path = self._resolve_prompt_path()
            if prompt_path is not None and prompt_path.is_file():
                raw_text = prompt_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            raw_text = default_extraction_prompt_template()
        raw_text = raw_text.replace("{{PAGE_IMAGE}}", str(page_path))
        raw_text = raw_text.replace("{{IMAGE_NAME}}", page_path.name)
        return raw_text

    def _resolve_prompt_path(self) -> Path | None:
        candidates = [
            Path.cwd() / "prompts" / "extraction_prompt.txt",
            Path(__file__).resolve().parents[3] / "prompts" / "extraction_prompt.txt",
            Path.cwd() / "prompt.txt",
            Path(__file__).resolve().parents[3] / "prompt.txt",
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    def _resolve_qwen_config_path(self) -> Path | None:
        candidates = [
            Path.cwd() / "configs" / "qwen_ui_runpod_queue.yaml",
            Path.cwd() / "configs" / "finetune_qwen35a3_vl.yaml",
            Path(__file__).resolve().parents[3] / "configs" / "qwen_ui_runpod_queue.yaml",
            Path(__file__).resolve().parents[3] / "configs" / "finetune_qwen35a3_vl.yaml",
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    def _resolve_model_name(self, request: ApiExtractionRequest) -> str:
        candidate = str(request.model or "").strip()
        if candidate:
            return candidate
        if request.provider == "gemini":
            return "gemini-3-flash-preview"
        return "qwen-flash-gt"

    def _load_few_shot_examples(self, preset: str | None) -> list[dict[str, Any]]:
        repo_roots = build_repo_roots(cwd=Path.cwd(), anchor_file=Path(__file__).resolve())
        if preset == FEW_SHOT_PRESET_EXTENDED:
            examples, _warnings = load_complex_few_shot_examples(
                repo_roots=repo_roots,
                selections=DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
            )
            return examples
        examples, _warnings = load_test_pdf_few_shot_examples(
            repo_roots=repo_roots,
            page_names=DEFAULT_TEST_FEW_SHOT_PAGES,
        )
        return examples
