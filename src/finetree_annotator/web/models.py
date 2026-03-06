from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..fact_ordering import normalize_document_meta
from ..schema_contract import schema_snapshot
from ..schemas import BBox, ExtractedFact, PageMeta


class ApiDocumentMeta(BaseModel):
    language: str | None = None
    reading_direction: str | None = None
    company_name: str | None = None
    company_id: str | None = None
    report_year: int | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> dict[str, Any]:
        return normalize_document_meta(value)


class ApiBBox(BBox):
    pass


class ApiFact(ExtractedFact):
    bbox: ApiBBox


class ApiIssue(BaseModel):
    severity: str
    code: str
    message: str
    page_image: str
    fact_index: int | None = None
    field_name: str | None = None
    model_config = ConfigDict(extra="forbid")


class ApiPageIssueSummary(BaseModel):
    page_image: str
    reg_flag_count: int = 0
    warning_count: int = 0
    issues: list[ApiIssue] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiDocumentIssueSummary(BaseModel):
    reg_flag_count: int = 0
    warning_count: int = 0
    pages_with_reg_flags: int = 0
    pages_with_warnings: int = 0
    page_summaries: dict[str, ApiPageIssueSummary] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ApiFormatFinding(BaseModel):
    page: str | None = None
    fact_index: int | None = None
    issue_codes: list[str] = Field(default_factory=list)
    message: str | None = None
    model_config = ConfigDict(extra="forbid")


class ApiWorkspaceDocumentSummary(BaseModel):
    doc_id: str
    source_pdf: str | None = None
    images_dir: str
    annotations_path: str
    thumbnail_path: str | None = None
    page_count: int
    annotated_page_count: int
    progress_pct: int
    status: str
    updated_at: float | None = None
    annotated_token_count: int = 0
    reg_flag_count: int = 0
    warning_count: int = 0
    pages_with_reg_flags: int = 0
    pages_with_warnings: int = 0
    checked: bool = False
    reviewed: bool = False
    model_config = ConfigDict(extra="forbid")


class ApiDocumentPage(BaseModel):
    image: str
    image_path: str
    width: int
    height: int
    meta: PageMeta
    facts: list[ApiFact] = Field(default_factory=list)
    annotated: bool = False
    issue_summary: ApiPageIssueSummary
    model_config = ConfigDict(extra="forbid")


class ApiDocumentDetail(BaseModel):
    doc_id: str
    source_pdf: str | None = None
    images_dir: str
    annotations_path: str
    document_meta: ApiDocumentMeta
    pages: list[ApiDocumentPage] = Field(default_factory=list)
    issue_summary: ApiDocumentIssueSummary
    checked: bool = False
    reviewed: bool = False
    status: str
    annotated_page_count: int
    page_count: int
    progress_pct: int
    updated_at: float | None = None
    save_warnings: list[ApiFormatFinding] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiDocumentSaveRequest(BaseModel):
    document_meta: ApiDocumentMeta = Field(default_factory=ApiDocumentMeta)
    pages: list["ApiPageInput"] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiPageInput(BaseModel):
    image: str
    meta: PageMeta
    facts: list[ApiFact] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiDocumentSaveResponse(BaseModel):
    document: ApiDocumentDetail
    changed: bool
    save_warnings: list[ApiFormatFinding] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiDocumentValidateResponse(BaseModel):
    issue_summary: ApiDocumentIssueSummary
    save_warnings: list[ApiFormatFinding] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiImportJsonRequest(BaseModel):
    payload: Any
    normalized_1000: bool = True
    default_page_image: str | None = None
    model_config = ConfigDict(extra="forbid")


class ApiApplyEntityRequest(BaseModel):
    entity_name: str
    overwrite_existing: bool = False
    model_config = ConfigDict(extra="forbid")


class ApiBooleanToggleRequest(BaseModel):
    value: bool
    model_config = ConfigDict(extra="forbid")


class ApiUploadPdfRequest(BaseModel):
    filename: str
    content_b64: str
    dpi: int = 200
    model_config = ConfigDict(extra="forbid")


class ApiExtractionRequest(BaseModel):
    provider: Literal["gemini", "qwen"]
    prompt: str | None = None
    model: str | None = None
    few_shot_enabled: bool = False
    few_shot_preset: str | None = None
    enable_thinking: bool | None = None
    model_config = ConfigDict(extra="forbid")


class ApiExtractionResponse(BaseModel):
    provider: Literal["gemini", "qwen"]
    model: str
    page_image: str
    prompt: str
    extraction: "ApiPageExtraction"
    model_config = ConfigDict(extra="forbid")


class ApiPageExtraction(BaseModel):
    meta: PageMeta
    facts: list[ApiFact] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ApiSchemaResponse(BaseModel):
    schema_payload: dict[str, Any] = Field(alias="schema")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @classmethod
    def build(cls) -> "ApiSchemaResponse":
        return cls(schema_payload=schema_snapshot())


class ApiAppConfig(BaseModel):
    startup_doc_id: str | None = None
    frontend_dist: str | None = None
    data_root: str
    schema_payload: dict[str, Any] = Field(alias="schema")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @classmethod
    def build(
        cls,
        *,
        startup_doc_id: str | None,
        frontend_dist: Path | None,
        data_root: Path,
    ) -> "ApiAppConfig":
        frontend_value = str(frontend_dist.resolve()) if frontend_dist is not None else None
        return cls(
            startup_doc_id=startup_doc_id,
            frontend_dist=frontend_value,
            data_root=str(Path(data_root).resolve()),
            schema_payload=schema_snapshot(),
        )


ApiDocumentSaveRequest.model_rebuild()
ApiExtractionResponse.model_rebuild()
