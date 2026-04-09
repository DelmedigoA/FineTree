from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal


ComparisonType = Literal["hard", "soft"]
ProviderName = Literal["gemini", "finetree_vllm"]


@dataclass(frozen=True)
class MetaFieldSpec:
    field: str
    comparison_type: ComparisonType


@dataclass(frozen=True)
class FactFieldSpec:
    field: str
    primary: bool
    score_formula: str


@dataclass(frozen=True)
class FieldExactMetrics:
    matches: int
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass
class MetaFieldResult:
    field: str
    comparison_type: ComparisonType
    gt_value: Any
    pred_value: Any
    score: float
    is_exact_match: bool


@dataclass
class FactFieldResult:
    field: str
    score: float
    exact_metrics: FieldExactMetrics
    string_similarity: float | None = None
    numeric_mae: float | None = None
    date_mae_days: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    doc_id: str
    page_index: int
    image_name: str
    meta_result: dict[str, MetaFieldResult]
    facts_result: dict[str, FactFieldResult]
    gt_fact_count: int
    pred_fact_count: int
    facts_applicable: bool
    facts_status: str
    meta_score: float
    facts_score: float
    page_score: float
    row_diff_diagnostics: list[dict[str, Any]]


@dataclass
class DocumentResult:
    doc_id: str
    meta_score: float
    facts_score: float
    document_score: float
    pages: list[PageResult]


@dataclass
class RunResult:
    run_score: float
    meta_score: float
    facts_score: float
    documents: list[DocumentResult]


@dataclass(frozen=True)
class PageInput:
    page_index: int
    image_name: str
    image_path: Path


@dataclass(frozen=True)
class DocumentInput:
    doc_id: str
    annotation_path: Path
    images_dir: Path
    pages: tuple[PageInput, ...]


@dataclass
class DatasetVersionInfo:
    version_id: str
    name: str
    created_at: float
    updated_at: float | None
    split_assignments: dict[str, str]
    split_stats: dict[str, dict[str, int]]
    export_config: dict[str, Any]
    path: Path


@dataclass
class ProviderPageOutput:
    page_index: int
    page_name: str
    assistant_text: str
    parsed_page: dict[str, Any] | None
    error: str | None
    received_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderDocumentOutput:
    doc_id: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    received_tokens: int
    fact_count: int
    page_outputs: list[ProviderPageOutput]
    failures: list[dict[str, str]]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderRunOutput:
    provider: ProviderName
    documents: list[ProviderDocumentOutput]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    phase: str
    provider: str
    dataset_version_id: str | None
    dataset_name: str | None
    split: str | None
    current_doc_id: str | None
    total_documents: int
    completed_documents: int
    total_pages: int
    completed_pages: int
    failed_pages: int
    fact_count: int
    total_tokens_received: int
    elapsed_seconds: float
    tokens_per_second: float
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value
