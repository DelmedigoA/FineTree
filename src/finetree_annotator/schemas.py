from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .date_normalization import normalize_date

CURRENT_SCHEMA_VERSION = 2


def _normalize_note_reference_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_note_name_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_page_annotation_note_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_page_annotation_status_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    aliases = {
        "approved": "approved",
        "approve": "approved",
        "flagged": "flagged",
        "flag": "flagged",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError("annotation_status must be 'approved', 'flagged', or null.")
    return normalized


def _normalize_equation_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip()
    return text or None


def _normalize_fact_num_value(value: Any) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        raise ValueError("fact_num must be an integer >= 1 or null.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("fact_num must be an integer >= 1 or null.")
        return value
    if isinstance(value, float) and float(value).is_integer():
        parsed = int(value)
        if parsed < 1:
            raise ValueError("fact_num must be an integer >= 1 or null.")
        return parsed
    text = str(value).strip()
    if text.isdigit():
        parsed = int(text)
        if parsed < 1:
            raise ValueError("fact_num must be an integer >= 1 or null.")
        return parsed
    raise ValueError("fact_num must be an integer >= 1 or null.")


def _normalize_note_num_value(value: Any) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        raise ValueError("note_num must be an integer or null.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    raise ValueError("note_num must be an integer or null.")


def _normalize_path_value(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("path must be a list of strings.")
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ValueError("path entries must be non-empty strings.")
        out.append(text)
    return out


def _validate_canonical_date(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip()
    normalized, warnings = normalize_date(text)
    if warnings or normalized != text:
        raise ValueError("date must be YYYY, YYYY-MM, or YYYY-MM-DD.")
    return text


def _validate_day_date(value: Any) -> str | None:
    text = _validate_canonical_date(value)
    if text is None:
        return None
    if len(text) != 10:
        raise ValueError("period_start and period_end must be YYYY-MM-DD or null.")
    return text


def _normalize_doc_language_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"he", "en"}:
        return text
    raise ValueError("language must be 'he', 'en', or null.")


def _normalize_reading_direction_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"rtl", "ltr"}:
        return text
    raise ValueError("reading_direction must be 'rtl', 'ltr', or null.")


def _normalize_company_id_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_company_name_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_report_year_value(value: Any) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        raise ValueError("report_year must be an integer or null.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    raise ValueError("report_year must be an integer or null.")


def _normalize_schema_version_value(value: Any) -> int:
    if value in ("", None):
        return CURRENT_SCHEMA_VERSION
    if isinstance(value, bool):
        raise ValueError("schema_version must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    raise ValueError("schema_version must be an integer.")


def _normalize_entity_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    aliases = {
        "state_owned_enterprise": "state_owned_enterprise",
        "public_company": "public_company",
        "private_company": "private_company",
        "registered_nonprofit": "registered_nonprofit",
        "nonprofit_npo": "nonprofit_npo",
        "public_benefit_company": "public_benefit_company",
        "partnership": "partnership",
        "limited_partnership": "limited_partnership",
        "limited_liability_company": "limited_liability_company",
        "other": "other",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("entity_type must be a supported enum value or null.")


def _normalize_report_scope_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "null":
        return None
    normalized = lowered.replace(" ", "_").replace("-", "_")
    aliases = {
        "proforma": "pro_forma",
        "pro_forma": "pro_forma",
        "pro forma": "pro_forma",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "separate",
        "consolidated",
        "combined",
        "pro_forma",
        "other",
    }
    if normalized in allowed:
        return normalized
    raise ValueError(
        "report_scope must be separate, consolidated, combined, pro_forma, other, or null."
    )


def _normalize_page_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    aliases = {
        "title": "title",
        "contents": "contents",
        "contents_page": "contents",
        "declaration": "declaration",
        "statements": "statements",
        "statement": "statements",
        "other": "other",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("page_type must be 'title', 'contents', 'declaration', 'statements', 'other', or null.")


def _normalize_statement_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    aliases = {
        "balance_sheet": "balance_sheet",
        "income_statement": "income_statement",
        "profits": "income_statement",
        "cash_flow": "cash_flow_statement",
        "cash_flow_statement": "cash_flow_statement",
        "statement_of_changes_in_equity": "statement_of_changes_in_equity",
        "notes": "notes_to_financial_statements",
        "notes_to_financial_statements": "notes_to_financial_statements",
        "board_of_directors_report": "board_of_directors_report",
        "auditor_report": "auditors_report",
        "auditors_report": "auditors_report",
        "statement_of_activities": "statement_of_activities",
        "activities": "statement_of_activities",
        "other_declaration": "other_declaration",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("statement_type must be a supported enum value or null.")


def _normalize_value_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"%", "percent", "percentage"}:
        return "percent"
    if lowered in {"amount", "regular"}:
        return "amount"
    if lowered in {"ratio", "count", "points"}:
        return lowered
    raise ValueError("value_type must be 'amount', 'percent', 'ratio', 'count', 'points', or null.")


def _normalize_value_context_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"textual", "tabular", "mixed"}:
        return text
    raise ValueError("value_context must be 'textual', 'tabular', 'mixed', or null.")


def _normalize_balance_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"debit", "credit"}:
        return text
    raise ValueError("balance_type must be 'debit', 'credit', or null.")


def _normalize_natural_sign_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    aliases = {
        "plus": "positive",
        "+": "positive",
        "positive": "positive",
        "minus": "negative",
        "-": "negative",
        "negative": "negative",
    }
    normalized = aliases.get(text)
    if normalized is not None:
        return normalized
    raise ValueError("natural_sign must be 'positive', 'negative', or null.")


def _normalize_aggregation_role_value(value: Any) -> str | None:
    if value in ("", None):
        return "additive"
    text = str(value).strip().lower()
    aliases = {
        "+": "additive",
        "plus": "additive",
        "additive": "additive",
        "-": "subtractive",
        "minus": "subtractive",
        "subtractive": "subtractive",
        # Backward compatibility with legacy polarity buckets.
        "total": "additive",
        "unknown": "additive",
    }
    normalized = aliases.get(text)
    if normalized is not None:
        return normalized
    raise ValueError("aggregation_role must be 'additive' or 'subtractive'.")


def _normalize_row_role_value(value: Any) -> str | None:
    if value in ("", None):
        return "detail"
    text = str(value).strip().lower()
    aliases = {
        "detail": "detail",
        "details": "detail",
        "child": "detail",
        "line": "detail",
        "total": "total",
        "subtotal": "total",
        "net": "total",
        "summary": "total",
    }
    normalized = aliases.get(text)
    if normalized is not None:
        return normalized
    raise ValueError("row_role must be 'detail' or 'total'.")


def _derive_natural_sign_from_value(value: str) -> str | None:
    text = str(value or "").strip()
    if text == "-":
        return None
    if "(" in text and ")" in text:
        return "negative"
    return "positive"


def _normalize_required_value(value: Any) -> str:
    if value is None:
        raise ValueError("value must be a non-empty string.")
    text = str(value).strip()
    if not text:
        raise ValueError("value must be a non-empty string.")
    return text


def _normalize_period_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"instant", "duration", "expected"}:
        return text
    raise ValueError("period_type must be 'instant', 'duration', 'expected', or null.")


def _normalize_duration_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"recurring", "recurrent"}:
        return "recurrent"
    raise ValueError("duration_type must be 'recurrent' or null.")


def _normalize_recurring_period_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"daily", "quarterly", "monthly", "yearly"}:
        return text
    raise ValueError("recurring_period must be daily, quarterly, monthly, yearly, or null.")


def _normalize_path_source_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"observed", "inferred"}:
        return text
    raise ValueError("path_source must be 'observed', 'inferred', or null.")


_LEGACY_PAGE_TYPE_MAPPING: dict[str, tuple[str, str | None]] = {
    "title": ("title", None),
    "contents_page": ("contents", None),
    "contents": ("contents", None),
    "declaration": ("declaration", None),
    "balance_sheet": ("statements", "balance_sheet"),
    "income_statement": ("statements", "income_statement"),
    "profits": ("statements", "income_statement"),
    "cash_flow": ("statements", "cash_flow_statement"),
    "cash_flow_statement": ("statements", "cash_flow_statement"),
    "activities": ("statements", "statement_of_activities"),
    "statement_of_activities": ("statements", "statement_of_activities"),
    "notes": ("statements", "notes_to_financial_statements"),
    "notes_to_financial_statements": ("statements", "notes_to_financial_statements"),
    "board_of_directors_report": ("statements", "board_of_directors_report"),
    "auditor_report": ("statements", "auditors_report"),
    "auditors_report": ("statements", "auditors_report"),
    "other_declaration": ("statements", "other_declaration"),
    "statement_of_changes_in_equity": ("statements", "statement_of_changes_in_equity"),
    "other": ("other", None),
}


def is_legacy_page_type_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text in _LEGACY_PAGE_TYPE_MAPPING


def split_legacy_page_type(value: Any) -> tuple[str, str | None]:
    text = str(value or "").strip().lower()
    if not text:
        return PageType.other.value, None
    return _LEGACY_PAGE_TYPE_MAPPING.get(text, (PageType.other.value, None))


class PageType(str, Enum):
    title = "title"
    contents = "contents"
    declaration = "declaration"
    statements = "statements"
    other = "other"


class StatementType(str, Enum):
    balance_sheet = "balance_sheet"
    income_statement = "income_statement"
    cash_flow_statement = "cash_flow_statement"
    statement_of_changes_in_equity = "statement_of_changes_in_equity"
    notes_to_financial_statements = "notes_to_financial_statements"
    board_of_directors_report = "board_of_directors_report"
    auditors_report = "auditors_report"
    statement_of_activities = "statement_of_activities"
    other_declaration = "other_declaration"


class ValueType(str, Enum):
    amount = "amount"
    percent = "percent"
    ratio = "ratio"
    count = "count"
    points = "points"


class ValueContext(str, Enum):
    textual = "textual"
    tabular = "tabular"
    mixed = "mixed"


class BalanceType(str, Enum):
    debit = "debit"
    credit = "credit"


class NaturalSign(str, Enum):
    positive = "positive"
    negative = "negative"


class AggregationRole(str, Enum):
    additive = "additive"
    subtractive = "subtractive"


class RowRole(str, Enum):
    detail = "detail"
    total = "total"


class Currency(str, Enum):
    ils = "ILS"
    usd = "USD"
    eur = "EUR"
    gbp = "GBP"


class Scale(int, Enum):
    one = 1
    thousand = 1000
    million = 1000000


class DocLanguage(str, Enum):
    hebrew = "he"
    english = "en"


class ReadingDirection(str, Enum):
    rtl = "rtl"
    ltr = "ltr"


class EntityType(str, Enum):
    state_owned_enterprise = "state_owned_enterprise"
    public_company = "public_company"
    private_company = "private_company"
    registered_nonprofit = "registered_nonprofit"
    nonprofit_npo = "nonprofit_npo"
    public_benefit_company = "public_benefit_company"
    partnership = "partnership"
    limited_partnership = "limited_partnership"
    limited_liability_company = "limited_liability_company"
    other = "other"


class ReportScope(str, Enum):
    separate = "separate"
    consolidated = "consolidated"
    combined = "combined"
    pro_forma = "pro_forma"
    other = "other"


class PeriodType(str, Enum):
    instant = "instant"
    duration = "duration"
    expected = "expected"


class DurationType(str, Enum):
    recurrent = "recurrent"


class RecurringPeriod(str, Enum):
    daily = "daily"
    quarterly = "quarterly"
    monthly = "monthly"
    yearly = "yearly"


class AnnotationStatus(str, Enum):
    approved = "approved"
    flagged = "flagged"


class PathSource(str, Enum):
    observed = "observed"
    inferred = "inferred"


class PageMeta(BaseModel):
    entity_name: Optional[str] = None
    page_num: Optional[str] = None
    page_type: PageType = PageType.other
    statement_type: Optional[StatementType] = None
    title: Optional[str] = None
    annotation_note: Optional[str] = None
    annotation_status: Optional[AnnotationStatus] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_page_type_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_page_type = data.get("page_type")
        raw_statement_type = data.get("statement_type")
        raw_legacy_type = data.get("type")

        if raw_page_type not in ("", None):
            try:
                page_type = _normalize_page_type_value(raw_page_type)
            except ValueError:
                if is_legacy_page_type_value(raw_page_type):
                    page_type, inferred_statement_type = split_legacy_page_type(raw_page_type)
                    data["page_type"] = page_type
                    if raw_statement_type in ("", None) and inferred_statement_type is not None:
                        data["statement_type"] = inferred_statement_type
            else:
                data["page_type"] = page_type
        elif raw_statement_type not in ("", None):
            data["page_type"] = PageType.statements.value

        if raw_statement_type in ("", None) and raw_legacy_type not in ("", None) and raw_page_type in ("", None, PageType.other.value):
            if is_legacy_page_type_value(raw_legacy_type):
                page_type, statement_type = split_legacy_page_type(raw_legacy_type)
                data["page_type"] = page_type
                if statement_type is not None:
                    data["statement_type"] = statement_type
            else:
                data["page_type"] = raw_legacy_type
        elif raw_page_type in ("", None) and raw_statement_type in ("", None):
            if is_legacy_page_type_value(raw_legacy_type):
                page_type, statement_type = split_legacy_page_type(raw_legacy_type)
                data["page_type"] = page_type
                if statement_type is not None:
                    data["statement_type"] = statement_type

        data.pop("type", None)
        return data

    @field_validator("page_type", mode="before")
    @classmethod
    def _normalize_page_type(cls, value: Any) -> str:
        normalized = _normalize_page_type_value(value)
        return normalized or PageType.other.value

    @field_validator("statement_type", mode="before")
    @classmethod
    def _normalize_statement_type(cls, value: Any) -> str | None:
        return _normalize_statement_type_value(value)

    @field_validator("annotation_note", mode="before")
    @classmethod
    def _normalize_annotation_note(cls, value: Any) -> str | None:
        return _normalize_page_annotation_note_value(value)

    @field_validator("annotation_status", mode="before")
    @classmethod
    def _normalize_annotation_status(cls, value: Any) -> str | None:
        return _normalize_page_annotation_status_value(value)

    @property
    def type(self) -> StatementType | PageType:
        if self.statement_type is not None:
            return self.statement_type
        return self.page_type


class Fact(BaseModel):
    value: str
    fact_num: Optional[int] = None
    equation: Optional[str] = None
    fact_equation: Optional[str] = None
    balance_type: Optional[BalanceType] = None
    natural_sign: Optional[NaturalSign] = None
    row_role: RowRole = RowRole.detail
    aggregation_role: AggregationRole = AggregationRole.additive
    comment_ref: Optional[str] = Field(default=None, validation_alias=AliasChoices("comment_ref", "ref_comment", "comment"))
    note_flag: bool = Field(
        default=False,
        validation_alias=AliasChoices("note_flag", "is_note", "is_beur", "beur"),
    )
    note_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("note_name", "beur_name"))
    note_num: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("note_num", "note", "beur_num", "beur_number"),
    )
    note_ref: Optional[str] = Field(default=None, validation_alias=AliasChoices("note_ref", "ref_note", "note_reference"))
    date: Optional[str] = None
    period_type: Optional[PeriodType] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    duration_type: Optional[DurationType] = None
    recurring_period: Optional[RecurringPeriod] = None
    path: List[str]
    path_source: Optional[PathSource] = None
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    value_type: Optional[ValueType] = None
    value_context: Optional[ValueContext] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_row_and_aggregation_roles(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_aggregation_role = str(data.get("aggregation_role") or "").strip().lower()
        raw_row_role = str(data.get("row_role") or "").strip()
        if not raw_row_role and raw_aggregation_role == "total":
            data["row_role"] = "total"
        return data

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> str:
        return _normalize_required_value(value)

    @field_validator("comment_ref", mode="before")
    @classmethod
    def _normalize_comment_ref(cls, value: Any) -> str | None:
        return _normalize_note_reference_value(value)

    @field_validator("equation", mode="before")
    @classmethod
    def _normalize_equation(cls, value: Any) -> str | None:
        return _normalize_equation_value(value)

    @field_validator("fact_equation", mode="before")
    @classmethod
    def _normalize_fact_equation(cls, value: Any) -> str | None:
        return _normalize_equation_value(value)

    @field_validator("fact_num", mode="before")
    @classmethod
    def _normalize_fact_num(cls, value: Any) -> int | None:
        return _normalize_fact_num_value(value)

    @field_validator("note_ref", mode="before")
    @classmethod
    def _normalize_note_ref(cls, value: Any) -> str | None:
        return _normalize_note_reference_value(value)

    @field_validator("note_name", mode="before")
    @classmethod
    def _normalize_note_name(cls, value: Any) -> str | None:
        return _normalize_note_name_value(value)

    @field_validator("note_num", mode="before")
    @classmethod
    def _normalize_note_num(cls, value: Any) -> int | None:
        return _normalize_note_num_value(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> list[str]:
        return _normalize_path_value(value)

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date(cls, value: Any) -> str | None:
        return _validate_canonical_date(value)

    @field_validator("period_type", mode="before")
    @classmethod
    def _validate_period_type(cls, value: Any) -> str | None:
        return _normalize_period_type_value(value)

    @field_validator("period_start", mode="before")
    @classmethod
    def _validate_period_start(cls, value: Any) -> str | None:
        return _validate_day_date(value)

    @field_validator("period_end", mode="before")
    @classmethod
    def _validate_period_end(cls, value: Any) -> str | None:
        return _validate_day_date(value)

    @field_validator("duration_type", mode="before")
    @classmethod
    def _validate_duration_type(cls, value: Any) -> str | None:
        return _normalize_duration_type_value(value)

    @field_validator("recurring_period", mode="before")
    @classmethod
    def _validate_recurring_period(cls, value: Any) -> str | None:
        return _normalize_recurring_period_value(value)

    @field_validator("path_source", mode="before")
    @classmethod
    def _validate_path_source(cls, value: Any) -> str | None:
        return _normalize_path_source_value(value)

    @field_validator("value_type", mode="before")
    @classmethod
    def _validate_value_type(cls, value: Any) -> str | None:
        return _normalize_value_type_value(value)

    @field_validator("value_context", mode="before")
    @classmethod
    def _validate_value_context(cls, value: Any) -> str | None:
        return _normalize_value_context_value(value)

    @field_validator("balance_type", mode="before")
    @classmethod
    def _validate_balance_type(cls, value: Any) -> str | None:
        return _normalize_balance_type_value(value)

    @field_validator("natural_sign", mode="before")
    @classmethod
    def _validate_natural_sign(cls, value: Any) -> str | None:
        return _normalize_natural_sign_value(value)

    @field_validator("aggregation_role", mode="before")
    @classmethod
    def _validate_aggregation_role(cls, value: Any) -> str | None:
        return _normalize_aggregation_role_value(value)

    @field_validator("row_role", mode="before")
    @classmethod
    def _validate_row_role(cls, value: Any) -> str | None:
        return _normalize_row_role_value(value)

    @model_validator(mode="after")
    def _validate_note_num_requires_note_flag(self) -> "Fact":
        if self.note_num is not None and not self.note_flag:
            raise ValueError("note_num requires note_flag=true.")
        derived_sign = _derive_natural_sign_from_value(self.value)
        self.natural_sign = NaturalSign(derived_sign) if derived_sign is not None else None
        return self

    @property
    def ref_comment(self) -> str | None:
        return self.comment_ref

    @property
    def ref_note(self) -> str | None:
        return self.note_ref


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_list_bbox(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return {"x": value[0], "y": value[1], "w": value[2], "h": value[3]}
        return value


class ExtractedFact(Fact):
    bbox: BBox


class Page(BaseModel):
    image: Optional[str] = None
    meta: PageMeta
    facts: List[ExtractedFact]
    model_config = ConfigDict(extra="forbid")


class Metadata(BaseModel):
    language: Optional[DocLanguage] = None
    reading_direction: Optional[ReadingDirection] = None
    company_name: Optional[str] = None
    company_id: Optional[str] = None
    report_year: Optional[int] = None
    report_scope: Optional[ReportScope] = None
    entity_type: Optional[EntityType] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> str | None:
        return _normalize_doc_language_value(value)

    @field_validator("reading_direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> str | None:
        return _normalize_reading_direction_value(value)

    @field_validator("company_name", mode="before")
    @classmethod
    def _normalize_company_name(cls, value: Any) -> str | None:
        return _normalize_company_name_value(value)

    @field_validator("company_id", mode="before")
    @classmethod
    def _normalize_company_id(cls, value: Any) -> str | None:
        return _normalize_company_id_value(value)

    @field_validator("report_year", mode="before")
    @classmethod
    def _normalize_report_year(cls, value: Any) -> int | None:
        return _normalize_report_year_value(value)

    @field_validator("report_scope", mode="before")
    @classmethod
    def _normalize_report_scope(cls, value: Any) -> str | None:
        return _normalize_report_scope_value(value)

    @field_validator("entity_type", mode="before")
    @classmethod
    def _normalize_entity_type(cls, value: Any) -> str | None:
        return _normalize_entity_type_value(value)


DocumentMeta = Metadata


class Document(BaseModel):
    schema_version: int = CURRENT_SCHEMA_VERSION
    images_dir: Optional[str] = None
    metadata: Optional[Metadata] = Field(default=None, validation_alias=AliasChoices("metadata", "document_meta"))
    pages: List[Page] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Any) -> int:
        return _normalize_schema_version_value(value)


class PageExtraction(Document):
    @model_validator(mode="after")
    def _validate_single_page(self) -> "PageExtraction":
        if len(self.pages) != 1:
            raise ValueError("PageExtraction requires exactly one page.")
        page = self.pages[0]
        if page.meta.statement_type != StatementType.notes_to_financial_statements and any(
            fact.note_flag for fact in page.facts
        ):
            raise ValueError("Facts with note_flag=true are only allowed on notes_to_financial_statements pages.")
        return self

    @property
    def page(self) -> Page:
        return self.pages[0]

    @property
    def meta(self) -> PageMeta:
        return self.page.meta

    @property
    def facts(self) -> List[ExtractedFact]:
        return self.page.facts
