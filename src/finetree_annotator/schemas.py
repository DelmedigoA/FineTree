from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .date_normalization import normalize_date


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
    if lowered in {"ratio", "count"}:
        return lowered
    raise ValueError("value_type must be 'amount', 'percent', 'ratio', 'count', or null.")


def _normalize_period_type_value(value: Any) -> str | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"instant", "duration"}:
        return text
    raise ValueError("period_type must be 'instant', 'duration', or null.")


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


class ValueType(str, Enum):
    amount = "amount"
    percent = "percent"
    ratio = "ratio"
    count = "count"


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


class PeriodType(str, Enum):
    instant = "instant"
    duration = "duration"


class PathSource(str, Enum):
    observed = "observed"
    inferred = "inferred"


class PageMeta(BaseModel):
    entity_name: Optional[str] = None
    page_num: Optional[str] = None
    page_type: PageType = PageType.other
    statement_type: Optional[StatementType] = None
    title: Optional[str] = None
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

    @property
    def type(self) -> StatementType | PageType:
        if self.statement_type is not None:
            return self.statement_type
        return self.page_type


class Fact(BaseModel):
    value: str
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
    path: List[str]
    path_source: Optional[PathSource] = None
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    value_type: Optional[ValueType] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("comment_ref", mode="before")
    @classmethod
    def _normalize_comment_ref(cls, value: Any) -> str | None:
        return _normalize_note_reference_value(value)

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

    @field_validator("path_source", mode="before")
    @classmethod
    def _validate_path_source(cls, value: Any) -> str | None:
        return _normalize_path_source_value(value)

    @field_validator("value_type", mode="before")
    @classmethod
    def _validate_value_type(cls, value: Any) -> str | None:
        return _normalize_value_type_value(value)

    @model_validator(mode="after")
    def _validate_note_num_requires_note_flag(self) -> "Fact":
        if self.note_num is not None and not self.note_flag:
            raise ValueError("note_num requires note_flag=true.")
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

    @field_validator("entity_type", mode="before")
    @classmethod
    def _normalize_entity_type(cls, value: Any) -> str | None:
        return _normalize_entity_type_value(value)


DocumentMeta = Metadata


class Document(BaseModel):
    images_dir: Optional[str] = None
    metadata: Optional[Metadata] = Field(default=None, validation_alias=AliasChoices("metadata", "document_meta"))
    pages: List[Page] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


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
