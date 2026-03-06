from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .date_normalization import normalize_date


def _normalize_note_reference_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_note_name_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_note_num_value(value):
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


def _normalize_path_value(value):
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


def _validate_canonical_date(value):
    if value in ("", None):
        return None
    text = str(value).strip()
    normalized, warnings = normalize_date(text)
    if warnings or normalized != text:
        raise ValueError("date must be YYYY, YYYY-MM, or YYYY-MM-DD.")
    return text


def _normalize_doc_language_value(value):
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"he", "en"}:
        return text
    raise ValueError("language must be 'he', 'en', or null.")


def _normalize_reading_direction_value(value):
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"rtl", "ltr"}:
        return text
    raise ValueError("reading_direction must be 'rtl', 'ltr', or null.")


def _normalize_company_id_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_company_name_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_report_year_value(value):
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


class PageType(str, Enum):
    balance_sheet = "balance_sheet"
    income_statement = "income_statement"
    cash_flow = "cash_flow"
    activities = "activities"
    notes = "notes"
    contents_page = "contents_page"
    title = "title"
    declaration = "declaration"
    profits = "profits"
    other = "other"


class ValueType(str, Enum):
    percent = "%"
    regular = "amount"


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


class PageMeta(BaseModel):
    entity_name: Optional[str] = None
    page_num: Optional[str] = None
    type: PageType
    title: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Fact(BaseModel):
    value: str
    comment: Optional[str] = None
    note_flag: bool = Field(
        default=False,
        validation_alias=AliasChoices("note_flag", "is_note", "is_beur", "beur"),
    )
    note_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("note_name", "beur_name"))
    note_num: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("note_num", "note", "beur_num", "beur_number"),
    )
    note_reference: Optional[str] = None
    date: Optional[str] = None
    path: List[str]
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    value_type: Optional[ValueType] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("note_reference", mode="before")
    @classmethod
    def _normalize_note_reference(cls, value):
        return _normalize_note_reference_value(value)

    @field_validator("note_name", mode="before")
    @classmethod
    def _normalize_note_name(cls, value):
        return _normalize_note_name_value(value)

    @field_validator("note_num", mode="before")
    @classmethod
    def _normalize_note_num(cls, value):
        return _normalize_note_num_value(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value):
        return _normalize_path_value(value)

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date(cls, value):
        return _validate_canonical_date(value)


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_list_bbox(cls, value):
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return {"x": value[0], "y": value[1], "w": value[2], "h": value[3]}
        return value


class ExtractedFact(BaseModel):
    bbox: BBox
    value: str
    comment: Optional[str] = None
    note_flag: bool = Field(
        default=False,
        validation_alias=AliasChoices("note_flag", "is_note", "is_beur", "beur"),
    )
    note_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("note_name", "beur_name"))
    note_num: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("note_num", "note", "beur_num", "beur_number"),
    )
    note_reference: Optional[str] = None
    date: Optional[str] = None
    path: List[str]
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    value_type: Optional[ValueType] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("note_reference", mode="before")
    @classmethod
    def _normalize_note_reference(cls, value):
        return _normalize_note_reference_value(value)

    @field_validator("note_name", mode="before")
    @classmethod
    def _normalize_note_name(cls, value):
        return _normalize_note_name_value(value)

    @field_validator("note_num", mode="before")
    @classmethod
    def _normalize_note_num(cls, value):
        return _normalize_note_num_value(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value):
        return _normalize_path_value(value)

    @field_validator("date", mode="before")
    @classmethod
    def _validate_date(cls, value):
        return _validate_canonical_date(value)

    @model_validator(mode="after")
    def _validate_note_num_requires_is_note(self) -> "ExtractedFact":
        if self.note_num is not None and not self.note_flag:
            raise ValueError("note_num requires note_flag=true.")
        return self


class PageExtraction(BaseModel):
    meta: PageMeta
    facts: List[ExtractedFact]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_note_facts_match_page_type(self) -> "PageExtraction":
        if self.meta.type != PageType.notes and any(fact.note_flag for fact in self.facts):
            raise ValueError("Facts with note_flag=true are only allowed on notes pages.")
        return self


class Page(BaseModel):
    meta: PageMeta
    facts: List[Fact]
    model_config = ConfigDict(extra="forbid")


class DocumentMeta(BaseModel):
    language: Optional[DocLanguage] = None
    reading_direction: Optional[ReadingDirection] = None
    company_name: Optional[str] = None
    company_id: Optional[str] = None
    report_year: Optional[int] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value):
        return _normalize_doc_language_value(value)

    @field_validator("reading_direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value):
        return _normalize_reading_direction_value(value)

    @field_validator("company_name", mode="before")
    @classmethod
    def _normalize_company_name(cls, value):
        return _normalize_company_name_value(value)

    @field_validator("company_id", mode="before")
    @classmethod
    def _normalize_company_id(cls, value):
        return _normalize_company_id_value(value)

    @field_validator("report_year", mode="before")
    @classmethod
    def _normalize_report_year(cls, value):
        return _normalize_report_year_value(value)


class Document(BaseModel):
    document_meta: Optional[DocumentMeta] = None
    pages: List[Page] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
