from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PageType(str, Enum):
    balance_sheet = "balance_sheet"
    income_statement = "income_statement"
    cash_flow = "cash_flow"
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
    is_note: bool = False
    note: Optional[str] = None
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
        if value is None:
            return None
        text = str(value).strip()
        return text or None


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
    is_note: bool = False
    note: Optional[str] = None
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
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PageExtraction(BaseModel):
    meta: PageMeta
    facts: List[ExtractedFact]
    model_config = ConfigDict(extra="forbid")


class Page(BaseModel):
    meta: PageMeta
    facts: List[Fact]
    model_config = ConfigDict(extra="forbid")


class DocumentMeta(BaseModel):
    language: Optional[DocLanguage] = None
    reading_direction: Optional[ReadingDirection] = None
    model_config = ConfigDict(extra="forbid")


class Document(BaseModel):
    document_meta: Optional[DocumentMeta] = None
    pages: List[Page] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
