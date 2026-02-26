from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


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


class PageMeta(BaseModel):
    entity_name: Optional[str] = None
    page_num: Optional[str] = None
    type: PageType
    title: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Fact(BaseModel):
    value: str
    refference: str
    date: Optional[str] = None
    path: List[str]
    currency: Optional[Currency] = None
    scale: Optional[Scale] = None
    value_type: Optional[ValueType] = None
    model_config = ConfigDict(extra="forbid")


class Page(BaseModel):
    meta: PageMeta
    facts: List[Fact]
    model_config = ConfigDict(extra="forbid")


class Document(BaseModel):
    pages: List[Page] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
