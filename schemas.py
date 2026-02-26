from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple
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


class PageMeta(BaseModel):
    entity_name: Optional[str] = None
    page_num: Optional[str] = None
    type: PageType
    title: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class Fact(BaseModel):
    value: str
    date: str
    path: List[str]
    currency: str
    scale: int
    value_type: ValueType


class Page:
    meta: 
    facts: List[Fact]


class Page(BaseModel):
    meta: PageMeta
    facts: List[Fact] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Document(BaseModel):
    pages: List[Page] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")