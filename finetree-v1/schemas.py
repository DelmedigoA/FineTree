from pydantic import BaseModel


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    unit: str
    page: int


class Fact(BaseModel):
    doc_id: str
    page: int
    value: str
    path_nodes: list[str]
    bbox: BBox


class ExportBundle(BaseModel):
    facts: list[Fact]
