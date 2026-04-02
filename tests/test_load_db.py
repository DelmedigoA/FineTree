from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest


def _load_db_module():
    module_path = Path(__file__).resolve().parents[1] / "db" / "load_db.py"
    spec = importlib.util.spec_from_file_location("finetree_load_db", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_db = _load_db_module()


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(load_db.SCHEMA)
    conn.execute("INSERT INTO documents(filename) VALUES (?)", ("sample.json",))
    return conn


def _insert_page(conn: sqlite3.Connection, *, page_type: str, statement_type: str | None) -> None:
    conn.execute(
        """
        INSERT INTO pages (
            document_id,
            page_index,
            image,
            entity_name,
            page_num,
            page_type,
            statement_type,
            title,
            annotation_note,
            annotation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (1, 0, "page_0001.png", None, None, page_type, statement_type, None, None, None),
    )


def test_pages_schema_accepts_valid_page_type_statement_type_pairs() -> None:
    conn = _build_conn()

    _insert_page(conn, page_type="declaration", statement_type="auditors_report")
    _insert_page(conn, page_type="statements", statement_type="balance_sheet")
    _insert_page(conn, page_type="other", statement_type=None)


def test_pages_schema_rejects_incompatible_page_type_statement_type_pairs() -> None:
    conn = _build_conn()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_page(conn, page_type="declaration", statement_type="balance_sheet")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_page(conn, page_type="statements", statement_type="other_declaration")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_page(conn, page_type="other", statement_type="income_statement")
