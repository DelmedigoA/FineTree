"""
Load all annotation JSONs into a SQLite database.

Tables:
  documents  - one row per annotation JSON file
  pages      - one row per page in each document
  facts      - one row per extracted fact

Run:
  python db/load_db.py            -> writes db/finetree.db
  python db/load_db.py --out /tmp/ft.db
"""

import argparse
import json
import sqlite3
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent.parent / "data" / "annotations"

# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT NOT NULL,          -- e.g. "pdf_2.json"
    images_dir       TEXT,
    schema_version   INTEGER,
    language         TEXT,
    reading_direction TEXT,
    company_name     TEXT,
    report_year      INTEGER,
    company_id       TEXT,
    entity_type      TEXT,
    report_scope     TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id       INTEGER NOT NULL REFERENCES documents(id),
    page_index        INTEGER NOT NULL,      -- 0-based position in document
    image             TEXT,
    entity_name       TEXT,
    page_num          TEXT,
    page_type         TEXT,
    statement_type    TEXT,
    title             TEXT,
    annotation_note   TEXT,
    annotation_status TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id          INTEGER NOT NULL REFERENCES pages(id),
    document_id      INTEGER NOT NULL REFERENCES documents(id),
    fact_num         INTEGER,
    value            TEXT,
    natural_sign     TEXT,
    row_role         TEXT,
    comment_ref      TEXT,
    note_flag        INTEGER,                -- 0/1
    note_name        TEXT,
    note_num         TEXT,
    note_ref         TEXT,
    date             TEXT,
    period_type      TEXT,
    period_start     TEXT,
    period_end       TEXT,
    duration_type    TEXT,
    recurring_period TEXT,
    path             TEXT,                   -- JSON array, e.g. '["Assets","Current"]'
    path_str         TEXT,                   -- human-readable: "Assets > Current"
    path_depth       INTEGER,
    path_leaf        TEXT,
    path_source      TEXT,
    currency         TEXT,
    scale            INTEGER,
    value_type       TEXT,
    value_context    TEXT,
    bbox_x           REAL,
    bbox_y           REAL,
    bbox_w           REAL,
    bbox_h           REAL,
    equations        TEXT                    -- raw JSON string or NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_page     ON facts(page_id);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _meta(doc: dict, key: str):
    return doc.get("metadata", {}).get(key)


def _page_meta(page: dict, key: str):
    return page.get("meta", {}).get(key)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_file(conn: sqlite3.Connection, json_path: Path) -> None:
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    cur = conn.cursor()

    # ── document ──────────────────────────────────────────────────────────────
    cur.execute(
        """
        INSERT INTO documents
          (filename, images_dir, schema_version,
           language, reading_direction,
           company_name, report_year, company_id,
           entity_type, report_scope)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            json_path.name,
            doc.get("images_dir"),
            doc.get("schema_version"),
            _meta(doc, "language"),
            _meta(doc, "reading_direction"),
            _meta(doc, "company_name"),
            _meta(doc, "report_year"),
            _meta(doc, "company_id"),
            _meta(doc, "entity_type"),
            _meta(doc, "report_scope"),
        ),
    )
    document_id = cur.lastrowid

    # ── pages & facts ─────────────────────────────────────────────────────────
    for page_index, page in enumerate(doc.get("pages", [])):
        cur.execute(
            """
            INSERT INTO pages
              (document_id, page_index, image,
               entity_name, page_num, page_type, statement_type,
               title, annotation_note, annotation_status)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                document_id,
                page_index,
                page.get("image"),
                _page_meta(page, "entity_name"),
                _page_meta(page, "page_num"),
                _page_meta(page, "page_type"),
                _page_meta(page, "statement_type"),
                _page_meta(page, "title"),
                _page_meta(page, "annotation_note"),
                _page_meta(page, "annotation_status"),
            ),
        )
        page_id = cur.lastrowid

        for fact in page.get("facts", []):
            path = fact.get("path") or []
            bbox = fact.get("bbox") or [None, None, None, None]
            equations = fact.get("equations")

            cur.execute(
                """
                INSERT INTO facts
                  (page_id, document_id,
                   fact_num, value, natural_sign, row_role,
                   comment_ref, note_flag, note_name, note_num, note_ref,
                   date, period_type, period_start, period_end,
                   duration_type, recurring_period,
                   path, path_str, path_depth, path_leaf, path_source,
                   currency, scale, value_type, value_context,
                   bbox_x, bbox_y, bbox_w, bbox_h,
                   equations)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    page_id, document_id,
                    fact.get("fact_num"),
                    fact.get("value"),
                    fact.get("natural_sign"),
                    fact.get("row_role"),
                    fact.get("comment_ref"),
                    int(bool(fact.get("note_flag"))),
                    fact.get("note_name"),
                    fact.get("note_num"),
                    fact.get("note_ref"),
                    fact.get("date"),
                    fact.get("period_type"),
                    fact.get("period_start"),
                    fact.get("period_end"),
                    fact.get("duration_type"),
                    fact.get("recurring_period"),
                    json.dumps(path, ensure_ascii=False),
                    " > ".join(path),
                    len(path),
                    path[-1] if path else None,
                    fact.get("path_source"),
                    fact.get("currency"),
                    fact.get("scale"),
                    fact.get("value_type"),
                    fact.get("value_context"),
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    json.dumps(equations, ensure_ascii=False) if equations is not None else None,
                ),
            )

    conn.commit()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load FineTree annotations into SQLite")
    parser.add_argument("--out", default=str(Path(__file__).parent / "finetree.db"),
                        help="Output SQLite file path")
    parser.add_argument("--annotations-dir", default=str(ANNOTATIONS_DIR),
                        help="Directory containing annotation JSON files")
    args = parser.parse_args()

    db_path = Path(args.out)
    annotations_dir = Path(args.annotations_dir)

    # Wipe and recreate every time
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    json_files = sorted(
        f for f in annotations_dir.glob("*.json")
        if not f.name.startswith("_")
    )

    if not json_files:
        print(f"No JSON files found in {annotations_dir}")
        return

    for json_path in json_files:
        print(f"  loading {json_path.name} ...", end=" ", flush=True)
        try:
            load_file(conn, json_path)
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_docs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM pages")
    n_pages = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM facts")
    n_facts = cur.fetchone()[0]

    conn.close()
    print(f"\nDone: {n_docs} documents, {n_pages} pages, {n_facts} facts")
    print(f"Database: {db_path.resolve()}")


if __name__ == "__main__":
    main()
