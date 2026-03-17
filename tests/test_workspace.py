from __future__ import annotations

import json
import re
from pathlib import Path

from finetree_annotator.annotation_core import PageState, default_page_meta
from finetree_annotator import workspace


class _DummyPage:
    def save(self, target: Path | str, format: str = "PNG") -> None:
        _ = format
        Path(target).write_bytes(b"png")


def _legacy_sanitize_doc_id(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_name or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "document"


def test_page_has_annotation_requires_title() -> None:
    state = PageState(meta=default_page_meta(0), facts=[])
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["type"] = "balance_sheet"
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["entity_name"] = "Acme Ltd"
    assert workspace.page_has_annotation(state, 0) is False


def test_page_has_annotation_ignores_non_entity_title_metadata() -> None:
    state = PageState(
        meta={
            **default_page_meta(0),
            "doc_language": "Auto",
            "reading_direction": "LTR",
        },
        facts=[],
    )
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["page_num"] = "5"
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["annotation_note"] = "revisit later"
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["title"] = "Annual Report"
    assert workspace.page_has_annotation(state, 0) is True


def test_page_has_annotation_accepts_title() -> None:
    state = PageState(
        meta={
            **default_page_meta(0),
            "title": "Annual report",
        },
        facts=[],
    )
    assert workspace.page_has_annotation(state, 0) is True


def test_page_has_annotation_accepts_flagged_annotation_status() -> None:
    state = PageState(
        meta={
            **default_page_meta(0),
            "annotation_status": "flagged",
        },
        facts=[],
    )
    assert workspace.page_has_annotation(state, 0) is True


def test_build_document_summary_counts_progress(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_a"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_a.pdf").write_bytes(b"%PDF-1.4")
    for idx in (1, 2):
        (images_dir / f"page_{idx:04d}.png").write_bytes(b"png")

    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": default_page_meta(0),
                "facts": [],
            },
            {
                "image": "page_0002.png",
                "meta": {
                    **default_page_meta(1),
                    "title": "Income Statement",
                    "annotation_status": "approved",
                },
                "facts": [{"bbox": [1, 2, 3, 4], "value": "42"}],
            },
        ]
    }
    (data_root / "annotations" / "doc_a.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = workspace.build_document_summary("doc_a", data_root=data_root)
    assert summary.page_count == 2
    assert summary.annotated_page_count == 1
    assert summary.approved_page_count == 1
    assert summary.fact_count == 1
    assert summary.annotated_token_count > 0
    assert summary.progress_pct == 50
    assert summary.status == "In progress"
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 0


def test_build_document_summary_counts_flagged_pages_as_annotated(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_flagged"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_flagged.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_flagged.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {**default_page_meta(0), "annotation_status": "flagged"},
                        "facts": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = workspace.build_document_summary("doc_flagged", data_root=data_root)
    assert summary.annotated_page_count == 1
    assert summary.progress_pct == 100

    workspace.set_document_checked("doc_flagged", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_flagged", data_root=data_root)
    assert summary.checked is True


def test_import_pdf_to_workspace_reuses_existing_doc(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    source_pdf = tmp_path / "doc_a.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    managed_pdf = data_root / "raw_pdfs" / "doc_a.pdf"
    managed_pdf.parent.mkdir(parents=True)
    managed_pdf.write_bytes(b"%PDF-1.4-existing")

    monkeypatch.setattr(workspace, "pdfinfo_from_path", lambda _path: {"Pages": 2})

    calls: list[tuple[int, int]] = []

    def _fake_convert(_pdf: str, *, dpi: int, use_pdftocairo: bool, first_page: int, last_page: int):
        _ = dpi, use_pdftocairo
        calls.append((first_page, last_page))
        return [_DummyPage() for _ in range(last_page - first_page + 1)]

    monkeypatch.setattr(workspace, "convert_from_path", _fake_convert)

    result = workspace.import_pdf_to_workspace(source_pdf, data_root=data_root, dpi=200)
    assert result.opened_existing is True
    assert result.copied_pdf is False
    assert result.document.doc_id == "doc_a"
    assert result.document.page_count == 2
    assert calls == [(1, 2)]


def test_discover_workspace_documents_merges_legacy_sanitized_images_with_original_pdf_name(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw_pdfs"
    images_root = data_root / "pdf_images"
    ann_root = data_root / "annotations"
    raw_root.mkdir(parents=True)
    images_root.mkdir(parents=True)
    ann_root.mkdir(parents=True)

    original_pdf = raw_root / "דוח כספי 2014 - אגודת הסטודנטים.pdf"
    original_pdf.write_bytes(b"%PDF-1.4")
    legacy_doc_id = _legacy_sanitize_doc_id(original_pdf.stem)
    legacy_images_dir = images_root / legacy_doc_id
    legacy_images_dir.mkdir()
    (legacy_images_dir / "page_0001.png").write_bytes(b"png")

    summaries = workspace.discover_workspace_documents(data_root=data_root)

    assert len(summaries) == 1
    assert summaries[0].doc_id == legacy_doc_id
    assert summaries[0].source_pdf == original_pdf
    assert summaries[0].images_dir == legacy_images_dir
    assert summaries[0].status == "Ready"


def test_discover_workspace_documents_keeps_distinct_hebrew_raw_pdfs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw_pdfs"
    raw_root.mkdir(parents=True)
    (data_root / "pdf_images").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)

    pdf_a = raw_root / "דוח כספי 2024 - בצוותא.pdf"
    pdf_b = raw_root / "דוח כספי 2024 - סחלבים.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    summaries = workspace.discover_workspace_documents(data_root=data_root)
    by_pdf = {summary.source_pdf.name: summary for summary in summaries if summary.source_pdf is not None}

    assert pdf_a.name in by_pdf
    assert pdf_b.name in by_pdf
    assert by_pdf[pdf_a.name].doc_id != by_pdf[pdf_b.name].doc_id
    assert by_pdf[pdf_a.name].status == "Needs extraction"
    assert by_pdf[pdf_b.name].status == "Needs extraction"


def test_discover_workspace_documents_avoids_legacy_duplicate_when_modern_doc_exists(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw_pdfs"
    images_root = data_root / "pdf_images"
    raw_root.mkdir(parents=True)
    images_root.mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)

    source_pdf = raw_root / "דוח כספי 2024 - בצוותא.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    modern_doc_id = workspace.sanitize_doc_id(source_pdf.stem)
    legacy_doc_id = _legacy_sanitize_doc_id(source_pdf.stem)
    assert modern_doc_id != legacy_doc_id

    modern_images_dir = images_root / modern_doc_id
    modern_images_dir.mkdir()
    (modern_images_dir / "page_0001.png").write_bytes(b"png")

    summaries = workspace.discover_workspace_documents(data_root=data_root)
    doc_ids = {summary.doc_id for summary in summaries}
    assert modern_doc_id in doc_ids
    assert legacy_doc_id not in doc_ids

    matching = [summary for summary in summaries if summary.doc_id == modern_doc_id]
    assert len(matching) == 1
    assert matching[0].source_pdf == source_pdf
    assert matching[0].images_dir == modern_images_dir


def test_reviewed_document_state_persists_in_workspace_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_review"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_review.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_review.json").write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": {**default_page_meta(0), "title": "Reviewed Page"}, "facts": [{"bbox": [1, 2, 3, 4], "value": "10"}]}]}),
        encoding="utf-8",
    )

    workspace.set_document_reviewed("doc_review", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_review", data_root=data_root)
    assert summary.checked is True
    assert summary.reviewed is True

    workspace.set_document_reviewed("doc_review", False, data_root=data_root)
    summary = workspace.build_document_summary("doc_review", data_root=data_root)
    assert summary.checked is True
    assert summary.reviewed is False


def test_delete_workspace_document_removes_pdf_images_annotations_and_state(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_delete"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    pdf_path = data_root / "raw_pdfs" / "doc_delete.pdf"
    annotations_path = data_root / "annotations" / "doc_delete.json"
    pdf_path.write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    annotations_path.write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": default_page_meta(0), "facts": []}]}),
        encoding="utf-8",
    )

    workspace.set_document_checked("doc_delete", True, data_root=data_root)
    workspace.set_document_reviewed("doc_delete", True, data_root=data_root)

    workspace.delete_workspace_document("doc_delete", data_root=data_root)

    checked_doc_ids, reviewed_doc_ids = workspace.load_workspace_check_state(data_root)
    assert not pdf_path.exists()
    assert not images_dir.exists()
    assert not annotations_path.exists()
    assert "doc_delete" not in checked_doc_ids
    assert "doc_delete" not in reviewed_doc_ids


def test_checked_document_state_persists_in_workspace_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_checked"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_checked.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_checked.json").write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": {**default_page_meta(0), "title": "Checked Page"}, "facts": [{"bbox": [1, 2, 3, 4], "value": "10"}]}]}),
        encoding="utf-8",
    )

    workspace.set_document_checked("doc_checked", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_checked", data_root=data_root)
    assert summary.checked is True
    assert summary.reviewed is False


def test_incomplete_document_cannot_remain_checked_in_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_incomplete_checked"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_incomplete_checked.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_incomplete_checked.json").write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": default_page_meta(0), "facts": []}]}),
        encoding="utf-8",
    )

    workspace.set_document_checked("doc_incomplete_checked", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_incomplete_checked", data_root=data_root)
    assert summary.checked is False
    assert summary.reviewed is False


def test_unchecking_document_clears_reviewed_state(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_stage"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_stage.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_stage.json").write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": default_page_meta(0), "facts": []}]}),
        encoding="utf-8",
    )

    workspace.set_document_reviewed("doc_stage", True, data_root=data_root)
    workspace.set_document_checked("doc_stage", False, data_root=data_root)
    summary = workspace.build_document_summary("doc_stage", data_root=data_root)
    assert summary.checked is False
    assert summary.reviewed is False


def test_build_document_summary_keeps_review_for_warning_only_issue(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_flags"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_flags.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_flags.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {"type": "notes", "title": "Notes Page"},
                        "facts": [{"bbox": [1, 2, 3, 4], "value": "10", "note_flag": False, "path": []}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    workspace.set_document_reviewed("doc_flags", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_flags", data_root=data_root)
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 1
    assert summary.pages_with_reg_flags == 0
    assert summary.pages_with_warnings == 1
    assert summary.reviewed is True


def test_build_document_summary_blocks_review_for_real_reg_flags(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_hard_flags"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_hard_flags.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_hard_flags.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {"type": "other", "entity_name": "Acme Ltd"},
                        "facts": [{"bbox": [1, 2, 3, 4], "value": "10", "note_flag": False, "note_num": 3, "path": []}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    workspace.set_document_reviewed("doc_hard_flags", True, data_root=data_root)
    summary = workspace.build_document_summary("doc_hard_flags", data_root=data_root)
    assert summary.reg_flag_count == 1
    assert summary.pages_with_reg_flags == 1
    assert summary.reviewed is False


def test_build_document_summary_includes_document_level_warnings(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_sequence"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_sequence.pdf").write_bytes(b"%PDF-1.4")
    for idx in (1, 2, 3):
        (images_dir / f"page_{idx:04d}.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_sequence.json").write_text(
        json.dumps(
            {
                "pages": [
                    {"image": "page_0001.png", "meta": {"entity_name": "Acme Ltd", "page_num": "5"}, "facts": []},
                    {"image": "page_0002.png", "meta": {"entity_name": "", "page_num": "5"}, "facts": []},
                    {"image": "page_0003.png", "meta": {"entity_name": "Acme Ltd", "page_num": "7"}, "facts": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = workspace.build_document_summary("doc_sequence", data_root=data_root)
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 4
    assert summary.pages_with_warnings == 3


def test_reset_document_approved_pages_clears_only_approved_statuses(tmp_path: Path) -> None:
    annotations_path = tmp_path / "doc.json"
    annotations_path.write_text(
        json.dumps(
            {
                "pages": [
                    {"image": "page_0001.png", "meta": {"annotation_status": "approved"}, "facts": []},
                    {"image": "page_0002.png", "meta": {"annotation_status": "flagged"}, "facts": []},
                    {"image": "page_0003.png", "meta": {"annotation_status": None}, "facts": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    changed = workspace.reset_document_approved_pages(annotations_path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))

    assert changed == 1
    assert payload["pages"][0]["meta"]["annotation_status"] is None
    assert payload["pages"][1]["meta"]["annotation_status"] == "flagged"
    assert payload["pages"][2]["meta"]["annotation_status"] is None
