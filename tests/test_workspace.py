from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.annotation_core import PageState, default_page_meta
from finetree_annotator import workspace


class _DummyPage:
    def save(self, target: Path | str, format: str = "PNG") -> None:
        _ = format
        Path(target).write_bytes(b"png")


def test_page_has_annotation_requires_entity_or_title() -> None:
    state = PageState(meta=default_page_meta(0), facts=[])
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["type"] = "balance_sheet"
    assert workspace.page_has_annotation(state, 0) is False

    state.meta["entity_name"] = "Acme Ltd"
    assert workspace.page_has_annotation(state, 0) is True


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

    state.meta["title"] = "Annual Report"
    assert workspace.page_has_annotation(state, 0) is True


def test_page_has_annotation_accepts_entity_or_title_only() -> None:
    state = PageState(
        meta={
            **default_page_meta(0),
            "entity_name": "Entity",
            "title": "Annual report",
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
                "meta": {**default_page_meta(1), "title": "Income Statement"},
                "facts": [{"bbox": [1, 2, 3, 4], "value": "42"}],
            },
        ]
    }
    (data_root / "annotations" / "doc_a.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = workspace.build_document_summary("doc_a", data_root=data_root)
    assert summary.page_count == 2
    assert summary.annotated_page_count == 1
    assert summary.fact_count == 1
    assert summary.annotated_token_count > 0
    assert summary.progress_pct == 50
    assert summary.status == "In progress"
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 0


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


def test_checked_document_state_persists_in_workspace_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "pdf_images" / "doc_checked"
    images_dir.mkdir(parents=True)
    (data_root / "raw_pdfs").mkdir(parents=True)
    (data_root / "annotations").mkdir(parents=True)
    (data_root / "raw_pdfs" / "doc_checked.pdf").write_bytes(b"%PDF-1.4")
    (images_dir / "page_0001.png").write_bytes(b"png")
    (data_root / "annotations" / "doc_checked.json").write_text(
        json.dumps({"pages": [{"image": "page_0001.png", "meta": {**default_page_meta(0), "entity_name": "Checked Entity"}, "facts": [{"bbox": [1, 2, 3, 4], "value": "10"}]}]}),
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
                        "facts": [{"bbox": [1, 2, 3, 4], "value": "", "note_flag": False, "path": []}],
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
