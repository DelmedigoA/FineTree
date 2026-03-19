from __future__ import annotations

from pathlib import Path

import pytest

from finetree_annotator import app as app_mod


class _DummyPage:
    def save(self, target: Path | str, format: str = "PNG") -> None:
        _ = format
        Path(target).write_bytes(b"png")


def test_resolve_startup_context_pdf_mode_defaults(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ctx = app_mod._resolve_startup_context(str(pdf_path), None, None)
    assert ctx.mode == "pdf"
    assert ctx.pdf_path == pdf_path.resolve()
    assert ctx.images_dir == (Path("data/pdf_images") / "doc").resolve()
    assert ctx.annotations_path == Path("data/annotations/doc.json")


def test_resolve_startup_context_images_dir_mode_from_input_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)

    ctx = app_mod._resolve_startup_context(str(images_dir), None, None)
    assert ctx.mode == "images-dir"
    assert ctx.images_dir == images_dir.resolve()
    assert ctx.annotations_path == Path("data/annotations/pages.json")
    assert ctx.pdf_path is None


def test_resolve_startup_context_conflict_input_and_images_dir() -> None:
    with pytest.raises(ValueError, match="Cannot provide both positional input_path and --images-dir"):
        app_mod._resolve_startup_context("doc.pdf", "data/pdf_images/test", None)


def test_resolve_startup_context_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(ValueError, match="Input path not found"):
        app_mod._resolve_startup_context(str(missing), None, None)


def test_resolve_startup_context_rejects_unsupported_file_type(tmp_path: Path) -> None:
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input path"):
        app_mod._resolve_startup_context(str(txt_path), None, None)


def test_ensure_pdf_images_reuses_complete_folder(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc_images"
    images_dir.mkdir(parents=True)
    for idx in (1, 2, 3):
        (images_dir / f"page_{idx:04d}.png").write_bytes(b"png")

    monkeypatch.setattr(app_mod.workspace_mod, "pdfinfo_from_path", lambda _path: {"Pages": 3})

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("convert_from_path should not run for complete image folders")

    monkeypatch.setattr(app_mod.workspace_mod, "convert_from_path", _should_not_run)

    summary = app_mod._ensure_pdf_images(pdf_path, images_dir, dpi=200)
    assert summary["action"] == "reused"
    assert summary["created_pages"] == 0
    assert summary["reused_pages"] == 3
    assert summary["missing_before"] == 0


def test_ensure_pdf_images_creates_all_pages_when_folder_missing(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc_images"

    monkeypatch.setattr(app_mod.workspace_mod, "pdfinfo_from_path", lambda _path: {"Pages": 3})

    calls: list[tuple[int, int]] = []

    def _fake_convert(_pdf: str, *, dpi: int, use_pdftocairo: bool, first_page: int, last_page: int):
        _ = dpi, use_pdftocairo
        calls.append((first_page, last_page))
        return [_DummyPage() for _ in range(last_page - first_page + 1)]

    monkeypatch.setattr(app_mod.workspace_mod, "convert_from_path", _fake_convert)

    summary = app_mod._ensure_pdf_images(pdf_path, images_dir, dpi=200)
    assert calls == [(1, 3)]
    assert summary["action"] == "created"
    assert summary["created_pages"] == 3
    assert summary["reused_pages"] == 0
    for idx in (1, 2, 3):
        assert (images_dir / f"page_{idx:04d}.png").is_file()


def test_ensure_pdf_images_heals_missing_subset_only(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc_images"
    images_dir.mkdir(parents=True)
    (images_dir / "page_0001.png").write_bytes(b"png")
    (images_dir / "page_0003.png").write_bytes(b"png")

    monkeypatch.setattr(app_mod.workspace_mod, "pdfinfo_from_path", lambda _path: {"Pages": 3})

    calls: list[tuple[int, int]] = []

    def _fake_convert(_pdf: str, *, dpi: int, use_pdftocairo: bool, first_page: int, last_page: int):
        _ = dpi, use_pdftocairo
        calls.append((first_page, last_page))
        return [_DummyPage()]

    monkeypatch.setattr(app_mod.workspace_mod, "convert_from_path", _fake_convert)

    summary = app_mod._ensure_pdf_images(pdf_path, images_dir, dpi=200)
    assert calls == [(2, 2)]
    assert summary["action"] == "healed"
    assert summary["created_pages"] == 1
    assert summary["reused_pages"] == 2
    assert (images_dir / "page_0002.png").is_file()


def test_ensure_pdf_images_conversion_failure_raises(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc_images"

    monkeypatch.setattr(app_mod.workspace_mod, "pdfinfo_from_path", lambda _path: {"Pages": 2})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_mod.workspace_mod, "convert_from_path", _boom)

    with pytest.raises(RuntimeError, match="Failed converting pages"):
        app_mod._ensure_pdf_images(pdf_path, images_dir, dpi=200)
