from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

from finetree_annotator import app as app_mod
from finetree_annotator import dashboard
from finetree_annotator.workspace import WorkspaceDocumentSummary

_APP: QApplication | None = None


def _qt_app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


@pytest.fixture(autouse=True)
def _stub_dashboard_workspace(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "discover_workspace_documents", lambda: [])


class _FakeAnnotationWindow(QWidget):
    annotations_saved = pyqtSignal(object)
    document_auto_annotate_page_completed = pyqtSignal(str, int)
    document_auto_annotate_page_failed = pyqtSignal(str, str)
    document_auto_annotate_page_stopped = pyqtSignal(str, str)

    def __init__(self, images_dir: Path, annotations_path: Path) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.confirm_close_result = True
        self.confirm_close_calls: list[bool] = []
        self.import_calls: list[dict[str, object]] = []
        self.save_calls: list[bool] = []

    def save_annotations(self, *, allow_locked: bool = False) -> bool:
        self.save_calls.append(bool(allow_locked))
        self.annotations_saved.emit(self.annotations_path)
        return True

    def import_annotations_json_text(
        self,
        raw_text: str,
        *,
        bbox_mode: str = "original_pixels",
        max_pixels=None,
        run_align_after_import: bool = False,
        default_page_name=None,
        show_info_messages: bool = True,
    ) -> dict[str, object]:
        self.import_calls.append(
            {
                "raw_text": raw_text,
                "bbox_mode": bbox_mode,
                "max_pixels": max_pixels,
                "run_align_after_import": run_align_after_import,
                "default_page_name": default_page_name,
                "show_info_messages": show_info_messages,
            }
        )
        return {
            "imported_count": 1,
            "parsed_page_count": 1,
            "extra_page_count": 0,
            "parse_result": SimpleNamespace(recovered=False, message=None),
            "sanity_message": None,
        }

    def show_help_dialog(self) -> None:
        return None

    def confirm_close(self, *, prepare_for_close: bool = False) -> bool:
        self.confirm_close_calls.append(bool(prepare_for_close))
        return self.confirm_close_result

    def cancel_pending_close(self) -> None:
        return None

    def has_unsaved_changes(self) -> bool:
        return False

    def set_document_auto_annotate_locked(self, locked: bool) -> None:
        return None

    def start_document_auto_annotate_page(self, page_name: str) -> bool:
        return True


class _FakeSettings:
    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self.values = dict(initial or {})

    def value(self, key: str, default=None, type=None):
        value = self.values.get(key, default)
        if type is bool:
            return bool(value)
        return value

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


def test_resolve_startup_context_no_args_opens_home() -> None:
    ctx = app_mod._resolve_startup_context(None, None, None)
    assert ctx.mode == "home"
    assert ctx.images_dir is None
    assert ctx.annotations_path is None


def test_dashboard_consumes_images_dir_startup(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    monkeypatch.setattr(dashboard, "AnnotationWindow", _FakeAnnotationWindow)

    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    annotations_path = tmp_path / "pages.json"
    ctx = app_mod.StartupContext(mode="images-dir", images_dir=images_dir, annotations_path=annotations_path)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    window._consume_startup_context()

    current = window.annotator_host.current_context()
    assert current is not None
    assert current.images_dir == images_dir
    assert current.annotations_path == annotations_path
    assert window.stack.currentWidget() is window.annotator_host
    window.close()


def test_dashboard_close_is_blocked_when_embedded_annotator_refuses(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    monkeypatch.setattr(dashboard, "AnnotationWindow", _FakeAnnotationWindow)

    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    annotations_path = tmp_path / "pages.json"
    ctx = app_mod.StartupContext(mode="images-dir", images_dir=images_dir, annotations_path=annotations_path)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    window.show()
    window._consume_startup_context()
    _qt_app().processEvents()

    current = window.annotator_host.current_window()
    assert current is not None
    current.confirm_close_result = False

    window.close()
    _qt_app().processEvents()

    assert window.isVisible() is True
    assert current.confirm_close_calls == [True]

    current.confirm_close_result = True
    window.close()


def test_embedded_annotator_exit_button_closes_dashboard(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")

    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    image = dashboard.QPixmap(32, 32)
    image.fill(dashboard.QColor("#ffffff"))
    assert image.save(str(images_dir / "page_0001.png"), "PNG")
    annotations_path = tmp_path / "pages.json"
    ctx = app_mod.StartupContext(mode="images-dir", images_dir=images_dir, annotations_path=annotations_path)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    window.show()
    window._consume_startup_context()
    _qt_app().processEvents()

    current = window.annotator_host.current_window()
    assert current is not None

    current.exit_btn.click()
    _qt_app().processEvents()

    assert window.isVisible() is False


def test_push_view_builds_cli_args() -> None:
    _qt_app()
    view = dashboard.PushView()
    view.resize(1400, 900)
    view.show()
    _qt_app().processEvents()
    view.config_edit.setText("configs/custom.yaml")
    view.repo_id_edit.setText("user/repo")
    view.repo_id_train_edit.setText("user/repo-train")
    view.repo_id_validation_edit.setText("user/repo-val")
    view.export_dir_edit.setText("artifacts/custom_export")
    view.token_edit.setText("hf_test")
    view.instruction_mode_combo.setCurrentText("minimal")
    view.min_pixels_spin.setValue(128)
    view.max_pixels_spin.setValue(512)
    view.exclude_doc_ids_edit.setText("doc1,doc2")
    view.public_check.setChecked(True)
    view.compact_tokens_check.setChecked(True)
    view.aggressive_compact_check.setChecked(True)
    view.push_all_variants_check.setChecked(True)
    view.push_split_repos_check.setChecked(True)
    view.allow_duplicate_check.setChecked(True)
    view.allow_ordering_check.setChecked(True)
    view.allow_format_check.setChecked(True)
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_reviewed_a",
                source_pdf=Path("/tmp/doc_reviewed_a.pdf"),
                images_dir=Path("/tmp/doc_reviewed_a"),
                annotations_path=Path("/tmp/doc_reviewed_a.json"),
                thumbnail_path=None,
                page_count=5,
                annotated_page_count=5,
                progress_pct=100,
                status="Complete",
                updated_at=None,
                approved_page_count=5,
                checked=True,
                reviewed=True,
            ),
            WorkspaceDocumentSummary(
                doc_id="doc_reviewed_b",
                source_pdf=Path("/tmp/doc_reviewed_b.pdf"),
                images_dir=Path("/tmp/doc_reviewed_b"),
                annotations_path=Path("/tmp/doc_reviewed_b.json"),
                thumbnail_path=None,
                page_count=4,
                annotated_page_count=3,
                progress_pct=75,
                status="In progress",
                updated_at=None,
                approved_page_count=3,
                checked=True,
                reviewed=True,
            ),
        ]
    )
    assert view.train_docs_list.count() == 2
    assert view.validation_docs_list.count() == 2
    first_item = view.validation_docs_list.item(0)
    assert first_item is not None
    first_item.setCheckState(2)
    train_first_item = view.train_docs_list.item(0)
    assert train_first_item is not None
    assert train_first_item.checkState() == 0

    argv = view._build_argv()
    assert argv[:6] == ["--config", "configs/custom.yaml", "--export-dir", "artifacts/custom_export", "--instruction-mode", "minimal"]
    assert "--include-doc-ids" in argv
    assert "doc_reviewed_a,doc_reviewed_b" in argv
    assert "--validation-doc-ids" in argv
    assert "doc_reviewed_a" in argv
    assert "--repo-id" in argv
    assert "--token" in argv
    assert "--private" not in argv
    assert "--approved-pages-only" in argv
    assert "--page-meta-keys" in argv
    assert "--fact-keys" in argv
    assert "--compact_tokens" in argv
    assert "--aggressive-compact-tokens" in argv
    assert "--push-all-variants" in argv
    assert "--push-train-val-separately" in argv
    assert "--allow-duplicate-facts" in argv
    assert "--allow-ordering-issues" in argv
    assert "--allow-format-issues" in argv
    assert "--repo-id-train" in argv
    assert "--repo-id-validation" in argv
    assert view.form_scroll.objectName() == "pushFormScroll"
    assert view.form_content.maximumWidth() == 720
    assert view.form_scroll.maximumWidth() == 1100
    assert view.basics_card.objectName() == "surfaceCard"
    assert view.content_splitter.objectName() == "pushContentSplitter"
    assert view.results_card.minimumWidth() == 320
    assert view.results_card.maximumWidth() == 560
    assert '"date"' not in view.schema_preview_view.toPlainText()
    assert "Selected fact keys" in view.prompt_preview_view.toPlainText()
    assert "Skipped non-approved pages: 1" in view.preview_summary_label.text()
    view.public_check.setChecked(False)
    argv_private = view._build_argv()
    assert "--private" in argv_private
    view.close()


def test_push_view_allows_approved_pdf_without_review_flag() -> None:
    _qt_app()
    view = dashboard.PushView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_approved_only",
                source_pdf=Path("/tmp/doc_approved_only.pdf"),
                images_dir=Path("/tmp/doc_approved_only"),
                annotations_path=Path("/tmp/doc_approved_only.json"),
                thumbnail_path=None,
                page_count=3,
                annotated_page_count=3,
                progress_pct=100,
                status="Complete",
                updated_at=None,
                approved_page_count=2,
                checked=True,
                reviewed=False,
            ),
        ]
    )
    assert view.train_docs_list.count() == 1
    assert view.validation_docs_list.count() == 1
    argv = view._build_argv()
    assert "--include-doc-ids" in argv
    assert "doc_approved_only" in argv
    assert "1 eligible PDF" in view.reviewed_docs_summary.text()
    assert "Approved pages to push: 2" in view.preview_summary_label.text()
    view.close()


def test_push_view_allows_approved_pdf_without_checked_or_complete_status() -> None:
    _qt_app()
    view = dashboard.PushView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_partial_approved",
                source_pdf=Path("/tmp/doc_partial_approved.pdf"),
                images_dir=Path("/tmp/doc_partial_approved"),
                annotations_path=Path("/tmp/doc_partial_approved.json"),
                thumbnail_path=None,
                page_count=6,
                annotated_page_count=2,
                progress_pct=33,
                status="In progress",
                updated_at=None,
                approved_page_count=2,
                checked=False,
                reviewed=False,
            ),
        ]
    )
    assert view.train_docs_list.count() == 1
    assert view.validation_docs_list.count() == 1
    assert view.selected_train_doc_ids() == ["doc_partial_approved"]
    argv = view._build_argv()
    assert "--include-doc-ids" in argv
    assert "doc_partial_approved" in argv
    assert "Approved pages to push: 2" in view.preview_summary_label.text()
    assert "Skipped non-approved pages: 4" in view.preview_summary_label.text()
    assert "1 eligible PDF" in view.reviewed_docs_summary.text()
    view.close()


def test_push_view_train_and_validation_lists_are_mutually_exclusive() -> None:
    _qt_app()
    view = dashboard.PushView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_a",
                source_pdf=Path("/tmp/doc_a.pdf"),
                images_dir=Path("/tmp/doc_a"),
                annotations_path=Path("/tmp/doc_a.json"),
                thumbnail_path=None,
                page_count=3,
                annotated_page_count=3,
                progress_pct=100,
                status="Complete",
                updated_at=None,
                approved_page_count=2,
                checked=True,
                reviewed=False,
            ),
            WorkspaceDocumentSummary(
                doc_id="doc_b",
                source_pdf=Path("/tmp/doc_b.pdf"),
                images_dir=Path("/tmp/doc_b"),
                annotations_path=Path("/tmp/doc_b.json"),
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=2,
                progress_pct=100,
                status="Complete",
                updated_at=None,
                approved_page_count=1,
                checked=True,
                reviewed=False,
            ),
        ]
    )
    assert view.train_docs_list.count() == 2
    assert view.validation_docs_list.count() == 2
    assert view.selected_train_doc_ids() == ["doc_a", "doc_b"]
    assert view.selected_validation_doc_ids() == []

    first_validation_item = view.validation_docs_list.item(0)
    assert first_validation_item is not None
    first_validation_item.setCheckState(2)
    assert view.selected_validation_doc_ids() == ["doc_a"]
    assert view.selected_train_doc_ids() == ["doc_b"]

    view.select_all_train_btn.click()
    assert view.selected_train_doc_ids() == ["doc_a", "doc_b"]
    assert view.selected_validation_doc_ids() == []

    view.select_all_validation_btn.click()
    assert view.selected_train_doc_ids() == []
    assert view.selected_validation_doc_ids() == ["doc_a", "doc_b"]

    view.clear_all_validation_btn.click()
    assert view.selected_validation_doc_ids() == []
    assert view.selected_train_doc_ids() == []

    view.select_all_train_btn.click()
    assert view.selected_included_doc_ids() == ["doc_a", "doc_b"]
    assert "Approved pages to push: 3" in view.preview_summary_label.text()
    view.close()


def test_push_view_blocks_tiny_pixel_budget_values(monkeypatch) -> None:
    _qt_app()
    view = dashboard.PushView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_budget",
                source_pdf=Path("/tmp/doc_budget.pdf"),
                images_dir=Path("/tmp/doc_budget"),
                annotations_path=Path("/tmp/doc_budget.json"),
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=1,
                progress_pct=50,
                status="In progress",
                updated_at=None,
                approved_page_count=1,
                checked=False,
                reviewed=False,
            ),
        ]
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )
    view.max_pixels_spin.setValue(1200)

    view.run_pipeline()

    assert warnings
    assert warnings[0][0] == "Pixel budget too small"
    assert "1200" in warnings[0][1]
    assert "28x28" in warnings[0][1]
    assert view._worker_thread is None
    view.close()


def test_dashboard_nav_buttons_use_larger_icon_size() -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    assert window.home_btn.iconSize().width() == 26
    assert window.push_btn.iconSize().height() == 26
    window.close()


def test_home_document_card_shows_prepare_for_unprepared_pdf(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=0,
        annotated_page_count=0,
        progress_pct=0,
        status="Needs extraction",
        updated_at=None,
    )

    card = dashboard.HomeDocumentCard(summary)
    prepared: list[str] = []
    card.prepare_requested.connect(prepared.append)
    card.show()
    _qt_app().processEvents()

    assert card.status_label.isVisible()
    assert card.action_btn.text() == "Prepare"
    assert card.action_btn.isEnabled()
    assert card.auto_annotate_btn.isEnabled()

    card.action_btn.click()
    assert prepared == ["doc"]
    card.close()


def test_home_document_card_emits_auto_annotate_request(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )

    card = dashboard.HomeDocumentCard(summary)
    requested: list[str] = []
    card.auto_annotate_requested.connect(requested.append)
    card.show()
    _qt_app().processEvents()

    assert card.auto_annotate_btn.text() == "Auto-Annotate"
    card.auto_annotate_btn.click()

    assert requested == ["doc"]
    card.close()


def test_home_document_card_emits_push_hf_request(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )

    card = dashboard.HomeDocumentCard(summary)
    requested: list[str] = []
    card.push_hf_requested.connect(requested.append)
    card.show()
    _qt_app().processEvents()

    assert card.push_hf_btn.text() == "Push to HF"
    assert card.push_hf_btn.isEnabled()
    card.push_hf_btn.click()

    assert requested == ["doc"]
    card.close()


def test_home_document_card_emits_load_entire_json_request(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )

    card = dashboard.HomeDocumentCard(summary)
    requested: list[str] = []
    card.load_entire_json_requested.connect(requested.append)
    card.show()
    _qt_app().processEvents()

    assert card.load_entire_json_btn.text() == "Load Entire JSON"
    assert card.load_entire_json_btn.isEnabled()
    card.load_entire_json_btn.click()

    assert requested == ["doc"]
    card.close()


def test_inference_push_dialog_defaults_match_full_training_schema() -> None:
    _qt_app()
    dialog = dashboard.InferencePushDialog("pdf_7")
    dialog.show()
    _qt_app().processEvents()

    options = dialog.options()
    assert options.max_pixels == dashboard.push_inference_dataset.DEFAULT_MAX_PIXELS
    assert options.page_meta_keys == dashboard.push_inference_dataset.DEFAULT_PAGE_META_KEYS
    assert options.fact_keys == dashboard.push_inference_dataset.DEFAULT_FACT_KEYS
    assert dashboard.push_inference_dataset.inference_dataset_name("pdf_7", options.max_pixels) in dialog.dataset_name_label.text()
    assert '"date"' not in dialog.schema_preview.toPlainText()
    assert '"bbox"' not in dialog.schema_preview.toPlainText()
    dialog.close()


def test_load_entire_json_dialog_reports_detected_pages() -> None:
    _qt_app()
    dialog = dashboard.LoadEntireJsonDialog(doc_title="doc", page_count=2)
    dialog.show()
    dialog.text_edit.setPlainText(
        '[{"meta":{"page_num":"1"},"facts":[{"value":"10","path":[]}]},{"meta":{"page_num":"2"},"facts":[{"value":"20","path":[]}]}]'
    )
    _qt_app().processEvents()

    assert "Detected imported pages: 2" in dialog.status_label.text()
    assert "Workspace pages: 2" in dialog.status_label.text()
    dialog.close()


def test_home_document_card_shows_runtime_auto_annotate_progress(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=10,
        annotated_page_count=2,
        progress_pct=20,
        status="In progress",
        updated_at=None,
    )

    progress = dashboard.AutoAnnotateProgress(
        doc_id="doc",
        stage="annotating",
        total_pages=10,
        completed_pages=4,
        fact_count=125,
        current_page="page_0005.png",
    )
    card = dashboard.HomeDocumentCard(summary, runtime_progress=progress)
    card.show()
    _qt_app().processEvents()

    assert card.status_label.text() == "Auto-Annotating"
    assert card.progress.value() == 40
    assert card.progress.format() == "40% • 4/10 pages • 125 facts"
    assert not card.auto_annotate_btn.isEnabled()
    card.close()


def test_home_view_refresh_deletes_old_cards_without_detaching_windows(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"

    first = WorkspaceDocumentSummary(
        doc_id="doc-one",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=1,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )
    second = WorkspaceDocumentSummary(
        doc_id="doc-two",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=1,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )

    view = dashboard.HomeView()
    view.show()
    view.set_documents([first])
    _qt_app().processEvents()

    old_card = view._cards["doc-one"]
    assert old_card.isVisible() is True
    assert old_card.parent() is not None

    view.set_documents([second])
    _qt_app().processEvents()

    assert "doc-two" in view._cards
    assert old_card.parent() is None
    assert old_card.isVisible() is False
    view.close()


def test_home_document_card_emits_delete_request(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=1,
        annotated_page_count=1,
        progress_pct=100,
        status="Complete",
        updated_at=None,
        checked=True,
        reviewed=False,
    )

    card = dashboard.HomeDocumentCard(summary)
    deleted: list[str] = []
    card.delete_requested.connect(deleted.append)
    card.show()
    _qt_app().processEvents()

    assert card.delete_btn.text() == "Remove"
    assert card.delete_btn.property("variant") == "danger"
    card.delete_btn.click()
    assert deleted == ["doc"]
    card.close()


def test_home_document_card_prefers_source_pdf_stem_as_title(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "דוח כספי 2014 - אגודת הסטודנטים.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "2014"
    annotations_path = tmp_path / "2014.json"
    summary = WorkspaceDocumentSummary(
        doc_id="2014",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=1,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )

    card = dashboard.HomeDocumentCard(summary)
    card.show()
    _qt_app().processEvents()

    assert card.title_label.text() == source_pdf.stem
    assert card.meta_label.text().startswith(source_pdf.name)
    card.close()


def test_home_document_card_emits_review_toggle_for_prepared_doc(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=1,
        progress_pct=50,
        status="In progress",
        updated_at=None,
        checked=True,
        reviewed=False,
    )

    card = dashboard.HomeDocumentCard(summary)
    reviewed: list[tuple[str, bool]] = []
    card.review_toggled.connect(lambda doc_id, checked: reviewed.append((doc_id, checked)))
    card.show()
    _qt_app().processEvents()

    assert card.review_btn.isEnabled()
    card.review_btn.click()
    assert reviewed == [("doc", True)]
    card.close()


def test_home_document_card_disables_review_until_checked(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=1,
        progress_pct=50,
        status="In progress",
        updated_at=None,
        checked=False,
        reviewed=False,
    )

    card = dashboard.HomeDocumentCard(summary)
    card.show()
    _qt_app().processEvents()

    assert not card.checked_btn.isEnabled()
    assert "Finish annotating every page" in card.checked_btn.toolTip()
    assert not card.review_btn.isEnabled()
    assert "checked before review" in card.review_btn.toolTip()
    card.close()


def test_home_document_card_disables_review_when_reg_flags_exist(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=annotations_path,
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=2,
        progress_pct=100,
        status="Complete",
        updated_at=None,
        reg_flag_count=2,
        warning_count=1,
        pages_with_reg_flags=1,
        pages_with_warnings=1,
        checked=True,
        reviewed=False,
    )

    card = dashboard.HomeDocumentCard(summary)
    card.show()
    _qt_app().processEvents()

    assert card.pages_label.text() == "Approved 0/2 pages"
    assert card.reg_flags_label.isVisible()
    assert card.reg_flags_label.text() == "⚑ 2 Reg Flags"
    assert card.reg_flags_label.property("tone") == "danger"
    assert card.warnings_label.isVisible()
    assert card.warnings_label.text() == "⚠ 1 Warning"
    assert card.warnings_label.property("tone") == "accent"
    assert not card.no_issues_label.isVisible()
    assert not card.review_btn.isEnabled()
    assert "Fix 2 reg flag(s)" in card.review_btn.toolTip()
    card.close()


def test_home_view_sorts_documents_by_checked_then_reg_flags_then_warnings(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"

    view = dashboard.HomeView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="warn_only",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=2,
                progress_pct=100,
                status="Complete",
                updated_at=10.0,
                reg_flag_count=0,
                warning_count=5,
                checked=False,
            ),
            WorkspaceDocumentSummary(
                doc_id="red_first",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=2,
                progress_pct=100,
                status="Complete",
                updated_at=20.0,
                reg_flag_count=2,
                warning_count=1,
                checked=True,
            ),
            WorkspaceDocumentSummary(
                doc_id="red_more_warn",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=2,
                progress_pct=100,
                status="Complete",
                updated_at=15.0,
                reg_flag_count=2,
                warning_count=3,
                checked=False,
            ),
        ]
    )
    view.sort_filter.setCurrentText("Issues Desc")
    _qt_app().processEvents()

    rendered_doc_ids = [
        view.cards_layout.itemAt(index).widget().summary.doc_id
        for index in range(view.cards_layout.count())
        if isinstance(view.cards_layout.itemAt(index).widget(), dashboard.HomeDocumentCard)
    ]
    assert rendered_doc_ids == ["red_more_warn", "warn_only", "red_first"]
    view.close()


def test_home_view_sorts_documents_by_approved_pages_ascending(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"

    view = dashboard.HomeView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="three",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=5,
                annotated_page_count=5,
                approved_page_count=3,
                progress_pct=100,
                status="Complete",
                updated_at=10.0,
            ),
            WorkspaceDocumentSummary(
                doc_id="zero_b",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=4,
                annotated_page_count=2,
                approved_page_count=0,
                progress_pct=50,
                status="In progress",
                updated_at=20.0,
            ),
            WorkspaceDocumentSummary(
                doc_id="zero_a",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=1,
                approved_page_count=0,
                progress_pct=50,
                status="In progress",
                updated_at=30.0,
            ),
            WorkspaceDocumentSummary(
                doc_id="one",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=3,
                annotated_page_count=3,
                approved_page_count=1,
                progress_pct=100,
                status="Complete",
                updated_at=40.0,
            ),
        ]
    )
    view.sort_filter.setCurrentText("Approved Asc")
    _qt_app().processEvents()

    rendered_doc_ids = [
        view.cards_layout.itemAt(index).widget().summary.doc_id
        for index in range(view.cards_layout.count())
        if isinstance(view.cards_layout.itemAt(index).widget(), dashboard.HomeDocumentCard)
    ]
    assert rendered_doc_ids == ["zero_a", "zero_b", "one", "three"]
    view.close()


def test_home_view_stats_include_annotated_image_and_token_totals(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"

    view = dashboard.HomeView()
    view.set_documents(
        [
            WorkspaceDocumentSummary(
                doc_id="doc_a",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=2,
                annotated_page_count=2,
                progress_pct=100,
                status="Complete",
                updated_at=None,
                fact_count=12,
                annotated_token_count=450_000,
            ),
            WorkspaceDocumentSummary(
                doc_id="doc_b",
                source_pdf=source_pdf,
                images_dir=images_dir,
                annotations_path=annotations_path,
                thumbnail_path=None,
                page_count=3,
                annotated_page_count=1,
                progress_pct=33,
                status="In progress",
                updated_at=None,
                fact_count=21,
                annotated_token_count=300_000,
            ),
        ]
    )
    _qt_app().processEvents()

    stats: dict[str, tuple[str, str]] = {}
    for index in range(view.stats_grid.count()):
        card = view.stats_grid.itemAt(index).widget()
        if card is None:
            continue
        layout = card.layout()
        if layout is None:
            continue
        title = layout.itemAt(0).widget().text()
        value = layout.itemAt(1).widget().text()
        caption = layout.itemAt(2).widget().text()
        stats[title] = (value, caption)

    assert stats["Annotated Images"] == ("3", "Across workspace")
    assert stats["Facts"] == ("33", "Across workspace")
    assert stats["Annotated Tokens"] == ("750,000", "75% of 1,000,000 target")
    view.close()


def test_home_view_initializes_approved_metric_at_zero() -> None:
    _qt_app()
    view = dashboard.HomeView()
    view.set_documents([])
    _qt_app().processEvents()

    stats: dict[str, tuple[str, str]] = {}
    for index in range(view.stats_grid.count()):
        card = view.stats_grid.itemAt(index).widget()
        if card is None or card.layout() is None:
            continue
        layout = card.layout()
        title = layout.itemAt(0).widget().text()
        value = layout.itemAt(1).widget().text()
        caption = layout.itemAt(2).widget().text()
        stats[title] = (value, caption)

    assert stats["% Approved Pages"] == ("0%", "0/0 across workspace")
    assert view.reset_approved_btn.isEnabled() is False
    view.close()


def test_home_document_card_uses_state_based_open_label(tmp_path: Path) -> None:
    _qt_app()
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    annotations_path = tmp_path / "doc.json"

    ready = dashboard.HomeDocumentCard(
        WorkspaceDocumentSummary(
            doc_id="ready_doc",
            source_pdf=source_pdf,
            images_dir=images_dir,
            annotations_path=annotations_path,
            thumbnail_path=None,
            page_count=4,
            annotated_page_count=0,
            progress_pct=0,
            status="Ready",
            updated_at=None,
        )
    )
    in_progress = dashboard.HomeDocumentCard(
        WorkspaceDocumentSummary(
            doc_id="progress_doc",
            source_pdf=source_pdf,
            images_dir=images_dir,
            annotations_path=annotations_path,
            thumbnail_path=None,
            page_count=4,
            annotated_page_count=2,
            progress_pct=50,
            status="In progress",
            updated_at=None,
        )
    )
    complete = dashboard.HomeDocumentCard(
        WorkspaceDocumentSummary(
            doc_id="complete_doc",
            source_pdf=source_pdf,
            images_dir=images_dir,
            annotations_path=annotations_path,
            thumbnail_path=None,
            page_count=4,
            annotated_page_count=4,
            progress_pct=100,
            status="Complete",
            updated_at=None,
        )
    )

    assert ready.action_btn.text() == "Open"
    assert in_progress.action_btn.text() == "Resume"
    assert complete.action_btn.text() == "Review"

    ready.close()
    in_progress.close()
    complete.close()


def test_dashboard_prepare_workspace_document_uses_background_import(tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=0,
        annotated_page_count=0,
        progress_pct=0,
        status="Needs extraction",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    calls: list[tuple[Path, bool]] = []

    def _fake_start_import(pdf_path: Path, *, open_after: bool = True) -> None:
        calls.append((pdf_path, open_after))

    window._start_import = _fake_start_import  # type: ignore[method-assign]
    window.prepare_workspace_document("doc")

    assert calls == [(source_pdf, False)]
    window.close()


def test_dashboard_load_entire_json_imports_and_saves_document(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    class _FakeDialog:
        def __init__(self, *, doc_title: str, page_count: int, parent=None) -> None:
            _ = doc_title, page_count, parent

        def exec_(self) -> int:
            return dashboard.QDialog.Accepted

        def json_text(self) -> str:
            return '[{"meta":{"page_num":"1"},"facts":[{"value":"10","path":[]}]}]'

    class _RunWindow:
        def __init__(self) -> None:
            self.import_calls: list[dict[str, object]] = []
            self.save_calls = 0

        def import_annotations_json_text(self, raw_text: str, **kwargs) -> dict[str, object]:
            self.import_calls.append({"raw_text": raw_text, **kwargs})
            return {
                "imported_count": 1,
                "parsed_page_count": 1,
                "extra_page_count": 0,
                "parse_result": SimpleNamespace(recovered=False, message=None),
                "sanity_message": None,
            }

        def save_annotations(self) -> bool:
            self.save_calls += 1
            return True

    run_window = _RunWindow()
    info_calls: list[tuple[str, str]] = []
    reloads: list[str] = []
    monkeypatch.setattr(dashboard, "LoadEntireJsonDialog", _FakeDialog)
    monkeypatch.setattr(window.annotator_host, "open_document", lambda context, activate=False: run_window)
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "information",
        lambda _parent, title, text: info_calls.append((title, text)),
    )
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.load_workspace_document_entire_json("doc")

    assert len(run_window.import_calls) == 1
    assert run_window.import_calls[0]["raw_text"] == '[{"meta":{"page_num":"1"},"facts":[{"value":"10","path":[]}]}]'
    assert run_window.import_calls[0]["bbox_mode"] == "original_pixels"
    assert run_window.import_calls[0]["run_align_after_import"] is False
    assert run_window.import_calls[0]["show_info_messages"] is False
    assert run_window.save_calls == 1
    assert reloads == ["reload"]
    assert info_calls
    assert info_calls[-1][0] == "Load Entire JSON complete"
    assert "Imported pages: 1/1" in info_calls[-1][1]
    window.close()


def test_dashboard_load_entire_json_prepares_before_import(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=0,
        annotated_page_count=0,
        progress_pct=0,
        status="Needs extraction",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    class _FakeDialog:
        def __init__(self, *, doc_title: str, page_count: int, parent=None) -> None:
            _ = doc_title, page_count, parent

        def exec_(self) -> int:
            return dashboard.QDialog.Accepted

        def json_text(self) -> str:
            return '[{"meta":{"page_num":"1"},"facts":[{"value":"10","path":[]}]}]'

    calls: list[tuple[Path, bool]] = []

    def _fake_start_import(
        pdf_path: Path,
        *,
        open_after: bool = True,
        success_callback=None,
        failure_callback=None,
    ) -> bool:
        _ = failure_callback
        calls.append((pdf_path, open_after))
        if callable(success_callback):
            success_callback(None)
        return True

    continued: list[tuple[str, str]] = []
    monkeypatch.setattr(dashboard, "LoadEntireJsonDialog", _FakeDialog)
    window._start_import = _fake_start_import  # type: ignore[method-assign]
    window._continue_load_workspace_document_entire_json = lambda doc_id, raw_text: continued.append((doc_id, raw_text))  # type: ignore[method-assign]

    window.load_workspace_document_entire_json("doc")

    assert calls == [(source_pdf, False)]
    assert continued == [("doc", '[{"meta":{"page_num":"1"},"facts":[{"value":"10","path":[]}]}]')]
    window.close()


def test_dashboard_auto_annotate_document_chains_prepare_and_starts_first_page(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=0,
        annotated_page_count=0,
        progress_pct=0,
        status="Needs extraction",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    class _RunWindow:
        def __init__(self) -> None:
            self.page_images = [tmp_path / "page_0001.png", tmp_path / "page_0002.png"]
            self.lock_calls: list[bool] = []
            self.started_pages: list[str] = []

        def has_unsaved_changes(self) -> bool:
            return False

        def set_document_auto_annotate_locked(self, locked: bool) -> None:
            self.lock_calls.append(bool(locked))

        def start_document_auto_annotate_page(self, page_name: str) -> bool:
            self.started_pages.append(page_name)
            return True

        def save_annotations(self, *, allow_locked: bool = False) -> bool:
            return True

    run_window = _RunWindow()
    monkeypatch.setattr(window.annotator_host, "open_document", lambda context, activate=False: run_window)

    import_calls: list[tuple[Path, bool]] = []

    def _fake_start_import(
        pdf_path: Path,
        *,
        open_after: bool = True,
        success_callback=None,
        failure_callback=None,
    ) -> bool:
        import_calls.append((pdf_path, open_after))
        if success_callback is not None:
            success_callback(None)
        return True

    window._start_import = _fake_start_import  # type: ignore[method-assign]

    window.auto_annotate_workspace_document("doc")
    _qt_app().processEvents()

    assert import_calls == [(source_pdf, False)]
    assert run_window.lock_calls == [True]
    assert run_window.started_pages == ["page_0001.png"]
    progress = window._runtime_auto_annotate_progress["doc"]
    assert progress.stage == "annotating"
    assert progress.total_pages == 2
    window.close()


def test_dashboard_auto_annotate_progress_advances_and_unlocks_on_completion(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    class _RunWindow:
        def __init__(self) -> None:
            self.page_images = [images_dir / "page_0001.png", images_dir / "page_0002.png"]
            self.lock_calls: list[bool] = []
            self.started_pages: list[str] = []
            self.save_calls: list[bool] = []

        def has_unsaved_changes(self) -> bool:
            return False

        def set_document_auto_annotate_locked(self, locked: bool) -> None:
            self.lock_calls.append(bool(locked))

        def start_document_auto_annotate_page(self, page_name: str) -> bool:
            self.started_pages.append(page_name)
            return True

        def save_annotations(self, *, allow_locked: bool = False) -> bool:
            self.save_calls.append(bool(allow_locked))
            return True

    run_window = _RunWindow()
    monkeypatch.setattr(window.annotator_host, "open_document", lambda context, activate=False: run_window)
    reloads: list[str] = []
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.auto_annotate_workspace_document("doc")
    _qt_app().processEvents()

    context = dashboard.DocumentContext.from_summary(summary)
    window._on_document_auto_annotate_page_completed(context, "page_0001.png", 7)
    _qt_app().processEvents()

    assert run_window.started_pages == ["page_0001.png", "page_0002.png"]
    progress = window._runtime_auto_annotate_progress["doc"]
    assert progress.completed_pages == 1
    assert progress.fact_count == 7
    assert progress.progress_pct == 50

    window._on_document_auto_annotate_page_completed(context, "page_0002.png", 9)
    _qt_app().processEvents()

    assert window._runtime_auto_annotate_progress == {}
    assert run_window.lock_calls == [True, False]
    assert run_window.save_calls == [True, True, True]
    assert reloads == ["reload"]
    window.close()


def test_dashboard_auto_annotate_waits_for_runner_to_become_idle_before_next_page(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    source_pdf = tmp_path / "doc.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    images_dir = tmp_path / "doc"
    images_dir.mkdir()
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=source_pdf,
        images_dir=images_dir,
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=0,
        progress_pct=0,
        status="Ready",
        updated_at=None,
    )
    window._documents_by_id = {"doc": summary}

    class _BusyController:
        def __init__(self) -> None:
            self.calls = 0

        def is_running(self) -> bool:
            self.calls += 1
            return self.calls == 1

    class _RunWindow:
        def __init__(self) -> None:
            self.page_images = [images_dir / "page_0001.png", images_dir / "page_0002.png"]
            self._ai_controller = _BusyController()
            self.started_pages: list[str] = []

        def has_unsaved_changes(self) -> bool:
            return False

        def set_document_auto_annotate_locked(self, locked: bool) -> None:
            return None

        def start_document_auto_annotate_page(self, page_name: str) -> bool:
            self.started_pages.append(page_name)
            return True

        def save_annotations(self, *, allow_locked: bool = False) -> bool:
            return True

    run_window = _RunWindow()
    monkeypatch.setattr(window.annotator_host, "open_document", lambda context, activate=False: run_window)

    window.auto_annotate_workspace_document("doc")
    _qt_app().processEvents()
    _qt_app().processEvents()

    assert run_window.started_pages == ["page_0001.png"]
    window.close()


def test_dashboard_refuses_review_for_documents_with_reg_flags(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=2,
        progress_pct=67,
        status="In progress",
        updated_at=None,
        reg_flag_count=3,
        warning_count=0,
        pages_with_reg_flags=2,
        pages_with_warnings=0,
        checked=True,
        reviewed=False,
    )
    window._documents_by_id = {"doc": summary}

    persisted: list[tuple[str, bool]] = []
    warnings: list[tuple[str, str]] = []
    reloads: list[str] = []
    monkeypatch.setattr(dashboard, "set_workspace_document_reviewed", lambda doc_id, reviewed: persisted.append((doc_id, reviewed)))
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.set_document_reviewed("doc", True)

    assert persisted == []
    assert reloads == ["reload"]
    assert warnings
    assert warnings[0][0] == "Cannot mark reviewed"
    assert "3 reg flag(s)" in warnings[0][1]
    window.close()


def test_dashboard_refuses_review_for_unchecked_documents(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=2,
        progress_pct=67,
        status="In progress",
        updated_at=None,
        checked=False,
        reviewed=False,
    )
    window._documents_by_id = {"doc": summary}

    persisted: list[tuple[str, bool]] = []
    warnings: list[tuple[str, str]] = []
    reloads: list[str] = []
    monkeypatch.setattr(dashboard, "set_workspace_document_reviewed", lambda doc_id, reviewed: persisted.append((doc_id, reviewed)))
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.set_document_reviewed("doc", True)

    assert persisted == []
    assert reloads == ["reload"]
    assert warnings == [("Cannot mark reviewed", "Mark this PDF as checked before review.")]
    window.close()


def test_dashboard_marks_document_checked(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=3,
        progress_pct=100,
        status="Complete",
        updated_at=None,
        checked=False,
        reviewed=False,
    )
    window._documents_by_id = {"doc": summary}

    persisted: list[tuple[str, bool]] = []
    reloads: list[str] = []
    monkeypatch.setattr(dashboard, "set_workspace_document_checked", lambda doc_id, checked: persisted.append((doc_id, checked)))
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.set_document_checked("doc", True)

    assert persisted == [("doc", True)]
    assert reloads == ["reload"]
    assert "marked as checked" in window.statusBar().currentMessage()
    window.close()


def test_dashboard_shows_saved_message_in_status_pane(tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    reloads: list[str] = []
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window._on_document_saved(tmp_path / "doc.json")

    assert reloads == ["reload"]
    assert window.statusBar().currentMessage() == "Workspace document saved."
    window.close()


def test_dashboard_shows_save_warnings_in_status_pane(tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    reloads: list[str] = []
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window._on_document_saved(
        {
            "annotations_path": tmp_path / "doc.json",
            "no_changes": False,
            "warning_count": 3,
            "backup_path": tmp_path / "doc.backup.json",
        }
    )

    assert reloads == ["reload"]
    message = window.statusBar().currentMessage()
    assert "Workspace document saved" in message
    assert "with 3 warning(s)" in message
    assert "Legacy backup created." in message
    window.close()


def test_dashboard_resets_all_approved_pages_workspace_wide(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary_a = WorkspaceDocumentSummary(
        doc_id="doc_a",
        source_pdf=tmp_path / "doc_a.pdf",
        images_dir=tmp_path / "doc_a",
        annotations_path=tmp_path / "doc_a.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=3,
        approved_page_count=2,
        progress_pct=100,
        status="Complete",
        updated_at=None,
    )
    summary_b = WorkspaceDocumentSummary(
        doc_id="doc_b",
        source_pdf=tmp_path / "doc_b.pdf",
        images_dir=tmp_path / "doc_b",
        annotations_path=tmp_path / "doc_b.json",
        thumbnail_path=None,
        page_count=2,
        annotated_page_count=1,
        approved_page_count=1,
        progress_pct=50,
        status="In progress",
        updated_at=None,
    )
    window._documents_by_id = {summary_a.doc_id: summary_a, summary_b.doc_id: summary_b}
    window.home_view.set_documents([summary_a, summary_b])

    confirmations: list[str] = []
    resets: list[Path] = []
    reloads: list[str] = []
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "question",
        lambda _parent, _title, message, *_args: (confirmations.append(message), dashboard.QMessageBox.Yes)[1],
    )
    monkeypatch.setattr(window.annotator_host, "managed_windows", lambda: [])
    monkeypatch.setattr(dashboard, "reset_document_approved_pages", lambda path: resets.append(path) or (2 if path == summary_a.annotations_path else 1))
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.reset_all_approved_pages()

    assert confirmations == ["Are you sure you want to disapprove 3 pages?"]
    assert resets == [summary_a.annotations_path, summary_b.annotations_path]
    assert reloads == ["reload"]
    assert window.statusBar().currentMessage() == "Disapproved 3 page(s)."
    window.close()


def test_dashboard_refuses_checked_for_incomplete_documents(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=1,
        progress_pct=33,
        status="In progress",
        updated_at=None,
        checked=False,
        reviewed=False,
    )
    window._documents_by_id = {"doc": summary}

    persisted: list[tuple[str, bool]] = []
    reloads: list[str] = []
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(dashboard, "set_workspace_document_checked", lambda doc_id, checked: persisted.append((doc_id, checked)))
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    window.set_document_checked("doc", True)

    assert persisted == []
    assert reloads == ["reload"]
    assert warnings == [("Cannot mark checked", "Finish annotating every page before marking this PDF as checked.")]
    window.close()


def test_dashboard_deletes_document_bundle_after_confirmation(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=1,
        annotated_page_count=1,
        progress_pct=100,
        status="Complete",
        updated_at=None,
        checked=True,
        reviewed=False,
    )
    window._documents_by_id = {"doc": summary}

    confirmations: list[str] = []
    deleted: list[str] = []
    close_calls: list[str] = []
    reloads: list[str] = []
    show_home_calls: list[str] = []
    monkeypatch.setattr(
        dashboard.QMessageBox,
        "question",
        lambda _parent, _title, message, *_args: (confirmations.append(message), dashboard.QMessageBox.Yes)[1],
    )
    monkeypatch.setattr(dashboard, "delete_workspace_document_files", lambda doc_id: deleted.append(doc_id))
    monkeypatch.setattr(window.annotator_host, "confirm_close_document", lambda doc_id: close_calls.append(f"confirm:{doc_id}") or True)
    monkeypatch.setattr(window.annotator_host, "close_document", lambda doc_id: close_calls.append(f"close:{doc_id}") or True)
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]
    window.show_home = lambda: show_home_calls.append("home")  # type: ignore[method-assign]

    window.delete_workspace_document("doc")

    assert confirmations and "Delete doc.pdf" in confirmations[0]
    assert deleted == ["doc"]
    assert close_calls == ["confirm:doc", "close:doc"]
    assert reloads == ["reload"]
    assert show_home_calls == ["home"]
    window.close()


def test_dashboard_auto_unreviews_document_when_live_reg_flags_appear(monkeypatch, tmp_path: Path) -> None:
    _qt_app()
    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)
    summary = WorkspaceDocumentSummary(
        doc_id="doc",
        source_pdf=tmp_path / "doc.pdf",
        images_dir=tmp_path / "doc",
        annotations_path=tmp_path / "doc.json",
        thumbnail_path=None,
        page_count=3,
        annotated_page_count=3,
        progress_pct=100,
        status="Complete",
        updated_at=None,
        checked=True,
        reviewed=True,
    )
    window._documents_by_id = {"doc": summary}

    persisted: list[tuple[str, bool]] = []
    reloads: list[str] = []
    monkeypatch.setattr(dashboard, "set_workspace_document_reviewed", lambda doc_id, reviewed: persisted.append((doc_id, reviewed)))
    window.reload_workspace = lambda: reloads.append("reload")  # type: ignore[method-assign]

    context = dashboard.DocumentContext.from_summary(summary)
    live_summary = SimpleNamespace(
        reg_flag_count=1,
        warning_count=2,
        pages_with_reg_flags=1,
        pages_with_warnings=1,
    )

    window._on_document_issues_changed(context, live_summary)

    assert persisted == [("doc", False)]
    assert reloads == ["reload"]
    assert "auto-unreviewed" in window.statusBar().currentMessage()
    window.close()


def test_dashboard_can_hide_nav_panel_and_persist_setting(monkeypatch) -> None:
    _qt_app()
    settings = _FakeSettings({"ui/nav_visible": True})
    monkeypatch.setattr(dashboard, "app_settings", lambda: settings)

    ctx = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    window = dashboard.DashboardWindow(ctx, dpi=200)

    assert not window.nav_rail.isHidden()
    assert window.show_nav_action.isChecked()
    assert not window.hide_nav_btn.isHidden()
    assert window.show_nav_btn.isHidden()

    window.set_nav_visible(False)

    assert window.nav_rail.isHidden()
    assert settings.values["ui/nav_visible"] is False
    assert not window.show_nav_action.isChecked()
    assert not window.show_nav_btn.isHidden()

    window.toggle_nav_visible()

    assert not window.nav_rail.isHidden()
    assert settings.values["ui/nav_visible"] is True
    assert window.show_nav_action.isChecked()
    assert window.show_nav_btn.isHidden()
    window.close()
