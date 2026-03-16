from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QKeyEvent, QPainter
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from finetree_annotator.annotation_core import PageState, default_page_meta
import finetree_annotator.app as app_mod
from finetree_annotator.app import AnnotRectItem, AnnotationWindow, item_scene_rect

_APP: QApplication | None = None


def _qt_app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def _write_test_png(
    path: Path,
    *,
    width: int = 32,
    height: int = 32,
    dark_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#ffffff"))
    if dark_rects:
        painter = QPainter(image)
        painter.fillRect(0, 0, width, height, QColor("#ffffff"))
        for x, y, w, h in dark_rects:
            painter.fillRect(x, y, w, h, QColor("#101010"))
        painter.end()
    assert image.save(str(path), "PNG")


class _SceneMouseEvent:
    def __init__(self, pos: QPointF, *, button=Qt.LeftButton, modifiers=Qt.NoModifier) -> None:
        self._pos = QPointF(pos)
        self._button = button
        self._modifiers = modifiers
        self.accepted = False

    def button(self):
        return self._button

    def scenePos(self):
        return QPointF(self._pos)

    def modifiers(self):
        return self._modifiers

    def accept(self) -> None:
        self.accepted = True


class _FakePainter:
    def __init__(self) -> None:
        self.pen_colors: list[str] = []
        self.pen_styles: list[int | None] = []
        self.pen_widths: list[int | None] = []
        self.draw_text_calls: list[tuple[object, object, object]] = []

    def save(self) -> None:
        return None

    def restore(self) -> None:
        return None

    def setPen(self, pen) -> None:
        color = pen.color() if hasattr(pen, "color") else pen
        self.pen_styles.append(pen.style() if hasattr(pen, "style") else None)
        self.pen_widths.append(pen.width() if hasattr(pen, "width") else None)
        if hasattr(color, "name"):
            self.pen_colors.append(color.name())
        else:
            self.pen_colors.append(str(color))

    def setBrush(self, _brush) -> None:
        return None

    def drawRect(self, _rect) -> None:
        return None

    def drawRoundedRect(self, _rect, _rx, _ry) -> None:
        return None

    def drawText(self, rect, flags, text) -> None:
        self.draw_text_calls.append((rect, flags, text))


@pytest.fixture(autouse=True)
def _auto_discard_unsaved_close(monkeypatch):
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")


def test_annotation_window_defaults_to_hidden_batch_panel_and_text_toolbar(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    assert not window.batch_box.isVisible()
    assert window.batch_toggle_btn.text() == "Show Batch Edit"
    assert window.save_btn.text() == "Save"
    assert window.ai_btn.text() == "AI"
    assert window.copy_image_btn.text() == "Copy Image"
    assert window.page_json_btn.text() == "JSON"
    assert window.exit_btn.text() == "Exit"
    assert window.page_jump_spin.minimumWidth() == 58
    assert not window.ai_btn.icon().isNull()
    assert window.page_thumb_list.iconSize().width() == 82
    assert window.thumb_panel.maximumWidth() == 152
    assert window.facts_list.objectName() == "factsList"
    assert window.facts_list.maximumHeight() == 176
    assert window.fact_editor_box.objectName() == "inspectorSubsection"
    assert window.fact_editor_box.layout().count() == 11
    assert window.apply_equation_btn.parentWidget() is not window.fact_editor_box
    assert not hasattr(window, "gemini_gt_btn")
    assert not hasattr(window, "gemini_complete_btn")
    assert not hasattr(window, "gemini_fill_btn")
    assert not hasattr(window, "qwen_gt_btn")
    assert window._ai_controller.dialog is None
    assert window.fact_is_beur_combo.maximumWidth() > 500
    assert window.page_annotation_status_label.text() == "Unclassified"
    assert window.page_flag_btn.text().endswith("Flag")
    assert window.facts_count_label.text() == "No facts"
    assert window.statusBar().isHidden()
    assert window.statusBar().maximumHeight() == 0

    window.close()


def test_multi_selection_scalar_edits_apply_to_all_selected_bboxes(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "currency": "USD", "ref_comment": "a"})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "currency": "ILS", "ref_comment": "b"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert not window.batch_box.isVisible()
    assert window.batch_toggle_btn.text() == "Show Batch Edit"
    assert window.fact_editor_box.title() == "Selected Facts (2)"
    assert window.fact_bbox_label.text() == "2 selected"
    assert window.fact_value_edit.placeholderText() == "Multiple values"
    assert window.fact_path_list.isEnabled()
    assert window.path_add_btn.isEnabled() is True

    window.batch_toggle_btn.click()
    _qt_app().processEvents()
    assert window.batch_box.isVisible()
    assert window.batch_toggle_btn.text() == "Hide Batch Edit"

    window.fact_value_edit.setText("shared")
    window.fact_value_edit.setModified(True)
    window._on_fact_editor_field_edited("value")

    assert item_a.fact_data["value"] == "shared"
    assert item_b.fact_data["value"] == "shared"

    idx = window.fact_currency_combo.findText("USD")
    window.fact_currency_combo.setCurrentIndex(idx)
    window._on_fact_editor_field_edited("currency")

    assert item_a.fact_data["currency"] == "USD"
    assert item_b.fact_data["currency"] == "USD"

    window.close()


def test_alt_f_focuses_fact_annotation_panel(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    window.view.setFocus()
    _qt_app().processEvents()
    QTest.keyClick(window, Qt.Key_F, Qt.AltModifier)
    _qt_app().processEvents()

    assert QApplication.focusWidget() is window.fact_value_edit
    window.close()


def test_ai_dialog_uses_supported_gemini_model_dropdown(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    options = [dialog.model_combo.itemText(index) for index in range(dialog.model_combo.count())]
    assert "gemini-3-flash-preview" in options
    assert "gemini-3.1-pro-preview" in options
    assert "gemini-2.5-flash" in options
    assert "gemini-flash-hf-tuned" in options
    assert dialog.current_provider() == app_mod.AIProvider.GEMINI
    dialog.close()
    window.close()


def test_ai_fix_dialog_uses_supported_gemini_model_dropdown(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.FIX_SELECTED)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    options = [dialog.model_combo.itemText(index) for index in range(dialog.model_combo.count())]
    assert "gemini-3-flash-preview" in options
    assert "gemini-3.1-flash-lite" in options
    assert "gemini-2.5-pro" in options
    assert "gemini-flash-hf-tuned" in options
    dialog.close()
    window.close()


def test_ai_ground_truth_dialog_defaults_max_facts_to_zero(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.max_facts_spin.isHidden() is False
    assert dialog.max_facts_spin.value() == 0
    dialog.close()
    window.close()


def test_ai_dialog_prompt_section_starts_collapsed(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.prompt_toggle.isChecked() is False
    assert dialog.prompt_frame.isVisible() is False
    dialog.close()
    window.close()


def test_ai_dialog_running_state_disables_config_and_enables_stop(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None

    window._ai_controller._set_status("Running...", fact_count=2, running=True)

    assert dialog.provider_combo.isEnabled() is False
    assert dialog.action_combo.isEnabled() is False
    assert dialog.model_combo.isEnabled() is False
    assert dialog.run_btn.isEnabled() is False
    assert dialog.stop_btn.isEnabled() is True

    dialog.close()
    window.close()


def test_ai_autocomplete_dialog_shows_few_shot_controls_disabled_by_default(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "period_type": "instant"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    _qt_app().processEvents()
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.AUTO_COMPLETE)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.few_shot_check.isChecked() is False
    assert dialog.few_shot_preset_combo.currentData() == app_mod.FEW_SHOT_PRESET_CLASSIC
    assert dialog.max_facts_spin.value() == 0
    dialog.close()
    window.close()


def test_ai_ground_truth_dialog_defaults_to_two_shot_preset(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.open_ai_dialog()
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.current_provider() == app_mod.AIProvider.GEMINI
    assert dialog.current_action() == app_mod.AIActionKind.GROUND_TRUTH
    assert dialog.current_model() == "gemini-3-flash-preview"
    assert dialog.thinking_check.isChecked() is False
    assert dialog.thinking_level_combo.currentText().lower() == "minimal"
    assert dialog.few_shot_check.isChecked() is True
    assert dialog.few_shot_preset_combo.currentData() == app_mod.FEW_SHOT_PRESET_2015_TWO_SHOT
    dialog.close()
    window.close()


def test_ai_fix_dialog_defaults_only_period_fields_checked(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.FIX_SELECTED)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.selected_fix_fields() == {"period_type", "period_start", "period_end"}
    assert "equations" in dialog._fix_field_checks
    dialog.close()
    window.close()


def test_ai_fix_dialog_clear_all_unchecks_every_fact_field(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.FIX_SELECTED)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    dialog._fix_field_checks["currency"].setChecked(True)
    dialog._fix_field_checks["date"].setChecked(True)
    dialog.clear_all_fields_btn.click()
    assert dialog.selected_fix_fields() == set()
    dialog.close()
    window.close()


def test_gemini_autocomplete_request_payload_includes_locked_page_state(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.statement_type_combo.setCurrentText("income_statement")
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "100",
            "period_type": "instant",
            "path": ["Assets"],
            "equations": [{"equation": "70 + 30", "fact_equation": "f2 + f3"}],
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window._capture_current_state()
    _qt_app().processEvents()

    payload = window._build_gemini_autocomplete_request_payload(
        page_name=window.page_images[window.current_index].name,
        ordered_items=window._sorted_fact_items(),
    )

    assert payload["pages"][0]["meta"]["statement_type"] == "income_statement"
    assert len(payload["pages"][0]["facts"]) == 1
    assert payload["pages"][0]["facts"][0]["fact_num"] == 1
    assert payload["pages"][0]["facts"][0]["equations"] == [{"equation": "70 + 30", "fact_equation": "f2 + f3"}]
    assert "metadata" not in payload
    assert "images_dir" not in payload
    assert "request" not in payload

    window.close()


def test_gemini_autocomplete_prompt_mentions_locked_facts(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    prompt = window._build_gemini_autocomplete_prompt_from_template(
        app_mod.default_gemini_autocomplete_prompt_template(),
        request_payload={
            "pages": [{"image": "page_0001.png", "meta": {}, "facts": []}],
        },
    )

    assert "must stay locked" in prompt
    assert "return only new missing facts" in prompt.lower()
    assert "original image pixel coordinates" in prompt.lower()
    assert "current image size: 32 x 32 pixels." in prompt.lower()
    assert "start with this" in prompt.lower()
    assert '"image": "page_0001.png"' in prompt
    assert "runtime rebuilds final contiguous numbering" in prompt.lower()
    assert '"metadata"' not in prompt

    window.close()


def test_ai_autocomplete_dialog_shows_empty_page_validation(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.AUTO_COMPLETE)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    assert dialog.validation_label.text() == "Auto Complete requires at least one existing fact on the current page."
    assert dialog.run_btn.isEnabled() is False
    dialog.close()
    window.close()


def test_gemini_autocomplete_passes_few_shot_examples_when_enabled(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "period_type": "instant", "path": ["Assets"]})
    window.scene.addItem(item)
    window.refresh_facts_list()
    _qt_app().processEvents()

    captured_stream_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        "finetree_annotator.gemini_vlm.ensure_gemini_backend_credentials",
        lambda _model_name, explicit_api_key=None: ("test-key", None),
    )
    monkeypatch.setattr(
        window,
        "_load_gemini_few_shot_examples",
        lambda *, preset: ([{"role": "user", "parts": ["example"]}], []),
    )
    monkeypatch.setattr(window._ai_controller, "_start_gemini_stream", lambda **kwargs: captured_stream_kwargs.update(kwargs))

    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.AUTO_COMPLETE)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    dialog.prompt_edit.setPlainText("autocomplete prompt")
    dialog.few_shot_check.setChecked(True)
    dialog.few_shot_preset_combo.setCurrentIndex(dialog.few_shot_preset_combo.findData(app_mod.FEW_SHOT_PRESET_ONE_SHOT))
    window._ai_controller.run_from_dialog()

    assert captured_stream_kwargs["few_shot_examples"] == [{"role": "user", "parts": ["example"]}]
    assert captured_stream_kwargs["mode"] == "autocomplete"

    window.close()


def test_ai_dialog_closes_after_run_starts(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    monkeypatch.setattr(
        "finetree_annotator.gemini_vlm.ensure_gemini_backend_credentials",
        lambda _model_name, explicit_api_key=None: ("test-key", None),
    )

    def _start_stream(**kwargs) -> None:
        _ = kwargs
        window._gemini_stream_thread = object()

    monkeypatch.setattr(window._ai_controller, "_start_gemini_stream", _start_stream)

    window.open_ai_dialog()
    dialog = window._ai_controller.dialog
    assert dialog is not None
    dialog.prompt_edit.setPlainText("ground truth prompt")

    try:
        window._ai_controller.run_from_dialog()
        assert dialog.isVisible() is False
    finally:
        window._gemini_stream_thread = None

    window.close()


def test_gemini_autocomplete_passes_max_facts_to_stream(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "period_type": "instant", "path": ["Assets"]})
    window.scene.addItem(item)
    window.refresh_facts_list()
    _qt_app().processEvents()

    captured_stream_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        "finetree_annotator.gemini_vlm.ensure_gemini_backend_credentials",
        lambda _model_name, explicit_api_key=None: ("test-key", None),
    )
    monkeypatch.setattr(window._ai_controller, "_start_gemini_stream", lambda **kwargs: captured_stream_kwargs.update(kwargs))

    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.AUTO_COMPLETE)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    dialog.prompt_edit.setPlainText("autocomplete prompt")
    dialog.max_facts_spin.setValue(2)
    window._ai_controller.run_from_dialog()

    assert captured_stream_kwargs["max_facts"] == 2
    assert captured_stream_kwargs["mode"] == "autocomplete"

    window.close()


def test_gemini_gt_vertex_model_does_not_require_standard_api_key(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)

    captured_stream_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        "finetree_annotator.gemini_vlm.ensure_gemini_backend_credentials",
        lambda _model_name, explicit_api_key=None: (None, None),
    )
    monkeypatch.setattr(window._ai_controller, "_start_gemini_stream", lambda **kwargs: captured_stream_kwargs.update(kwargs))

    window.open_ai_dialog(provider=app_mod.AIProvider.GEMINI, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    dialog.prompt_edit.setPlainText("vertex prompt")
    dialog.model_combo.setCurrentText("gemini-flash-hf-tuned")
    window._ai_controller.run_from_dialog()

    assert captured_stream_kwargs["model_name"] == "gemini-flash-hf-tuned"
    assert captured_stream_kwargs["gemini_api_key"] is None

    window.close()


def test_gemini_autocomplete_ignores_meta_and_merges_only_missing_facts(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.statement_type_combo.setCurrentText("income_statement")
    locked_item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "period_type": "instant", "path": ["Assets"]})
    window.scene.addItem(locked_item)
    window.refresh_facts_list()
    window._capture_current_state()
    _qt_app().processEvents()

    page_name = window.page_images[window.current_index].name
    ordered_items = window._sorted_fact_items()
    locked_payload = window._fact_payload_from_item(locked_item)
    locked_before = dict(locked_item.fact_data)
    window._gemini_autocomplete_snapshot = {
        "page_name": page_name,
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
        "locked_fact_payloads": [app_mod.deepcopy(window._fact_payload_from_item(item)) for item in ordered_items],
    }
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "autocomplete"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = {window._fact_uniqueness_key(locked_payload)}
    window._gemini_stream_fact_count = 0
    window._gemini_autocomplete_buffered_facts = []

    window._on_gemini_stream_meta({"statement_type": "balance_sheet"})

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "balance_sheet",
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(model_dump=lambda mode="json", payload=locked_payload: payload),
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [40, 10, 20, 20],
                    "value": "200",
                    "equation": None,
                    "value_type": None,
                    "value_context": None,
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": None,
                    "scale": None,
                    "date": None,
                    "period_type": "duration",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["Revenue"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            ),
        ],
    )

    window._on_gemini_stream_completed(extraction)

    assert locked_item.fact_data == locked_before
    assert window.statement_type_combo.currentText() == "income_statement"
    assert window._gemini_stream_fact_count == 1
    assert len(window._fact_items) == 2
    assert window._fact_items[1].fact_data["value"] == "200"
    assert window._fact_items[1].fact_data["period_type"] == "duration"

    window.close()


def test_gemini_gt_stream_fact_is_buffered_until_completion(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=1617, height=2384)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_seen_facts = set()
    window._gemini_stream_apply_meta = False

    added = window._apply_stream_fact(
        page_name,
        {
            "bbox": [474, 731, 58, 30],
            "value": "18,763",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["A"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        stream_source="gemini",
    )

    assert added is True
    assert len(window._fact_items) == 0
    assert len(window._gemini_gt_buffered_facts) == 1

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [474, 731, 58, 30],
                    "value": "18,763",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["A"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    rect = item_scene_rect(window._fact_items[0])
    assert len(window._fact_items) == 1
    assert rect.x() == pytest.approx(474.0)
    assert rect.y() == pytest.approx(731.0)
    assert rect.width() == pytest.approx(58.0)
    assert rect.height() == pytest.approx(30.0)
    assert window._gemini_gt_last_bbox_mode == app_mod.BBOX_MODE_PIXEL_AS_IS

    window.close()


def test_gemini_gt_stream_renders_live_after_lock_threshold(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(
        images_dir / "page_0001.png",
        width=200,
        height=200,
        dark_rects=[(120, 20, 20, 10), (120, 40, 20, 10), (120, 60, 20, 10), (120, 80, 20, 10)],
    )
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_seen_facts = set()
    window._gemini_stream_apply_meta = False
    window._gemini_stream_fact_count = 0

    payloads = [
        {
            "bbox": [120, 20, 20, 10],
            "value": "10",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["A"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [120, 40, 20, 10],
            "value": "11",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["B"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [120, 60, 20, 10],
            "value": "12",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["C"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [120, 80, 20, 10],
            "value": "13",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["D"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
    ]

    for payload in payloads[:3]:
        window._on_gemini_stream_fact(payload)
    assert len(window._fact_items) == 0
    assert window._gemini_gt_live_bbox_mode_locked is False

    window._on_gemini_stream_fact(payloads[3])

    assert len(window._fact_items) == 4
    assert window._gemini_gt_live_bbox_mode_locked is True
    assert window._gemini_gt_live_applied is True
    rect = item_scene_rect(window._fact_items[0])
    assert rect.x() == pytest.approx(120.0)
    assert rect.y() == pytest.approx(20.0)

    window.close()


def test_gemini_gt_finalize_reconciles_mode_after_live_lock(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_seen_facts = set()
    window._gemini_stream_apply_meta = False
    window._gemini_stream_fact_count = 0

    calls = {"count": 0}
    def _fake_resolve(target_page: str, fact_payloads: list[dict[str, object]]):
        calls["count"] += 1
        if calls["count"] == 1:
            window._gemini_autocomplete_last_bbox_scores = {
                app_mod.BBOX_MODE_PIXEL_AS_IS: 0.9,
                app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.1,
            }
            return (
                window._gemini_gt_payloads_for_bbox_mode(
                    page_name=target_page,
                    fact_payloads=fact_payloads,
                    mode=app_mod.BBOX_MODE_PIXEL_AS_IS,
                ),
                app_mod.BBOX_MODE_PIXEL_AS_IS,
            )
        window._gemini_autocomplete_last_bbox_scores = {
            app_mod.BBOX_MODE_PIXEL_AS_IS: 0.1,
            app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.9,
        }
        return (
            window._gemini_gt_payloads_for_bbox_mode(
                page_name=target_page,
                fact_payloads=fact_payloads,
                mode=app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
            ),
            app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
        )

    window._resolve_autocomplete_bbox_mode = _fake_resolve  # type: ignore[assignment]

    payloads = [
        {
            "bbox": [100, 100, 40, 20],
            "value": "10",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["A"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [100, 140, 40, 20],
            "value": "11",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["B"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [150, 100, 40, 20],
            "value": "12",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["C"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
        {
            "bbox": [150, 140, 40, 20],
            "value": "13",
            "equation": None,
            "value_type": "amount",
            "value_context": "tabular",
            "natural_sign": "positive",
            "row_role": "detail",
            "currency": "ILS",
            "scale": 1,
            "date": "2014",
            "period_type": "duration",
            "period_start": "2014-01-01",
            "period_end": "2014-12-31",
            "duration_type": None,
            "recurring_period": None,
            "note_flag": False,
            "note_num": None,
            "note_name": None,
            "path": ["D"],
            "path_source": "observed",
            "note_ref": None,
            "comment_ref": None,
        },
    ]

    for payload in payloads:
        window._on_gemini_stream_fact(payload)

    assert len(window._fact_items) == 4
    live_rect = item_scene_rect(window._fact_items[0])
    assert live_rect.x() == pytest.approx(100.0)
    assert window._gemini_gt_live_bbox_mode_locked is True

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[SimpleNamespace(model_dump=lambda mode="json", _payload=payload: _payload) for payload in payloads],
    )

    window._on_gemini_stream_completed(extraction)

    assert window._gemini_gt_last_bbox_mode == app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL
    final_rect = item_scene_rect(window._fact_items[0])
    assert final_rect.x() == pytest.approx(20.0)
    assert final_rect.y() == pytest.approx(20.0)
    assert final_rect.width() == pytest.approx(8.0)
    assert final_rect.height() == pytest.approx(4.0)
    assert window._gemini_stream_fact_count == 4

    window.close()


def test_gemini_gt_bbox_mode_converts_normalized_1000_when_ink_score_is_better(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(
        images_dir / "page_0001.png",
        width=200,
        height=200,
        dark_rects=[(100, 100, 40, 20)],
    )
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = set()

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [500, 500, 200, 100],
                    "value": "12",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["A"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    rect = item_scene_rect(window._fact_items[0])
    assert rect.x() == pytest.approx(100.0)
    assert rect.y() == pytest.approx(100.0)
    assert rect.width() == pytest.approx(40.0)
    assert rect.height() == pytest.approx(20.0)
    assert window._gemini_gt_last_bbox_mode == app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL
    assert window._gemini_gt_last_bbox_scores[app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL] > window._gemini_gt_last_bbox_scores[app_mod.BBOX_MODE_PIXEL_AS_IS]

    window.close()


def test_gemini_gt_bbox_mode_keeps_pixel_when_pixel_score_is_better(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(
        images_dir / "page_0001.png",
        width=200,
        height=200,
        dark_rects=[(120, 110, 30, 20)],
    )
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = set()

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [120, 110, 30, 20],
                    "value": "55",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["A"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    rect = item_scene_rect(window._fact_items[0])
    assert rect.x() == pytest.approx(120.0)
    assert rect.y() == pytest.approx(110.0)
    assert rect.width() == pytest.approx(30.0)
    assert rect.height() == pytest.approx(20.0)
    assert window._gemini_gt_last_bbox_mode == app_mod.BBOX_MODE_PIXEL_AS_IS
    assert window._gemini_gt_last_bbox_scores[app_mod.BBOX_MODE_PIXEL_AS_IS] >= window._gemini_gt_last_bbox_scores[app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL]

    window.close()


def test_gemini_gt_bbox_mode_defaults_to_pixel_when_scores_are_ambiguous(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = set()

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [120, 110, 30, 20],
                    "value": "55",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["A"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    assert window._gemini_gt_last_bbox_mode == app_mod.BBOX_MODE_PIXEL_AS_IS

    window.close()


def test_gemini_gt_dedupes_facts_across_stream_and_finalize(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200, dark_rects=[(120, 110, 30, 20)])
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "gt"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = set()
    window._gemini_stream_fact_count = 0

    payload = {
        "bbox": [120, 110, 30, 20],
        "value": "55",
        "equation": None,
        "value_type": "amount",
        "value_context": "tabular",
        "natural_sign": "positive",
        "row_role": "detail",
        "currency": "ILS",
        "scale": 1,
        "date": "2014",
        "period_type": "duration",
        "period_start": "2014-01-01",
        "period_end": "2014-12-31",
        "duration_type": None,
        "recurring_period": None,
        "note_flag": False,
        "note_num": None,
        "note_name": None,
        "path": ["A"],
        "path_source": "observed",
        "note_ref": None,
        "comment_ref": None,
    }

    window._on_gemini_stream_fact(payload)
    assert len(window._fact_items) == 0
    assert len(window._gemini_gt_buffered_facts) == 1

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "statements",
                "statement_type": "income_statement",
                "title": None,
            }
        ),
        facts=[SimpleNamespace(model_dump=lambda mode="json", _payload=payload: _payload)],
    )

    window._on_gemini_stream_completed(extraction)

    assert len(window._fact_items) == 1
    assert window._gemini_stream_fact_count == 1

    window.close()


def test_autocomplete_bbox_mode_converts_normalized_1000_when_ink_score_is_better(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(
        images_dir / "page_0001.png",
        width=200,
        height=200,
        dark_rects=[(100, 100, 40, 20)],
    )
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name

    resolved_payloads, bbox_mode = window._resolve_autocomplete_bbox_mode(
        page_name,
        [
            {
                "bbox": [500, 500, 200, 100],
                "value": "12",
                "row_role": "detail",
                "path": ["A"],
            }
        ],
    )

    assert bbox_mode == app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL
    resolved_bbox = resolved_payloads[0]["bbox"]
    assert resolved_bbox[0] == pytest.approx(100.0)
    assert resolved_bbox[1] == pytest.approx(100.0)
    assert resolved_bbox[2] == pytest.approx(40.0)
    assert resolved_bbox[3] == pytest.approx(20.0)
    assert window._gemini_autocomplete_last_bbox_scores[app_mod.BBOX_MODE_NORMALIZED_1000_TO_PIXEL] > window._gemini_autocomplete_last_bbox_scores[app_mod.BBOX_MODE_PIXEL_AS_IS]

    window.close()


def test_autocomplete_bbox_mode_keeps_pixel_when_pixel_score_is_better(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(
        images_dir / "page_0001.png",
        width=200,
        height=200,
        dark_rects=[(120, 110, 30, 20)],
    )
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name

    resolved_payloads, bbox_mode = window._resolve_autocomplete_bbox_mode(
        page_name,
        [
            {
                "bbox": [120, 110, 30, 20],
                "value": "55",
                "row_role": "detail",
                "path": ["A"],
            }
        ],
    )

    assert bbox_mode == app_mod.BBOX_MODE_PIXEL_AS_IS
    assert resolved_payloads[0]["bbox"][0] == pytest.approx(120.0)
    assert resolved_payloads[0]["bbox"][1] == pytest.approx(110.0)

    window.close()


def test_autocomplete_bbox_mode_defaults_to_pixel_when_scores_are_ambiguous(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    page_name = window.page_images[window.current_index].name

    resolved_payloads, bbox_mode = window._resolve_autocomplete_bbox_mode(
        page_name,
        [
            {
                "bbox": [120, 110, 30, 20],
                "value": "55",
                "row_role": "detail",
                "path": ["A"],
            }
        ],
    )

    assert bbox_mode == app_mod.BBOX_MODE_PIXEL_AS_IS
    assert resolved_payloads[0]["bbox"][0] == pytest.approx(120.0)
    assert resolved_payloads[0]["bbox"][1] == pytest.approx(110.0)

    window.close()


def test_gemini_autocomplete_merges_buffered_facts_between_locked_prefix_and_suffix(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=400, height=400)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    locked_values = ["A", "B", "X", "Y", "Z"]
    locked_ys = [10, 40, 220, 260, 300]
    for value, y in zip(locked_values, locked_ys):
        window.scene.addItem(AnnotRectItem(QRectF(20, y, 40, 20), {"value": value, "path": [value]}))
    window.refresh_facts_list()
    _qt_app().processEvents()

    page_name = window.page_images[window.current_index].name
    ordered_items = window._sorted_fact_items()
    window._gemini_autocomplete_snapshot = {
        "page_name": page_name,
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
        "locked_fact_payloads": [app_mod.deepcopy(window._fact_payload_from_item(item)) for item in ordered_items],
    }
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "autocomplete"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = {
        window._fact_uniqueness_key(window._fact_payload_from_item(item)) for item in ordered_items
    }
    window._gemini_stream_fact_count = 0
    window._gemini_autocomplete_buffered_facts = []

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "other",
                "statement_type": None,
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [20, 120, 40, 20],
                    "value": "M1",
                    "equation": "A + B",
                    "fact_equation": "f1 + f2",
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "total",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["M1"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            ),
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [20, 170, 40, 20],
                    "value": "M2",
                    "equation": None,
                    "fact_equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": "ILS",
                    "scale": 1,
                    "date": "2014",
                    "period_type": "duration",
                    "period_start": "2014-01-01",
                    "period_end": "2014-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["M2"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            ),
        ],
    )

    window._on_gemini_stream_completed(extraction)

    assert [item.fact_data["value"] for item in window._fact_items] == ["A", "B", "M1", "M2", "X", "Y", "Z"]
    assert [item.fact_data["fact_num"] for item in window._fact_items] == [1, 2, 3, 4, 5, 6, 7]
    assert window._fact_items[2].fact_data["equations"] == [{"equation": "A + B", "fact_equation": "f1 + f2"}]

    window.close()


def test_gemini_autocomplete_rejects_stale_snapshot_without_changes(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    locked_item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "path": ["A"]})
    window.scene.addItem(locked_item)
    window.refresh_facts_list()
    _qt_app().processEvents()

    page_name = window.page_images[window.current_index].name
    ordered_items = window._sorted_fact_items()
    window._gemini_autocomplete_snapshot = {
        "page_name": page_name,
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
        "locked_fact_payloads": [app_mod.deepcopy(window._fact_payload_from_item(item)) for item in ordered_items],
    }
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "autocomplete"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = {
        window._fact_uniqueness_key(window._fact_payload_from_item(item)) for item in ordered_items
    }
    window._gemini_stream_fact_count = 0
    window._gemini_autocomplete_buffered_facts = []

    warnings: list[str] = []
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "warning",
        lambda *_args: warnings.append(str(_args[2]) if len(_args) > 2 else "warning"),
    )

    locked_item.fact_data = {**locked_item.fact_data, "value": "changed"}
    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "other",
                "statement_type": None,
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [40, 10, 20, 20],
                    "value": "200",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": None,
                    "scale": None,
                    "date": None,
                    "period_type": "duration",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["Revenue"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    assert [item.fact_data["value"] for item in window._fact_items] == ["changed"]
    assert warnings
    assert "Current page facts changed during Gemini Auto Complete" in warnings[0]

    window.close()


def test_gemini_autocomplete_remaps_locked_equation_refs_after_geometry_resequence(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=400, height=400)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    child = AnnotRectItem(QRectF(20, 80, 40, 20), {"value": "100", "path": ["child"], "fact_num": 1})
    total = AnnotRectItem(
        QRectF(20, 140, 40, 20),
        {
            "value": "100",
            "path": ["total"],
            "fact_num": 2,
            "equation": "100",
            "fact_equation": "f1",
            "row_role": "total",
        },
    )
    window.scene.addItem(child)
    window.scene.addItem(total)
    window.refresh_facts_list()
    _qt_app().processEvents()

    page_name = window.page_images[window.current_index].name
    ordered_items = window._sorted_fact_items()
    window._gemini_autocomplete_snapshot = {
        "page_name": page_name,
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
        "locked_fact_payloads": [app_mod.deepcopy(window._fact_payload_from_item(item)) for item in ordered_items],
    }
    window._gemini_stream_target_page = page_name
    window._gemini_stream_mode = "autocomplete"
    window._gemini_stream_apply_meta = False
    window._gemini_stream_seen_facts = {
        window._fact_uniqueness_key(window._fact_payload_from_item(item)) for item in ordered_items
    }
    window._gemini_stream_fact_count = 0
    window._gemini_autocomplete_buffered_facts = []

    extraction = SimpleNamespace(
        meta=SimpleNamespace(
            model_dump=lambda mode="json": {
                "entity_name": None,
                "page_num": None,
                "page_type": "other",
                "statement_type": None,
                "title": None,
            }
        ),
        facts=[
            SimpleNamespace(
                model_dump=lambda mode="json": {
                    "bbox": [20, 20, 40, 20],
                    "value": "50",
                    "equation": None,
                    "value_type": "amount",
                    "value_context": "tabular",
                    "natural_sign": "positive",
                    "row_role": "detail",
                    "currency": None,
                    "scale": None,
                    "date": None,
                    "period_type": None,
                    "period_start": None,
                    "period_end": None,
                    "duration_type": None,
                    "recurring_period": None,
                    "note_flag": False,
                    "note_num": None,
                    "note_name": None,
                    "path": ["new"],
                    "path_source": "observed",
                    "note_ref": None,
                    "comment_ref": None,
                }
            )
        ],
    )

    window._on_gemini_stream_completed(extraction)

    assert [item.fact_data["value"] for item in window._fact_items] == ["50", "100", "100"]
    assert window._fact_items[2].fact_data["equations"][0]["fact_equation"] == "f2"

    window.close()


def test_ai_dialog_qwen_gt_accepts_thinking_controls(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    window.open_ai_dialog(provider=app_mod.AIProvider.QWEN, action=app_mod.AIActionKind.GROUND_TRUTH)
    dialog = window._ai_controller.dialog
    assert dialog is not None
    options = [dialog.model_combo.itemText(index) for index in range(dialog.model_combo.count())]
    assert "qwen-flash-gt" in options
    assert "qwen3.5-flash" in options
    assert "qwen3.5-plus" in options
    assert dialog.current_provider() == app_mod.AIProvider.QWEN
    assert dialog.thinking_check.isVisible() is True
    assert dialog.thinking_level_combo.isVisible() is False
    dialog.close()
    window.close()


def test_gemini_fill_completed_applies_equations_patch(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "fact_num": 1, "path": ["Total"]})
    window.scene.addItem(item)
    window.refresh_facts_list()
    item.setSelected(True)
    _qt_app().processEvents()

    ordered_items = window._sorted_fact_items()
    window._gemini_fill_target_page = window.page_images[window.current_index].name
    window._gemini_fill_snapshot = {
        "page_name": window._gemini_fill_target_page,
        "selected_fact_nums": [1],
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
    }
    window._gemini_fill_include_statement_type = False
    monkeypatch.setattr(app_mod.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._on_gemini_fill_completed(
        {
            "meta_updates": {},
            "fact_updates": [
                {
                    "fact_num": 1,
                    "updates": {
                        "row_role": "total",
                        "equations": [{"equation": "100 + 20", "fact_equation": "f2 + f3"}],
                    },
                }
            ],
        }
    )

    assert item.fact_data["row_role"] == "total"
    assert item.fact_data["equations"] == [{"equation": "100 + 20", "fact_equation": "f2 + f3"}]
    window.close()


def test_gemini_autocomplete_generated_fact_payload_preserves_equations(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    payload = window._normalized_autocomplete_generated_fact_payload(
        {
            "bbox": [10, 10, 20, 20],
            "value": "120",
            "row_role": "total",
            "equations": [{"equation": "100 + 20", "fact_equation": "f1 + f2"}],
            "path": ["Total"],
        }
    )

    assert payload is not None
    assert payload["equations"] == [{"equation": "100 + 20", "fact_equation": "f1 + f2"}]
    window.close()


def test_note_num_clear_sets_null(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "note_flag": True, "note_num": 9})
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_beur_num_edit.text() == "9"
    window.fact_beur_num_edit.setText("")
    window.fact_beur_num_edit.setModified(True)
    assert window.fact_beur_num_edit.hasAcceptableInput() is True
    window._on_fact_editor_field_edited("note_num")

    assert item.fact_data["note_num"] is None
    window.close()


def test_page_revisit_note_persists_and_shows_in_thumbnail_tooltip(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    monkeypatch.setattr(app_mod.QMessageBox, "information", lambda *args, **kwargs: app_mod.QMessageBox.Ok)
    monkeypatch.setattr(app_mod.QMessageBox, "warning", lambda *args, **kwargs: app_mod.QMessageBox.Ok)
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    window.page_annotation_note_edit.setText("revisit disclosures")
    window._on_meta_edited()
    _qt_app().processEvents()

    page_name = window.page_images[window.current_index].name
    assert window.page_states[page_name].meta["annotation_note"] == "revisit disclosures"
    thumb_item = window.page_thumb_list.item(window.current_index)
    assert thumb_item is not None
    assert "Revisit: revisit disclosures" in thumb_item.toolTip()

    assert window.save_annotations() is True
    window.close()

    reloaded = AnnotationWindow(images_dir, annotations_path)
    reloaded.show()
    _qt_app().processEvents()
    reloaded_page_name = reloaded.page_images[reloaded.current_index].name
    assert reloaded.page_states[reloaded_page_name].meta["annotation_note"] == "revisit disclosures"
    assert reloaded.page_annotation_note_edit.text() == "revisit disclosures"
    reloaded.close()


def test_page_approval_and_flag_controls_persist_in_natural_thumbnail_order(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    monkeypatch.setattr(app_mod.QMessageBox, "information", lambda *args, **kwargs: app_mod.QMessageBox.Ok)
    monkeypatch.setattr(app_mod.QMessageBox, "warning", lambda *args, **kwargs: app_mod.QMessageBox.Ok)
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    _write_test_png(images_dir / "page_0002.png")
    _write_test_png(images_dir / "page_0003.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    assert window.current_index == 0
    window.page_approve_continue_btn.click()
    _qt_app().processEvents()
    assert window.page_states["page_0001.png"].meta["annotation_status"] == "approved"
    assert window.current_index == 1

    window.page_flag_btn.click()
    _qt_app().processEvents()
    assert window.page_states["page_0002.png"].meta["annotation_status"] == "flagged"
    assert window.page_annotation_status_label.text() == "Flagged"
    order = [window.page_thumb_list.item(i).data(Qt.UserRole) for i in range(window.page_thumb_list.count())]
    assert order == [0, 1, 2]

    assert window.save_annotations() is True
    window.close()

    reloaded = AnnotationWindow(images_dir, annotations_path)
    reloaded.show()
    _qt_app().processEvents()
    assert reloaded.page_states["page_0001.png"].meta["annotation_status"] == "approved"
    assert reloaded.page_states["page_0002.png"].meta["annotation_status"] == "flagged"
    reloaded.close()


def test_gemini_fill_request_payload_redacts_only_requested_fields(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.statement_type_combo.setCurrentText("income_statement")
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "100",
            "equations": [{"equation": "40 + 60", "fact_equation": None}],
            "row_role": "total",
            "period_type": "instant",
            "period_start": None,
            "period_end": "2024-12-31",
            "path_source": "observed",
        },
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {
            "value": "200",
            "equations": [{"equation": "120 + 80", "fact_equation": None}],
            "row_role": "total",
            "period_type": "duration",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "path_source": "inferred",
        },
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    ordered_items = window._sorted_fact_items()
    selected_fact_nums = [item.fact_data["fact_num"] for item in ordered_items if item in {item_a, item_b}]
    payload = window._build_gemini_fill_request_payload(
        page_name=window.page_images[window.current_index].name,
        selected_fact_nums=selected_fact_nums,
        selected_fact_fields={"period_type", "period_start", "period_end"},
        include_statement_type=True,
        ordered_items=ordered_items,
    )

    facts = payload["pages"][0]["facts"]
    assert len(facts) == 2
    assert payload["pages"][0]["meta"]["statement_type"] is None
    assert "metadata" not in payload
    assert "images_dir" not in payload
    assert "request" not in payload
    assert all(fact["period_type"] is None for fact in facts)
    assert all(fact["period_start"] is None for fact in facts)
    assert all(fact["period_end"] is None for fact in facts)
    assert {fact["path_source"] for fact in facts} == {"observed", "inferred"}
    assert item_a.fact_data["equations"][0]["equation"] == "40 + 60"
    assert item_a.fact_data["period_type"] == "instant"
    assert item_b.fact_data["equations"][0]["equation"] == "120 + 80"
    assert item_b.fact_data["period_type"] == "duration"

    window.close()


def test_fact_editor_shows_row_role_and_deterministic_natural_sign(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "(120)", "path": []},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_row_role_combo.currentText() == "detail"
    assert window.fact_natural_sign_label.text() == "negative"

    idx = window.fact_row_role_combo.findText("total")
    window.fact_row_role_combo.setCurrentIndex(idx)
    window._on_fact_editor_field_edited("row_role")
    assert item.fact_data["row_role"] == "total"

    window.fact_value_edit.setText("-")
    window.fact_value_edit.setModified(True)
    window._on_fact_editor_field_edited("value")
    _qt_app().processEvents()

    assert item.fact_data["natural_sign"] is None
    assert window.fact_natural_sign_label.text() == "-"
    window.close()


def test_gemini_fill_prompt_omits_statement_type_schema_when_not_requested(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    prompt = window._build_gemini_fill_prompt_from_template(
        "Requested meta fields:\n{{META_FIELDS}}\nSchema:\n{{META_UPDATES_SCHEMA}}\n{{REQUEST_JSON}}",
        request_payload={"pages": [{"image": "page_0001.png", "meta": {}, "facts": []}]},
        selected_fact_fields={"period_type"},
        include_statement_type=False,
    )

    assert "Requested meta fields:\nnone" in prompt
    assert "Schema:\n{}" in prompt
    assert '"statement_type"' not in prompt

    window.close()


def test_arrow_keys_nudge_selected_fact_by_one_pixel(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(80, 80, 20, 20), {"value": "100", "path": []})
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    window.view.setFocus()
    _qt_app().processEvents()

    original = item_scene_rect(item)
    QTest.keyClick(window.view.viewport(), Qt.Key_Right)
    _qt_app().processEvents()
    moved = item_scene_rect(item)
    assert moved.x() == pytest.approx(original.x() + 1.0)
    assert moved.y() == pytest.approx(original.y())

    QTest.keyClick(window.view.viewport(), Qt.Key_Left)
    _qt_app().processEvents()
    moved = item_scene_rect(item)
    assert moved.x() == pytest.approx(original.x())
    assert moved.y() == pytest.approx(original.y())

    QTest.keyClick(window.view.viewport(), Qt.Key_Up)
    _qt_app().processEvents()
    moved = item_scene_rect(item)
    assert moved.x() == pytest.approx(original.x())
    assert moved.y() == pytest.approx(original.y() - 1.0)

    QTest.keyClick(window.view.viewport(), Qt.Key_Down)
    _qt_app().processEvents()
    moved = item_scene_rect(item)
    assert moved.x() == pytest.approx(original.x())
    assert moved.y() == pytest.approx(original.y())
    window.close()


def test_fact_list_arrow_navigation_still_works_and_shift_arrow_nudges(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=200, height=200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    top = AnnotRectItem(QRectF(40, 30, 20, 20), {"value": "100", "path": []})
    bottom = AnnotRectItem(QRectF(42, 100, 20, 20), {"value": "200", "path": []})
    window.scene.addItem(top)
    window.scene.addItem(bottom)
    window.refresh_facts_list()
    window.show()
    top.setSelected(True)
    _qt_app().processEvents()

    initial_rect = item_scene_rect(top)
    window.facts_list.setFocus()
    QTest.keyClick(window.facts_list, Qt.Key_Down)
    _qt_app().processEvents()
    assert bottom.isSelected() is True

    QTest.keyClick(window.view.viewport(), Qt.Key_Right, Qt.ShiftModifier)
    _qt_app().processEvents()
    nudged_rect = item_scene_rect(bottom)
    assert nudged_rect.x() == pytest.approx(52.0)
    assert nudged_rect.y() == pytest.approx(100.0)
    assert item_scene_rect(top) == initial_rect
    window.close()


def test_shift_left_and_right_nudge_keep_same_selected_item(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    left_item = AnnotRectItem(QRectF(20, 40, 20, 20), {"value": "10", "path": []})
    right_item = AnnotRectItem(QRectF(70, 40, 20, 20), {"value": "20", "path": []})
    window.scene.addItem(left_item)
    window.scene.addItem(right_item)
    window.refresh_facts_list()
    window.show()
    right_item.setSelected(True)
    window.view.setFocus()
    _qt_app().processEvents()

    original = item_scene_rect(right_item)
    QTest.keyClick(window.view.viewport(), Qt.Key_Left, Qt.ShiftModifier)
    _qt_app().processEvents()
    moved_left = item_scene_rect(right_item)
    assert right_item.isSelected() is True
    assert moved_left.x() == pytest.approx(original.x() - 10.0)

    QTest.keyClick(window.view.viewport(), Qt.Key_Right, Qt.ShiftModifier)
    _qt_app().processEvents()
    moved_right = item_scene_rect(right_item)
    assert right_item.isSelected() is True
    assert moved_right.x() == pytest.approx(original.x())
    window.close()


def test_fact_editor_shows_recurring_period_only_for_recurrent_duration_type(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "120", "duration_type": "recurrent", "recurring_period": "monthly", "path": []},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_duration_type_combo.currentText() == "recurrent"
    assert window.fact_recurring_period_combo.currentText() == "monthly"
    assert window.fact_recurring_period_block.isVisible() is True

    window.fact_duration_type_combo.setCurrentIndex(0)
    window._on_fact_editor_field_edited("duration_type")
    _qt_app().processEvents()

    assert item.fact_data["duration_type"] is None
    assert window.fact_recurring_period_block.isVisible() is False

    idx = window.fact_duration_type_combo.findText("recurrent")
    window.fact_duration_type_combo.setCurrentIndex(idx)
    window._on_fact_editor_field_edited("duration_type")
    _qt_app().processEvents()

    assert item.fact_data["duration_type"] == "recurrent"
    assert window.fact_recurring_period_block.isVisible() is True
    window.close()


def test_clearing_duration_type_does_not_backfill_period_end_from_date(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "path": []})
    item.fact_data = {
        **item.fact_data,
        "date": "2024",
        "period_type": None,
        "period_start": None,
        "period_end": None,
        "duration_type": "recurrent",
        "recurring_period": "monthly",
    }
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    window.fact_duration_type_combo.setCurrentIndex(0)
    window._on_fact_editor_field_edited("duration_type")
    _qt_app().processEvents()

    assert item.fact_data["duration_type"] is None
    assert item.fact_data["period_end"] is None
    assert item.fact_data["period_start"] is None
    assert item.fact_data["period_type"] is None
    window.close()


def test_date_field_hidden_in_ui_but_kept_in_fact_payload(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "date": "2024-12-31", "path": []})
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_date_edit.isVisible() is False
    assert window.batch_date_edit.isVisible() is False
    assert window.fact_date_edit.text() == "2024-12-31"
    assert window._fact_data_from_editor()["date"] == "2024-12-31"
    assert item.fact_data["date"] == "2024-12-31"
    window.close()


def test_gemini_fill_apply_updates_selected_facts_meta_and_history(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "date": "2024-12-31"})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "date": "2024"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    ordered_items = window._sorted_fact_items()
    selected_fact_nums = [item.fact_data["fact_num"] for item in ordered_items if item in {item_a, item_b}]
    window._gemini_fill_target_page = window.page_images[window.current_index].name
    window._gemini_fill_snapshot = {
        "page_name": window._gemini_fill_target_page,
        "selected_fact_nums": selected_fact_nums,
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
    }
    window._gemini_fill_include_statement_type = True
    history_before = window._history_index
    info_messages: list[str] = []
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "information",
        lambda *_args: info_messages.append(str(_args[2]) if len(_args) > 2 else "info"),
    )

    window._on_gemini_fill_completed(
        {
            "meta_updates": {"statement_type": "income_statement"},
            "fact_updates": [
                {
                    "fact_num": selected_fact_nums[0],
                    "updates": {
                        "period_type": "instant",
                        "period_start": None,
                        "period_end": "2024-12-31",
                    },
                },
                {
                    "fact_num": selected_fact_nums[1],
                    "updates": {
                        "period_type": "duration",
                        "period_start": "2024-01-01",
                        "period_end": "2024-12-31",
                        "path_source": "observed",
                    },
                },
            ],
        }
    )

    assert item_a.fact_data["period_type"] == "instant"
    assert item_a.fact_data["period_end"] == "2024-12-31"
    assert item_b.fact_data["period_type"] == "duration"
    assert item_b.fact_data["period_start"] == "2024-01-01"
    assert item_b.fact_data["path_source"] == "observed"
    assert window.statement_type_combo.currentText() == "income_statement"
    assert window._history_index > history_before
    assert info_messages
    assert "Gemini Auto-Fix finished." in info_messages[0]

    window.close()


def test_gemini_fill_rejects_stale_snapshot_without_partial_apply(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    item.setSelected(True)
    _qt_app().processEvents()

    ordered_items = window._sorted_fact_items()
    window._gemini_fill_target_page = window.page_images[window.current_index].name
    window._gemini_fill_snapshot = {
        "page_name": window._gemini_fill_target_page,
        "selected_fact_nums": [1],
        "ordered_fact_signature": window._current_page_fact_snapshot_signature(ordered_items),
    }
    window._gemini_fill_include_statement_type = False

    warnings: list[str] = []
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "warning",
        lambda *_args: warnings.append(str(_args[2]) if len(_args) > 2 else "warning"),
    )

    item.fact_data = {**item.fact_data, "value": "changed"}
    window._on_gemini_fill_completed(
        {
            "meta_updates": {},
            "fact_updates": [{"fact_num": 1, "updates": {"period_type": "duration"}}],
        }
    )

    assert item.fact_data["period_type"] is None
    assert warnings

    window.close()


def test_multi_selection_path_preview_highlights_shared_prefix_and_variant_leaf(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "100", "path": ["עלות", "יתרה ליום 1 בינואר 2024", "ביאור 8 - רכוש אחר"]},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "200", "path": ["עלות", "יתרה ליום 1 בינואר 2024", "ביאור 9 - ציוד"]},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_path_list.count() == 3
    assert window.fact_path_list.item(0).text() == "עלות"
    assert window.fact_path_list.item(1).text() == "יתרה ליום 1 בינואר 2024"
    assert "ביאור 8 - רכוש אחר" in window.fact_path_list.item(2).toolTip()
    assert "ביאור 9 - ציוד" in window.fact_path_list.item(2).toolTip()
    assert window.fact_path_list.item(0).background().color() == window.fact_path_list.item(1).background().color()
    assert window.fact_path_list.item(0).background().color() != window.fact_path_list.item(2).background().color()
    assert "highlighted in green" in window.fact_path_list.toolTip().lower()
    assert window.path_add_btn.isEnabled() is False

    window.fact_path_list.item(1).setText("יתרה ליום 31 בדצמבר 2024")
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["עלות", "יתרה ליום 31 בדצמבר 2024", "ביאור 8 - רכוש אחר"]
    assert item_b.fact_data["path"] == ["עלות", "יתרה ליום 31 בדצמבר 2024", "ביאור 9 - ציוד"]

    window.close()


def test_multi_selection_identical_path_allows_direct_path_editing(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "path": ["הון", "פתיחה"]})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "path": ["הון", "פתיחה"]})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert window.path_add_btn.isEnabled() is True

    window.fact_path_list.item(0).setText("הון מניות")
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["הון מניות", "פתיחה"]
    assert item_b.fact_data["path"] == ["הון מניות", "פתיחה"]

    window.close()


def test_multi_selection_shared_prefix_allows_direct_path_removal(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "100", "path": ["מאזן", "רכוש שוטף", "מזומנים"]},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "200", "path": ["מאזן", "רכוש שוטף", "לקוחות"]},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.fact_path_list.setCurrentRow(1)
    assert window.path_remove_btn.isEnabled() is True

    window.remove_selected_path_level()
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["מאזן", "מזומנים"]
    assert item_b.fact_data["path"] == ["מאזן", "לקוחות"]

    window.close()


def test_multi_selection_shared_prefix_allows_move_up(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "100", "path": ["מאזן", "רכוש שוטף", "מזומנים"]},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "200", "path": ["מאזן", "רכוש שוטף", "לקוחות"]},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.fact_path_list.setCurrentRow(1)
    assert window.path_up_btn.isEnabled() is True

    window.move_selected_path_up()
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["רכוש שוטף", "מאזן", "מזומנים"]
    assert item_b.fact_data["path"] == ["רכוש שוטף", "מאזן", "לקוחות"]

    window.close()


def test_multi_selection_variant_leaf_allows_move_up(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "100", "path": ["מאזן", "רכוש שוטף", "מזומנים"]},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "200", "path": ["מאזן", "רכוש שוטף", "לקוחות"]},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.fact_path_list.setCurrentRow(2)
    assert window.path_up_btn.isEnabled() is True

    window.move_selected_path_up()
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["מאזן", "מזומנים", "רכוש שוטף"]
    assert item_b.fact_data["path"] == ["מאזן", "לקוחות", "רכוש שוטף"]

    window.close()


def test_multi_selection_variant_middle_allows_move_up_past_shared_suffix_hint(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "100", "path": ["מאזן", "רכוש שוטף", "מזומנים", "2024"]},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "200", "path": ["מאזן", "רכוש שוטף", "לקוחות", "2024"]},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_path_list.item(2).text() == "2024"
    assert window.fact_path_list.item(3).text() == "Different path tails"

    window.fact_path_list.setCurrentRow(3)
    assert window.path_up_btn.isEnabled() is True

    window.move_selected_path_up()
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["מאזן", "מזומנים", "רכוש שוטף", "2024"]
    assert item_b.fact_data["path"] == ["מאזן", "לקוחות", "רכוש שוטף", "2024"]

    window.close()


def test_multi_selection_invert_path_reverses_each_selected_fact(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "path": ["A", "B", "C"]})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "path": ["X", "Y"]})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert window.path_invert_btn.isEnabled() is True
    window.invert_selected_path_levels()
    _qt_app().processEvents()

    assert item_a.fact_data["path"] == ["C", "B", "A"]
    assert item_b.fact_data["path"] == ["Y", "X"]

    window.close()


def test_shift_click_adds_separate_bboxes_to_selection(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(60, 10, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    point_a = window.view.mapFromScene(item_a.mapRectToScene(item_a.rect()).center())
    point_b = window.view.mapFromScene(item_b.mapRectToScene(item_b.rect()).center())

    QTest.mouseClick(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, point_a)
    _qt_app().processEvents()
    assert set(window._selected_fact_items()) == {item_a}

    QTest.mouseClick(window.view.viewport(), Qt.LeftButton, Qt.ShiftModifier, point_b)
    _qt_app().processEvents()
    assert set(window._selected_fact_items()) == {item_a, item_b}
    assert len(window.facts_list.selectedItems()) == 2

    window.close()


def test_click_without_shift_selects_only_clicked_bbox(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(60, 10, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()
    assert set(window._selected_fact_items()) == {item_a, item_b}

    point_b = window.view.mapFromScene(item_b.mapRectToScene(item_b.rect()).center())
    QTest.mouseClick(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, point_b)
    _qt_app().processEvents()

    assert set(window._selected_fact_items()) == {item_b}
    window.close()


def test_drag_selects_multiple_bboxes_without_shift(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(20, 20, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(70, 20, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(10, 10), modifiers=Qt.NoModifier))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(110, 60), modifiers=Qt.NoModifier))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(110, 60), modifiers=Qt.NoModifier))
    _qt_app().processEvents()

    assert set(window._selected_fact_items()) == {item_a, item_b}
    window.close()


def test_shift_drag_adds_bboxes_to_existing_selection(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(20, 20, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(70, 20, 20, 20), {"value": "200"})
    item_c = AnnotRectItem(QRectF(120, 20, 20, 20), {"value": "300"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.scene.addItem(item_c)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    item_a.setSelected(True)
    _qt_app().processEvents()

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(60, 10), modifiers=Qt.ShiftModifier))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(160, 60), modifiers=Qt.ShiftModifier))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(160, 60), modifiers=Qt.ShiftModifier))
    _qt_app().processEvents()

    assert set(window._selected_fact_items()) == {item_a, item_b, item_c}
    window.close()


def test_drag_selection_ignores_deleted_temp_select_item(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(10, 10), modifiers=Qt.NoModifier))
    assert window.scene._temp_select_item is not None
    temp_item = window.scene._temp_select_item
    window.scene.removeItem(temp_item)
    sip.delete(temp_item)

    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(110, 60), modifiers=Qt.NoModifier))
    _qt_app().processEvents()

    assert window.scene._temp_select_item is None
    assert window.scene._selecting is False
    window.close()


def test_click_empty_page_clears_selection(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(20, 20, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(70, 20, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()
    assert len(window._selected_fact_items()) == 2

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(150, 150), modifiers=Qt.NoModifier))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(150, 150), modifiers=Qt.NoModifier))
    _qt_app().processEvents()

    assert window._selected_fact_items() == []
    window.close()


def test_batch_expand_selected_applies_to_all_selected_bboxes(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(60, 10, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.batch_resize_step_spin.setValue(10)
    window.batch_expand_selected("right")

    width_a = item_a.rect().width()
    width_b = item_b.rect().width()
    assert width_a == 30.0
    assert width_b == 30.0

    window.close()


def test_handle_resize_updates_all_selected_bboxes_visually(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=260, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(20, 20, 20, 20), {"value": "100"})
    item_b = AnnotRectItem(QRectF(70, 20, 20, 20), {"value": "200"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.scene._begin_group_resize(item_a, item_a._H_RIGHT)
    anchor_start = item_scene_rect(item_a)
    widened_anchor = QRectF(anchor_start)
    widened_anchor.setRight(anchor_start.right() + 12)
    item_a._clamp_and_apply_resize(widened_anchor, item_a._H_RIGHT)
    window.scene._update_group_resize_from_anchor(item_a)
    window.scene._end_group_resize(item_a)
    _qt_app().processEvents()

    assert item_a.rect().width() == 32.0
    assert item_b.rect().width() == 32.0
    window.close()


def test_box_geometry_change_updates_list_without_issue_recompute(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    _qt_app().processEvents()

    issue_refresh_calls: list[bool] = []
    window._refresh_current_page_issues = lambda *, use_current_fact_items=False: issue_refresh_calls.append(bool(use_current_fact_items))  # type: ignore[method-assign]

    item.setPos(QPointF(5, 0))
    window._on_box_geometry_changed(item)

    assert issue_refresh_calls == []
    assert "[15,10,20,20]" in window.facts_list.item(0).text()
    window.close()


def test_percent_range_value_stays_in_value_field(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "5", "note_ref": None, "value_type": "amount"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    item.setSelected(True)
    _qt_app().processEvents()

    value_type_idx = window.fact_value_type_combo.findText("percent")
    window.fact_value_type_combo.setCurrentIndex(value_type_idx)
    window._on_fact_editor_field_edited("value_type")
    window.fact_value_edit.setText("7-10")
    window.fact_value_edit.setModified(True)
    window._on_fact_editor_field_edited("value")

    assert item.fact_data["value"] == "7-10"
    assert item.fact_data["note_ref"] is None
    assert item.fact_data["value_type"] == "percent"
    window.close()


def test_page_issue_panel_updates_for_invalid_notes_page(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "10", "note_flag": False})
    window.scene.addItem(item)
    window.type_combo.setCurrentText("statements")
    window.statement_type_combo.setCurrentText("notes_to_financial_statements")
    window._on_meta_edited()
    window.refresh_facts_list()

    assert window.page_reg_flags_label.text() == "Reg Flags: 0"
    assert window.page_warnings_label.text() == "Warnings: 1"
    assert window.page_issues_list.count() == 1
    assert "none of the facts are marked as note facts" in window.page_issues_list.item(0).text().lower()
    window.close()


def test_page_issue_panel_updates_for_document_level_duplicate_page_number(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    _write_test_png(images_dir / "page_0002.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.page_states["page_0002.png"] = PageState(
        meta={**default_page_meta(1), "page_num": "5"},
        facts=[],
    )
    window.page_num_edit.setText("5")
    window._on_meta_edited()

    issue_texts = [window.page_issues_list.item(idx).text() for idx in range(window.page_issues_list.count())]
    assert any("page number '5' appears on multiple pages" in text.lower() for text in issue_texts)
    assert window.page_warnings_label.text() == "Warnings: 1"
    window.close()


def test_annotation_view_uses_a_and_d_for_page_navigation(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    _write_test_png(images_dir / "page_0002.png")
    _write_test_png(images_dir / "page_0003.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    assert window.current_index == 0

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_D, Qt.NoModifier))
    assert window.current_index == 1

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert window.current_index == 0

    zoom_before = window.view.transform().m11()
    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Plus, Qt.ShiftModifier, "+"))
    zoom_after_plus = window.view.transform().m11()
    assert zoom_after_plus > zoom_before

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Minus, Qt.NoModifier, "-"))
    zoom_after_minus = window.view.transform().m11()
    assert zoom_after_minus < zoom_after_plus

    window.close()


def test_annotation_window_auto_fits_when_shown(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    fit_calls: list[str] = []
    monkeypatch.setattr(window, "_fit_view_height", lambda: fit_calls.append("fit"))

    window.show()
    _qt_app().processEvents()

    assert fit_calls == ["fit"]
    window.close()


def test_command_a_selects_all_bboxes_on_current_page(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "10"})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "11"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.ControlModifier))

    assert set(window._selected_fact_items()) == {item_a, item_b}
    assert len(window.facts_list.selectedItems()) == 2
    window.close()


def test_empty_space_drag_does_not_create_bbox_without_command_modifier(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    start = window.view.mapFromScene(QPointF(20, 20))
    end = window.view.mapFromScene(QPointF(80, 80))
    QTest.mousePress(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(window.view.viewport(), end)
    QTest.mouseRelease(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, end)
    _qt_app().processEvents()

    assert window._selected_fact_items() == []
    assert [item for item in window.scene.items() if isinstance(item, AnnotRectItem)] == []
    window.close()


def test_command_drag_creates_one_bbox(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    start = QPointF(24, 24)
    end = QPointF(88, 92)
    window.scene.mousePressEvent(_SceneMouseEvent(start, modifiers=Qt.MetaModifier))
    window.scene.mouseMoveEvent(_SceneMouseEvent(end, modifiers=Qt.MetaModifier))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(end, modifiers=Qt.MetaModifier))
    _qt_app().processEvents()

    items = [item for item in window.scene.items() if isinstance(item, AnnotRectItem)]
    assert len(items) == 1
    window.close()


def test_equation_builder_uses_natural_sign_and_operator_for_polarity() -> None:
    candidate_text, result_text, fact_candidate_text, invalid_values, structured_terms = app_mod._build_equation_candidate_from_facts(
        [
            {"fact_num": 1, "value": "100", "natural_sign": "positive", "operator": "+"},
            {"fact_num": 2, "value": "30", "natural_sign": "positive", "operator": "+"},
            {"fact_num": 3, "value": "5", "natural_sign": "positive", "operator": "-"},
            {"fact_num": 4, "value": "(7)", "natural_sign": "negative", "operator": "-"},
            {"fact_num": 5, "value": "-", "natural_sign": None, "operator": "+"},
        ]
    )

    assert candidate_text == "100 + 30 - 5 + 7 + 0"
    assert result_text == "132"
    assert fact_candidate_text == "f1 + f2 - f3 - f4 + f5"
    assert invalid_values == []
    assert [term.get("effective_normalized_value") for term in structured_terms] == [100, 30, -5, 7, 0]
    assert [term.get("contribution_sign") for term in structured_terms] == [1, 1, -1, 1, 1]


def test_equation_result_match_state_applies_target_natural_sign() -> None:
    tone, message = app_mod._equation_result_match_state("-100", "100", "negative")
    assert tone == "ok"
    assert message == "Matches target value."

    tone, message = app_mod._equation_result_match_state("100", "100", "positive")
    assert tone == "ok"
    assert message == "Matches target value."


def test_evaluate_equation_string_uses_left_side_when_equals_present() -> None:
    assert app_mod._evaluate_equation_string("971771 + 599659 = 599659") == "1571430"


def test_equation_builder_defaults_to_additive_when_operator_missing() -> None:
    facts = [
        app_mod.normalize_fact_data({"fact_num": 10, "value": "269968", "path": ["רכוש קבוע", "עלות"]}),
        app_mod.normalize_fact_data({"fact_num": 12, "value": "209255", "path": ["רכוש קבוע", "בניכוי - פחת שנצבר"]}),
    ]

    candidate_text, result_text, fact_candidate_text, invalid_values, structured_terms = app_mod._build_equation_candidate_from_facts(
        facts
    )

    assert candidate_text == "269968 + 209255"
    assert result_text == "479223"
    assert fact_candidate_text == "f10 + f12"
    assert invalid_values == []
    assert [term.get("operator") for term in structured_terms] == ["+", "+"]


def test_equation_builder_preserves_operator_for_dash_placeholder_terms() -> None:
    facts = [
        app_mod.normalize_fact_data({"fact_num": 1, "value": "100", "path": ["A"]}),
        app_mod.normalize_fact_data({"fact_num": 2, "value": "-", "path": ["A"]}),
    ]

    candidate_text, result_text, fact_candidate_text, invalid_values, _structured_terms = app_mod._build_equation_candidate_from_facts(
        [
            {**facts[0], "operator": "+"},
            {**facts[1], "operator": "-"},
        ]
    )

    assert candidate_text == "100 - 0"
    assert fact_candidate_text == "f1 - f2"
    assert result_text == "100"
    assert invalid_values == []


def test_c_drag_builds_equation_preview_and_apply_persists_it(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"
    monkeypatch.setattr(app_mod.QMessageBox, "information", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_mod.QMessageBox, "warning", lambda *_args, **_kwargs: app_mod.QMessageBox.Ok)

    window = AnnotationWindow(images_dir, annotations_path)
    save_statuses: list[dict[str, object]] = []
    window.annotations_save_status.connect(save_statuses.append)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "999", "fact_num": 9})
    ref_high = AnnotRectItem(QRectF(30, 60, 20, 20), {"value": "20", "fact_num": 5})
    ref_low = AnnotRectItem(QRectF(30, 90, 20, 20), {"value": "100", "fact_num": 1})
    ref_invalid = AnnotRectItem(QRectF(30, 120, 20, 20), {"value": "abc", "fact_num": 3})
    ref_negative = AnnotRectItem(QRectF(30, 150, 20, 20), {"value": "(5)", "fact_num": 2})
    ref_dash = AnnotRectItem(QRectF(30, 180, 20, 20), {"value": "-", "fact_num": 4})
    window.scene.addItem(target)
    window.scene.addItem(ref_high)
    window.scene.addItem(ref_low)
    window.scene.addItem(ref_invalid)
    window.scene.addItem(ref_negative)
    window.scene.addItem(ref_dash)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 50)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(80, 180)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 180)))
    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    assert target.fact_data.get("equations") is None
    assert window.fact_equation_edit.text() == "20 + 100 - 5 + 0"
    assert window.fact_equation_result_label.text() == "115"
    assert "#b7791f" in window.fact_equation_result_label.styleSheet()
    assert "Does not match target value" in window.fact_equation_status_label.text()
    assert "Ignored 1 invalid value" in window.fact_equation_status_label.text()
    assert {"fact_num": 6, "fact_reference": "f6", "normalized_value": 0, "raw_value": "-", "status": "normalized_dash"} in [
        {
            "fact_num": term.get("fact_num"),
            "fact_reference": term.get("fact_reference"),
            "normalized_value": term.get("normalized_value"),
            "raw_value": term.get("raw_value"),
            "status": term.get("status"),
        }
        for term in window._equation_candidate_terms
    ]
    assert window.apply_equation_btn.isEnabled() is True

    window.apply_equation_btn.click()
    _qt_app().processEvents()

    assert target.fact_data["equations"][0]["equation"] == "20 + 100 - 5 + 0"
    assert target.fact_data["equations"][0]["fact_equation"] == "f2 + f3 + f5 + f6"
    assert window.apply_equation_btn.isEnabled() is False
    assert window.fact_equation_edit.text() == "20 + 100 - 5 + 0"
    assert window.fact_equation_result_label.text() == "115"
    assert "#b7791f" in window.fact_equation_result_label.styleSheet()

    assert window.save_annotations() is True
    assert save_statuses
    assert save_statuses[-1]["warning_count"] >= 1
    assert save_statuses[-1]["equation_warning_count"] >= 1
    assert any(
        finding.get("code") == "equation_arithmetic_mismatch"
        for finding in save_statuses[-1]["equation_findings"]
    )
    window.close()


def test_save_shortcut_triggers_single_save(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    window.activateWindow()
    window.setFocus(Qt.OtherFocusReason)
    _qt_app().processEvents()

    calls: list[str] = []
    monkeypatch.setattr(window, "save_annotations", lambda: calls.append("save") or True)

    handled = window.event(QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.ControlModifier))
    _qt_app().processEvents()

    assert handled is True
    assert calls == ["save"]
    window.close()


def test_page_json_dialog_supports_search_and_selected_fact_focus(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "fact_num": 1, "path": ["Revenue"]})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "fact_num": 2, "path": ["Expense"]})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    item_b.setSelected(True)
    _qt_app().processEvents()

    window.show_current_page_json()
    _qt_app().processEvents()

    dialog = window._page_json_dialog
    assert dialog is not None
    assert dialog.isVisible() is True

    page_text = dialog.text_view.toPlainText()
    expected_pos = page_text.rfind("{", 0, page_text.index('"fact_num": 2') + 1)
    assert dialog.text_view.textCursor().position() == expected_pos

    QTest.keyClick(dialog, Qt.Key_F, Qt.ControlModifier)
    _qt_app().processEvents()
    assert QApplication.focusWidget() is dialog.search_edit

    dialog.search_edit.setText('"fact_num": 1')
    dialog.find_next()
    _qt_app().processEvents()
    assert dialog.text_view.textCursor().selectedText() == '"fact_num": 1'

    item_a.setSelected(True)
    item_b.setSelected(False)
    _qt_app().processEvents()

    updated_text = dialog.text_view.toPlainText()
    updated_pos = updated_text.rfind("{", 0, updated_text.index('"fact_num": 1') + 1)
    assert dialog.text_view.textCursor().position() == updated_pos
    window.close()


def test_alt_shift_approves_equation_candidate(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "fact_num": 9})
    ref_a = AnnotRectItem(QRectF(30, 60, 20, 20), {"value": "100", "fact_num": 1})
    ref_b = AnnotRectItem(QRectF(30, 90, 20, 20), {"value": "20", "fact_num": 2})
    window.scene.addItem(target)
    window.scene.addItem(ref_a)
    window.scene.addItem(ref_b)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 50)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 120)))
    _qt_app().processEvents()

    assert target.fact_data.get("equations") is None
    assert window.apply_equation_btn.isEnabled() is True

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Shift, Qt.AltModifier, ""))
    _qt_app().processEvents()

    assert target.fact_data["equations"][0]["equation"] == "100 + 20"
    assert target.fact_data["equations"][0]["fact_equation"] == "f2 + f3"
    assert window.apply_equation_btn.isEnabled() is False
    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    window.close()


def test_alt_shift_apply_exits_equation_mode_and_next_target_requires_alt_again(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=260, height=260)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "fact_num": 9})
    target_b = AnnotRectItem(QRectF(10, 45, 20, 20), {"value": "120", "fact_num": 10})
    ref_a = AnnotRectItem(QRectF(30, 100, 20, 20), {"value": "100", "fact_num": 1})
    ref_b = AnnotRectItem(QRectF(30, 130, 20, 20), {"value": "20", "fact_num": 2})
    window.scene.addItem(target_a)
    window.scene.addItem(target_b)
    window.scene.addItem(ref_a)
    window.scene.addItem(ref_b)
    window.refresh_facts_list()
    window.show()
    target_a.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 95)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(85, 160)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(85, 160)))
    _qt_app().processEvents()

    assert window.apply_equation_btn.isEnabled() is True
    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Shift, Qt.AltModifier, ""))
    _qt_app().processEvents()
    assert target_a.fact_data["equations"][0]["equation"] == "100 + 20"
    assert window.view._calculate_drag_active is False
    assert window.scene._calculate_drag_active is False

    window.scene.clearSelection()
    target_b.setSelected(True)
    _qt_app().processEvents()

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 95)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(85, 160)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(85, 160)))
    _qt_app().processEvents()

    assert window._equation_target_item is None
    assert window.apply_equation_btn.isEnabled() is False

    window.scene.clearSelection()
    target_b.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 95)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(85, 160)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(85, 160)))
    _qt_app().processEvents()

    assert window._equation_target_item is target_b
    assert window.fact_equation_edit.text() == "100 + 20"
    assert window.apply_equation_btn.isEnabled() is True

    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    window.close()


def test_clear_equation_button_clears_equation_without_deleting_fact(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "120", "equation": "100 + 20", "fact_equation": "f1 + f2", "fact_num": 1},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.clear_equation_btn.isEnabled() is True
    window.clear_equation_btn.click()
    _qt_app().processEvents()

    assert len(window._fact_items) == 1
    assert item.scene() is window.scene
    assert item.fact_data.get("value") == "120"
    assert item.fact_data.get("equations") is None
    assert window.fact_equation_edit.text() == ""
    assert window.clear_equation_btn.isEnabled() is False
    window.close()


def test_clear_equation_button_clears_for_multi_selection(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "120", "equation": "100 + 20", "fact_equation": "f1 + f2", "fact_num": 1},
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {"value": "55", "equation": "60 - 5", "fact_equation": "f3 - f4", "fact_num": 2},
    )
    item_c = AnnotRectItem(
        QRectF(70, 10, 20, 20),
        {"value": "999", "equation": None, "fact_equation": None, "fact_num": 3},
    )
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.scene.addItem(item_c)
    window.refresh_facts_list()
    window.show()
    item_a.setSelected(True)
    item_b.setSelected(True)
    item_c.setSelected(True)
    _qt_app().processEvents()

    assert window.clear_equation_btn.isEnabled() is True
    window.clear_equation_btn.click()
    _qt_app().processEvents()

    assert len(window._fact_items) == 3
    assert item_a.fact_data.get("equations") is None
    assert item_b.fact_data.get("equations") is None
    assert item_c.fact_data.get("equations") is None
    assert window.clear_equation_btn.isEnabled() is False
    window.close()


def test_clear_equation_button_removes_all_saved_equation_variants(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_variants_list.count() == 2
    window.clear_equation_btn.click()
    _qt_app().processEvents()

    assert item.fact_data.get("equations") is None
    window.close()


def test_equation_variants_list_switches_active_equation(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_variants_list.count() == 2
    variant_item = window.fact_equation_variants_list.item(1)
    monkeypatch.setattr(app_mod.QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))
    window._on_equation_variant_item_clicked(variant_item)
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "80 + 40"
    assert item.fact_data["equations"][0]["fact_equation"] == "f3 + f4"
    assert item.fact_data["equations"][0]["equation"] == "80 + 40"
    assert item.fact_data["equations"][1]["equation"] == "100 + 20"
    window.close()


def test_add_equation_variant_button_appends_candidate_and_sets_active(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f2 + f3",
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    window._equation_target_item = item
    window._equation_candidate_text = "90 + 30"
    window._equation_candidate_fact_text = "f4 + f5"
    window._equation_candidate_result_text = "120"
    window._equation_candidate_terms = [
        {"equation_child": {"fact_num": 4, "operator": "+"}},
        {"equation_child": {"fact_num": 5, "operator": "+"}},
    ]
    window._refresh_equation_panel()
    assert window.equation_add_variant_btn.isEnabled() is True

    window.equation_add_variant_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "90 + 30"
    assert item.fact_data["equations"][0]["fact_equation"] == "f4 + f5"
    assert item.fact_data["equations"][0]["equation"] == "90 + 30"
    assert item.fact_data["equations"][1]["equation"] == "100 + 20"
    window.close()


def test_delete_selected_saved_equation_button_removes_active_variant(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_variants_list.count() == 2
    assert window.equation_delete_variant_btn.isEnabled() is True
    window.equation_delete_variant_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "80 + 40"
    assert item.fact_data["equations"][0]["fact_equation"] == "f3 + f4"
    assert item.fact_data["equations"] == [{"equation": "80 + 40", "fact_equation": "f3 + f4"}]
    focus_widget = QApplication.focusWidget()
    assert focus_widget is not window.fact_note_edit
    assert focus_widget in {window.fact_equation_variants_list, window.fact_equation_variants_list.viewport()}
    window.close()


def test_delete_saved_equation_works_with_current_row_even_without_explicit_selection(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    # Simulate a focused row with no explicit selected items.
    window.fact_equation_variants_list.clearSelection()
    window.fact_equation_variants_list.setCurrentRow(1, app_mod.QItemSelectionModel.NoUpdate)
    _qt_app().processEvents()

    window.equation_delete_variant_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "100 + 20"
    assert item.fact_data["equations"] == [{"equation": "100 + 20", "fact_equation": "f1 + f2"}]
    window.close()


def test_delete_key_in_equation_variants_list_deletes_selected_variant(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    window.fact_equation_variants_list.clearSelection()
    row_1 = window.fact_equation_variants_list.item(1)
    assert row_1 is not None
    row_1.setSelected(True)
    window.fact_equation_variants_list.setCurrentRow(1)
    window.fact_equation_variants_list.setFocus()
    _qt_app().processEvents()

    window._delete_selected_fact_shortcut()
    _qt_app().processEvents()

    assert len(window._fact_items) == 1
    assert item.scene() is window.scene
    assert len(item.fact_data["equations"]) == 1
    assert item.fact_data["equations"][0] == {"equation": "100 + 20", "fact_equation": "f1 + f2"}
    window.close()


def test_equation_variant_order_buttons_reorder_saved_equations(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.equation_variant_down_btn.isEnabled() is True
    window.equation_variant_down_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "80 + 40"
    assert item.fact_data["equations"][0]["equation"] == "80 + 40"
    assert item.fact_data["equations"][1]["equation"] == "100 + 20"
    window.close()


def test_delete_selected_saved_equations_removes_multiple_variants(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {"equation": "100 + 20", "fact_equation": "f1 + f2"},
                {"equation": "80 + 40", "fact_equation": "f3 + f4"},
                {"equation": "70 + 50", "fact_equation": "f5 + f6"},
            ],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    row_0 = window.fact_equation_variants_list.item(0)
    row_1 = window.fact_equation_variants_list.item(1)
    row_2 = window.fact_equation_variants_list.item(2)
    row_0.setSelected(False)
    row_1.setSelected(True)
    row_2.setSelected(True)
    _qt_app().processEvents()

    assert window.equation_delete_variant_btn.isEnabled() is True
    window.equation_delete_variant_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"][0]["equation"] == "100 + 20"
    assert item.fact_data["equations"] == [{"equation": "100 + 20", "fact_equation": "f1 + f2"}]
    window.close()


def test_readding_dash_subtractive_equation_does_not_restore_deleted_additive_variant(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "value": "100",
            "equation": "100 + 0",
            "fact_equation": "f2 + f3",
            "equations": [{"equation": "100 + 0", "fact_equation": "f2 + f3"}],
            "fact_num": 1,
        },
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_variants_list.count() == 1
    window.equation_delete_variant_btn.click()
    _qt_app().processEvents()
    assert item.fact_data.get("equations") is None

    window._equation_target_item = item
    window._equation_candidate_text = "100 - 0"
    window._equation_candidate_fact_text = "f2 - f3"
    window._equation_candidate_result_text = "100"
    window._equation_candidate_terms = [
        {"equation_child": {"fact_num": 2, "operator": "+"}},
        {"equation_child": {"fact_num": 3, "operator": "-"}},
    ]
    window._refresh_equation_panel()
    assert window.equation_add_variant_btn.isEnabled() is True

    window.equation_add_variant_btn.click()
    _qt_app().processEvents()

    assert item.fact_data["equations"] == [{"equation": "100 - 0", "fact_equation": "f2 - f3"}]
    window.close()


def test_deleting_dash_additive_variant_does_not_return_after_refresh(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {
            "fact_num": 1,
            "value": "100",
            "equation": "100 + 0",
            "fact_equation": "f2 + f3",
            "equations": [
                {"equation": "100 + 0", "fact_equation": "f2 + f3"},
                {"equation": "100 - 0", "fact_equation": "f2 - f3"},
            ],
        },
    )
    child_amount = AnnotRectItem(QRectF(40, 10, 20, 20), {"fact_num": 2, "value": "100"})
    child_dash = AnnotRectItem(QRectF(70, 10, 20, 20), {"fact_num": 3, "value": "-"})
    window.scene.addItem(target)
    window.scene.addItem(child_amount)
    window.scene.addItem(child_dash)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_variants_list.count() == 2
    first_row = window.fact_equation_variants_list.item(0)
    assert first_row is not None
    first_row.setSelected(True)
    window.fact_equation_variants_list.setCurrentItem(first_row)
    _qt_app().processEvents()

    window.equation_delete_variant_btn.click()
    _qt_app().processEvents()

    assert target.fact_data["equations"] == [{"equation": "100 - 0", "fact_equation": "f2 - f3"}]

    # Force multiple refresh cycles (same path used during normal UI interactions).
    window.refresh_facts_list()
    _qt_app().processEvents()
    target.setSelected(True)
    window.refresh_facts_list()
    _qt_app().processEvents()
    target.setSelected(True)
    _qt_app().processEvents()

    assert target.fact_data["equations"] == [{"equation": "100 - 0", "fact_equation": "f2 - f3"}]
    assert window.fact_equation_variants_list.count() == 1
    assert window.fact_equation_variants_list.item(0).text().endswith("100 - 0")
    window.close()


def test_delete_shortcut_ignored_while_equation_field_focused(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "120", "equation": "100 + 20", "fact_equation": "f1 + f2", "fact_num": 1},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    window.fact_equation_edit.setFocus()
    _qt_app().processEvents()
    QTest.keyClick(window.fact_equation_edit, Qt.Key_Backspace)
    _qt_app().processEvents()

    assert len(window._fact_items) == 1
    assert item.scene() is window.scene

    window.view.setFocus()
    _qt_app().processEvents()
    window._delete_selected_fact_shortcut()
    _qt_app().processEvents()
    assert len(window._fact_items) == 0
    window.close()


def test_equation_preview_clears_on_selection_change_without_persisting(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "999"})
    ref_a = AnnotRectItem(QRectF(30, 60, 20, 20), {"value": "40"})
    ref_b = AnnotRectItem(QRectF(30, 90, 20, 20), {"value": "2"})
    window.scene.addItem(target)
    window.scene.addItem(ref_a)
    window.scene.addItem(ref_b)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 50)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    assert [item.fact_data["fact_num"] for item in window._fact_items] == [1, 2, 3]
    assert window.fact_equation_edit.text() == "40 + 2"
    assert target.fact_data.get("equations") is None

    ref_a.setSelected(True)
    _qt_app().processEvents()

    assert target.fact_data.get("equations") is None
    assert window.apply_equation_btn.isEnabled() is False
    assert window.fact_equation_edit.text() == ""
    window.close()


def test_refresh_facts_list_resequences_fact_nums_contiguously(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "fact_num": 7})
    b = AnnotRectItem(QRectF(10, 50, 20, 20), {"value": "200", "fact_num": 7})
    c = AnnotRectItem(QRectF(10, 90, 20, 20), {"value": "300", "fact_num": 11})
    window.scene.addItem(a)
    window.scene.addItem(b)
    window.scene.addItem(c)
    window.refresh_facts_list()

    assert [item.fact_data["fact_num"] for item in window._fact_items] == [1, 2, 3]
    window.close()


def test_refresh_facts_list_remaps_fact_equation_refs_when_fact_nums_shift(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=240)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    inserted = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "900", "fact_num": 4})
    child_a = AnnotRectItem(QRectF(10, 50, 20, 20), {"value": "100", "fact_num": 1})
    child_b = AnnotRectItem(QRectF(10, 90, 20, 20), {"value": "20", "fact_num": 2})
    target = AnnotRectItem(
        QRectF(10, 130, 20, 20),
        {"value": "120", "fact_num": 3, "equation": "100 + 20", "fact_equation": "f1 + f2"},
    )
    window.scene.addItem(inserted)
    window.scene.addItem(child_a)
    window.scene.addItem(child_b)
    window.scene.addItem(target)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    assert [item.fact_data["fact_num"] for item in window._fact_items] == [1, 2, 3, 4]
    assert target.fact_data["equations"][0]["fact_equation"] == "f2 + f3"
    assert target.fact_data["equations"][0]["equation"] == "100 + 20"
    assert window.fact_equation_edit.text() == "100 + 20"
    assert window.fact_equation_result_label.text() == "120"
    assert "#027a48" in window.fact_equation_result_label.styleSheet()
    assert "Matches target value." in window.fact_equation_status_label.text()
    window.close()


def test_saved_mismatching_equation_displays_raw_expression_with_preview_result(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=240)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "3595", "equation": "13639 + 10044", "fact_num": 1},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "13639 + 10044"
    assert window.fact_equation_result_label.text() == "23683"
    assert "Does not match target value (3595)." in window.fact_equation_status_label.text()
    window.close()


def test_refresh_facts_list_marks_only_matching_saved_equations_as_green(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=240)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    ok_item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "120", "equation": "100 + 20", "fact_num": 1},
    )
    bad_item = AnnotRectItem(
        QRectF(10, 50, 20, 20),
        {"value": "120", "equation": "100 + 30", "fact_num": 2},
    )
    no_equation = AnnotRectItem(QRectF(10, 90, 20, 20), {"value": "120", "fact_num": 3})
    window.scene.addItem(ok_item)
    window.scene.addItem(bad_item)
    window.scene.addItem(no_equation)
    window.refresh_facts_list()

    assert ok_item._equation_match_ok is True
    assert bad_item._equation_match_ok is False
    assert no_equation._equation_match_ok is False
    window.close()


def test_equation_match_bbox_paint_has_no_top_v_badge() -> None:
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "equation": "100 + 20", "fact_num": 1})
    item.set_equation_match_ok(True)
    painter = _FakePainter()

    item.paint(painter, None)

    assert painter.draw_text_calls == []
    assert painter.pen_colors[0] == "#14804a"


def test_selected_equation_match_bbox_uses_standard_blue_outline() -> None:
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "120", "equation": "100 + 20", "fact_num": 1})
    item.set_equation_match_ok(True)
    item.setSelected(True)
    painter = _FakePainter()

    item.paint(painter, None)

    assert painter.pen_colors.count("#175cd3") == 1
    assert Qt.DashLine in painter.pen_styles


def test_c_mode_accumulates_clicks_and_multiple_rectangles_until_key_release(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=240)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "103", "fact_num": 9})
    ref_a = AnnotRectItem(QRectF(30, 60, 20, 20), {"value": "100", "fact_num": 1})
    ref_b = AnnotRectItem(QRectF(70, 60, 20, 20), {"value": "(5)", "fact_num": 2})
    ref_c = AnnotRectItem(QRectF(30, 110, 20, 20), {"value": "8", "fact_num": 3})
    ref_d = AnnotRectItem(QRectF(70, 110, 20, 20), {"value": "2", "fact_num": 4})
    window.scene.addItem(target)
    window.scene.addItem(ref_a)
    window.scene.addItem(ref_b)
    window.scene.addItem(ref_c)
    window.scene.addItem(ref_d)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(25, 55)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(55, 85)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(55, 85)))

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(80, 70)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 70)))

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(25, 105)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(95, 135)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(95, 135)))

    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 120)))

    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 - 5 + 8"
    assert window.fact_equation_result_label.text() == "103"
    assert "#027a48" in window.fact_equation_result_label.styleSheet()
    assert "Matches target value." in window.fact_equation_status_label.text()
    assert window._equation_candidate_fact_text == "f2 + f3 + f4"
    assert sorted(item.fact_data["fact_num"] for item in window._equation_reference_preview_items) == [2, 3, 4]
    assert window.apply_equation_btn.isEnabled() is True

    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 - 5 + 8"
    assert window._equation_candidate_fact_text == "f2 + f3 + f4"
    assert target.fact_data.get("equations") is None
    window.close()


def test_equation_click_selection_preserves_chronological_order(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=260, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "30", "fact_num": 9})
    left_ref = AnnotRectItem(QRectF(20, 70, 20, 20), {"value": "10", "fact_num": 8})
    right_ref = AnnotRectItem(QRectF(160, 70, 20, 20), {"value": "20", "fact_num": 2})
    window.scene.addItem(target)
    window.scene.addItem(left_ref)
    window.scene.addItem(right_ref)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window._equation_target_item = target
    window._on_equation_reference_selection_changed([left_ref, right_ref])

    left_num = window._fact_num_for_item(left_ref)
    right_num = window._fact_num_for_item(right_ref)
    ordered_fact_nums = [
        int(term["fact_num"])
        for term in window._equation_candidate_terms
        if isinstance(term, dict) and isinstance(term.get("equation_child"), dict) and isinstance(term.get("fact_num"), int)
    ]
    assert ordered_fact_nums == [left_num, right_num]
    window.close()


def test_equation_drag_selection_preserves_first_enter_order(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=260, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "30", "fact_num": 9})
    left_ref = AnnotRectItem(QRectF(20, 70, 20, 20), {"value": "10", "fact_num": 8})
    right_ref = AnnotRectItem(QRectF(160, 70, 20, 20), {"value": "20", "fact_num": 2})
    window.scene.addItem(target)
    window.scene.addItem(left_ref)
    window.scene.addItem(right_ref)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(15, 65)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(205, 110)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(205, 110)))
    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    left_num = window._fact_num_for_item(left_ref)
    right_num = window._fact_num_for_item(right_ref)
    ordered_fact_nums = [
        int(term["fact_num"])
        for term in window._equation_candidate_terms
        if isinstance(term, dict) and isinstance(term.get("equation_child"), dict) and isinstance(term.get("fact_num"), int)
    ]
    assert ordered_fact_nums == [left_num, right_num]
    window.close()


def test_equation_operator_override_updates_preview_and_saved_children(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=240)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "95", "fact_num": 9})
    ref_a = AnnotRectItem(QRectF(30, 60, 20, 20), {"value": "100", "fact_num": 1})
    ref_b = AnnotRectItem(QRectF(30, 90, 20, 20), {"value": "5", "fact_num": 2})
    window.scene.addItem(target)
    window.scene.addItem(ref_a)
    window.scene.addItem(ref_b)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier, ""))
    window.scene.mousePressEvent(_SceneMouseEvent(QPointF(20, 50)))
    window.scene.mouseMoveEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.scene.mouseReleaseEvent(_SceneMouseEvent(QPointF(80, 120)))
    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 + 5"
    assert window.fact_equation_terms_list.isVisible() is True
    assert window.fact_equation_terms_list.count() == 2

    second_term = window.fact_equation_terms_list.item(1)
    assert second_term is not None
    second_term.setSelected(True)
    _qt_app().processEvents()
    assert window.equation_mark_subtract_btn.isEnabled() is True

    window.equation_mark_subtract_btn.click()
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 - 5"
    assert window.fact_equation_result_label.text() == "95"
    assert "Matches target value." in window.fact_equation_status_label.text()
    assert window._equation_candidate_fact_text == "f2 - f3"

    window.apply_equation_btn.click()
    _qt_app().processEvents()

    assert target.fact_data["equations"][0]["equation"] == "100 - 5"
    assert target.fact_data["equations"][0]["fact_equation"] == "f2 - f3"
    assert target.fact_data["equations"][0]["fact_equation"] == "f2 - f3"
    window.close()


def test_same_child_can_use_different_operators_under_different_targets(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=260, height=260)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "95", "fact_num": 11})
    target_b = AnnotRectItem(QRectF(10, 40, 20, 20), {"value": "105", "fact_num": 12})
    shared = AnnotRectItem(QRectF(30, 90, 20, 20), {"value": "100", "fact_num": 1})
    adjustment = AnnotRectItem(QRectF(30, 120, 20, 20), {"value": "5", "fact_num": 2})
    window.scene.addItem(target_a)
    window.scene.addItem(target_b)
    window.scene.addItem(shared)
    window.scene.addItem(adjustment)
    window.refresh_facts_list()
    window.show()

    window.scene.clearSelection()
    target_a.setSelected(True)
    window._equation_target_item = target_a
    window._on_equation_reference_selection_changed([shared, adjustment])
    _qt_app().processEvents()

    target_a_adjustment = window.fact_equation_terms_list.item(1)
    assert target_a_adjustment is not None
    target_a_adjustment.setSelected(True)
    _qt_app().processEvents()
    window.equation_mark_subtract_btn.click()
    _qt_app().processEvents()
    window.apply_equation_btn.click()
    _qt_app().processEvents()

    window.scene.clearSelection()
    target_b.setSelected(True)
    window._equation_target_item = target_b
    window._on_equation_reference_selection_changed([shared, adjustment])
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 + 5"
    window.apply_equation_btn.click()
    _qt_app().processEvents()

    assert target_a.fact_data["equations"][0]["fact_equation"] == "f3 - f4"
    assert target_b.fact_data["equations"][0]["fact_equation"] == "f3 + f4"
    window.close()


def test_invalid_saved_equation_shows_cannot_calculate_in_red(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    target = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "10", "equation": "abc", "fact_equation": "f1 + f2"})
    window.scene.addItem(target)
    window.refresh_facts_list()
    window.show()
    target.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "abc"
    assert window.fact_equation_result_label.text() == "cannot calculate"
    assert "#b7791f" in window.fact_equation_result_label.styleSheet()
    assert "cannot be calculated" in window.fact_equation_status_label.text().lower()
    window.close()


def test_close_with_unsaved_changes_cancel_keeps_window_open(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    window.entity_name_edit.setText("Acme")
    window._on_meta_edited()
    assert window._has_unsaved_changes() is True

    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "cancel")

    window.close()
    _qt_app().processEvents()

    assert window.isVisible() is True
    window.hide()
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")
    window.close()


def test_page_issue_warning_points_to_outlier_fact(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "10", "scale": 1000})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "11", "scale": 1000})
    item_c = AnnotRectItem(QRectF(70, 10, 20, 20), {"value": "12", "scale": 1})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.scene.addItem(item_c)
    window.refresh_facts_list()

    assert window.page_warnings_label.text() == "Warnings: 1"
    assert window.page_issues_list.count() == 1

    issue_item = window.page_issues_list.item(0)
    assert issue_item is not None
    assert "most facts use '1000'" in issue_item.text()

    window._on_page_issue_clicked(issue_item)

    assert item_c.isSelected()
    assert window.facts_list.currentRow() == 2
    window.close()


def test_fact_list_selection_keeps_current_view_while_selecting_bbox(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=1200, height=1200)
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.resize(1400, 1000)
    window.show()
    _qt_app().processEvents()

    item = AnnotRectItem(QRectF(40, 60, 24, 24), {"value": "100"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    _qt_app().processEvents()

    zoom_before = window.view.transform().m11()
    view_center_before = window.view.mapToScene(window.view.viewport().rect().center())
    assert window.facts_list.item(0) is not None
    window.facts_list.item(0).setSelected(True)
    window.facts_list.setCurrentRow(0)
    _qt_app().processEvents()

    zoom_after = window.view.transform().m11()
    view_center_after = window.view.mapToScene(window.view.viewport().rect().center())

    assert zoom_after == pytest.approx(zoom_before)
    assert abs(view_center_after.x() - view_center_before.x()) < 0.5
    assert abs(view_center_after.y() - view_center_before.y()) < 0.5
    assert item.isSelected() is True
    window.close()
