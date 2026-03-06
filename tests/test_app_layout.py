from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QKeyEvent
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


def _write_test_png(path: Path, *, width: int = 32, height: int = 32) -> None:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#ffffff"))
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


@pytest.fixture(autouse=True)
def _auto_discard_unsaved_close(monkeypatch):
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")


def test_annotation_window_defaults_to_hidden_batch_panel_and_text_toolbar(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    window.show()
    _qt_app().processEvents()

    assert not window.batch_box.isVisible()
    assert window.batch_toggle_btn.text() == "Show Batch Edit"
    assert window.save_btn.text() == "Save"
    assert window.gemini_gt_btn.text() == "Gemini"
    assert window.copy_image_btn.text() == "Copy Image"
    assert window.page_json_btn.text() == "JSON"
    assert window.exit_btn.text() == "Exit"
    assert window.page_jump_spin.minimumWidth() == 58
    assert not window.gemini_gt_btn.icon().isNull()
    assert not window.qwen_gt_btn.icon().isNull()
    assert window.page_thumb_list.iconSize().width() == 82
    assert window.thumb_panel.maximumWidth() == 152
    assert window.facts_list.objectName() == "factsList"
    assert window.facts_list.maximumHeight() == 190
    assert window.fact_editor_box.objectName() == "inspectorSubsection"
    assert window.fact_editor_box.layout().count() == 7
    assert window.fact_is_beur_combo.maximumWidth() == 220
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
    item_a = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100", "currency": "USD", "comment": "a"})
    item_b = AnnotRectItem(QRectF(40, 10, 20, 20), {"value": "200", "currency": "ILS", "comment": "b"})
    window.scene.addItem(item_a)
    window.scene.addItem(item_b)
    window.refresh_facts_list()
    window.show()
    item_a.setSelected(True)
    item_b.setSelected(True)
    _qt_app().processEvents()

    assert window.batch_box.isVisible()
    assert window.batch_toggle_btn.text() == "Hide Batch Edit"
    assert window.fact_editor_box.title() == "Selected Facts (2)"
    assert window.fact_bbox_label.text() == "2 selected"
    assert window.fact_value_edit.placeholderText() == "Multiple values"
    assert window.fact_path_list.isEnabled()
    assert window.path_add_btn.isEnabled() is True

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
    item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "5", "note_reference": None, "value_type": "amount"})
    window.scene.addItem(item)
    window.refresh_facts_list()
    item.setSelected(True)
    _qt_app().processEvents()

    value_type_idx = window.fact_value_type_combo.findText("%")
    window.fact_value_type_combo.setCurrentIndex(value_type_idx)
    window._on_fact_editor_field_edited("value_type")
    window.fact_value_edit.setText("7-10")
    window.fact_value_edit.setModified(True)
    window._on_fact_editor_field_edited("value")

    assert item.fact_data["value"] == "7-10"
    assert item.fact_data["note_reference"] is None
    assert item.fact_data["value_type"] == "%"
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
    window.type_combo.setCurrentText("notes")
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


def test_fact_list_selection_zooms_to_bbox(tmp_path: Path) -> None:
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
    assert window.facts_list.item(0) is not None
    window.facts_list.item(0).setSelected(True)
    window.facts_list.setCurrentRow(0)
    _qt_app().processEvents()

    zoom_after = window.view.transform().m11()
    view_center = window.view.mapToScene(window.view.viewport().rect().center())
    bbox_center = item_scene_rect(item).center()

    assert zoom_after > zoom_before
    assert abs(view_center.x() - bbox_center.x()) < 40.0
    assert abs(view_center.y() - bbox_center.y()) < 40.0
    window.close()
