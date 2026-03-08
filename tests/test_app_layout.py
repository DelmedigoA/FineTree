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
    assert window.gemini_fill_btn.text() == "Auto-Fix"
    assert window.copy_image_btn.text() == "Copy Image"
    assert window.page_json_btn.text() == "JSON"
    assert window.exit_btn.text() == "Exit"
    assert window.page_jump_spin.minimumWidth() == 58
    assert not window.gemini_gt_btn.icon().isNull()
    assert not window.gemini_fill_btn.icon().isNull()
    assert not window.qwen_gt_btn.icon().isNull()
    assert window.page_thumb_list.iconSize().width() == 82
    assert window.thumb_panel.maximumWidth() == 152
    assert window.page_thumb_filter_combo.currentText() == "All Pages"
    assert window.page_thumb_sort_combo.currentText() == "Page Order"
    assert window.facts_list.objectName() == "factsList"
    assert window.facts_list.maximumHeight() == 176
    assert window.fact_editor_box.objectName() == "inspectorSubsection"
    assert window.fact_editor_box.layout().count() == 11
    assert window.apply_equation_btn.parentWidget() is not window.fact_editor_box
    assert window.show_order_labels_check.isChecked() is False
    assert window.gemini_fill_btn.isEnabled() is False
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


def test_page_approval_and_flag_controls_persist_and_drive_thumbnail_filter_sort(tmp_path: Path, monkeypatch) -> None:
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

    approved_idx = window.page_thumb_filter_combo.findText("Approved")
    window.page_thumb_filter_combo.setCurrentIndex(approved_idx)
    window._on_page_thumbnail_filter_or_sort_changed()
    _qt_app().processEvents()
    assert window.page_thumb_list.count() == 1
    assert window.page_thumb_list.item(0).data(Qt.UserRole) == 0
    assert window.current_index == 0

    all_idx = window.page_thumb_filter_combo.findText("All Pages")
    window.page_thumb_filter_combo.setCurrentIndex(all_idx)
    sort_idx = window.page_thumb_sort_combo.findText("Approved First")
    window.page_thumb_sort_combo.setCurrentIndex(sort_idx)
    window._on_page_thumbnail_filter_or_sort_changed()
    _qt_app().processEvents()
    order = [window.page_thumb_list.item(i).data(Qt.UserRole) for i in range(window.page_thumb_list.count())]
    assert order == [0, 2, 1]

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
            "equation": "40 + 60",
            "period_type": "instant",
            "period_start": None,
            "period_end": "2024-12-31",
            "balance_type": "debit",
            "path_source": "observed",
        },
    )
    item_b = AnnotRectItem(
        QRectF(40, 10, 20, 20),
        {
            "value": "200",
            "equation": "120 + 80",
            "period_type": "duration",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "balance_type": "credit",
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
        selected_fact_fields={"equation", "period_type", "period_start", "period_end", "balance_type"},
        include_statement_type=True,
        ordered_items=ordered_items,
    )

    facts = payload["pages"][0]["facts"]
    assert len(facts) == 2
    assert payload["pages"][0]["meta"]["statement_type"] is None
    assert all(fact["equation"] is None for fact in facts)
    assert all(fact["period_type"] is None for fact in facts)
    assert all(fact["period_start"] is None for fact in facts)
    assert all(fact["period_end"] is None for fact in facts)
    assert all(fact["balance_type"] is None for fact in facts)
    assert {fact["path_source"] for fact in facts} == {"observed", "inferred"}
    assert item_a.fact_data["equation"] == "40 + 60"
    assert item_a.fact_data["period_type"] == "instant"
    assert item_a.fact_data["balance_type"] == "debit"
    assert item_b.fact_data["equation"] == "120 + 80"
    assert item_b.fact_data["period_type"] == "duration"
    assert item_b.fact_data["balance_type"] == "credit"

    window.close()


def test_fact_editor_shows_balance_type_and_deterministic_natural_sign(tmp_path: Path) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    item = AnnotRectItem(
        QRectF(10, 10, 20, 20),
        {"value": "(120)", "balance_type": "credit", "path": []},
    )
    window.scene.addItem(item)
    window.refresh_facts_list()
    window.show()
    item.setSelected(True)
    _qt_app().processEvents()

    assert window.fact_balance_type_combo.currentText() == "credit"
    assert window.fact_natural_sign_label.text() == "negative"

    idx = window.fact_balance_type_combo.findText("debit")
    window.fact_balance_type_combo.setCurrentIndex(idx)
    window._on_fact_editor_field_edited("balance_type")
    assert item.fact_data["balance_type"] == "debit"

    window.fact_value_edit.setText("-")
    window.fact_value_edit.setModified(True)
    window._on_fact_editor_field_edited("value")
    _qt_app().processEvents()

    assert item.fact_data["natural_sign"] is None
    assert window.fact_natural_sign_label.text() == "-"
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


def test_equation_builder_applies_balance_type_sign_rules() -> None:
    candidate_text, result_text, fact_candidate_text, invalid_values, structured_terms = app_mod._build_equation_candidate_from_facts(
        [
            {"fact_num": 1, "value": "100", "balance_type": "debit"},
            {"fact_num": 2, "value": "30", "balance_type": "credit"},
            {"fact_num": 3, "value": "(5)", "balance_type": "credit"},
            {"fact_num": 4, "value": "-", "balance_type": "credit"},
        ]
    )

    assert candidate_text == "- 100 + 30 - 5 + 0"
    assert result_text == "-75"
    assert fact_candidate_text == "- f1 + f2 - f3 + f4"
    assert invalid_values == []
    assert [term.get("effective_normalized_value") for term in structured_terms] == [-100, 30, -5, 0]


def test_c_drag_builds_equation_preview_and_apply_persists_it(tmp_path: Path, monkeypatch) -> None:
    _qt_app()
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=220, height=220)
    annotations_path = tmp_path / "annotations.json"
    monkeypatch.setattr(app_mod.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window = AnnotationWindow(images_dir, annotations_path)
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

    assert target.fact_data.get("equation") is None
    assert target.fact_data.get("fact_equation") is None
    assert window.fact_equation_edit.text() == "100 - 5 + 0 + 20"
    assert window.fact_equation_result_label.text() == "115"
    assert "#b42318" in window.fact_equation_result_label.styleSheet()
    assert "Does not match target value" in window.fact_equation_status_label.text()
    assert "Ignored 1 invalid value" in window.fact_equation_status_label.text()
    assert {"fact_num": 4, "fact_reference": "f4", "normalized_value": 0, "raw_value": "-", "status": "normalized_dash"} in [
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

    assert target.fact_data["equation"] == "100 - 5 + 0 + 20"
    assert target.fact_data["fact_equation"] == "f1 - f2 + f4 + f5"
    assert window.apply_equation_btn.isEnabled() is False
    assert window.fact_equation_edit.text() == "100 - 5 + 0 + 20"
    assert window.fact_equation_result_label.text() == "115"
    assert "#b42318" in window.fact_equation_result_label.styleSheet()

    assert window.save_annotations() is True
    window.close()

    reloaded = AnnotationWindow(images_dir, annotations_path)
    reloaded.show()
    _qt_app().processEvents()

    reloaded_item = reloaded._fact_items[0]
    assert reloaded_item.fact_data["equation"] == "100 - 5 + 0 + 20"
    assert reloaded_item.fact_data["fact_equation"] == "f1 - f2 + f4 + f5"
    reloaded.close()


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

    assert target.fact_data.get("equation") is None
    assert target.fact_data.get("fact_equation") is None
    assert window.apply_equation_btn.isEnabled() is True

    window.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Shift, Qt.AltModifier, ""))
    _qt_app().processEvents()

    assert target.fact_data["equation"] == "100 + 20"
    assert target.fact_data["fact_equation"] == "f1 + f2"
    assert window.apply_equation_btn.isEnabled() is False
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
    assert item.fact_data.get("equation") is None
    assert item.fact_data.get("fact_equation") is None
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
    assert item_a.fact_data.get("equation") is None
    assert item_a.fact_data.get("fact_equation") is None
    assert item_b.fact_data.get("equation") is None
    assert item_b.fact_data.get("fact_equation") is None
    assert item_c.fact_data.get("equation") is None
    assert item_c.fact_data.get("fact_equation") is None
    assert window.clear_equation_btn.isEnabled() is False
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
    QTest.keyClick(window, Qt.Key_Delete)
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
    assert target.fact_data.get("equation") is None
    assert target.fact_data.get("fact_equation") is None

    ref_a.setSelected(True)
    _qt_app().processEvents()

    assert target.fact_data.get("equation") is None
    assert window.apply_equation_btn.isEnabled() is False
    assert window.fact_equation_edit.text() == ""
    window.close()


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
    assert window._equation_candidate_fact_text == "f1 - f2 + f3"
    assert sorted(item.fact_data["fact_num"] for item in window._equation_reference_preview_items) == [1, 2, 3]
    assert window.apply_equation_btn.isEnabled() is True

    window.view.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier, ""))
    _qt_app().processEvents()

    assert window.fact_equation_edit.text() == "100 - 5 + 8"
    assert window._equation_candidate_fact_text == "f1 - f2 + f3"
    assert target.fact_data.get("equation") is None
    assert target.fact_data.get("fact_equation") is None
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

    assert window.fact_equation_result_label.text() == "cannot calculate"
    assert "#b42318" in window.fact_equation_result_label.styleSheet()
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
