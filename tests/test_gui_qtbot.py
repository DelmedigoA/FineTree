from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QGroupBox, QLineEdit, QListWidget, QPushButton

import finetree_annotator.app as app_mod
from finetree_annotator.annotation_core import BoxRecord, PageState
from finetree_annotator.app import AnnotRectItem, AnnotationWindow, JsonImportDialog


def _write_test_png(path: Path, *, width: int = 64, height: int = 64) -> None:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#ffffff"))
    assert image.save(str(path), "PNG")


def _make_window(tmp_path: Path, qtbot) -> AnnotationWindow:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png")
    annotations_path = tmp_path / "annotations.json"

    window = AnnotationWindow(images_dir, annotations_path)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)
    return window


def _button_by_qa_name(window: AnnotationWindow, qa_name: str) -> QPushButton:
    for button in window.findChildren(QPushButton, "smallActionBtn"):
        if str(button.property("qaName") or "") == qa_name:
            return button
    raise AssertionError(f"Button not found by qaName: {qa_name}")


def _batch_box(window: AnnotationWindow) -> QGroupBox:
    for box in window.findChildren(QGroupBox, "inspectorSubsection"):
        if str(box.property("qaName") or "") == "batchEditBox":
            return box
    raise AssertionError("Batch edit box not found")


@pytest.fixture(autouse=True)
def _auto_discard_unsaved_close(monkeypatch):
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")


def test_qtbot_application_startup_has_core_widgets(tmp_path: Path, qtbot) -> None:
    window = _make_window(tmp_path, qtbot)

    assert window.objectName() == "annotatorWindow"
    assert window.findChild(QListWidget, "factsList") is not None
    assert window.findChild(QLineEdit, "factValueEdit") is not None
    assert window.findChild(QLineEdit, "factEquationEdit") is not None



def test_qtbot_clicking_batch_toggle_button_updates_visibility(tmp_path: Path, qtbot) -> None:
    window = _make_window(tmp_path, qtbot)
    toggle_btn = _button_by_qa_name(window, "batchToggleBtn")
    batch_box = _batch_box(window)

    assert batch_box.isVisible() is False

    qtbot.mouseClick(toggle_btn, Qt.LeftButton)
    qtbot.wait(20)

    assert batch_box.isVisible() is True
    assert toggle_btn.text() == "Hide Batch Edit"

    qtbot.mouseClick(toggle_btn, Qt.LeftButton)
    qtbot.wait(20)

    assert batch_box.isVisible() is False
    assert toggle_btn.text() == "Show Batch Edit"



def test_qtbot_value_edit_updates_selected_fact_after_signal(tmp_path: Path, qtbot) -> None:
    window = _make_window(tmp_path, qtbot)
    fact_item = AnnotRectItem(QRectF(10, 10, 20, 20), {"value": "100"})
    window.scene.addItem(fact_item)
    window.refresh_facts_list()
    qtbot.wait(20)

    facts_list = window.findChild(QListWidget, "factsList")
    assert facts_list is not None
    index = facts_list.model().index(0, 0)
    rect = facts_list.visualRect(index)
    qtbot.mouseClick(facts_list.viewport(), Qt.LeftButton, pos=rect.center())
    qtbot.wait(20)

    value_edit = window.findChild(QLineEdit, "factValueEdit")
    assert value_edit is not None
    value_edit.setFocus()
    value_edit.selectAll()
    qtbot.keyClicks(value_edit, "321")

    with qtbot.waitSignal(value_edit.editingFinished, timeout=1000):
        qtbot.keyClick(value_edit, Qt.Key_Return)
    qtbot.wait(20)

    assert fact_item.fact_data["value"] == "321"
    assert "321" in facts_list.item(0).text()


def test_json_import_dialog_defaults_to_original_pixels(tmp_path: Path, qtbot) -> None:
    _ = tmp_path
    dialog = JsonImportDialog(default_page_name="page_0001.png")
    qtbot.addWidget(dialog)

    assert dialog.selected_bbox_mode() == app_mod.IMPORT_BBOX_MODE_ORIGINAL_PIXELS
    assert dialog.import_max_pixels() is None
    assert dialog.max_pixels_spin.value() == 1_400_000
    assert dialog.max_pixels_spin.isEnabled() is False


def test_json_import_dialog_resized_mode_uses_entered_max_pixels(tmp_path: Path, qtbot) -> None:
    _ = tmp_path
    dialog = JsonImportDialog(default_page_name="page_0001.png")
    qtbot.addWidget(dialog)

    dialog.bbox_mode_combo.setCurrentIndex(2)
    dialog.max_pixels_spin.setValue(1_234_567)

    assert dialog.selected_bbox_mode() == app_mod.IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS
    assert dialog.import_max_pixels() == 1_234_567
    assert dialog.max_pixels_spin.isEnabled() is True


def test_import_annotations_json_replaces_page_state_and_sets_status_message(tmp_path: Path, qtbot, monkeypatch) -> None:
    window = _make_window(tmp_path, qtbot)
    window.page_states["page_0001.png"] = PageState(
        meta={},
        facts=[BoxRecord(bbox={"x": 9, "y": 9, "w": 9, "h": 9}, fact={"value": "old", "path": []})],
    )

    class _FakeDialog:
        def __init__(self, default_page_name: str, parent=None) -> None:
            _ = default_page_name, parent

        def exec_(self) -> int:
            return app_mod.QDialog.Accepted

        def json_text(self) -> str:
            return '{"meta":{"page_num":"2"},"facts":[{"bbox":[1,2,3,4],"value":"10","path":[]}]}'

        def selected_bbox_mode(self) -> str:
            return app_mod.IMPORT_BBOX_MODE_ORIGINAL_PIXELS

        def import_max_pixels(self) -> int | None:
            return None

    monkeypatch.setattr(app_mod, "JsonImportDialog", _FakeDialog)

    window.import_annotations_json()
    qtbot.wait(20)

    state = window.page_states["page_0001.png"]
    assert state.meta["page_num"] == "2"
    assert len(state.facts) == 1
    assert state.facts[0].bbox == {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    assert state.facts[0].fact["value"] == "10"
    assert window.statusBar().currentMessage() == "Imported annotations for 1 page(s)."


def test_import_annotations_json_shows_resize_sanity_check_for_resized_mode(tmp_path: Path, qtbot, monkeypatch) -> None:
    window = _make_window(tmp_path, qtbot)
    info_calls: list[tuple[str, str]] = []

    class _FakeDialog:
        def __init__(self, default_page_name: str, parent=None) -> None:
            _ = default_page_name, parent

        def exec_(self) -> int:
            return app_mod.QDialog.Accepted

        def json_text(self) -> str:
            return '{"meta":{"page_num":"2"},"facts":[{"bbox":[1,2,3,4],"value":"10","path":[]}]}'

        def selected_bbox_mode(self) -> str:
            return app_mod.IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS

        def import_max_pixels(self) -> int | None:
            return 1_400_000

    monkeypatch.setattr(app_mod, "JsonImportDialog", _FakeDialog)
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "information",
        lambda _parent, title, text: info_calls.append((title, text)),
    )

    window.import_annotations_json()
    qtbot.wait(20)

    assert info_calls
    title, text = info_calls[-1]
    assert title == "Import sanity check"
    assert "Max pixels: 1,400,000" in text
    assert "page_0001.png: 64 x 64 from original 64 x 64" in text


def test_import_annotations_json_recovers_truncated_llm_output(tmp_path: Path, qtbot, monkeypatch) -> None:
    window = _make_window(tmp_path, qtbot)
    info_calls: list[tuple[str, str]] = []

    class _FakeDialog:
        def __init__(self, default_page_name: str, parent=None) -> None:
            _ = default_page_name, parent

        def exec_(self) -> int:
            return app_mod.QDialog.Accepted

        def json_text(self) -> str:
            return (
                '{"meta":{"page_num":"2"},'
                '"facts":[{"bbox":[1,2,3,4],"value":"10","path":[]},'
                '{"bbox":[5,6,7'
            )

        def selected_bbox_mode(self) -> str:
            return app_mod.IMPORT_BBOX_MODE_ORIGINAL_PIXELS

        def import_max_pixels(self) -> int | None:
            return None

    monkeypatch.setattr(app_mod, "JsonImportDialog", _FakeDialog)
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "information",
        lambda _parent, title, text: info_calls.append((title, text)),
    )

    window.import_annotations_json()
    qtbot.wait(20)

    state = window.page_states["page_0001.png"]
    assert state.meta["page_num"] == "2"
    assert len(state.facts) == 1
    assert state.facts[0].fact["value"] == "10"
    assert info_calls
    title, text = info_calls[-1]
    assert title == "Recovered partial JSON"
    assert "Recovered import from invalid JSON." in text
    assert "Imported 1 complete fact(s)" in text
