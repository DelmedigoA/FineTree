from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QGroupBox, QLineEdit, QListWidget, QPushButton

import finetree_annotator.app as app_mod
from finetree_annotator.app import AnnotRectItem, AnnotationWindow


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
