from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QGroupBox, QLineEdit, QListWidget, QPushButton

import finetree_annotator.app as app_mod
from finetree_annotator.annotation_core import BoxRecord, PageState, make_placeholder_bbox
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
    assert dialog.run_align_bboxes_after_import() is True
    assert dialog.align_after_import_check.text() == "Align bounding boxes after import"
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

        def run_align_bboxes_after_import(self) -> bool:
            return False

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

        def run_align_bboxes_after_import(self) -> bool:
            return False

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

        def run_align_bboxes_after_import(self) -> bool:
            return False

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


def test_import_annotations_json_assigns_placeholder_bbox_without_auto_align(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    window = _make_window(tmp_path, qtbot)

    class _FakeDialog:
        def __init__(self, default_page_name: str, parent=None) -> None:
            _ = default_page_name, parent

        def exec_(self) -> int:
            return app_mod.QDialog.Accepted

        def json_text(self) -> str:
            return '{"facts":[{"value":"10","note_num":2,"path":[]}]}'

        def selected_bbox_mode(self) -> str:
            return app_mod.IMPORT_BBOX_MODE_ORIGINAL_PIXELS

        def import_max_pixels(self) -> int | None:
            return None

        def run_align_bboxes_after_import(self) -> bool:
            return False

    monkeypatch.setattr(app_mod, "JsonImportDialog", _FakeDialog)

    window.import_annotations_json()
    qtbot.wait(20)

    state = window.page_states["page_0001.png"]
    assert len(state.facts) == 1
    assert state.facts[0].bbox == {"x": 8.0, "y": 8.0, "w": 56.0, "h": 28.0}
    assert state.facts[0].fact["note_num"] == "2"


def test_import_annotations_json_runs_align_queue_for_imported_pages(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=120)
    _write_test_png(images_dir / "page_0002.png", width=240, height=120)
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)

    started: list[tuple[str, str]] = []

    class _FakeDialog:
        def __init__(self, default_page_name: str, parent=None) -> None:
            _ = default_page_name, parent

        def exec_(self) -> int:
            return app_mod.QDialog.Accepted

        def json_text(self) -> str:
            return (
                '{"pages":['
                '{"image":"page_0001.png","facts":[{"value":"10","path":[]}]},'
                '{"image":"page_0002.png","facts":[{"value":"20","path":[]}]}'
                ']}'
            )

        def selected_bbox_mode(self) -> str:
            return app_mod.IMPORT_BBOX_MODE_ORIGINAL_PIXELS

        def import_max_pixels(self) -> int | None:
            return None

        def run_align_bboxes_after_import(self) -> bool:
            return True

    def _fake_start_align(page_name: str, *, mode: str = "align") -> bool:
        started.append((page_name, mode))
        window._import_align_last_result = {"page_name": page_name, "matched_count": 1, "unmatched_count": 0}
        window.import_align_bboxes_page_completed.emit(page_name, 1)
        return True

    monkeypatch.setattr(app_mod, "JsonImportDialog", _FakeDialog)
    monkeypatch.setattr(window, "start_align_bboxes_for_page", _fake_start_align)

    window.import_annotations_json()
    qtbot.wait(20)

    assert started == [
        ("page_0001.png", "qwen_import"),
        ("page_0002.png", "qwen_import"),
    ]
    assert window._import_align_total_pages == 0


def test_import_annotations_json_text_replaces_only_imported_pages_by_order(
    tmp_path: Path,
    qtbot,
) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=120)
    _write_test_png(images_dir / "page_0002.png", width=240, height=120)
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)

    window.page_states["page_0001.png"] = PageState(
        meta={"page_num": "old-1"},
        facts=[BoxRecord(bbox={"x": 1, "y": 1, "w": 5, "h": 5}, fact={"value": "old-1", "path": []})],
    )
    window.page_states["page_0002.png"] = PageState(
        meta={"page_num": "old-2"},
        facts=[BoxRecord(bbox={"x": 2, "y": 2, "w": 5, "h": 5}, fact={"value": "old-2", "path": []})],
    )

    summary = window.import_annotations_json_text(
        json.dumps(
            [
                json.dumps({"meta": {"page_num": "1"}, "facts": [{"value": "10", "path": []}]}),
            ]
        ),
        run_align_after_import=False,
        show_info_messages=False,
    )
    qtbot.wait(20)

    assert summary["imported_count"] == 1
    assert summary["parsed_page_count"] == 1
    assert window.page_states["page_0001.png"].meta["page_num"] == "1"
    assert window.page_states["page_0001.png"].facts[0].fact["value"] == "10"
    assert window.page_states["page_0002.png"].meta["page_num"] == "old-2"
    assert window.page_states["page_0002.png"].facts[0].fact["value"] == "old-2"


def test_import_annotations_json_text_reports_ignored_extra_pages(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    window = _make_window(tmp_path, qtbot)
    info_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_mod.QMessageBox,
        "information",
        lambda _parent, title, text: info_calls.append((title, text)),
    )

    summary = window.import_annotations_json_text(
        json.dumps(
            [
                json.dumps({"meta": {"page_num": "1"}, "facts": [{"value": "10", "path": []}]}),
                json.dumps({"meta": {"page_num": "2"}, "facts": [{"value": "20", "path": []}]}),
            ]
        ),
        run_align_after_import=False,
        show_info_messages=True,
    )
    qtbot.wait(20)

    assert summary["extra_page_count"] == 1
    assert info_calls
    assert "Ignored 1 imported page(s)" in info_calls[-1][1]


def test_import_annotations_json_text_queues_qwen_align_for_imported_pages_only(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    images_dir = tmp_path / "pages"
    images_dir.mkdir(parents=True)
    _write_test_png(images_dir / "page_0001.png", width=240, height=120)
    _write_test_png(images_dir / "page_0002.png", width=240, height=120)
    window = AnnotationWindow(images_dir, tmp_path / "annotations.json")
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)

    window.page_states["page_0002.png"] = PageState(
        meta={"page_num": "existing"},
        facts=[BoxRecord(bbox={"x": 2, "y": 2, "w": 5, "h": 5}, fact={"value": "keep", "path": []})],
    )

    started: list[tuple[str, str]] = []

    def _fake_start_align(page_name: str, *, mode: str = "align") -> bool:
        started.append((page_name, mode))
        window._import_align_last_result = {
            "page_name": page_name,
            "matched_count": 0,
            "unmatched_count": 1,
        }
        window.import_align_bboxes_page_completed.emit(page_name, 1)
        return True

    monkeypatch.setattr(window, "start_align_bboxes_for_page", _fake_start_align)

    summary = window.import_annotations_json_text(
        json.dumps(
            [
                json.dumps({"meta": {"page_num": "1"}, "facts": [{"value": "10", "path": []}]}),
            ]
        ),
        run_align_after_import=True,
        show_info_messages=False,
    )
    qtbot.wait(20)

    assert summary["imported_page_names"] == ["page_0001.png"]
    assert started == [("page_0001.png", "qwen_import")]
    assert window._import_align_total_pages == 0
    assert window.page_states["page_0001.png"].facts[0].bbox == make_placeholder_bbox(0)
    assert window.page_states["page_0002.png"].meta["page_num"] == "existing"
    assert window.page_states["page_0002.png"].facts[0].fact["value"] == "keep"
