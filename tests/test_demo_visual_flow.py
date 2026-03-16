from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtWidgets import QApplication

import finetree_annotator.app as app_mod
from finetree_annotator import dashboard
from finetree_annotator.app import AnnotRectItem, item_scene_rect
from finetree_annotator.workspace import sanitize_doc_id


@pytest.fixture(autouse=True)
def _auto_discard_unsaved_close(monkeypatch):
    monkeypatch.setattr(app_mod, "_prompt_unsaved_close_action", lambda _parent: "discard")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_sample_pdf(repo_root: Path) -> Path:
    search_roots = (
        repo_root / "assets",
        repo_root / "test",
        repo_root / "tests",
        repo_root / "data",
    )
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*.pdf"):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    if not discovered:
        raise AssertionError("No sample PDFs found under assets/test/tests/data.")

    discovered.sort(key=lambda path: str(path).lower())
    images_root = repo_root / "data" / "pdf_images"
    for pdf_path in discovered:
        doc_id = sanitize_doc_id(pdf_path.stem)
        if (images_root / doc_id / "page_0001.png").is_file():
            return pdf_path
    return discovered[0]


def _bbox_items(window: app_mod.AnnotationWindow) -> list[AnnotRectItem]:
    return [item for item in window.scene.items() if isinstance(item, AnnotRectItem)]


def _close_open_ai_dialog() -> None:
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if isinstance(widget, app_mod.AIDialog):
            widget.close()


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "").strip().lower() == "offscreen",
    reason="Visual GUI demo requires a non-headless Qt backend.",
)
def test_demo_visual_flow(qtbot, monkeypatch) -> None:
    repo_root = _repo_root()
    sample_pdf = _find_sample_pdf(repo_root)

    monkeypatch.setattr(
        dashboard.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(sample_pdf), "PDF Files (*.pdf)"),
    )
    monkeypatch.setattr(dashboard.QMessageBox, "warning", lambda *_args, **_kwargs: dashboard.QMessageBox.Ok)
    monkeypatch.setattr(dashboard.QMessageBox, "information", lambda *_args, **_kwargs: dashboard.QMessageBox.Ok)

    startup = app_mod.StartupContext(mode="home", images_dir=None, annotations_path=None)
    shell = dashboard.DashboardWindow(startup, dpi=200)
    qtbot.addWidget(shell)
    shell.show()
    qtbot.waitForWindowShown(shell)
    assert shell.isVisible()

    qtbot.mouseClick(shell.home_view.import_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: shell.annotator_host.current_window() is not None, timeout=120000)
    qtbot.waitUntil(lambda: shell._import_thread is None, timeout=120000)

    annotator = shell.annotator_host.current_window()
    assert annotator is not None
    qtbot.waitUntil(lambda: annotator.isVisible(), timeout=10000)
    qtbot.waitUntil(
        lambda: annotator.scene.image_rect.width() > 20 and annotator.scene.image_rect.height() > 20,
        timeout=15000,
    )

    # Explicitly navigate to page 1 through the page jump controls.
    page_jump_edit = annotator.page_jump_spin.lineEdit()
    page_jump_edit.setFocus()
    page_jump_edit.selectAll()
    qtbot.keyClicks(page_jump_edit, "1")
    qtbot.mouseClick(annotator.page_jump_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: annotator.current_index == 0, timeout=5000)

    existing_items = _bbox_items(annotator)
    baseline_count = len(existing_items)
    existing_ids = {id(item) for item in existing_items}
    image_rect = annotator.scene.image_rect
    start_scene = image_rect.topLeft() + QPointF(image_rect.width() * 0.18, image_rect.height() * 0.20)
    end_scene = start_scene + QPointF(max(40.0, image_rect.width() * 0.22), max(28.0, image_rect.height() * 0.14))
    end_scene = QPointF(min(end_scene.x(), image_rect.right() - 4), min(end_scene.y(), image_rect.bottom() - 4))

    start_view = annotator.view.mapFromScene(start_scene)
    end_view = annotator.view.mapFromScene(end_scene)
    qtbot.mousePress(annotator.view.viewport(), Qt.LeftButton, Qt.ControlModifier, start_view)
    qtbot.mouseMove(annotator.view.viewport(), end_view)
    qtbot.mouseRelease(annotator.view.viewport(), Qt.LeftButton, Qt.ControlModifier, end_view)
    qtbot.waitUntil(lambda: len(_bbox_items(annotator)) >= baseline_count + 1, timeout=6000)

    current_items = _bbox_items(annotator)
    created_item = next((item for item in current_items if id(item) not in existing_ids), current_items[-1])
    created_rect = item_scene_rect(created_item)
    assert created_rect.width() > 1
    assert created_rect.height() > 1
    assert annotator.scene.image_rect.contains(created_rect.center())

    # Trigger AI action window (user touchpoint) and close it automatically.
    QTimer.singleShot(150, _close_open_ai_dialog)
    qtbot.mouseClick(annotator.ai_btn, Qt.LeftButton)
    qtbot.wait(350)
    assert annotator._gemini_stream_thread is None

    first_page_name = annotator.page_images[0].name
    qtbot.mouseClick(annotator.page_approve_continue_btn, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: (
            first_page_name in annotator.page_states
            and annotator.page_states[first_page_name].meta.get("annotation_status") == "approved"
        ),
        timeout=3000,
    )

    # Keep the app visible for manual observation.
    qtbot.wait(7000)

    qtbot.mouseClick(annotator.exit_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not shell.isVisible(), timeout=10000)
