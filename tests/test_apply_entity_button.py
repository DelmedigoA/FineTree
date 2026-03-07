from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from finetree_annotator import app as app_mod
from finetree_annotator.annotation_core import PageState


class _FakeLineEdit:
    def __init__(self, value: str) -> None:
        self._value = value

    def text(self) -> str:
        return self._value


class _FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self.messages.append((message, timeout))


def _make_window(entity_name: str, page_states: dict[str, PageState]) -> tuple[SimpleNamespace, _FakeStatusBar, dict[str, object]]:
    page_images = [Path("page_0001.png"), Path("page_0002.png"), Path("page_0003.png")]
    status_bar = _FakeStatusBar()
    calls: dict[str, object] = {
        "capture_count": 0,
        "history_count": 0,
        "shown_pages": [],
    }

    window = SimpleNamespace(
        page_images=page_images,
        page_states=page_states,
        entity_name_edit=_FakeLineEdit(entity_name),
        current_index=0,
    )

    def _capture_current_state() -> None:
        calls["capture_count"] = int(calls["capture_count"]) + 1
        page_name = page_images[window.current_index].name
        state = page_states.get(page_name, PageState(meta={}, facts=[]))
        meta = dict(state.meta)
        meta["entity_name"] = window.entity_name_edit.text().strip() or None
        page_states[page_name] = PageState(meta=meta, facts=list(state.facts))

    def _record_history_snapshot() -> None:
        calls["history_count"] = int(calls["history_count"]) + 1

    def _show_page(index: int) -> None:
        shown_pages = calls["shown_pages"]
        assert isinstance(shown_pages, list)
        shown_pages.append(index)

    window._capture_current_state = _capture_current_state
    window._record_history_snapshot = _record_history_snapshot
    window.show_page = _show_page
    window.statusBar = lambda: status_bar
    return window, status_bar, calls


def test_apply_entity_button_backfills_missing_pages_after_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "_prompt_entity_apply_mode", lambda *_args, **_kwargs: "missing_only")

    page_states = {
        "page_0001.png": PageState(meta={"page_type": "other", "statement_type": None}, facts=[]),
        "page_0002.png": PageState(
            meta={"page_type": "statements", "statement_type": "notes_to_financial_statements", "title": "n2"},
            facts=[],
        ),
        "page_0003.png": PageState(meta={"entity_name": "KEEP", "page_type": "other", "statement_type": None}, facts=[]),
    }
    window, status_bar, calls = _make_window("ACME LTD", page_states)

    app_mod.AnnotationWindow.apply_entity_name_to_all_missing_pages(window)

    assert page_states["page_0001.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["title"] == "n2"
    assert page_states["page_0003.png"].meta["entity_name"] == "KEEP"
    assert calls["capture_count"] == 1
    assert calls["history_count"] == 1
    assert calls["shown_pages"] == [0]
    assert status_bar.messages == [("Applied entity_name to 1 page(s) with empty entity_name.", 5000)]


def test_apply_entity_button_does_nothing_when_confirmation_declined(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "_prompt_entity_apply_mode", lambda *_args, **_kwargs: None)

    page_states = {
        "page_0001.png": PageState(meta={"page_type": "other", "statement_type": None}, facts=[]),
        "page_0002.png": PageState(meta={"page_type": "other", "statement_type": None}, facts=[]),
    }
    window, status_bar, calls = _make_window("ACME LTD", page_states)

    app_mod.AnnotationWindow.apply_entity_name_to_all_missing_pages(window)

    assert "entity_name" not in page_states["page_0001.png"].meta
    assert "entity_name" not in page_states["page_0002.png"].meta
    assert calls["capture_count"] == 0
    assert calls["history_count"] == 0
    assert calls["shown_pages"] == []
    assert status_bar.messages == []


def test_apply_entity_button_requires_non_empty_entity_name(monkeypatch) -> None:
    info_calls: list[tuple[str, str]] = []

    def _information(_parent, title: str, message: str) -> int:
        info_calls.append((title, message))
        return app_mod.QMessageBox.Ok

    monkeypatch.setattr(app_mod.QMessageBox, "information", _information)
    monkeypatch.setattr(
        app_mod,
        "_prompt_entity_apply_mode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompt should not be called")),
    )

    page_states = {
        "page_0001.png": PageState(meta={"page_type": "other", "statement_type": None}, facts=[]),
    }
    window, status_bar, calls = _make_window("   ", page_states)

    app_mod.AnnotationWindow.apply_entity_name_to_all_missing_pages(window)

    assert info_calls == [
        (
            "Entity name required",
            "Enter an entity_name first, then click 'Apply Entity'.",
        )
    ]
    assert calls["capture_count"] == 0
    assert calls["history_count"] == 0
    assert calls["shown_pages"] == []
    assert status_bar.messages == []


def test_apply_entity_button_can_force_overwrite_all_pages(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "_prompt_entity_apply_mode", lambda *_args, **_kwargs: "force_all")

    page_states = {
        "page_0001.png": PageState(meta={"page_type": "other", "statement_type": None}, facts=[]),
        "page_0002.png": PageState(
            meta={"page_type": "statements", "statement_type": "notes_to_financial_statements", "title": "n2"},
            facts=[],
        ),
        "page_0003.png": PageState(meta={"entity_name": "KEEP", "page_type": "other", "statement_type": None}, facts=[]),
    }
    window, status_bar, calls = _make_window("ACME LTD", page_states)

    app_mod.AnnotationWindow.apply_entity_name_to_all_missing_pages(window)

    assert page_states["page_0001.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["title"] == "n2"
    assert page_states["page_0003.png"].meta["entity_name"] == "ACME LTD"
    assert calls["capture_count"] == 1
    assert calls["history_count"] == 1
    assert calls["shown_pages"] == [0]
    assert status_bar.messages == [("Applied entity_name to 2 page(s) across the PDF.", 5000)]
