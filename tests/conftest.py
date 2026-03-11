from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtCore import qInstallMessageHandler
except Exception:  # pragma: no cover - only relevant when Qt is unavailable.
    qInstallMessageHandler = None

_IGNORED_QT_WARNING_SNIPPETS = (
    "This plugin does not support propagateSizeHints()",
)


@pytest.fixture(scope="session", autouse=True)
def _suppress_known_qt_plugin_warnings():
    """Keep pytest-qt stable in headless mode while preserving real warnings."""
    if qInstallMessageHandler is None:
        yield
        return

    def _handler(msg_type, context, message):
        text = str(message or "")
        if any(snippet in text for snippet in _IGNORED_QT_WARNING_SNIPPETS):
            return
        if previous_handler is not None:
            previous_handler(msg_type, context, message)

    previous_handler = qInstallMessageHandler(_handler)
    try:
        yield
    finally:
        qInstallMessageHandler(previous_handler)
