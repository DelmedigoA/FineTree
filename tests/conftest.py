from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (_REPO_ROOT / "src", _REPO_ROOT):
    candidate_text = str(candidate)
    if candidate.is_dir() and candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

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
