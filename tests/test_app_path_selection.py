from __future__ import annotations

from finetree_annotator import app as app_mod


def test_shared_path_prefix_returns_common_leading_levels() -> None:
    path_signatures = [
        ("assets", "current", "cash"),
        ("assets", "current", "receivables"),
        ("assets", "current", "inventory"),
    ]

    assert app_mod._shared_path_prefix(path_signatures) == ["assets", "current"]


def test_shared_path_elements_include_non_prefix_levels() -> None:
    path_signatures = [
        ("assets", "current", "cash"),
        ("liabilities", "current", "cash"),
        ("equity", "current", "cash"),
    ]

    assert app_mod._shared_path_prefix(path_signatures) == []
    assert app_mod._shared_path_elements(path_signatures) == ["current", "cash"]
