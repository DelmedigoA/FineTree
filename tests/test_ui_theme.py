from __future__ import annotations

from finetree_annotator import ui_theme


class _FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def value(self, key: str, default=None, type=None):
        _ = type
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


def test_theme_choice_round_trip() -> None:
    settings = _FakeSettings()
    assert ui_theme.load_theme_choice(settings) == "light"
    assert ui_theme.save_theme_choice("dark", settings) == "dark"
    assert ui_theme.load_theme_choice(settings) == "dark"
    assert ui_theme.save_theme_choice("unknown", settings) == "light"


def test_stylesheet_includes_checkbox_indicator_rules() -> None:
    stylesheet = ui_theme._app_stylesheet(ui_theme.theme_palette("light"))
    assert "QCheckBox::indicator" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet


def test_stylesheet_includes_inspector_rules() -> None:
    stylesheet = ui_theme._app_stylesheet(ui_theme.theme_palette("light"))
    assert "QGroupBox#inspectorSection" in stylesheet
    assert "QListWidget#factsList" in stylesheet
    assert "QListWidget#pageIssuesList" in stylesheet
    assert "QLabel#factBboxLabel" in stylesheet
    assert "QWidget#inspectorFieldBlock" in stylesheet
