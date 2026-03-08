from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QStyleFactory

DEFAULT_THEME = "light"
THEME_SETTING_KEY = "ui/theme"
SETTINGS_ORG = "FineTree"
SETTINGS_APP = "FineTreeAnnotator"

_LIGHT = {
    "font_heading": "Avenir Next, Manrope, 'Segoe UI', sans-serif",
    "font_body": "'IBM Plex Sans', 'SF Pro Text', 'Segoe UI', sans-serif",
    "font_mono": "'IBM Plex Mono', 'SF Mono', Menlo, monospace",
    "bg": "#e6edf5",
    "bg_alt": "#dbe5ef",
    "surface": "#f8fbff",
    "surface_alt": "#edf3f9",
    "surface_raised": "#ffffff",
    "surface_border": "#c6d4e3",
    "text": "#172334",
    "text_muted": "#5c6d82",
    "text_soft": "#7b8a9d",
    "accent": "#0f9fb6",
    "accent_soft": "#d7f4f8",
    "accent_strong": "#0b7285",
    "selection": "#c2eef4",
    "nav_bg": "#122033",
    "nav_surface": "#172a42",
    "nav_text": "#d6e8f4",
    "nav_muted": "#8eabc1",
    "nav_selected": "#0f9fb6",
    "ok": "#17855c",
    "warn": "#b7791f",
    "danger": "#b94949",
    "shadow": "rgba(8, 19, 35, 0.08)",
    "canvas": "#d6dee7",
}

_DARK = {
    "font_heading": "Avenir Next, Manrope, 'Segoe UI', sans-serif",
    "font_body": "'IBM Plex Sans', 'SF Pro Text', 'Segoe UI', sans-serif",
    "font_mono": "'IBM Plex Mono', 'SF Mono', Menlo, monospace",
    "bg": "#101923",
    "bg_alt": "#0d141d",
    "surface": "#162332",
    "surface_alt": "#1b2b3d",
    "surface_raised": "#1f3248",
    "surface_border": "#32465d",
    "text": "#edf6ff",
    "text_muted": "#a6bacf",
    "text_soft": "#7f98b1",
    "accent": "#33c6d9",
    "accent_soft": "#153744",
    "accent_strong": "#5ce3f6",
    "selection": "#13404a",
    "nav_bg": "#0a1119",
    "nav_surface": "#111d2c",
    "nav_text": "#d5e4f3",
    "nav_muted": "#7f95ab",
    "nav_selected": "#33c6d9",
    "ok": "#39b980",
    "warn": "#e0aa47",
    "danger": "#ff7a7a",
    "shadow": "rgba(0, 0, 0, 0.22)",
    "canvas": "#213140",
}

THEMES = {"light": _LIGHT, "dark": _DARK}


def app_settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def normalize_theme_name(name: Optional[str]) -> str:
    key = str(name or "").strip().lower()
    return key if key in THEMES else DEFAULT_THEME


def load_theme_choice(settings: Optional[QSettings] = None) -> str:
    settings = settings or app_settings()
    return normalize_theme_name(settings.value(THEME_SETTING_KEY, DEFAULT_THEME, type=str))


def save_theme_choice(theme_name: str, settings: Optional[QSettings] = None) -> str:
    resolved = normalize_theme_name(theme_name)
    settings = settings or app_settings()
    settings.setValue(THEME_SETTING_KEY, resolved)
    return resolved


def available_themes() -> tuple[str, ...]:
    return tuple(THEMES.keys())


def theme_palette(theme_name: str) -> dict[str, str]:
    return dict(THEMES[normalize_theme_name(theme_name)])


def _load_font_assets() -> None:
    fonts_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    if not fonts_dir.is_dir():
        return
    for ext in ("*.ttf", "*.otf"):
        for font_path in fonts_dir.glob(ext):
            QFontDatabase.addApplicationFont(str(font_path))


def _app_stylesheet(tokens: dict[str, str]) -> str:
    return f"""
    QWidget {{
        color: {tokens['text']};
        background: {tokens['bg']};
        font-family: {tokens['font_body']};
        font-size: 11pt;
    }}
    QLabel {{
        background: transparent;
    }}
    QCheckBox {{
        background: transparent;
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {tokens['surface_border']};
        border-radius: 5px;
        background: {tokens['surface_raised']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {tokens['accent']};
    }}
    QCheckBox::indicator:checked {{
        background: {tokens['accent']};
        border-color: {tokens['accent']};
    }}
    QCheckBox::indicator:disabled {{
        background: {tokens['surface_alt']};
        border-color: {tokens['surface_border']};
    }}
    QWidget#shellRoot,
    QMainWindow#shellWindow,
    QWidget#homeView,
    QWidget#pushView,
    QWidget#annotatorHost,
    QWidget#inspectorPanel {{
        background: {tokens['bg']};
    }}
    QLabel#eyebrowLabel {{
        color: {tokens['text_soft']};
        font-size: 9pt;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    QLabel#sectionTitle,
    QLabel#homeTitle,
    QLabel#pushTitle {{
        color: {tokens['text']};
        font-family: {tokens['font_heading']};
        font-size: 20pt;
        font-weight: 700;
        background: transparent;
    }}
    QLabel#annotatorTitle {{
        color: {tokens['text']};
        font-family: {tokens['font_heading']};
        font-size: 13pt;
        font-weight: 700;
        background: transparent;
    }}
    QLabel#subtitleLabel,
    QLabel#homeSubtitle,
    QLabel#pushSubtitle,
    QLabel#statusMetaLabel,
    QLabel#outputPathLabel,
    QLabel#hintText {{
        color: {tokens['text_muted']};
        background: transparent;
    }}
    QLabel#toolbarTitle {{
        color: {tokens['text_soft']};
        background: transparent;
        font-family: {tokens['font_heading']};
        font-size: 7pt;
        font-weight: 700;
        letter-spacing: 0.05em;
    }}
    QLabel#inspectorFieldLabel,
    QLabel#inspectorFieldLabelRequired,
    QLabel#inspectorFieldLabelCompact,
    QLabel#inspectorFieldLabelCompactRequired {{
        color: {tokens['text_soft']};
        background: transparent;
        font-family: {tokens['font_heading']};
        font-size: 9pt;
        font-weight: 700;
    }}
    QLabel#inspectorFieldLabel,
    QLabel#inspectorFieldLabelRequired {{
        padding-top: 6px;
    }}
    QLabel#inspectorFieldLabelCompact,
    QLabel#inspectorFieldLabelCompactRequired {{
        padding-top: 0;
        padding-bottom: 1px;
        font-size: 8.7pt;
    }}
    QWidget#inspectorFieldBlock {{
        background: transparent;
        border: none;
    }}
    QLabel#monoLabel,
    QLabel#statusLogLabel {{
        color: {tokens['text_muted']};
        background: transparent;
        font-family: {tokens['font_mono']};
    }}
    QFrame#navRail {{
        background: {tokens['nav_bg']};
        border: none;
        border-radius: 18px;
    }}
    QLabel#navBrand {{
        color: {tokens['nav_text']};
        background: transparent;
        font-family: {tokens['font_heading']};
        font-size: 18pt;
        font-weight: 700;
    }}
    QLabel#navCaption {{
        color: {tokens['nav_muted']};
        background: transparent;
        font-size: 10pt;
    }}
    QPushButton#navButton {{
        background: transparent;
        color: {tokens['nav_text']};
        border: 1px solid transparent;
        border-radius: 16px;
        text-align: left;
        padding: 16px 18px;
        min-height: 58px;
        font-family: {tokens['font_heading']};
        font-size: 11.5pt;
        font-weight: 600;
    }}
    QPushButton#navButton:hover {{
        background: {tokens['nav_surface']};
        border-color: {tokens['nav_surface']};
    }}
    QPushButton#navButton:checked {{
        background: {tokens['accent']};
        color: #ffffff;
        border-color: {tokens['accent']};
    }}
    QPushButton#shellChromeBtn {{
        background: {tokens['surface_raised']};
        color: {tokens['text']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 8px;
        min-height: 18px;
        max-height: 20px;
        padding: 0 7px;
        font-family: {tokens['font_heading']};
        font-size: 8.5pt;
        font-weight: 600;
    }}
    QPushButton#shellChromeBtn:hover {{
        border-color: {tokens['accent']};
        background: {tokens['surface']};
    }}
    QMenuBar {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 14px;
        padding: 4px 8px;
        font-family: {tokens['font_heading']};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 8px 12px;
        border-radius: 10px;
    }}
    QMenuBar::item:selected {{
        background: {tokens['accent_soft']};
    }}
    QMenu {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        padding: 8px;
    }}
    QMenu::item {{
        padding: 8px 18px;
        border-radius: 8px;
    }}
    QMenu::item:selected {{
        background: {tokens['accent_soft']};
    }}
    QFrame#surfaceCard,
    QWidget#surfaceCard,
    QGroupBox,
    QFrame#docCard,
    QFrame#statCard,
    QFrame#emptyStateCard {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 16px;
    }}
    QFrame#toolbarGroup {{
        background: transparent;
        border: none;
    }}
    QFrame#annotatorHeaderCard,
    QFrame#toolbarStrip {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 12px;
    }}
    QFrame#toolbarDivider {{
        background: {tokens['surface_border']};
        min-width: 1px;
        max-width: 1px;
        margin: 2px 2px;
    }}
    QGroupBox#inspectorSection {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 14px;
        margin-top: 10px;
        padding-top: 12px;
    }}
    QGroupBox#inspectorSubsection {{
        background: {tokens['surface']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 12px;
        margin-top: 8px;
        padding-top: 10px;
    }}
    QFrame#docCard[statusTone="complete"] {{
        border-color: {tokens['ok']};
    }}
    QFrame#docCard[statusTone="progress"] {{
        border-color: {tokens['accent']};
    }}
    QFrame#docCard[statusTone="attention"] {{
        border-color: {tokens['warn']};
    }}
    QGroupBox {{
        margin-top: 16px;
        padding-top: 14px;
        font-family: {tokens['font_heading']};
        font-size: 13pt;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 6px;
        color: {tokens['text']};
        background: transparent;
    }}
    QGroupBox#inspectorSection::title,
    QGroupBox#inspectorSubsection::title {{
        left: 14px;
        padding: 0;
        color: {tokens['text']};
        font-family: {tokens['font_heading']};
        font-size: 10.5pt;
        font-weight: 700;
    }}
    QLabel#statusPill {{
        background: {tokens['surface_alt']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 9pt;
        font-weight: 700;
        color: {tokens['text_muted']};
    }}
    QLabel#statusPill[tone="accent"] {{
        background: {tokens['accent_soft']};
        border-color: {tokens['accent']};
        color: {tokens['accent_strong']};
    }}
    QLabel#statusPill[tone="ok"] {{
        background: {tokens['surface_alt']};
        border-color: {tokens['ok']};
        color: {tokens['ok']};
    }}
    QLabel#statusPill[tone="warn"] {{
        background: {tokens['surface_alt']};
        border-color: {tokens['warn']};
        color: {tokens['warn']};
    }}
    QLabel#statusPill[tone="danger"] {{
        background: {tokens['surface_alt']};
        border-color: {tokens['danger']};
        color: {tokens['danger']};
    }}
    QLineEdit,
    QComboBox,
    QSpinBox,
    QPlainTextEdit,
    QListWidget,
    QGraphicsView {{
        background: {tokens['surface']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 12px;
        selection-background-color: {tokens['selection']};
        selection-color: {tokens['text']};
    }}
    QScrollArea {{
        background: {tokens['surface']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 12px;
    }}
    QScrollArea#toolbarScroll,
    QScrollArea#pushFormScroll,
    QScrollArea#homeCardsScroll,
    QScrollArea#inspectorScroll {{
        background: transparent;
        border: none;
    }}
    QLineEdit,
    QComboBox,
    QSpinBox {{
        min-height: 38px;
        padding: 4px 10px;
    }}
    QPlainTextEdit {{
        padding: 10px;
        font-family: {tokens['font_mono']};
    }}
    QLineEdit:focus,
    QComboBox:focus,
    QSpinBox:focus,
    QPlainTextEdit:focus,
    QListWidget:focus,
    QGraphicsView:focus {{
        border: 2px solid {tokens['accent']};
    }}
    QPushButton {{
        background: {tokens['surface_alt']};
        color: {tokens['text']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 10px;
        min-height: 36px;
        padding: 6px 12px;
        font-family: {tokens['font_heading']};
        font-size: 10.5pt;
        font-weight: 600;
    }}
    QPushButton:hover {{
        border-color: {tokens['accent']};
    }}
    QPushButton:pressed {{
        background: {tokens['accent_soft']};
    }}
    QPushButton:disabled {{
        color: {tokens['text_soft']};
        border-color: {tokens['surface_border']};
    }}
    QPushButton[variant="primary"] {{
        background: {tokens['accent']};
        color: #ffffff;
        border-color: {tokens['accent']};
    }}
    QPushButton[variant="primary"]:hover {{
        background: {tokens['accent_strong']};
        border-color: {tokens['accent_strong']};
    }}
    QPushButton[variant="ghost"] {{
        background: transparent;
    }}
    QPushButton[variant="danger"] {{
        background: {tokens['danger']};
        color: #ffffff;
        border-color: {tokens['danger']};
    }}
    QPushButton#toolbarActionBtn {{
        min-height: 24px;
        padding: 0 7px;
        border-radius: 8px;
        font-size: 8pt;
    }}
    QPushButton#smallActionBtn {{
        min-height: 30px;
        font-size: 9.5pt;
        padding: 3px 9px;
    }}
    QListWidget {{
        padding: 6px;
    }}
    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 8px;
        margin: 2px 0;
    }}
    QListWidget::item:selected {{
        background: {tokens['selection']};
        color: {tokens['text']};
    }}
    QListWidget#thumbList {{
        background: {tokens['surface_alt']};
        padding: 10px;
    }}
    QListWidget#factsList,
    QListWidget#pathList,
    QListWidget#pageIssuesList {{
        background: {tokens['surface_alt']};
        border-radius: 14px;
        padding: 8px;
    }}
    QListWidget#factsList::item,
    QListWidget#pageIssuesList::item {{
        background: {tokens['surface_raised']};
        border: 1px solid transparent;
        min-height: 30px;
        padding: 8px 10px;
        margin: 4px 0;
    }}
    QListWidget#factsList::item:selected,
    QListWidget#pageIssuesList::item:selected {{
        background: {tokens['surface_raised']};
        border-color: {tokens['accent']};
        color: {tokens['text']};
    }}
    QListWidget#thumbList::item {{
        background: {tokens['surface_raised']};
        border: 1px solid {tokens['surface_border']};
        min-height: 134px;
        padding: 4px;
        margin: 3px 0;
    }}
    QListWidget#thumbList::item:selected {{
        border: 2px solid {tokens['accent']};
        background: {tokens['surface_raised']};
    }}
    QLabel#factBboxLabel {{
        background: {tokens['surface_alt']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 8px;
        color: {tokens['text_muted']};
        font-family: {tokens['font_mono']};
        padding: 5px 8px;
        min-height: 18px;
    }}
    QWidget#inspectorPanel QLabel#inspectorFieldLabel,
    QWidget#inspectorPanel QLabel#inspectorFieldLabelRequired {{
        padding-top: 3px;
        font-size: 8.7pt;
    }}
    QWidget#inspectorPanel QLabel#inspectorFieldLabelCompact,
    QWidget#inspectorPanel QLabel#inspectorFieldLabelCompactRequired {{
        font-size: 8.2pt;
        padding-bottom: 0;
    }}
    QWidget#inspectorPanel QLineEdit,
    QWidget#inspectorPanel QComboBox,
    QWidget#inspectorPanel QSpinBox {{
        min-height: 30px;
        padding: 2px 8px;
        border-radius: 9px;
        font-size: 10pt;
    }}
    QWidget#inspectorPanel QPushButton {{
        min-height: 30px;
        padding: 4px 10px;
        font-size: 9.8pt;
    }}
    QWidget#inspectorPanel QPushButton#smallActionBtn {{
        min-height: 26px;
        padding: 2px 8px;
        font-size: 9pt;
    }}
    QWidget#inspectorPanel QListWidget#factsList::item,
    QWidget#inspectorPanel QListWidget#pageIssuesList::item {{
        min-height: 24px;
        padding: 6px 8px;
        margin: 3px 0;
    }}
    QCheckBox#inspectorOption {{
        color: {tokens['text_muted']};
        font-size: 10pt;
        font-weight: 600;
    }}
    QGraphicsView {{
        background: {tokens['canvas']};
    }}
    QProgressBar {{
        background: {tokens['surface_alt']};
        border: 1px solid {tokens['surface_border']};
        border-radius: 999px;
        text-align: center;
        min-height: 18px;
        color: {tokens['text_muted']};
    }}
    QProgressBar::chunk {{
        background: {tokens['accent']};
        border-radius: 999px;
    }}
    QStatusBar {{
        background: {tokens['surface_raised']};
        border-top: 1px solid {tokens['surface_border']};
        color: {tokens['text_muted']};
    }}
    QSplitter::handle {{
        background: {tokens['bg_alt']};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
        margin: 8px 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {tokens['surface_border']};
        border-radius: 6px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0;
    }}
    """


def apply_theme(app: QApplication, theme_name: str) -> str:
    resolved = normalize_theme_name(theme_name)
    _load_font_assets()
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    tokens = theme_palette(resolved)
    base_font = QFont()
    base_font.setFamilies([font.strip() for font in tokens["font_body"].replace("'", "").split(",") if font.strip()])
    if base_font.pointSize() < 11:
        base_font.setPointSize(11)
    app.setFont(base_font)
    app.setStyleSheet(_app_stylesheet(tokens))
    app.setProperty("finetreeTheme", resolved)
    return resolved
