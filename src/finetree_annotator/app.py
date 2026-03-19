"""Simple PyQt5 app for annotating PDF page images with schema-based facts."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError
from PyQt5 import sip
from PyQt5.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, QItemSelectionModel, QEvent, pyqtSignal, QRegularExpression
from PyQt5.QtGui import QBrush, QColor, QCloseEvent, QFont, QIcon, QImage, QIntValidator, QKeyEvent, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QResizeEvent, QShowEvent, QTextCursor, QTextDocument, QTransform, QRegularExpressionValidator
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QShortcut,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .numeric_text import normalize_angle_bracketed_numeric_text
from .ai.bbox import (
    BBOX_MODE_MIXED_AUTO,
    BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
    BBOX_MODE_PIXEL_AS_IS,
    bbox_looks_normalized_1000,
    normalize_ai_fact_payload,
    ordered_fact_payloads_by_geometry,
    payloads_for_bbox_mode,
    resolve_bbox_mode,
    score_bbox_candidate_payloads,
    score_bbox_payload_ink,
)
from .ai.controller import AIWorkflowController
from .ai.dialog import AIDialog
from .ai.payloads import (
    build_gemini_autocomplete_prompt as build_ai_gemini_autocomplete_prompt,
    build_gemini_autocomplete_request_payload as build_ai_gemini_autocomplete_request_payload,
    build_gemini_fill_prompt as build_ai_gemini_fill_prompt,
    build_gemini_fill_request_payload as build_ai_gemini_fill_request_payload,
    build_page_prompt_payload as build_ai_page_prompt_payload,
)
from .ai.types import (
    AIActionKind,
    AIPageContext,
    AIProvider,
    FEW_SHOT_PRESET_2015_TWO_SHOT,
    FEW_SHOT_PRESET_CLASSIC,
    FEW_SHOT_PRESET_EXTENDED,
    FEW_SHOT_PRESET_ONE_SHOT,
)
from .annotation_core import (
    BoxRecord,
    CURRENCY_OPTIONS,
    PageState,
    SCALE_OPTIONS,
    apply_entity_name_to_pages,
    bbox_to_list,
    build_annotations_payload_with_findings,
    denormalize_bbox_from_1000,
    default_page_meta,
    extract_document_meta,
    load_page_states,
    parse_import_payload,
    normalize_bbox_data,
    normalize_fact_data,
    serialize_annotations_json,
)
from .annotation_backups import atomic_write_text, create_annotation_backup
from .equation_integrity import audit_and_rebuild_financial_facts, resequence_fact_numbers_and_remap_fact_equations
from .fact_ordering import canonical_fact_order_indices, compact_document_meta, normalize_document_meta, resolve_reading_direction
from .fact_normalization import normalize_annotation_payload
from .finetune.config import load_finetune_config
from .gemini_vlm import DEFAULT_GEMINI_MODEL
from .gemini_few_shot import (
    DEFAULT_2015_TWO_SHOT_SELECTIONS,
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_ONE_SHOT_PAGE,
    DEFAULT_TEST_FEW_SHOT_PAGES,
    build_repo_roots,
    load_complex_few_shot_examples,
    load_test_pdf_few_shot_examples,
)
from .page_issues import DocumentIssueSummary, PageIssue, PageIssueSummary, validate_document_issues
from .provider_workers import GeminiFillWorker, GeminiStreamWorker, QwenStreamWorker
from .schema_contract import (
    default_gemini_autocomplete_prompt_template,
    default_gemini_fill_prompt_template,
    default_extraction_prompt_template,
)
from .schema_io import load_any_schema, payload_requires_migration, payload_uses_legacy_aliases
from .schema_registry import SchemaRegistry
from .schema_ui import enum_options
from .schemas import PageExtraction, PageMeta, PageType
from .startup import (
    StartupContext as _StartupContext,
    default_annotations_path as _default_annotations_path_impl,
    resolve_startup_context as _resolve_startup_context_impl,
)
from . import workspace as workspace_mod
from .workspace import page_has_annotation

GEMINI_GT_BBOX_LOCK_MIN_FACTS = 4
GEMINI_GENERATED_BBOX_POLICY_DEFAULT = "auto"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_ROOT = PACKAGE_ROOT / "assets"
ICONS_ROOT = ASSETS_ROOT / "icons"
GEMINI_BUTTON_ICON = ICONS_ROOT / "gemini.png"
MULTI_VALUE_PLACEHOLDER = "Multiple values"
PATH_LEVEL_INDEX_ROLE = Qt.UserRole + 17
EQUATION_TERM_INDEX_ROLE = Qt.UserRole + 18
PATH_OPERATION_INDEX_ROLE = Qt.UserRole + 19
EQUATION_VARIANT_INDEX_ROLE = Qt.UserRole + 20
EQUATION_VARIANT_SIGNATURE_ROLE = Qt.UserRole + 21
VALUE_TYPE_OPTIONS: tuple[str, ...] = enum_options("fact", "value_type")
VALUE_CONTEXT_OPTIONS: tuple[str, ...] = enum_options("fact", "value_context")
BALANCE_TYPE_OPTIONS: tuple[str, ...] = ("",)
ROW_ROLE_OPTIONS: tuple[str, ...] = enum_options("fact", "row_role")
AGGREGATION_ROLE_OPTIONS: tuple[str, ...] = ("",)
STATEMENT_TYPE_OPTIONS: tuple[str, ...] = enum_options("page_meta", "statement_type")
ENTITY_TYPE_OPTIONS: tuple[str, ...] = enum_options("metadata", "entity_type")
PERIOD_TYPE_OPTIONS: tuple[str, ...] = enum_options("fact", "period_type")
PATH_SOURCE_OPTIONS: tuple[str, ...] = enum_options("fact", "path_source")
DURATION_TYPE_OPTIONS: tuple[str, ...] = enum_options("fact", "duration_type")
RECURRING_PERIOD_OPTIONS: tuple[str, ...] = enum_options("fact", "recurring_period")
REPORT_SCOPE_OPTIONS: tuple[str, ...] = enum_options("metadata", "report_scope")
GEMINI_THINKING_LEVEL_OPTIONS: tuple[str, ...] = ("minimal", "low", "medium", "high")
GEMINI_FILL_FACT_FIELDS: tuple[str, ...] = tuple(
    SchemaRegistry.get_prompt_contract("gemini_fill")["fact_patch_fields"]
)
_GEMINI_AUTO_FIX_DEFAULT_FIELDS = {"period_type", "period_start", "period_end"}
GEMINI_AUTO_FIX_FIELD_CHOICES: tuple[tuple[str, bool], ...] = tuple(
    (field_name, field_name in _GEMINI_AUTO_FIX_DEFAULT_FIELDS)
    for field_name in GEMINI_FILL_FACT_FIELDS
)
_EQUATION_NUMERIC_VALUE_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
_SAFE_EQUATION_CHARS_RE = re.compile(r"^[\d,\.\+\-\s]+$")
_SAFE_EQUATION_TERM_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_FACT_EQUATION_RE = re.compile(r"^\s*[+-]?\s*f\d+(?:\s*[+-]\s*f\d+)*\s*$", flags=re.IGNORECASE)
_FACT_EQUATION_TOKEN_RE = re.compile(r"([+-]?)\s*f(\d+)", flags=re.IGNORECASE)


def _shared_path_prefix(path_signatures: List[tuple[str, ...]]) -> list[str]:
    prefix: list[str] = list(path_signatures[0]) if path_signatures else []
    for path in path_signatures[1:]:
        common = 0
        while common < len(prefix) and common < len(path) and prefix[common] == path[common]:
            common += 1
        prefix = prefix[:common]
    return prefix


def _shared_path_elements(path_signatures: List[tuple[str, ...]]) -> list[str]:
    if not path_signatures:
        return []
    ordered_unique: list[str] = []
    for part in path_signatures[0]:
        if part and part not in ordered_unique:
            ordered_unique.append(part)
    return [part for part in ordered_unique if all(part in path for path in path_signatures[1:])]


def _prompt_entity_apply_mode(parent: QWidget, entity_name: str) -> Optional[str]:
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("Apply Entity Name")
    msg.setText(f"Apply entity_name '{entity_name}' across this PDF.")
    msg.setInformativeText(
        "Choose whether to fill only empty pages or overwrite entity_name on every page."
    )
    missing_btn = msg.addButton("Missing Only", QMessageBox.AcceptRole)
    force_btn = msg.addButton("Force All Pages", QMessageBox.DestructiveRole)
    msg.addButton(QMessageBox.Cancel)
    msg.setDefaultButton(missing_btn)
    msg.exec_()
    clicked = msg.clickedButton()
    if clicked is missing_btn:
        return "missing_only"
    if clicked is force_btn:
        return "force_all"
    return None


def _prompt_unsaved_close_action(parent: QWidget) -> str:
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Unsaved Changes")
    msg.setText("You have unsaved annotation changes.")
    msg.setInformativeText("Save before closing the annotator?")
    save_btn = msg.addButton("Save", QMessageBox.AcceptRole)
    discard_btn = msg.addButton("Discard", QMessageBox.DestructiveRole)
    msg.addButton(QMessageBox.Cancel)
    msg.setDefaultButton(save_btn)
    msg.exec_()

    clicked = msg.clickedButton()
    if clicked is save_btn:
        return "save"
    if clicked is discard_btn:
        return "discard"
    return "cancel"


def bbox_to_dict(rect: QRectF) -> Dict[str, float]:
    return normalize_bbox_data(
        {
            "x": rect.x(),
            "y": rect.y(),
            "w": rect.width(),
            "h": rect.height(),
        }
    )


def dict_to_rect(data: Dict[str, Any]) -> QRectF:
    bbox = normalize_bbox_data(data)
    return QRectF(bbox["x"], bbox["y"], bbox["w"], bbox["h"])


def item_scene_rect(item: QGraphicsRectItem) -> QRectF:
    try:
        return item.mapRectToScene(item.rect())
    except RuntimeError:
        # Can happen if selection callbacks run while scene items are being cleared.
        return QRectF()


def _is_alive_graphics_item(item: Optional[QGraphicsRectItem], scene: Optional[QGraphicsScene] = None) -> bool:
    if item is None:
        return False
    try:
        if sip.isdeleted(item):
            return False
        if scene is not None:
            return item.scene() is scene
        return item.scene() is not None
    except RuntimeError:
        return False


def _is_save_key_event(event: QKeyEvent) -> bool:
    if event.matches(QKeySequence.Save):
        return True
    if event.key() != Qt.Key_S:
        return False
    modifiers = event.modifiers()
    return bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))


def _format_decimal_plain(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _decimal_to_number(value: Decimal) -> int | float:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return int(normalized)
    return float(normalized)


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float) and float(value).is_integer():
        parsed = int(value)
        return parsed if parsed >= 1 else None
    text = str(value or "").strip()
    if text.isdigit():
        parsed = int(text)
        return parsed if parsed >= 1 else None
    return None


def _normalize_natural_sign_for_equation(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "+": "positive",
        "plus": "positive",
        "positive": "positive",
        "-": "negative",
        "minus": "negative",
        "negative": "negative",
    }
    return aliases.get(text)


def _normalize_equation_operator(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "+": "+",
        "plus": "+",
        "additive": "+",
        "-": "-",
        "minus": "-",
        "subtractive": "-",
    }
    return aliases.get(text, "+")


def _natural_sign_multiplier(value: Any) -> int:
    return -1 if _normalize_natural_sign_for_equation(value) == "negative" else 1


def _operator_multiplier(value: Any) -> int:
    return -1 if _normalize_equation_operator(value) == "-" else 1


def _normalize_page_annotation_status(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"approved", "flagged"}:
        return text
    return None


def _parse_fact_value_for_equation(value: Any) -> tuple[Decimal | None, str | None, dict[str, Any]]:
    raw = normalize_angle_bracketed_numeric_text(value)
    if not raw:
        return None, None, {"raw_value": "", "normalized_value": None, "status": "invalid"}

    if raw == "-":
        return Decimal("0"), "0", {"raw_value": "-", "normalized_value": 0, "status": "normalized_dash"}

    negative = False
    text = raw
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.startswith("+"):
        text = text[1:].strip()
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()

    if not _EQUATION_NUMERIC_VALUE_RE.fullmatch(text):
        return None, None, {"raw_value": raw, "normalized_value": None, "status": "invalid"}

    try:
        parsed = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None, None, {"raw_value": raw, "normalized_value": None, "status": "invalid"}
    normalized = -parsed if negative else parsed
    return normalized, text, {
        "raw_value": raw,
        "normalized_value": _decimal_to_number(normalized),
        "status": "ok",
    }


def _build_equation_candidate_from_facts(
    facts: list[dict[str, Any]],
) -> tuple[str | None, str | None, str | None, list[str], list[dict[str, Any]]]:
    numeric_terms: list[str] = []
    fact_terms: list[str] = []
    total = Decimal("0")
    invalid_values: list[str] = []
    structured_terms: list[dict[str, Any]] = []
    valid_count = 0

    for fact in facts:
        fact_num = _coerce_positive_int(fact.get("fact_num"))
        raw_value = fact.get("value")
        natural_sign = _normalize_natural_sign_for_equation(fact.get("natural_sign"))
        operator = _normalize_equation_operator(fact.get("operator"))
        parsed, display, term_meta = _parse_fact_value_for_equation(raw_value)
        magnitude = parsed.copy_abs() if parsed is not None else None
        contribution_sign = _natural_sign_multiplier(natural_sign) * _operator_multiplier(operator)
        effective_value = magnitude * Decimal(contribution_sign) if magnitude is not None else None
        term_meta = {
            **term_meta,
            "fact_num": fact_num,
            "fact_reference": f"f{fact_num}" if fact_num is not None else None,
            "natural_sign": natural_sign,
            "operator": operator,
            "contribution_sign": contribution_sign,
            "effective_normalized_value": _decimal_to_number(effective_value) if effective_value is not None else None,
            "equation_child": {"fact_num": fact_num, "operator": operator} if fact_num is not None else None,
        }
        if parsed is None or display is None or fact_num is None:
            rendered = str(raw_value or "").strip() or "<empty>"
            invalid_values.append(rendered)
            structured_terms.append(term_meta)
            continue

        status = str(term_meta.get("status") or "")
        force_operator_sign = status == "normalized_dash"
        prefix = ""
        if valid_count > 0:
            if force_operator_sign:
                prefix = "- " if contribution_sign < 0 else "+ "
            else:
                prefix = "- " if effective_value < 0 else "+ "
        elif force_operator_sign:
            prefix = "- " if contribution_sign < 0 else ""
        elif effective_value < 0:
            prefix = "- "

        numeric_terms.append(f"{prefix}{display}" if prefix else display)
        fact_prefix = ""
        if valid_count > 0:
            fact_prefix = "- " if operator == "-" else "+ "
        elif operator == "-":
            fact_prefix = "- "
        fact_terms.append(f"{fact_prefix}f{fact_num}" if fact_prefix else f"f{fact_num}")
        total += effective_value
        structured_terms.append(term_meta)
        valid_count += 1

    if valid_count == 0:
        return None, None, None, invalid_values, structured_terms
    return (
        " ".join(numeric_terms),
        _format_decimal_plain(total),
        " ".join(fact_terms),
        invalid_values,
        structured_terms,
    )


def _fact_equation_operator_map(value: Any) -> dict[int, str]:
    text = str(value or "").strip()
    if not text or not _FACT_EQUATION_RE.fullmatch(text):
        return {}
    out: dict[int, str] = {}
    for sign_text, fact_num_text in _FACT_EQUATION_TOKEN_RE.findall(text):
        out[int(fact_num_text)] = "-" if sign_text.strip() == "-" else "+"
    return out


def _effective_target_value_for_equation(
    target_value: Any,
    target_natural_sign: Any,
) -> tuple[Decimal | None, str | None]:
    target_value_decimal, _target_display, _meta = _parse_fact_value_for_equation(target_value)
    if target_value_decimal is None:
        return None, None
    target_value_decimal = target_value_decimal.copy_abs() * Decimal(_natural_sign_multiplier(target_natural_sign))
    return target_value_decimal, _format_decimal_plain(target_value_decimal)


def _equation_result_match_state(
    result_text: str | None,
    target_value: Any,
    target_natural_sign: Any = None,
) -> tuple[str, str]:
    if result_text is None:
        return "danger", "Cannot calculate preview."

    result_value, _display, _meta = _parse_fact_value_for_equation(result_text)
    if result_value is None:
        return "danger", "Cannot calculate preview."

    target_value_decimal, target_display = _effective_target_value_for_equation(
        target_value,
        target_natural_sign,
    )
    if target_value_decimal is None or target_display is None:
        return "neutral", "Preview only; target value is non-numeric."
    if result_value == target_value_decimal:
        return "ok", "Matches target value."
    return "danger", f"Does not match target value ({target_display})."


def _evaluate_equation_string(equation: Any) -> str | None:
    text = str(equation or "").strip()
    if "=" in text:
        text = text.split("=", 1)[0].strip()
    if not text or not _SAFE_EQUATION_CHARS_RE.fullmatch(text):
        return None
    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return None
    terms = _SAFE_EQUATION_TERM_RE.findall(normalized)
    if not terms or "".join(terms) != normalized:
        return None

    total = Decimal("0")
    for term in terms:
        try:
            total += Decimal(term.replace(",", ""))
        except InvalidOperation:
            return None
    return _format_decimal_plain(total)


def _normalize_equation_bundle_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return None
        return {"equation": text, "fact_equation": None}
    if not isinstance(value, dict):
        return None
    equation = str(value.get("equation") or "").strip()
    if not equation:
        return None
    fact_equation = str(value.get("fact_equation") or "").strip() or None
    return {
        "equation": equation,
        "fact_equation": fact_equation,
    }


def _equation_bundle_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    equation = str(entry.get("equation") or "")
    fact_equation = str(entry.get("fact_equation") or "")
    return equation, fact_equation


def _equation_bundles_from_fact_payload(fact_payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_fact = normalize_fact_data(fact_payload)
    bundles: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    raw_equations = normalized_fact.get("equations")
    if isinstance(raw_equations, list):
        for raw_entry in raw_equations:
            normalized_entry = _normalize_equation_bundle_payload(raw_entry)
            if normalized_entry is None:
                continue
            signature = _equation_bundle_signature(normalized_entry)
            if signature in seen:
                continue
            seen.add(signature)
            bundles.append(normalized_entry)
    return bundles


def _fact_payload_with_active_equation_bundle(
    fact_payload: dict[str, Any],
    equation_bundles: list[dict[str, Any]],
    *,
    active_index: int = 0,
) -> dict[str, Any]:
    base_fact = normalize_fact_data(fact_payload)
    normalized_bundles: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in equation_bundles:
        normalized_entry = _normalize_equation_bundle_payload(entry)
        if normalized_entry is None:
            continue
        signature = _equation_bundle_signature(normalized_entry)
        if signature in seen:
            continue
        seen.add(signature)
        normalized_bundles.append(normalized_entry)

    if not normalized_bundles:
        return normalize_fact_data(
            {
                **base_fact,
                "equations": None,
            }
        )

    clamped_index = max(0, min(active_index, len(normalized_bundles) - 1))
    active_bundle = normalized_bundles[clamped_index]
    ordered_bundles = [
        active_bundle,
        *[entry for idx, entry in enumerate(normalized_bundles) if idx != clamped_index],
    ]
    return normalize_fact_data(
        {
            **base_fact,
            "equation": active_bundle.get("equation"),
            "fact_equation": active_bundle.get("fact_equation"),
            "equations": ordered_bundles,
        }
    )


def _active_equation_bundle_from_fact_payload(fact_payload: dict[str, Any]) -> dict[str, Any] | None:
    bundles = _equation_bundles_from_fact_payload(fact_payload)
    if not bundles:
        return None
    return bundles[0]


class AnnotationView(QGraphicsView):
    zoom_requested = pyqtSignal(float)
    nudge_selected_requested = pyqtSignal(int, int)
    resize_selected_requested = pyqtSignal(str, int)
    select_all_requested = pyqtSignal()
    previous_page_requested = pyqtSignal()
    next_page_requested = pyqtSignal()
    calculate_drag_active_changed = pyqtSignal(bool)
    equation_approval_requested = pyqtSignal()
    _PAN_STEP = 60

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._is_view_panning = False
        self._pan_last = QPoint()
        self._pan_button: Optional[int] = None
        self._lens_enabled = False
        self._lens_zoom = 2.8
        self._lens_radius = 74
        self._lens_view_pos: Optional[QPoint] = None
        self._calculate_drag_active = False
        # Grouped bbox drags generate many small repaints; bounding-rect updates are cheaper here.
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_lens_enabled(self, enabled: bool) -> None:
        self._lens_enabled = bool(enabled)
        if not self._lens_enabled:
            self._lens_view_pos = None
        self.viewport().update()

    def disable_calculate_drag_mode(self) -> None:
        if not self._calculate_drag_active:
            return
        self._calculate_drag_active = False
        self.calculate_drag_active_changed.emit(False)

    def _update_lens_pos(self, pos: QPoint) -> None:
        if not self._lens_enabled:
            return
        if self.viewport().rect().contains(pos):
            self._lens_view_pos = QPoint(pos)
        else:
            self._lens_view_pos = None
        self.viewport().update()

    def pan_by_view_pixels(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        center_view = self.viewport().rect().center()
        target_scene = self.mapToScene(center_view + QPoint(dx, dy))
        self.centerOn(target_scene)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                factor = 1.2 if delta > 0 else 1 / 1.2
                self.zoom_requested.emit(factor)
                event.accept()
                return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.setFocus()
            self._is_view_panning = True
            self._pan_button = int(event.button())
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self._lens_view_pos = None
            self.viewport().update()
            event.accept()
            return
        self.setFocus()
        self._update_lens_pos(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._update_lens_pos(event.pos())
        if self._is_view_panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.pan_by_view_pixels(-delta.x(), -delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._is_view_panning and self._pan_button == int(event.button()):
            self._is_view_panning = False
            self._pan_button = None
            self.setCursor(Qt.ArrowCursor)
            self._update_lens_pos(event.pos())
            event.accept()
            return
        self._update_lens_pos(event.pos())
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        modifiers = int(event.modifiers())
        zoom_modifiers = {
            int(Qt.NoModifier),
            int(Qt.ShiftModifier),
            int(Qt.KeypadModifier),
            int(Qt.ShiftModifier | Qt.KeypadModifier),
        }
        if key == Qt.Key_Alt and not event.isAutoRepeat():
            if not self._calculate_drag_active:
                self._calculate_drag_active = True
                self.calculate_drag_active_changed.emit(True)
            event.accept()
            return
        if key == Qt.Key_Shift and self._calculate_drag_active and not event.isAutoRepeat():
            self.equation_approval_requested.emit()
            event.accept()
            return
        if key == Qt.Key_A and event.modifiers() in (Qt.ControlModifier, Qt.MetaModifier):
            self.select_all_requested.emit()
            event.accept()
            return
        if event.modifiers() == Qt.NoModifier:
            if key == Qt.Key_A:
                self.previous_page_requested.emit()
                event.accept()
                return
            if key == Qt.Key_D:
                self.next_page_requested.emit()
                event.accept()
                return
        if key == Qt.Key_Plus and modifiers in zoom_modifiers:
            self.zoom_requested.emit(1.2)
            event.accept()
            return
        if key == Qt.Key_Minus and modifiers in {int(Qt.NoModifier), int(Qt.KeypadModifier)}:
            self.zoom_requested.emit(1 / 1.2)
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if event.modifiers() & Qt.AltModifier:
                direction = "left"
                if key == Qt.Key_Right:
                    direction = "right"
                elif key == Qt.Key_Up:
                    direction = "up"
                elif key == Qt.Key_Down:
                    direction = "down"
                multiplier = 10 if event.modifiers() & Qt.ShiftModifier else 1
                self.resize_selected_requested.emit(direction, multiplier)
                event.accept()
                return
            if event.modifiers() & Qt.ControlModifier:
                step = self._PAN_STEP * (3 if event.modifiers() & Qt.ShiftModifier else 1)
                if key == Qt.Key_Left:
                    self.pan_by_view_pixels(-step, 0)
                elif key == Qt.Key_Right:
                    self.pan_by_view_pixels(step, 0)
                elif key == Qt.Key_Up:
                    self.pan_by_view_pixels(0, -step)
                elif key == Qt.Key_Down:
                    self.pan_by_view_pixels(0, step)
                event.accept()
                return
            step = 10 if event.modifiers() & Qt.ShiftModifier else 1
            dx = 0
            dy = 0
            if key == Qt.Key_Left:
                dx = -step
            elif key == Qt.Key_Right:
                dx = step
            elif key == Qt.Key_Up:
                dy = -step
            elif key == Qt.Key_Down:
                dy = step
            self.nudge_selected_requested.emit(dx, dy)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Alt and not event.isAutoRepeat():
            if self._calculate_drag_active:
                self._calculate_drag_active = False
                self.calculate_drag_active_changed.emit(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._lens_view_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        self._paint_zoom_lens()

    def _paint_zoom_lens(self) -> None:
        if not self._lens_enabled or self._lens_view_pos is None:
            return
        scene = self.scene()
        if scene is None:
            return
        view_rect = self.viewport().rect()
        if not view_rect.contains(self._lens_view_pos):
            return

        radius = int(self._lens_radius)
        lens_rect = QRect(
            self._lens_view_pos.x() - radius,
            self._lens_view_pos.y() - radius,
            radius * 2,
            radius * 2,
        )
        if lens_rect.left() < view_rect.left():
            lens_rect.moveLeft(view_rect.left())
        if lens_rect.top() < view_rect.top():
            lens_rect.moveTop(view_rect.top())
        if lens_rect.right() > view_rect.right():
            lens_rect.moveRight(view_rect.right())
        if lens_rect.bottom() > view_rect.bottom():
            lens_rect.moveBottom(view_rect.bottom())

        source_half = max(6, int(radius / max(1.2, self._lens_zoom)))
        source_view_rect = QRect(
            self._lens_view_pos.x() - source_half,
            self._lens_view_pos.y() - source_half,
            source_half * 2,
            source_half * 2,
        ).intersected(view_rect)
        if source_view_rect.width() < 2 or source_view_rect.height() < 2:
            return

        source_scene_rect = self.mapToScene(source_view_rect).boundingRect()
        sampled = QPixmap(source_view_rect.size())
        sampled.fill(Qt.transparent)
        sample_painter = QPainter(sampled)
        sample_painter.setRenderHint(QPainter.Antialiasing, True)
        sample_painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        scene.render(
            sample_painter,
            QRectF(0, 0, sampled.width(), sampled.height()),
            source_scene_rect,
            Qt.IgnoreAspectRatio,
        )
        sample_painter.end()

        magnified = sampled.scaled(lens_rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, True)
        clip_path = QPainterPath()
        clip_path.addEllipse(QRectF(lens_rect))
        painter.setClipPath(clip_path)
        painter.drawPixmap(lens_rect, magnified)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(34, 53, 88), 2))
        painter.drawEllipse(QRectF(lens_rect).adjusted(1, 1, -1, -1))
        center = lens_rect.center()
        painter.setPen(QPen(QColor(34, 53, 88, 180), 1))
        painter.drawLine(center.x() - 8, center.y(), center.x() + 8, center.y())
        painter.drawLine(center.x(), center.y() - 8, center.x(), center.y() + 8)
        painter.end()


class EquationVariantsListWidget(QListWidget):
    delete_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key_Delete, Qt.Key_Backspace}:
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AnnotRectItem(QGraphicsRectItem):
    _RESIZE_MARGIN = 8.0
    _MIN_SIDE = 4.0

    _H_TOP = "top"
    _H_BOTTOM = "bottom"
    _H_LEFT = "left"
    _H_RIGHT = "right"
    _H_TOP_LEFT = "top_left"
    _H_TOP_RIGHT = "top_right"
    _H_BOTTOM_LEFT = "bottom_left"
    _H_BOTTOM_RIGHT = "bottom_right"

    def __init__(self, rect: QRectF, fact_data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(rect)
        self.fact_data: Dict[str, Any] = normalize_fact_data(fact_data)
        self.setBrush(Qt.transparent)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)
        self._active_handle: Optional[str] = None
        self._resize_start_rect: Optional[QRectF] = None
        self._resize_start_scene_pos: Optional[QPointF] = None
        self._press_scene_rect: Optional[QRectF] = None
        self._order_label: Optional[int] = None
        self._show_order_label = True
        self._equation_reference_preview = False
        self._equation_match_ok = False

    def set_equation_reference_preview(self, enabled: bool) -> None:
        next_value = bool(enabled)
        if self._equation_reference_preview == next_value:
            return
        self._equation_reference_preview = next_value
        self.update()

    def set_equation_match_ok(self, enabled: bool) -> None:
        next_value = bool(enabled)
        if self._equation_match_ok == next_value:
            return
        self._equation_match_ok = next_value
        self.update()

    def set_order_label(self, order: Optional[int], *, visible: bool = True) -> None:
        next_order = int(order) if order is not None else None
        next_visible = bool(visible)
        if next_order == self._order_label and next_visible == self._show_order_label:
            return
        self._order_label = next_order
        self._show_order_label = next_visible
        self.update()

    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        pen_color = QColor("#d92d20")
        pen_width = 1
        pen_style = Qt.SolidLine
        if self.isSelected():
            pen_color = QColor("#175cd3")
            pen_width = 2
            pen_style = Qt.DashLine
        elif self._equation_reference_preview:
            pen_color = QColor("#14804a")
            pen_width = 2
        elif self._equation_match_ok:
            pen_color = QColor("#14804a")
            pen_width = 2
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        pen.setStyle(pen_style)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.transparent)
        painter.drawRect(self.rect())
        painter.restore()
        if not self._show_order_label or self._order_label is None:
            return

        text = str(self._order_label)
        painter.save()
        font = QFont(painter.font())
        font.setPointSize(max(7, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)

        fm = painter.fontMetrics()
        pad_x = 4
        pad_y = 2
        badge_w = fm.horizontalAdvance(text) + (pad_x * 2)
        badge_h = fm.height() + (pad_y * 2)

        rect = self.rect()
        badge_rect = QRectF(
            rect.left() + 1.0,
            rect.top() + 1.0,
            float(badge_w),
            float(badge_h),
        )
        painter.setPen(QPen(QColor(34, 53, 88, 220), 1))
        painter.setBrush(QColor(255, 255, 255, 220))
        painter.drawRoundedRect(badge_rect, 3.0, 3.0)
        painter.setPen(QPen(QColor(34, 53, 88), 1))
        painter.drawText(badge_rect, Qt.AlignCenter, text)
        painter.restore()

    def _geometry_changed_since_press(self) -> bool:
        if self._press_scene_rect is None:
            return False
        curr = item_scene_rect(self)
        prev = self._press_scene_rect
        eps = 0.01
        return (
            abs(curr.x() - prev.x()) > eps
            or abs(curr.y() - prev.y()) > eps
            or abs(curr.width() - prev.width()) > eps
            or abs(curr.height() - prev.height()) > eps
        )

    def _handle_at(self, pos: QPointF) -> Optional[str]:
        rect = self.rect()
        margin = self._RESIZE_MARGIN
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        near_left = abs(pos.x() - left) <= margin
        near_right = abs(pos.x() - right) <= margin
        near_top = abs(pos.y() - top) <= margin
        near_bottom = abs(pos.y() - bottom) <= margin

        inside_x = (left + margin) < pos.x() < (right - margin)
        inside_y = (top + margin) < pos.y() < (bottom - margin)

        if near_left and near_top:
            return self._H_TOP_LEFT
        if near_right and near_top:
            return self._H_TOP_RIGHT
        if near_left and near_bottom:
            return self._H_BOTTOM_LEFT
        if near_right and near_bottom:
            return self._H_BOTTOM_RIGHT
        if near_left and inside_y:
            return self._H_LEFT
        if near_right and inside_y:
            return self._H_RIGHT
        if near_top and inside_x:
            return self._H_TOP
        if near_bottom and inside_x:
            return self._H_BOTTOM
        return None

    def _cursor_for_handle(self, handle: Optional[str]):
        if handle in (self._H_TOP_LEFT, self._H_BOTTOM_RIGHT):
            return Qt.SizeFDiagCursor
        if handle in (self._H_TOP_RIGHT, self._H_BOTTOM_LEFT):
            return Qt.SizeBDiagCursor
        if handle in (self._H_LEFT, self._H_RIGHT):
            return Qt.SizeHorCursor
        if handle in (self._H_TOP, self._H_BOTTOM):
            return Qt.SizeVerCursor
        return Qt.OpenHandCursor

    def _clamp_and_apply_resize(self, scene_rect: QRectF, handle: str) -> None:
        moving_left = handle in (self._H_LEFT, self._H_TOP_LEFT, self._H_BOTTOM_LEFT)
        moving_right = handle in (self._H_RIGHT, self._H_TOP_RIGHT, self._H_BOTTOM_RIGHT)
        moving_top = handle in (self._H_TOP, self._H_TOP_LEFT, self._H_TOP_RIGHT)
        moving_bottom = handle in (self._H_BOTTOM, self._H_BOTTOM_LEFT, self._H_BOTTOM_RIGHT)

        min_side = self._MIN_SIDE
        if scene_rect.width() < min_side:
            if moving_left and not moving_right:
                scene_rect.setLeft(scene_rect.right() - min_side)
            elif moving_right and not moving_left:
                scene_rect.setRight(scene_rect.left() + min_side)
        if scene_rect.height() < min_side:
            if moving_top and not moving_bottom:
                scene_rect.setTop(scene_rect.bottom() - min_side)
            elif moving_bottom and not moving_top:
                scene_rect.setBottom(scene_rect.top() + min_side)

        scene = self.scene()
        if isinstance(scene, AnnotationScene):
            bounds = scene.image_rect
            if moving_left and scene_rect.left() < bounds.left():
                scene_rect.setLeft(bounds.left())
            if moving_right and scene_rect.right() > bounds.right():
                scene_rect.setRight(bounds.right())
            if moving_top and scene_rect.top() < bounds.top():
                scene_rect.setTop(bounds.top())
            if moving_bottom and scene_rect.bottom() > bounds.bottom():
                scene_rect.setBottom(bounds.bottom())

            if scene_rect.width() < min_side:
                if moving_left:
                    scene_rect.setLeft(scene_rect.right() - min_side)
                elif moving_right:
                    scene_rect.setRight(scene_rect.left() + min_side)
            if scene_rect.height() < min_side:
                if moving_top:
                    scene_rect.setTop(scene_rect.bottom() - min_side)
                elif moving_bottom:
                    scene_rect.setBottom(scene_rect.top() + min_side)

        pos = self.pos()
        self.setRect(
            QRectF(
                scene_rect.left() - pos.x(),
                scene_rect.top() - pos.y(),
                scene_rect.width(),
                scene_rect.height(),
            )
        )

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange and self.scene() is not None:
            scene = self.scene()
            if isinstance(scene, AnnotationScene):
                new_pos = value if isinstance(value, QPointF) else QPointF(value)
                rect = self.rect()
                moved = QRectF(
                    rect.left() + new_pos.x(),
                    rect.top() + new_pos.y(),
                    rect.width(),
                    rect.height(),
                )
                bounds = scene.image_rect

                dx = 0.0
                dy = 0.0
                if moved.left() < bounds.left():
                    dx = bounds.left() - moved.left()
                elif moved.right() > bounds.right():
                    dx = bounds.right() - moved.right()

                if moved.top() < bounds.top():
                    dy = bounds.top() - moved.top()
                elif moved.bottom() > bounds.bottom():
                    dy = bounds.bottom() - moved.bottom()

                if dx or dy:
                    return QPointF(new_pos.x() + dx, new_pos.y() + dy)
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event) -> None:
        if self._active_handle is None:
            self.setCursor(self._cursor_for_handle(self._handle_at(event.pos())))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._active_handle is None:
            self.setCursor(Qt.OpenHandCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self._press_scene_rect = item_scene_rect(self)
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self.setSelected(not self.isSelected())
                event.accept()
                return
            handle = self._handle_at(event.pos())
            if handle is not None:
                scene = self.scene()
                self._active_handle = handle
                self._resize_start_rect = item_scene_rect(self)
                self._resize_start_scene_pos = event.scenePos()
                self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
                self.setCursor(self._cursor_for_handle(handle))
                if isinstance(scene, AnnotationScene):
                    scene._begin_group_resize(self, handle)
                event.accept()
                return
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle and self._resize_start_rect is not None and self._resize_start_scene_pos is not None:
            delta = event.scenePos() - self._resize_start_scene_pos
            rect = QRectF(self._resize_start_rect)
            handle = self._active_handle

            if handle in (self._H_LEFT, self._H_TOP_LEFT, self._H_BOTTOM_LEFT):
                rect.setLeft(rect.left() + delta.x())
            if handle in (self._H_RIGHT, self._H_TOP_RIGHT, self._H_BOTTOM_RIGHT):
                rect.setRight(rect.right() + delta.x())
            if handle in (self._H_TOP, self._H_TOP_LEFT, self._H_TOP_RIGHT):
                rect.setTop(rect.top() + delta.y())
            if handle in (self._H_BOTTOM, self._H_BOTTOM_LEFT, self._H_BOTTOM_RIGHT):
                rect.setBottom(rect.bottom() + delta.y())

            self._clamp_and_apply_resize(rect, handle)
            scene = self.scene()
            if isinstance(scene, AnnotationScene):
                scene._update_group_resize_from_anchor(self)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._active_handle is not None:
            scene = self.scene()
            self._active_handle = None
            self._resize_start_rect = None
            self._resize_start_scene_pos = None
            self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
            self.setCursor(Qt.OpenHandCursor)
            if isinstance(scene, AnnotationScene):
                scene._end_group_resize(self)
                if self._geometry_changed_since_press():
                    scene.box_moved.emit(self)
            self._press_scene_rect = None
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self.setCursor(Qt.OpenHandCursor)
        scene = self.scene()
        if isinstance(scene, AnnotationScene) and self._geometry_changed_since_press():
            scene.box_moved.emit(self)
        self._press_scene_rect = None


class AnnotationScene(QGraphicsScene):
    box_created = pyqtSignal(object)
    box_duplicated = pyqtSignal(object)
    box_double_clicked = pyqtSignal(object)
    box_moved = pyqtSignal(object)
    equation_reference_selection_started = pyqtSignal()
    equation_reference_selection_changed = pyqtSignal(object)
    equation_reference_approval_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_rect = QRectF()
        self._drawing = False
        self._draw_start = None
        self._temp_rect_item: Optional[QGraphicsRectItem] = None
        self._selecting = False
        self._select_start = None
        self._temp_select_item: Optional[QGraphicsRectItem] = None
        self._calculate_drag_active = False
        self._equation_session_active = False
        self._equation_reference_items: list[AnnotRectItem] = []
        self._equation_drag_reference_items: list[AnnotRectItem] = []
        self._equation_drag_reference_item_ids: set[int] = set()
        self._equation_selecting = False
        self._equation_click_candidate: Optional[AnnotRectItem] = None
        self._equation_interaction_start: Optional[QPointF] = None
        self._equation_select_start = None
        self._temp_equation_select_item: Optional[QGraphicsRectItem] = None
        self._pending_toggle_selection: Optional[tuple[AnnotRectItem, ...]] = None
        self._select_append_mode = False
        self._group_resize_anchor: Optional[AnnotRectItem] = None
        self._group_resize_handle: Optional[str] = None
        self._group_resize_anchor_start_rect: Optional[QRectF] = None
        self._group_resize_other_start_rects: list[tuple[AnnotRectItem, QRectF]] = []

    def set_image_rect(self, rect: QRectF) -> None:
        self.image_rect = rect

    def reset_equation_reference_session(self) -> None:
        if self._temp_equation_select_item is not None:
            self.removeItem(self._temp_equation_select_item)
            self._temp_equation_select_item = None
        self._equation_session_active = False
        self._equation_reference_items = []
        self._equation_drag_reference_items = []
        self._equation_drag_reference_item_ids = set()
        self._equation_selecting = False
        self._equation_click_candidate = None
        self._equation_interaction_start = None
        self._equation_select_start = None

    def set_calculate_drag_active(self, active: bool) -> None:
        self._calculate_drag_active = bool(active)
        if self._calculate_drag_active:
            return
        self.reset_equation_reference_session()

    def _selected_annot_items(self) -> list[AnnotRectItem]:
        return [item for item in self.selectedItems() if isinstance(item, AnnotRectItem)]

    def _reset_group_resize_state(self) -> None:
        self._group_resize_anchor = None
        self._group_resize_handle = None
        self._group_resize_anchor_start_rect = None
        self._group_resize_other_start_rects = []

    def _begin_group_resize(self, anchor_item: AnnotRectItem, handle: str) -> None:
        selected_items = [
            item
            for item in self._selected_annot_items()
            if item is not anchor_item and item.scene() is self
        ]
        if not selected_items:
            self._reset_group_resize_state()
            return
        self._group_resize_anchor = anchor_item
        self._group_resize_handle = handle
        self._group_resize_anchor_start_rect = item_scene_rect(anchor_item)
        self._group_resize_other_start_rects = [
            (item, item_scene_rect(item))
            for item in selected_items
        ]

    def _update_group_resize_from_anchor(self, anchor_item: AnnotRectItem) -> None:
        if self._group_resize_anchor is not anchor_item:
            return
        if self._group_resize_handle is None or self._group_resize_anchor_start_rect is None:
            return

        handle = self._group_resize_handle
        anchor_start = self._group_resize_anchor_start_rect
        anchor_current = item_scene_rect(anchor_item)

        moving_left = handle in (
            AnnotRectItem._H_LEFT,
            AnnotRectItem._H_TOP_LEFT,
            AnnotRectItem._H_BOTTOM_LEFT,
        )
        moving_right = handle in (
            AnnotRectItem._H_RIGHT,
            AnnotRectItem._H_TOP_RIGHT,
            AnnotRectItem._H_BOTTOM_RIGHT,
        )
        moving_top = handle in (
            AnnotRectItem._H_TOP,
            AnnotRectItem._H_TOP_LEFT,
            AnnotRectItem._H_TOP_RIGHT,
        )
        moving_bottom = handle in (
            AnnotRectItem._H_BOTTOM,
            AnnotRectItem._H_BOTTOM_LEFT,
            AnnotRectItem._H_BOTTOM_RIGHT,
        )

        delta_left = anchor_current.left() - anchor_start.left() if moving_left else 0.0
        delta_right = anchor_current.right() - anchor_start.right() if moving_right else 0.0
        delta_top = anchor_current.top() - anchor_start.top() if moving_top else 0.0
        delta_bottom = anchor_current.bottom() - anchor_start.bottom() if moving_bottom else 0.0

        for item, start_rect in self._group_resize_other_start_rects:
            if item.scene() is not self:
                continue
            updated = QRectF(start_rect)
            if moving_left:
                updated.setLeft(start_rect.left() + delta_left)
            if moving_right:
                updated.setRight(start_rect.right() + delta_right)
            if moving_top:
                updated.setTop(start_rect.top() + delta_top)
            if moving_bottom:
                updated.setBottom(start_rect.bottom() + delta_bottom)
            item._clamp_and_apply_resize(updated, handle)

    def _end_group_resize(self, anchor_item: AnnotRectItem) -> None:
        if self._group_resize_anchor is not anchor_item:
            return
        self._reset_group_resize_state()

    def _begin_equation_selection_session(self) -> None:
        if self._equation_session_active:
            return
        self._equation_session_active = True
        self._equation_reference_items = []
        self._equation_drag_reference_items = []
        self._equation_drag_reference_item_ids = set()
        self.equation_reference_selection_started.emit()
        self.equation_reference_selection_changed.emit([])

    def _equation_order_items_by_drag_vector(
        self,
        items: list[AnnotRectItem],
        *,
        current_pos: QPointF | None,
    ) -> list[AnnotRectItem]:
        deduped = self._dedupe_equation_items(items)
        start = self._equation_select_start or self._equation_interaction_start
        if start is None or current_pos is None:
            return sorted(
                deduped,
                key=lambda item: (
                    round(item_scene_rect(item).center().y(), 2),
                    round(item_scene_rect(item).center().x(), 2),
                    id(item),
                ),
            )

        dx = float(current_pos.x() - start.x())
        dy = float(current_pos.y() - start.y())
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return sorted(
                deduped,
                key=lambda item: (
                    round(item_scene_rect(item).center().y(), 2),
                    round(item_scene_rect(item).center().x(), 2),
                    id(item),
                ),
            )

        ux = dx / norm
        uy = dy / norm

        def _projection_key(item: AnnotRectItem) -> tuple[float, float, float, int]:
            center = item_scene_rect(item).center()
            rel_x = float(center.x() - start.x())
            rel_y = float(center.y() - start.y())
            along = rel_x * ux + rel_y * uy
            lateral = rel_x * (-uy) + rel_y * ux
            return (round(along, 4), round(abs(lateral), 4), round(lateral, 4), id(item))

        return sorted(deduped, key=_projection_key)

    def _dedupe_equation_items(self, items: list[AnnotRectItem]) -> list[AnnotRectItem]:
        unique_items: list[AnnotRectItem] = []
        seen_ids: set[int] = set()
        for item in items:
            item_id = id(item)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            unique_items.append(item)
        return unique_items

    def _set_equation_reference_items(self, items: list[AnnotRectItem]) -> None:
        self._equation_reference_items = self._dedupe_equation_items(items)
        self.equation_reference_selection_changed.emit(list(self._equation_reference_items))

    def _toggle_equation_reference_item(self, item: AnnotRectItem) -> None:
        current_ids = {id(current) for current in self._equation_reference_items}
        if id(item) in current_ids:
            self._set_equation_reference_items(
                [current for current in self._equation_reference_items if current is not item]
            )
            return
        self._set_equation_reference_items([*self._equation_reference_items, item])

    def _equation_drag_exceeds_threshold(self, pos: QPointF) -> bool:
        if self._equation_interaction_start is None:
            return False
        delta = pos - self._equation_interaction_start
        return abs(delta.x()) >= 4 or abs(delta.y()) >= 4

    def _equation_selection_items(self, rect: QRectF) -> list[AnnotRectItem]:
        if rect.width() < 4 or rect.height() < 4:
            return []
        return [
            item
            for item in self.items(rect, Qt.IntersectsItemShape)
            if isinstance(item, AnnotRectItem)
        ]

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.image_rect.contains(event.scenePos()):
            modifiers = event.modifiers()
            if self._calculate_drag_active:
                self._begin_equation_selection_session()
                self._equation_drag_reference_items = []
                self._equation_drag_reference_item_ids = set()
                item = self.itemAt(event.scenePos(), QTransform())
                self._equation_selecting = False
                self._equation_interaction_start = event.scenePos()
                self._equation_select_start = event.scenePos()
                self._equation_click_candidate = item if isinstance(item, AnnotRectItem) else None
                event.accept()
                return
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, AnnotRectItem):
                next_selection = list(self._selected_annot_items())
                if modifiers & Qt.ShiftModifier:
                    if item not in next_selection:
                        next_selection.append(item)
                        self._pending_toggle_selection = tuple(next_selection)
                    event.accept()
                    return
            if modifiers in (Qt.NoModifier, Qt.ShiftModifier) and (item is None or isinstance(item, QGraphicsPixmapItem)):
                self._selecting = True
                self._select_append_mode = bool(modifiers & Qt.ShiftModifier)
                self._select_start = event.scenePos()
                self._temp_select_item = QGraphicsRectItem(QRectF(self._select_start, self._select_start))
                select_pen = QPen(Qt.darkGreen)
                select_pen.setWidth(1)
                select_pen.setStyle(Qt.DashLine)
                self._temp_select_item.setPen(select_pen)
                self._temp_select_item.setBrush(Qt.transparent)
                self.addItem(self._temp_select_item)
                event.accept()
                return
            duplicate_drag_mod = Qt.MetaModifier | Qt.ControlModifier
            if isinstance(item, AnnotRectItem) and (event.modifiers() & duplicate_drag_mod):
                rect = item_scene_rect(item).intersected(self.image_rect)
                if rect.width() >= 1 and rect.height() >= 1:
                    copy_item = AnnotRectItem(rect, deepcopy(item.fact_data))
                    self.addItem(copy_item)
                    self.clearSelection()
                    copy_item.setSelected(True)
                    self.box_duplicated.emit(copy_item)
                    # Keep the mouse press flow so users can Cmd/Ctrl-drag the new copy immediately.
                    super().mousePressEvent(event)
                    return
            draw_mod = Qt.MetaModifier | Qt.ControlModifier
            if (event.modifiers() & draw_mod) and (item is None or isinstance(item, QGraphicsPixmapItem)):
                self._drawing = True
                self._draw_start = event.scenePos()
                self._temp_rect_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
                temp_pen = QPen(Qt.blue)
                temp_pen.setWidth(1)
                temp_pen.setStyle(Qt.DashLine)
                self._temp_rect_item.setPen(temp_pen)
                self._temp_rect_item.setBrush(Qt.transparent)
                self.addItem(self._temp_rect_item)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not _is_alive_graphics_item(self._temp_equation_select_item, self):
            self._temp_equation_select_item = None
        if not _is_alive_graphics_item(self._temp_select_item, self):
            self._temp_select_item = None
        if not _is_alive_graphics_item(self._temp_rect_item, self):
            self._temp_rect_item = None
        if self._equation_selecting and self._temp_equation_select_item is None:
            self._equation_selecting = False
            self._equation_select_start = None
            self._equation_drag_reference_items = []
            self._equation_drag_reference_item_ids = set()
            event.accept()
            return
        if self._selecting and self._temp_select_item is None:
            self._selecting = False
            self._select_start = None
            event.accept()
            return
        if self._drawing and self._temp_rect_item is None:
            self._drawing = False
            self._draw_start = None
            event.accept()
            return
        if self._calculate_drag_active and self._equation_interaction_start is not None and self._equation_select_start is not None:
            if not self._equation_selecting and self._equation_drag_exceeds_threshold(event.scenePos()):
                self._equation_selecting = True
                self._temp_equation_select_item = QGraphicsRectItem(QRectF(self._equation_select_start, self._equation_select_start))
                select_pen = QPen(QColor("#14804a"))
                select_pen.setWidth(1)
                select_pen.setStyle(Qt.DashLine)
                self._temp_equation_select_item.setPen(select_pen)
                self._temp_equation_select_item.setBrush(Qt.transparent)
                self.addItem(self._temp_equation_select_item)
            if self._equation_selecting and self._temp_equation_select_item:
                rect = QRectF(self._equation_select_start, event.scenePos()).normalized().intersected(self.image_rect)
                self._temp_equation_select_item.setRect(rect)
                frame_items = self._equation_order_items_by_drag_vector(
                    self._equation_selection_items(rect),
                    current_pos=event.scenePos(),
                )
                for item in frame_items:
                    item_id = id(item)
                    if item_id in self._equation_drag_reference_item_ids:
                        continue
                    if any(item is existing for existing in self._equation_reference_items):
                        continue
                    self._equation_drag_reference_item_ids.add(item_id)
                    self._equation_drag_reference_items.append(item)
                self.equation_reference_selection_changed.emit(
                    self._dedupe_equation_items([*self._equation_reference_items, *self._equation_drag_reference_items])
                )
                event.accept()
                return
        if self._selecting and self._temp_select_item and self._select_start:
            rect = QRectF(self._select_start, event.scenePos()).normalized().intersected(self.image_rect)
            self._temp_select_item.setRect(rect)
            event.accept()
            return
        if self._drawing and self._temp_rect_item and self._draw_start:
            rect = QRectF(self._draw_start, event.scenePos()).normalized()
            rect = rect.intersected(self.image_rect)
            self._temp_rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if not _is_alive_graphics_item(self._temp_equation_select_item, self):
            self._temp_equation_select_item = None
        if not _is_alive_graphics_item(self._temp_select_item, self):
            self._temp_select_item = None
        if not _is_alive_graphics_item(self._temp_rect_item, self):
            self._temp_rect_item = None
        if self._equation_selecting and self._temp_equation_select_item is None:
            self._equation_selecting = False
            self._equation_select_start = None
            self._equation_interaction_start = None
            self._equation_drag_reference_items = []
            self._equation_drag_reference_item_ids = set()
            event.accept()
            return
        if self._selecting and self._temp_select_item is None:
            self._selecting = False
            self._select_start = None
            self._select_append_mode = False
            event.accept()
            return
        if self._drawing and self._temp_rect_item is None:
            self._drawing = False
            self._draw_start = None
            event.accept()
            return
        if self._equation_interaction_start is not None:
            if self._equation_selecting and self._temp_equation_select_item:
                rect = self._temp_equation_select_item.rect().normalized().intersected(self.image_rect)
                self.removeItem(self._temp_equation_select_item)
                self._temp_equation_select_item = None
                self._equation_selecting = False
                self._equation_select_start = None
                self._equation_click_candidate = None
                self._equation_interaction_start = None
                self._set_equation_reference_items([*self._equation_reference_items, *self._equation_drag_reference_items])
                self._equation_drag_reference_items = []
                self._equation_drag_reference_item_ids = set()
                if event.modifiers() & Qt.ShiftModifier:
                    self.equation_reference_approval_requested.emit()
                event.accept()
                return
            if self._equation_click_candidate is not None:
                click_item = self._equation_click_candidate
                self._equation_click_candidate = None
                self._equation_interaction_start = None
                self._equation_select_start = None
                self._equation_drag_reference_items = []
                self._equation_drag_reference_item_ids = set()
                self._toggle_equation_reference_item(click_item)
                if event.modifiers() & Qt.ShiftModifier:
                    self.equation_reference_approval_requested.emit()
                event.accept()
                return
            self._equation_interaction_start = None
            self._equation_select_start = None
            self._equation_drag_reference_items = []
            self._equation_drag_reference_item_ids = set()
            event.accept()
            return
        if self._pending_toggle_selection is not None:
            next_selection = self._pending_toggle_selection
            self._pending_toggle_selection = None
            self.clearSelection()
            for item in next_selection:
                if item.scene() is self:
                    item.setSelected(True)
            event.accept()
            return
        if self._selecting and self._temp_select_item:
            rect = self._temp_select_item.rect().normalized().intersected(self.image_rect)
            self.removeItem(self._temp_select_item)
            self._temp_select_item = None
            self._selecting = False
            self._select_start = None
            if rect.width() >= 4 and rect.height() >= 4:
                if not self._select_append_mode:
                    self.clearSelection()
                for item in self.items(rect, Qt.IntersectsItemShape):
                    if isinstance(item, AnnotRectItem):
                        item.setSelected(True)
            elif not self._select_append_mode:
                self.clearSelection()
            self._select_append_mode = False
            event.accept()
            return
        if self._drawing and self._temp_rect_item:
            rect = self._temp_rect_item.rect().normalized().intersected(self.image_rect)
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
            self._drawing = False
            self._draw_start = None
            if rect.width() >= 4 and rect.height() >= 4:
                item = AnnotRectItem(rect)
                self.addItem(item)
                self.clearSelection()
                item.setSelected(True)
                self.box_created.emit(item)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, AnnotRectItem):
            self.box_double_clicked.emit(item)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class JsonImportDialog(QDialog):
    def __init__(self, default_page_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Annotations JSON")
        self.resize(920, 640)

        root = QVBoxLayout(self)
        hint = QLabel(
            "Paste JSON to import.\n"
            "Supported shapes: "
            '{"pages":[{"image":"...","meta":{...},"facts":[...]}]} '
            'or legacy full-document {"images_dir":"...","metadata":{...},"pages":[...]} '
            'or legacy {"meta": {...}, "facts": [...]} / {"image": "...", "meta": {...}, "facts": [...]}.\n'
            f'If "image" is omitted, current page "{default_page_name}" is used.\n'
            'Use the checkbox below for Gemini-style normalized coordinates (0..1000).'
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.normalized_1000_check = QCheckBox("Import bbox as normalized 0..1000 (Gemini)")
        self.normalized_1000_check.setChecked(True)
        root.addWidget(self.normalized_1000_check)

        self.text_edit = QPlainTextEdit()
        root.addWidget(self.text_edit, 1)

        actions = QHBoxLayout()
        self.load_file_btn = QPushButton("Load From File...")
        actions.addWidget(self.load_file_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addWidget(self.button_box)

        self.load_file_btn.clicked.connect(self._load_from_file)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose JSON file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Import file error", str(exc))
            return
        self.text_edit.setPlainText(text)

    def json_text(self) -> str:
        return self.text_edit.toPlainText()

    def import_normalized_1000_enabled(self) -> bool:
        return self.normalized_1000_check.isChecked()


class PageJsonDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_text = ""
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(920, 640)

        root = QVBoxLayout(self)
        hint = QLabel("Current page representation in the annotations JSON schema.")
        hint.setWordWrap(True)
        root.addWidget(hint)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search JSON")
        self.search_prev_btn = QPushButton("Previous")
        self.search_next_btn = QPushButton("Next")
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.search_prev_btn)
        search_row.addWidget(self.search_next_btn)
        root.addLayout(search_row)

        self.text_view = QPlainTextEdit()
        self.text_view.setReadOnly(True)
        root.addWidget(self.text_view, 1)

        actions = QHBoxLayout()
        self.copy_btn = QPushButton("Copy JSON")
        self.close_btn = QPushButton("Close")
        actions.addWidget(self.copy_btn)
        actions.addStretch(1)
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.search_edit.returnPressed.connect(self.find_next)
        self.search_next_btn.clicked.connect(self.find_next)
        self.search_prev_btn.clicked.connect(self.find_previous)
        self.close_btn.clicked.connect(self.close)
        self._find_shortcut = QShortcut(QKeySequence.Find, self, activated=self.focus_search)
        self._find_shortcut.setContext(Qt.WindowShortcut)

    def set_page_json(self, *, title: str, page_text: str, fact_start_pos: Optional[int] = None) -> None:
        self.setWindowTitle(title)
        if page_text != self._current_text:
            self._current_text = page_text
            self.text_view.setPlainText(page_text)
        if fact_start_pos is not None and fact_start_pos >= 0:
            cursor = self.text_view.textCursor()
            cursor.setPosition(int(fact_start_pos))
            self.text_view.setTextCursor(cursor)
            self.text_view.centerCursor()

    def focus_search(self) -> None:
        self.search_edit.setFocus(Qt.ShortcutFocusReason)
        self.search_edit.selectAll()

    def find_next(self) -> bool:
        return self._find_text(backward=False)

    def find_previous(self) -> bool:
        return self._find_text(backward=True)

    def _find_text(self, *, backward: bool) -> bool:
        needle = self.search_edit.text().strip()
        if not needle:
            self.focus_search()
            return False
        flags = QTextDocument.FindBackward if backward else QTextDocument.FindFlags()
        if self.text_view.find(needle, flags):
            return True
        cursor = self.text_view.textCursor()
        cursor.movePosition(QTextCursor.End if backward else QTextCursor.Start)
        self.text_view.setTextCursor(cursor)
        return self.text_view.find(needle, flags)


class AnnotationWindow(QMainWindow):
    annotations_saved = pyqtSignal(object)
    annotations_save_status = pyqtSignal(object)
    document_issues_changed = pyqtSignal(object)

    def __init__(self, images_dir: Path, annotations_path: Path) -> None:
        super().__init__()
        self.setObjectName("annotatorWindow")
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.page_images: List[Path] = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        )
        self.page_states: Dict[str, PageState] = {}
        self.document_meta: Dict[str, Any] = {
            "language": None,
            "reading_direction": None,
            "company_name": None,
            "company_id": None,
            "report_year": None,
            "report_scope": None,
            "entity_type": None,
        }
        self._path_list_editable = True
        self._path_list_structure_editable = True
        self.current_index = -1
        self._fact_items: List[AnnotRectItem] = []
        self._syncing_selection = False
        self._fitting_view = False
        self._is_restoring_history = False
        self._is_loading_page = False
        self._is_loading_fact_editor = False
        self._history_limit = 200
        self._history: List[Dict[str, Any]] = []
        self._history_index = -1
        self._page_json_dialog: Optional[PageJsonDialog] = None
        self._save_shortcut_in_flight = False
        self._gemini_stream_thread: Optional[QThread] = None
        self._gemini_stream_worker: Optional[GeminiStreamWorker] = None
        self._gemini_stream_target_page: Optional[str] = None
        self._gemini_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._gemini_stream_fact_count = 0
        self._gemini_stream_cancel_requested = False
        self._gemini_stream_mode = "gt"
        self._gemini_stream_apply_meta = True
        self._gemini_stream_max_facts = 0
        self._gemini_stream_limit_reached = False
        self._gemini_autocomplete_snapshot: Optional[dict[str, Any]] = None
        self._gemini_autocomplete_buffered_facts: list[dict[str, Any]] = []
        self._gemini_autocomplete_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self._gemini_autocomplete_last_bbox_scores: dict[str, float] = {}
        self._gemini_gt_buffered_facts: list[dict[str, Any]] = []
        self._gemini_gt_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self._gemini_gt_last_bbox_scores: dict[str, float] = {}
        self._gemini_gt_live_lock_min_facts = GEMINI_GT_BBOX_LOCK_MIN_FACTS
        self._gemini_gt_live_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self._gemini_gt_live_bbox_mode_locked = False
        self._gemini_gt_live_applied = False
        self._gemini_fill_thread: Optional[QThread] = None
        self._gemini_fill_worker: Optional[GeminiFillWorker] = None
        self._gemini_fill_cancel_requested = False
        self._gemini_fill_target_page: Optional[str] = None
        self._gemini_fill_snapshot: Optional[dict[str, Any]] = None
        self._gemini_fill_selected_fact_fields: set[str] = set()
        self._gemini_fill_include_statement_type = False
        self._gemini_model_name = DEFAULT_GEMINI_MODEL
        self._gemini_temperature: Optional[float] = None
        self._gemini_enable_thinking = False
        self._gemini_thinking_level = "minimal"
        self._qwen_stream_thread: Optional[QThread] = None
        self._qwen_stream_worker: Optional[QwenStreamWorker] = None
        self._qwen_stream_target_page: Optional[str] = None
        self._qwen_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._qwen_stream_fact_count = 0
        self._qwen_stream_cancel_requested = False
        self._qwen_model_name = self._initial_qwen_model_name()
        self._qwen_enable_thinking = self._initial_qwen_enable_thinking()
        self._ai_controller = AIWorkflowController(
            self,
            thinking_levels=GEMINI_THINKING_LEVEL_OPTIONS,
            fix_field_choices=GEMINI_AUTO_FIX_FIELD_CHOICES,
        )
        self._page_issue_summaries: Dict[str, PageIssueSummary] = {}
        self._last_document_issue_signature: Optional[tuple[int, int, int, int]] = None
        self._last_saved_content: Dict[str, Any] = {"page_states": {}, "metadata": {}}
        self._pending_close_approved = False
        self._pending_auto_fit = False
        self._equation_target_item: Optional[AnnotRectItem] = None
        self._equation_candidate_text: Optional[str] = None
        self._equation_candidate_fact_text: Optional[str] = None
        self._equation_candidate_result_text: Optional[str] = None
        self._equation_candidate_invalid_values: list[str] = []
        self._equation_candidate_terms: list[dict[str, Any]] = []
        self._equation_reference_preview_items: list[AnnotRectItem] = []
        self._is_loading_equation_terms = False
        self._is_loading_equation_variants = False
        self._page_annotation_status: str | None = None
        self._thumbnail_page_indices: list[int] = []
        self._thumbnail_row_lookup: dict[int, int] = {}

        if not self.page_images:
            raise RuntimeError(f"No page images found under: {self.images_dir}")

        self.setWindowTitle("FineTree PDF Annotator")
        self._resize_to_available_screen()
        self._build_ui()
        self._load_existing_annotations()
        self._recompute_all_page_issues(emit=False)
        self.show_page(0)
        self._init_history()
        self._mark_saved_content()

    def _resize_to_available_screen(self) -> None:
        preferred_w, preferred_h = 1320, 860
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            self.resize(preferred_w, preferred_h)
            return

        available = screen.availableGeometry()
        width = min(preferred_w, max(1000, available.width() - 40))
        height = min(preferred_h, max(700, available.height() - 80))
        width = max(920, min(width, available.width()))
        height = max(620, min(height, available.height()))
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(x, y, width, height)

    def _make_badge_icon(self, label: str, bg_hex: str, fg_hex: str = "#ffffff") -> QIcon:
        pixmap = QPixmap(22, 22)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(bg_hex))
        painter.drawRoundedRect(QRectF(1, 1, 20, 20), 6, 6)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(8 if len(label) > 1 else 10)
        painter.setFont(font)
        painter.setPen(QColor(fg_hex))
        painter.drawText(QRectF(1, 1, 20, 20), Qt.AlignCenter, label)
        painter.end()
        return QIcon(pixmap)

    def _resolve_nav_icon(
        self,
        theme_names: tuple[str, ...],
        fallback: int | None = None,
        *,
        badge_label: str | None = None,
        badge_bg: str = "#5b8ff9",
        badge_fg: str = "#ffffff",
    ) -> QIcon:
        for theme_name in theme_names:
            icon = QIcon.fromTheme(theme_name)
            if not icon.isNull():
                return icon
        if fallback is not None:
            return self.style().standardIcon(fallback)
        if badge_label:
            return self._make_badge_icon(badge_label, badge_bg, badge_fg)
        return QIcon()

    def _configure_top_nav_button(
        self,
        button: QPushButton,
        text: str,
        description: str,
        icon: Optional[QIcon] = None,
    ) -> None:
        button.setText(text)
        button.setIcon(icon or QIcon())
        if icon is not None and not icon.isNull():
            button.setIconSize(QSize(14, 14))
        button.setToolTip(description)
        button.setStatusTip(description)
        button.setAccessibleName(description)
        button.setMinimumHeight(30)
        button.setMinimumWidth(max(58, len(text) * 7 + 18))
        button.setMaximumHeight(30)

    def _load_repo_icon(self, path: Path) -> QIcon:
        if not path.exists():
            return QIcon()
        return QIcon(str(path))

    def _toolbar_group(self, title: str, body_layout) -> QFrame:
        frame = QFrame()
        frame.setObjectName("toolbarGroup")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("toolbarTitle")
        layout.addWidget(title_label)
        layout.addLayout(body_layout)
        return frame

    def _toolbar_divider(self) -> QFrame:
        divider = QFrame()
        divider.setObjectName("toolbarDivider")
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Plain)
        divider.setFixedWidth(1)
        return divider

    def _configure_inspector_form(self, form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setContentsMargins(14, 14, 14, 12)

    def _inspector_label(self, text: str, *, required: bool = False) -> QLabel:
        label = QLabel(f"{text} *" if required else text)
        label.setObjectName("inspectorFieldLabelRequired" if required else "inspectorFieldLabel")
        label.setMinimumWidth(92)
        return label

    def _compact_inspector_label(self, text: str, *, required: bool = False) -> QLabel:
        label = QLabel(f"{text} *" if required else text)
        label.setObjectName(
            "inspectorFieldLabelCompactRequired" if required else "inspectorFieldLabelCompact"
        )
        return label

    def _inspector_field_block(
        self,
        text: str,
        widget: QWidget,
        *,
        required: bool = False,
    ) -> QWidget:
        block = QWidget()
        block.setObjectName("inspectorFieldBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._compact_inspector_label(text, required=required))
        layout.addWidget(widget)
        return block

    def _apply_top_nav_icons(self) -> None:
        nav_specs = (
            (self.prev_btn, "Prev", "Previous page", None),
            (self.next_btn, "Next", "Next page", None),
            (self.page_jump_btn, "Go", "Go to selected page", None),
            (self.undo_btn, "Undo", "Undo", None),
            (self.redo_btn, "Redo", "Redo", None),
            (self.import_btn, "Import", "Import annotations JSON", None),
            (self.ai_btn, "AI", "Open AI actions", self._load_repo_icon(GEMINI_BUTTON_ICON)),
            (self.delete_nav_btn, "Delete", "Delete selected bounding box", None),
            (self.zoom_out_btn, "Zoom -", "Zoom out", None),
            (self.zoom_in_btn, "Zoom +", "Zoom in", None),
            (self.lens_btn, "Lens", "Toggle zoom lens", None),
            (self.copy_image_btn, "Copy Image", "Copy current page image", None),
            (self.fit_btn, "Fit", "Fit page to view height", None),
            (self.page_json_btn, "JSON", "Show current page JSON", None),
            (self.apply_entity_all_btn, "Apply Entity", "Apply current entity across pages", None),
            (self.page_approve_continue_btn, "Approve + Next", "Mark page approved and move to next page", None),
            (self.page_flag_btn, "⚑ Flag", "Mark page for review", None),
            (self.help_btn, "Help", "Help and shortcuts", None),
            (self.save_btn, "Save", "Save annotations (Ctrl+S)", None),
            (self.exit_btn, "Exit", "Close annotator", None),
        )
        for button, text, description, icon in nav_specs:
            self._configure_top_nav_button(button, text, description, icon=icon)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        self._apply_ui_sizing()
        root = QVBoxLayout(central)
        root.setSpacing(2)
        root.setContentsMargins(6, 1, 6, 6)

        header_block = QWidget()
        header_layout = QVBoxLayout(header_block)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        toolbar_strip = QFrame()
        toolbar_strip.setObjectName("toolbarStrip")
        toolbar_strip_layout = QVBoxLayout(toolbar_strip)
        toolbar_strip_layout.setContentsMargins(8, 3, 8, 3)
        toolbar_strip_layout.setSpacing(2)

        chrome_row = QHBoxLayout()
        chrome_row.setContentsMargins(0, 0, 0, 0)
        chrome_row.setSpacing(6)
        self.annotator_title_label = QLabel(self.images_dir.name)
        self.annotator_title_label.setObjectName("annotatorTitle")
        chrome_row.addWidget(self.annotator_title_label, 1)
        self.output_path_label = QLabel(str(self.annotations_path))
        self.output_path_label.setObjectName("monoLabel")
        self.output_path_label.setToolTip(str(self.annotations_path))
        self.output_path_label.setMaximumWidth(260)
        chrome_row.addWidget(self.output_path_label, 0, Qt.AlignRight | Qt.AlignVCenter)
        toolbar_strip_layout.addLayout(chrome_row)

        nav_widget = QWidget()
        nav = QHBoxLayout(nav_widget)
        nav.setSpacing(4)
        nav.setContentsMargins(0, 0, 0, 0)
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.page_label = QLabel("-")
        self.page_label.setObjectName("monoLabel")
        self.page_jump_spin = QSpinBox()
        self.page_jump_spin.setRange(1, len(self.page_images))
        self.page_jump_spin.setKeyboardTracking(False)
        self.page_jump_spin.setValue(1)
        self.page_jump_spin.setMinimumWidth(58)
        self.page_jump_btn = QPushButton("Go")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.import_btn = QPushButton("Import JSON")
        self.ai_btn = QPushButton("AI")
        self.delete_nav_btn = QPushButton("Delete BBox")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.lens_btn = QPushButton("Lens")
        self.lens_btn.setCheckable(True)
        self.copy_image_btn = QPushButton("Copy Image")
        self.fit_btn = QPushButton("Fit")
        self.page_json_btn = QPushButton("Page JSON")
        self.apply_entity_all_btn = QPushButton("Apply Entity To Missing")
        self.page_approve_continue_btn = QPushButton("Approve + Next")
        self.page_flag_btn = QPushButton("\u2691 Flag")
        self.help_btn = QPushButton("Help")
        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.exit_btn = QPushButton("Exit")
        for btn in (
            self.prev_btn,
            self.next_btn,
            self.page_jump_btn,
            self.undo_btn,
            self.redo_btn,
            self.import_btn,
            self.ai_btn,
            self.delete_nav_btn,
            self.zoom_out_btn,
            self.zoom_in_btn,
            self.lens_btn,
            self.copy_image_btn,
            self.fit_btn,
            self.page_json_btn,
            self.apply_entity_all_btn,
            self.page_approve_continue_btn,
            self.page_flag_btn,
            self.help_btn,
            self.save_btn,
            self.exit_btn,
        ):
            btn.setObjectName("toolbarActionBtn")
        self._apply_top_nav_icons()
        self.ai_btn.setProperty("variant", "primary")
        self.save_btn.setProperty("variant", "primary")
        for button in (self.import_btn, self.page_json_btn, self.help_btn, self.apply_entity_all_btn, self.exit_btn):
            button.setProperty("variant", "ghost")
        self.page_flag_btn.setProperty("variant", "ghost")

        doc_layout = QHBoxLayout()
        doc_layout.setSpacing(6)
        doc_layout.addWidget(self.save_btn)
        doc_layout.addWidget(self.import_btn)
        doc_layout.addWidget(self.page_json_btn)
        doc_layout.addWidget(self.help_btn)
        doc_layout.addWidget(self.page_approve_continue_btn)
        doc_layout.addWidget(self.page_flag_btn)
        doc_layout.addWidget(self.exit_btn)
        nav.addWidget(self._toolbar_group("Document", doc_layout))
        nav.addWidget(self._toolbar_divider())

        navigation_layout = QHBoxLayout()
        navigation_layout.setSpacing(4)
        navigation_layout.addWidget(self.prev_btn)
        navigation_layout.addWidget(self.next_btn)
        navigation_layout.addWidget(self.page_label)
        jump_label = QLabel("Go")
        jump_label.setObjectName("subtitleLabel")
        navigation_layout.addWidget(jump_label)
        navigation_layout.addWidget(self.page_jump_spin)
        navigation_layout.addWidget(self.page_jump_btn)
        nav.addWidget(self._toolbar_group("Navigation", navigation_layout))
        nav.addWidget(self._toolbar_divider())

        gt_layout = QHBoxLayout()
        gt_layout.setSpacing(4)
        gt_layout.addWidget(self.ai_btn)
        gt_layout.addWidget(self.apply_entity_all_btn)
        nav.addWidget(self._toolbar_group("Generation", gt_layout))
        nav.addWidget(self._toolbar_divider())

        view_layout = QHBoxLayout()
        view_layout.setSpacing(4)
        view_layout.addWidget(self.fit_btn)
        view_layout.addWidget(self.copy_image_btn)
        nav.addWidget(self._toolbar_group("View", view_layout))
        nav.addWidget(self._toolbar_divider())

        edit_layout = QHBoxLayout()
        edit_layout.setSpacing(4)
        edit_layout.addWidget(self.undo_btn)
        edit_layout.addWidget(self.redo_btn)
        edit_layout.addWidget(self.delete_nav_btn)
        nav.addWidget(self._toolbar_group("Edit", edit_layout))
        nav.addStretch(1)

        nav_widget.setMinimumWidth(nav_widget.sizeHint().width())
        nav_widget.setMinimumHeight(max(48, nav_widget.sizeHint().height()))
        nav_scroll = QScrollArea()
        nav_scroll.setObjectName("toolbarScroll")
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        nav_scroll.setWidget(nav_widget)
        nav_scroll.setMinimumHeight(nav_widget.minimumHeight())
        toolbar_strip_layout.addWidget(nav_scroll)
        header_layout.addWidget(toolbar_strip)
        root.addWidget(header_block)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        self.scene = AnnotationScene(self)
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.scene.box_created.connect(self._on_box_created)
        self.scene.box_duplicated.connect(self._on_box_duplicated)
        self.scene.box_double_clicked.connect(self._on_box_double_clicked)
        self.scene.box_moved.connect(self._on_box_geometry_changed)
        self.scene.equation_reference_selection_started.connect(self._on_equation_reference_selection_started)
        self.scene.equation_reference_selection_changed.connect(self._on_equation_reference_selection_changed)
        self.scene.equation_reference_approval_requested.connect(self._on_equation_approval_requested)
        self.view = AnnotationView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.zoom_requested.connect(self._apply_zoom)
        self.view.nudge_selected_requested.connect(self._nudge_selected_facts)
        self.view.resize_selected_requested.connect(self.batch_expand_selected)
        self.view.select_all_requested.connect(self.select_all_bboxes)
        self.view.calculate_drag_active_changed.connect(self.scene.set_calculate_drag_active)
        self.view.equation_approval_requested.connect(self._on_equation_approval_requested)
        self.page_thumb_list = QListWidget()
        self.page_thumb_list.setObjectName("thumbList")
        self.page_thumb_list.setViewMode(QListView.IconMode)
        self.page_thumb_list.setFlow(QListView.TopToBottom)
        self.page_thumb_list.setMovement(QListView.Static)
        self.page_thumb_list.setResizeMode(QListView.Adjust)
        self.page_thumb_list.setWrapping(False)
        self.page_thumb_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.page_thumb_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.page_thumb_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page_thumb_list.setIconSize(QSize(82, 114))
        self.page_thumb_list.setSpacing(6)

        thumb_panel = QWidget()
        self.thumb_panel = thumb_panel
        thumb_panel.setMinimumWidth(96)
        thumb_panel.setMaximumWidth(152)
        thumb_layout = QVBoxLayout(thumb_panel)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(4)
        thumb_title = QLabel("Pages")
        thumb_title.setObjectName("hintText")
        thumb_layout.addWidget(thumb_title)
        thumb_layout.addWidget(self.page_thumb_list, 1)

        splitter.addWidget(thumb_panel)
        splitter.addWidget(self.view)

        right = QWidget()
        right.setObjectName("inspectorPanel")
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        doc_box = QGroupBox("Document Metadata")
        doc_box.setObjectName("inspectorSection")
        doc_form = QFormLayout(doc_box)
        self._configure_inspector_form(doc_form)

        page_meta_box = QGroupBox("Page Metadata")
        page_meta_box.setObjectName("inspectorSection")
        meta_form = QFormLayout(page_meta_box)
        self._configure_inspector_form(meta_form)
        self.entity_name_edit = QLineEdit()
        self.page_num_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems([p.value for p in PageType])
        self.statement_type_combo = QComboBox()
        self.statement_type_combo.addItems(list(STATEMENT_TYPE_OPTIONS))
        self.title_edit = QLineEdit()
        self.page_annotation_note_edit = QLineEdit()
        self.page_annotation_status_label = QLabel("Unclassified")
        self.page_annotation_status_label.setObjectName("statusPill")
        self.page_annotation_status_label.setProperty("tone", "accent")
        self.doc_language_combo = QComboBox()
        self.doc_language_combo.addItems(["Auto", "Hebrew (he)", "English (en)"])
        self.doc_direction_combo = QComboBox()
        self.doc_direction_combo.addItems(["Auto", "RTL", "LTR"])
        self.company_name_edit = QLineEdit()
        self.company_id_edit = QLineEdit()
        self.report_year_edit = QLineEdit()
        self.report_scope_combo = QComboBox()
        self.report_scope_combo.addItems(list(REPORT_SCOPE_OPTIONS))
        self.entity_type_combo = QComboBox()
        self.entity_type_combo.addItems(list(ENTITY_TYPE_OPTIONS))
        self.report_year_edit.setValidator(QIntValidator(0, 9999, self))
        field_min_width = 248
        self.entity_name_edit.setMinimumWidth(field_min_width)
        self.page_num_edit.setMinimumWidth(field_min_width)
        self.type_combo.setMinimumWidth(field_min_width)
        self.statement_type_combo.setMinimumWidth(field_min_width)
        self.title_edit.setMinimumWidth(field_min_width)
        self.page_annotation_note_edit.setMinimumWidth(field_min_width)
        self.doc_language_combo.setMinimumWidth(field_min_width)
        self.doc_direction_combo.setMinimumWidth(field_min_width)
        self.company_name_edit.setMinimumWidth(field_min_width)
        self.company_id_edit.setMinimumWidth(field_min_width)
        self.report_year_edit.setMinimumWidth(field_min_width)
        self.report_scope_combo.setMinimumWidth(field_min_width)
        self.entity_type_combo.setMinimumWidth(field_min_width)
        doc_form.addRow(self._inspector_label("Language"), self.doc_language_combo)
        doc_form.addRow(self._inspector_label("Direction"), self.doc_direction_combo)
        doc_form.addRow(self._inspector_label("Company Name"), self.company_name_edit)
        doc_form.addRow(self._inspector_label("Company ID"), self.company_id_edit)
        doc_form.addRow(self._inspector_label("Report Year"), self.report_year_edit)
        doc_form.addRow(self._inspector_label("Report Scope"), self.report_scope_combo)
        doc_form.addRow(self._inspector_label("Entity Type"), self.entity_type_combo)
        meta_form.addRow(self._inspector_label("Entity"), self.entity_name_edit)
        meta_form.addRow(self._inspector_label("Page"), self.page_num_edit)
        meta_form.addRow(self._inspector_label("Page Type", required=True), self.type_combo)
        meta_form.addRow(self._inspector_label("Statement Type"), self.statement_type_combo)
        meta_form.addRow(self._inspector_label("Title"), self.title_edit)
        self.page_annotation_note_edit.setPlaceholderText("Annotation-only reminder to revisit this page")
        meta_form.addRow(self._inspector_label("Revisit Note"), self.page_annotation_note_edit)
        meta_form.addRow(self._inspector_label("Page Review"), self.page_annotation_status_label)

        self.page_issues_box = QGroupBox("Page Issues")
        self.page_issues_box.setObjectName("inspectorSection")
        page_issues_layout = QVBoxLayout(self.page_issues_box)
        page_issues_layout.setContentsMargins(14, 14, 14, 12)
        page_issues_layout.setSpacing(8)
        page_issues_header = QHBoxLayout()
        page_issues_header.setSpacing(8)
        self.page_reg_flags_label = QLabel("Reg Flags: 0")
        self.page_reg_flags_label.setObjectName("statusPill")
        self.page_reg_flags_label.setProperty("tone", "danger")
        self.page_warnings_label = QLabel("Warnings: 0")
        self.page_warnings_label.setObjectName("statusPill")
        self.page_warnings_label.setProperty("tone", "warn")
        page_issues_header.addWidget(self.page_reg_flags_label)
        page_issues_header.addWidget(self.page_warnings_label)
        page_issues_header.addStretch(1)
        self.page_issues_hint_label = QLabel("No issues on this page.")
        self.page_issues_hint_label.setObjectName("subtitleLabel")
        self.page_issues_hint_label.setWordWrap(True)
        self.page_issues_list = QListWidget()
        self.page_issues_list.setObjectName("pageIssuesList")
        self.page_issues_list.setMinimumHeight(88)
        self.page_issues_list.setMaximumHeight(164)
        page_issues_layout.addLayout(page_issues_header)
        page_issues_layout.addWidget(self.page_issues_hint_label)
        page_issues_layout.addWidget(self.page_issues_list)
        fact_box = QGroupBox("Facts (Bounding Boxes)")
        fact_box.setObjectName("inspectorSection")
        fact_layout = QVBoxLayout(fact_box)
        fact_layout.setContentsMargins(14, 14, 14, 12)
        fact_layout.setSpacing(8)
        self.facts_count_label = QLabel("No facts")
        self.facts_count_label.setObjectName("statusPill")
        self.facts_count_label.setProperty("tone", "accent")
        self.facts_list = QListWidget()
        self.facts_list.setObjectName("factsList")
        self.facts_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.facts_list.setSpacing(4)
        self.facts_list.setMinimumHeight(120)
        self.facts_list.setMaximumHeight(176)
        self.fact_bbox_label = QLabel("-")
        self.fact_value_edit = QLineEdit()
        self.fact_value_edit.setObjectName("factValueEdit")
        self.fact_equation_edit = QLineEdit()
        self.fact_equation_edit.setObjectName("factEquationEdit")
        self.fact_equation_edit.setReadOnly(True)
        self.fact_note_edit = QLineEdit()
        self.fact_note_name_edit = QLineEdit()
        self.fact_is_beur_combo = QComboBox()
        self.fact_is_beur_combo.addItems(["false", "true"])
        self.fact_beur_num_edit = QLineEdit()
        self.fact_refference_edit = QLineEdit()
        self.fact_date_edit = QLineEdit()
        self.fact_period_type_combo = QComboBox()
        self.fact_period_type_combo.addItems(list(PERIOD_TYPE_OPTIONS))
        self.fact_duration_type_combo = QComboBox()
        self.fact_duration_type_combo.addItems(list(DURATION_TYPE_OPTIONS))
        self.fact_recurring_period_combo = QComboBox()
        self.fact_recurring_period_combo.addItems(list(RECURRING_PERIOD_OPTIONS))
        self.fact_period_start_edit = QLineEdit()
        self.fact_period_end_edit = QLineEdit()
        self.fact_currency_combo = QComboBox()
        self.fact_currency_combo.addItems(["", *CURRENCY_OPTIONS])
        self.fact_scale_combo = QComboBox()
        self.fact_scale_combo.addItems(["", *[str(s) for s in SCALE_OPTIONS]])
        self.fact_value_type_combo = QComboBox()
        self.fact_value_type_combo.addItems(list(VALUE_TYPE_OPTIONS))
        self.fact_value_context_combo = QComboBox()
        self.fact_value_context_combo.addItems(list(VALUE_CONTEXT_OPTIONS))
        self.fact_balance_type_combo = QComboBox()
        self.fact_balance_type_combo.addItems(list(BALANCE_TYPE_OPTIONS))
        self.fact_row_role_combo = QComboBox()
        self.fact_row_role_combo.addItems(list(ROW_ROLE_OPTIONS))
        self.fact_aggregation_role_combo = QComboBox()
        self.fact_aggregation_role_combo.addItems(list(AGGREGATION_ROLE_OPTIONS))
        self.fact_natural_sign_label = QLabel("-")
        self.fact_natural_sign_label.setObjectName("factBboxLabel")
        self.fact_path_source_combo = QComboBox()
        self.fact_path_source_combo.addItems(list(PATH_SOURCE_OPTIONS))
        self.fact_path_list = QListWidget()
        self.fact_path_list.setObjectName("pathList")
        self.fact_path_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.fact_path_list.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.fact_path_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.fact_path_list.setDefaultDropAction(Qt.MoveAction)
        self.fact_path_list.setDragEnabled(True)
        self.fact_path_list.viewport().setAcceptDrops(True)
        self.fact_path_list.setDropIndicatorShown(True)
        self.fact_path_list.setMinimumHeight(92)
        self.fact_path_list.setMaximumHeight(128)

        self.path_add_btn = QPushButton("+ Add Level")
        self.path_remove_btn = QPushButton("Remove")
        self.path_up_btn = QPushButton("Move Up")
        self.path_down_btn = QPushButton("Move Down")
        self.path_invert_btn = QPushButton("Invert")
        self.path_add_btn.setObjectName("smallActionBtn")
        self.path_remove_btn.setObjectName("smallActionBtn")
        self.path_up_btn.setObjectName("smallActionBtn")
        self.path_down_btn.setObjectName("smallActionBtn")
        self.path_invert_btn.setObjectName("smallActionBtn")

        path_actions = QHBoxLayout()
        path_actions.setSpacing(6)
        path_actions.addWidget(self.path_add_btn)
        path_actions.addWidget(self.path_remove_btn)
        path_actions.addWidget(self.path_up_btn)
        path_actions.addWidget(self.path_down_btn)
        path_actions.addWidget(self.path_invert_btn)
        path_actions.addStretch(1)

        path_panel = QWidget()
        path_panel_layout = QVBoxLayout(path_panel)
        path_panel_layout.setContentsMargins(0, 0, 0, 0)
        path_panel_layout.setSpacing(6)
        path_panel_layout.addWidget(self.fact_path_list)
        path_panel_layout.addLayout(path_actions)

        self.fact_editor_box = QGroupBox("Selected Fact")
        self.fact_editor_box.setObjectName("inspectorSubsection")
        fact_editor_layout = QVBoxLayout(self.fact_editor_box)
        fact_editor_layout.setContentsMargins(14, 12, 14, 12)
        fact_editor_layout.setSpacing(7)

        def add_fact_editor_row(
            left_block: QWidget,
            right_block: QWidget,
            *,
            left_stretch: int = 1,
            right_stretch: int = 1,
        ) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(left_block, left_stretch)
            row.addWidget(right_block, right_stretch)
            fact_editor_layout.addLayout(row)

        self.fact_bbox_label.setObjectName("factBboxLabel")
        fact_field_min_width = 132
        self.fact_bbox_label.setMinimumWidth(0)
        self.fact_value_edit.setMinimumWidth(fact_field_min_width)
        self.fact_equation_edit.setMinimumWidth(fact_field_min_width)
        self.fact_note_edit.setMinimumWidth(fact_field_min_width)
        self.fact_note_name_edit.setMinimumWidth(fact_field_min_width)
        self.fact_is_beur_combo.setMinimumWidth(112)
        self.fact_beur_num_edit.setMinimumWidth(fact_field_min_width)
        # Empty must stay acceptable so clearing note_num can be committed as null.
        self.fact_beur_num_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*$"), self))
        self.fact_refference_edit.setMinimumWidth(fact_field_min_width)
        self.fact_date_edit.setMinimumWidth(fact_field_min_width)
        self.fact_period_type_combo.setMinimumWidth(112)
        self.fact_period_start_edit.setMinimumWidth(fact_field_min_width)
        self.fact_period_end_edit.setMinimumWidth(fact_field_min_width)
        self.fact_currency_combo.setMinimumWidth(112)
        self.fact_scale_combo.setMinimumWidth(112)
        self.fact_value_type_combo.setMinimumWidth(112)
        self.fact_value_context_combo.setMinimumWidth(112)
        self.fact_balance_type_combo.setMinimumWidth(112)
        self.fact_row_role_combo.setMinimumWidth(112)
        self.fact_aggregation_role_combo.setMinimumWidth(112)
        self.fact_path_source_combo.setMinimumWidth(112)
        self.fact_duration_type_combo.setMinimumWidth(112)
        self.fact_recurring_period_combo.setMinimumWidth(112)
        self.fact_path_list.setMinimumWidth(0)
        self.fact_equation_result_label = QLabel("-")
        self.fact_equation_result_label.setObjectName("hintText")
        self.fact_equation_status_label = QLabel("")
        self.fact_equation_status_label.setObjectName("hintText")
        self.fact_equation_status_label.setWordWrap(True)
        self.fact_equation_variants_list = EquationVariantsListWidget()
        self.fact_equation_variants_list.setObjectName("equationVariantsList")
        self.fact_equation_variants_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fact_equation_variants_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fact_equation_variants_list.setMinimumHeight(66)
        self.fact_equation_variants_list.setMaximumHeight(108)
        self.equation_add_variant_btn = QPushButton("+")
        self.equation_add_variant_btn.setObjectName("smallActionBtn")
        self.equation_add_variant_btn.setMaximumWidth(30)
        self.equation_add_variant_btn.setToolTip("Add current preview as an additional saved equation")
        self.equation_add_variant_btn.setFocusPolicy(Qt.NoFocus)
        self.equation_delete_variant_btn = QPushButton("Delete")
        self.equation_delete_variant_btn.setObjectName("smallActionBtn")
        self.equation_delete_variant_btn.setToolTip("Delete selected saved equation")
        self.equation_delete_variant_btn.setFocusPolicy(Qt.NoFocus)
        self.equation_variant_up_btn = QPushButton("↑")
        self.equation_variant_up_btn.setObjectName("smallActionBtn")
        self.equation_variant_up_btn.setMaximumWidth(30)
        self.equation_variant_up_btn.setToolTip("Move selected equation up")
        self.equation_variant_up_btn.setFocusPolicy(Qt.NoFocus)
        self.equation_variant_down_btn = QPushButton("↓")
        self.equation_variant_down_btn.setObjectName("smallActionBtn")
        self.equation_variant_down_btn.setMaximumWidth(30)
        self.equation_variant_down_btn.setToolTip("Move selected equation down")
        self.equation_variant_down_btn.setFocusPolicy(Qt.NoFocus)
        self.fact_equation_terms_list = QListWidget()
        self.fact_equation_terms_list.setObjectName("equationTermsList")
        self.fact_equation_terms_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fact_equation_terms_list.setMinimumHeight(80)
        self.fact_equation_terms_list.setMaximumHeight(116)
        self.fact_equation_terms_list.setVisible(False)
        self.equation_mark_add_btn = QPushButton("Mark +")
        self.equation_mark_add_btn.setObjectName("smallActionBtn")
        self.equation_mark_add_btn.setFocusPolicy(Qt.NoFocus)
        self.equation_mark_subtract_btn = QPushButton("Mark -")
        self.equation_mark_subtract_btn.setObjectName("smallActionBtn")
        self.equation_mark_subtract_btn.setFocusPolicy(Qt.NoFocus)
        self.equation_mark_add_btn.setVisible(False)
        self.equation_mark_subtract_btn.setVisible(False)
        self.apply_equation_btn = QPushButton("Apply Equation")
        self.apply_equation_btn.setObjectName("smallActionBtn")
        self.apply_equation_btn.setFocusPolicy(Qt.NoFocus)
        self.clear_equation_btn = QPushButton("Clear")
        self.clear_equation_btn.setObjectName("smallActionBtn")
        self.clear_equation_btn.setProperty("variant", "ghost")
        self.clear_equation_btn.setFocusPolicy(Qt.NoFocus)

        equation_editor_panel = QWidget()
        equation_editor_layout = QVBoxLayout(equation_editor_panel)
        equation_editor_layout.setContentsMargins(0, 0, 0, 0)
        equation_editor_layout.setSpacing(5)
        equation_editor_layout.addWidget(self.fact_equation_edit)
        equation_variants_row = QHBoxLayout()
        equation_variants_row.setContentsMargins(0, 0, 0, 0)
        equation_variants_row.setSpacing(6)
        equation_variants_row.addWidget(self.fact_equation_variants_list, 1)
        equation_variants_actions = QVBoxLayout()
        equation_variants_actions.setContentsMargins(0, 0, 0, 0)
        equation_variants_actions.setSpacing(4)
        equation_variants_actions.addWidget(self.equation_add_variant_btn)
        equation_variants_actions.addWidget(self.equation_delete_variant_btn)
        equation_variants_actions.addWidget(self.equation_variant_up_btn)
        equation_variants_actions.addWidget(self.equation_variant_down_btn)
        equation_variants_actions.addStretch(1)
        equation_variants_row.addLayout(equation_variants_actions)
        equation_editor_layout.addLayout(equation_variants_row)

        equation_preview_panel = QWidget()
        equation_preview_layout = QVBoxLayout(equation_preview_panel)
        equation_preview_layout.setContentsMargins(0, 0, 0, 0)
        equation_preview_layout.setSpacing(2)
        equation_preview_header = QHBoxLayout()
        equation_preview_header.setContentsMargins(0, 0, 0, 0)
        equation_preview_header.setSpacing(6)
        equation_preview_header.addWidget(self.fact_equation_result_label, 0, Qt.AlignVCenter)
        equation_preview_header.addWidget(self.apply_equation_btn, 0, Qt.AlignVCenter)
        equation_preview_header.addWidget(self.clear_equation_btn, 0, Qt.AlignVCenter)
        equation_preview_header.addStretch(1)
        equation_preview_layout.addLayout(equation_preview_header)
        equation_preview_layout.addWidget(self.fact_equation_status_label)
        equation_terms_actions = QHBoxLayout()
        equation_terms_actions.setContentsMargins(0, 0, 0, 0)
        equation_terms_actions.setSpacing(6)
        equation_terms_actions.addWidget(self.equation_mark_add_btn, 0, Qt.AlignVCenter)
        equation_terms_actions.addWidget(self.equation_mark_subtract_btn, 0, Qt.AlignVCenter)
        equation_terms_actions.addStretch(1)
        equation_preview_layout.addWidget(self.fact_equation_terms_list)
        equation_preview_layout.addLayout(equation_terms_actions)

        bbox_block = self._inspector_field_block("BBox", self.fact_bbox_label)
        bbox_block.setMaximumWidth(132)
        note_flag_block = self._inspector_field_block("Note Flag", self.fact_is_beur_combo)
        currency_block = self._inspector_field_block("Currency", self.fact_currency_combo)
        scale_block = self._inspector_field_block("Scale", self.fact_scale_combo)
        value_type_block = self._inspector_field_block("Value Type", self.fact_value_type_combo)
        value_context_block = self._inspector_field_block("Value Context", self.fact_value_context_combo)
        balance_type_block = self._inspector_field_block("Balance Type", self.fact_balance_type_combo)
        row_role_block = self._inspector_field_block("Row Role", self.fact_row_role_combo)
        aggregation_role_block = self._inspector_field_block("Aggregation Role", self.fact_aggregation_role_combo)
        balance_type_block.setVisible(False)
        aggregation_role_block.setVisible(False)
        natural_sign_block = self._inspector_field_block("Natural Sign", self.fact_natural_sign_label)
        period_type_block = self._inspector_field_block("Period Type", self.fact_period_type_combo)
        duration_type_block = self._inspector_field_block("Duration Type", self.fact_duration_type_combo)
        recurring_period_block = self._inspector_field_block("Recurring Period", self.fact_recurring_period_combo)
        self.fact_recurring_period_block = recurring_period_block
        path_source_block = self._inspector_field_block("Path Source", self.fact_path_source_combo)
        value_with_sign_panel = QWidget()
        value_with_sign_layout = QHBoxLayout(value_with_sign_panel)
        value_with_sign_layout.setContentsMargins(0, 0, 0, 0)
        value_with_sign_layout.setSpacing(8)
        value_with_sign_layout.addWidget(
            self._inspector_field_block("Value", self.fact_value_edit, required=True),
            1,
        )
        value_with_sign_layout.addWidget(natural_sign_block, 0)
        value_with_sign_layout.addWidget(row_role_block, 0)
        value_with_sign_layout.addWidget(aggregation_role_block, 0)
        period_range_panel = QWidget()
        period_range_layout = QHBoxLayout(period_range_panel)
        period_range_layout.setContentsMargins(0, 0, 0, 0)
        period_range_layout.setSpacing(8)
        period_range_layout.addWidget(self.fact_period_start_edit)
        period_range_layout.addWidget(self.fact_period_end_edit)
        period_range_block = self._inspector_field_block("Period Range", period_range_panel)

        add_fact_editor_row(
            bbox_block,
            value_with_sign_panel,
            left_stretch=0,
            right_stretch=1,
        )
        add_fact_editor_row(
            self._inspector_field_block("Equations", equation_editor_panel),
            self._inspector_field_block("Result Preview", equation_preview_panel),
        )
        add_fact_editor_row(
            self._inspector_field_block("Comment Ref", self.fact_note_edit),
            note_flag_block,
            left_stretch=1,
            right_stretch=1,
        )
        add_fact_editor_row(
            self._inspector_field_block("Note Name", self.fact_note_name_edit),
            self._inspector_field_block("Note Num", self.fact_beur_num_edit),
        )
        note_ref_block = self._inspector_field_block("Note Ref", self.fact_refference_edit)
        note_ref_row = QHBoxLayout()
        note_ref_row.setContentsMargins(0, 0, 0, 0)
        note_ref_row.setSpacing(8)
        note_ref_row.addWidget(note_ref_block, 1)
        note_ref_row.addStretch(1)
        fact_editor_layout.addLayout(note_ref_row)
        add_fact_editor_row(currency_block, scale_block)
        add_fact_editor_row(value_type_block, value_context_block)
        add_fact_editor_row(balance_type_block, path_source_block)
        add_fact_editor_row(period_type_block, duration_type_block)
        add_fact_editor_row(recurring_period_block, period_range_block)
        fact_editor_layout.addWidget(self._inspector_field_block("Path", path_panel))

        self.dup_fact_btn = QPushButton("Duplicate")
        self.del_fact_btn = QPushButton("Delete")
        self.dup_fact_btn.setObjectName("smallActionBtn")
        self.del_fact_btn.setObjectName("smallActionBtn")
        self.del_fact_btn.setProperty("variant", "danger")
        facts_header = QHBoxLayout()
        facts_header.setSpacing(10)
        facts_header.addWidget(self.facts_count_label, 0, Qt.AlignRight)
        fact_layout.addLayout(facts_header)
        fact_layout.addWidget(self.facts_list, 1)
        fact_layout.addWidget(self.fact_editor_box)
        fact_action_row = QHBoxLayout()
        fact_action_row.setSpacing(8)
        fact_action_row.addWidget(self.dup_fact_btn)
        fact_action_row.addWidget(self.del_fact_btn)
        self.batch_toggle_btn = QPushButton("Show Batch Edit")
        self.batch_toggle_btn.setObjectName("smallActionBtn")
        self.batch_toggle_btn.setProperty("qaName", "batchToggleBtn")
        self.batch_toggle_btn.setProperty("variant", "ghost")
        fact_action_row.addWidget(self.batch_toggle_btn)
        fact_action_row.addStretch(1)
        fact_layout.addLayout(fact_action_row)
        self.batch_box = QGroupBox("Batch Edit Selected BBoxes")
        self.batch_box.setObjectName("inspectorSubsection")
        self.batch_box.setProperty("qaName", "batchEditBox")
        batch_layout = QVBoxLayout(self.batch_box)
        batch_layout.setContentsMargins(14, 14, 14, 12)
        batch_layout.setSpacing(8)
        self.batch_selected_label = QLabel("Selected: 0")
        self.batch_selected_label.setObjectName("hintText")
        self.batch_path_level_edit = QLineEdit()
        self.batch_path_level_edit.setPlaceholderText("Path level (e.g. נכסים)")
        self.batch_prepend_path_btn = QPushButton("Add As Parent")
        self.batch_append_path_btn = QPushButton("Add As Child")
        self.batch_insert_index_spin = QSpinBox()
        self.batch_insert_index_spin.setRange(1, 99)
        self.batch_insert_index_spin.setValue(2)
        self.batch_insert_path_btn = QPushButton("Insert At Position")
        self.batch_remove_first_btn = QPushButton("Remove First Level")
        self.batch_remove_last_btn = QPushButton("Remove Last Level")
        self.batch_value_edit = QLineEdit()
        self.batch_value_edit.setPlaceholderText("Value for selected bboxes")
        self.batch_set_value_btn = QPushButton("Set Value")
        self.batch_clear_value_btn = QPushButton("Clear Value")
        self.batch_refference_edit = QLineEdit()
        self.batch_refference_edit.setPlaceholderText("Note ref for selected bboxes")
        self.batch_set_refference_btn = QPushButton("Set Note Ref")
        self.batch_clear_refference_btn = QPushButton("Clear Note Ref")
        self.batch_note_edit = QLineEdit()
        self.batch_note_edit.setPlaceholderText("Comment ref text for selected bboxes")
        self.batch_set_note_btn = QPushButton("Set Comment Ref")
        self.batch_clear_note_btn = QPushButton("Clear Comment Ref")
        self.batch_note_name_edit = QLineEdit()
        self.batch_note_name_edit.setPlaceholderText("Note name for selected bboxes")
        self.batch_set_note_name_btn = QPushButton("Set Note Name")
        self.batch_clear_note_name_btn = QPushButton("Clear Note Name")
        self.batch_date_edit = QLineEdit()
        self.batch_date_edit.setPlaceholderText("Date for selected bboxes (e.g. 2024-12-31)")
        self.batch_set_date_btn = QPushButton("Set Date")
        self.batch_clear_date_btn = QPushButton("Clear Date")
        self.batch_period_type_combo = QComboBox()
        self.batch_period_type_combo.addItems(list(PERIOD_TYPE_OPTIONS))
        self.batch_set_period_type_btn = QPushButton("Set period_type")
        self.batch_clear_period_type_btn = QPushButton("Clear period_type")
        self.batch_duration_type_combo = QComboBox()
        self.batch_duration_type_combo.addItems(list(DURATION_TYPE_OPTIONS))
        self.batch_set_duration_type_btn = QPushButton("Set duration_type")
        self.batch_clear_duration_type_btn = QPushButton("Clear duration_type")
        self.batch_recurring_period_combo = QComboBox()
        self.batch_recurring_period_combo.addItems(list(RECURRING_PERIOD_OPTIONS))
        self.batch_set_recurring_period_btn = QPushButton("Set recurring_period")
        self.batch_clear_recurring_period_btn = QPushButton("Clear recurring_period")
        self.batch_period_start_edit = QLineEdit()
        self.batch_period_start_edit.setPlaceholderText("Period start (YYYY-MM-DD)")
        self.batch_set_period_start_btn = QPushButton("Set Period Start")
        self.batch_clear_period_start_btn = QPushButton("Clear Period Start")
        self.batch_period_end_edit = QLineEdit()
        self.batch_period_end_edit.setPlaceholderText("Period end (YYYY-MM-DD)")
        self.batch_set_period_end_btn = QPushButton("Set Period End")
        self.batch_clear_period_end_btn = QPushButton("Clear Period End")
        self.batch_is_beur_combo = QComboBox()
        self.batch_is_beur_combo.addItems(["false", "true"])
        self.batch_set_is_beur_btn = QPushButton("Apply note_flag")
        self.batch_clear_is_beur_btn = QPushButton("Set note_flag false")
        self.batch_beur_num_edit = QLineEdit()
        self.batch_beur_num_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.batch_beur_num_edit.setPlaceholderText("Note number for selected bboxes")
        self.batch_set_beur_num_btn = QPushButton("Set Note Num")
        self.batch_clear_beur_num_btn = QPushButton("Clear Note Num")
        self.batch_currency_combo = QComboBox()
        self.batch_currency_combo.addItems(["", *CURRENCY_OPTIONS])
        self.batch_set_currency_btn = QPushButton("Set currency")
        self.batch_clear_currency_btn = QPushButton("Clear currency")
        self.batch_scale_combo = QComboBox()
        self.batch_scale_combo.addItems(["", *[str(s) for s in SCALE_OPTIONS]])
        self.batch_set_scale_btn = QPushButton("Set scale")
        self.batch_clear_scale_btn = QPushButton("Clear scale")
        self.batch_value_type_combo = QComboBox()
        self.batch_value_type_combo.addItems(list(VALUE_TYPE_OPTIONS))
        self.batch_set_value_type_btn = QPushButton("Set value_type")
        self.batch_clear_value_type_btn = QPushButton("Clear value_type")
        self.batch_value_context_combo = QComboBox()
        self.batch_value_context_combo.addItems(list(VALUE_CONTEXT_OPTIONS))
        self.batch_set_value_context_btn = QPushButton("Set value_context")
        self.batch_clear_value_context_btn = QPushButton("Clear value_context")
        self.batch_balance_type_combo = QComboBox()
        self.batch_balance_type_combo.addItems(list(BALANCE_TYPE_OPTIONS))
        self.batch_set_balance_type_btn = QPushButton("Set balance_type")
        self.batch_clear_balance_type_btn = QPushButton("Clear balance_type")
        self.batch_path_source_combo = QComboBox()
        self.batch_path_source_combo.addItems(list(PATH_SOURCE_OPTIONS))
        self.batch_set_path_source_btn = QPushButton("Set path_source")
        self.batch_clear_path_source_btn = QPushButton("Clear path_source")
        self.batch_resize_step_spin = QSpinBox()
        self.batch_resize_step_spin.setRange(1, 500)
        self.batch_resize_step_spin.setSingleStep(1)
        self.batch_resize_step_spin.setValue(2)
        self.batch_expand_left_btn = QPushButton("Grow Left")
        self.batch_expand_right_btn = QPushButton("Grow Right")
        self.batch_expand_up_btn = QPushButton("Grow Up")
        self.batch_expand_down_btn = QPushButton("Grow Down")
        batch_row_1 = QHBoxLayout()
        batch_row_1.setSpacing(8)
        batch_row_1.addWidget(self.batch_prepend_path_btn)
        batch_row_1.addWidget(self.batch_append_path_btn)
        batch_row_insert = QHBoxLayout()
        batch_row_insert.setSpacing(8)
        batch_row_insert.addWidget(QLabel("Insert pos:"))
        batch_row_insert.addWidget(self.batch_insert_index_spin)
        batch_row_insert.addWidget(self.batch_insert_path_btn)
        batch_row_insert.addStretch(1)
        batch_row_2 = QHBoxLayout()
        batch_row_2.setSpacing(8)
        batch_row_2.addWidget(self.batch_remove_first_btn)
        batch_row_2.addWidget(self.batch_remove_last_btn)
        batch_value_row = QHBoxLayout()
        batch_value_row.setSpacing(8)
        batch_value_row.addWidget(self.batch_value_edit)
        batch_value_row.addWidget(self.batch_set_value_btn)
        batch_value_row.addWidget(self.batch_clear_value_btn)
        batch_refference_row = QHBoxLayout()
        batch_refference_row.setSpacing(8)
        batch_refference_row.addWidget(self.batch_refference_edit)
        batch_refference_row.addWidget(self.batch_set_refference_btn)
        batch_refference_row.addWidget(self.batch_clear_refference_btn)
        batch_note_row = QHBoxLayout()
        batch_note_row.setSpacing(8)
        batch_note_row.addWidget(self.batch_note_edit)
        batch_note_row.addWidget(self.batch_set_note_btn)
        batch_note_row.addWidget(self.batch_clear_note_btn)
        batch_note_name_row = QHBoxLayout()
        batch_note_name_row.setSpacing(8)
        batch_note_name_row.addWidget(self.batch_note_name_edit)
        batch_note_name_row.addWidget(self.batch_set_note_name_btn)
        batch_note_name_row.addWidget(self.batch_clear_note_name_btn)
        batch_period_type_row = QHBoxLayout()
        batch_period_type_row.setSpacing(8)
        batch_period_type_row.addWidget(QLabel("period_type:"))
        batch_period_type_row.addWidget(self.batch_period_type_combo)
        batch_period_type_row.addWidget(self.batch_set_period_type_btn)
        batch_period_type_row.addWidget(self.batch_clear_period_type_btn)
        batch_period_type_row.addStretch(1)
        batch_duration_type_row = QHBoxLayout()
        batch_duration_type_row.setSpacing(8)
        batch_duration_type_row.addWidget(QLabel("duration_type:"))
        batch_duration_type_row.addWidget(self.batch_duration_type_combo)
        batch_duration_type_row.addWidget(self.batch_set_duration_type_btn)
        batch_duration_type_row.addWidget(self.batch_clear_duration_type_btn)
        batch_duration_type_row.addStretch(1)
        batch_recurring_period_row = QHBoxLayout()
        batch_recurring_period_row.setSpacing(8)
        batch_recurring_period_row.addWidget(QLabel("recurring_period:"))
        batch_recurring_period_row.addWidget(self.batch_recurring_period_combo)
        batch_recurring_period_row.addWidget(self.batch_set_recurring_period_btn)
        batch_recurring_period_row.addWidget(self.batch_clear_recurring_period_btn)
        batch_recurring_period_row.addStretch(1)
        batch_period_start_row = QHBoxLayout()
        batch_period_start_row.setSpacing(8)
        batch_period_start_row.addWidget(self.batch_period_start_edit)
        batch_period_start_row.addWidget(self.batch_set_period_start_btn)
        batch_period_start_row.addWidget(self.batch_clear_period_start_btn)
        batch_period_end_row = QHBoxLayout()
        batch_period_end_row.setSpacing(8)
        batch_period_end_row.addWidget(self.batch_period_end_edit)
        batch_period_end_row.addWidget(self.batch_set_period_end_btn)
        batch_period_end_row.addWidget(self.batch_clear_period_end_btn)
        batch_is_beur_row = QHBoxLayout()
        batch_is_beur_row.setSpacing(8)
        batch_is_beur_row.addWidget(QLabel("note_flag:"))
        batch_is_beur_row.addWidget(self.batch_is_beur_combo)
        batch_is_beur_row.addWidget(self.batch_set_is_beur_btn)
        batch_is_beur_row.addWidget(self.batch_clear_is_beur_btn)
        batch_is_beur_row.addStretch(1)
        batch_beur_num_row = QHBoxLayout()
        batch_beur_num_row.setSpacing(8)
        batch_beur_num_row.addWidget(self.batch_beur_num_edit)
        batch_beur_num_row.addWidget(self.batch_set_beur_num_btn)
        batch_beur_num_row.addWidget(self.batch_clear_beur_num_btn)
        batch_currency_row = QHBoxLayout()
        batch_currency_row.setSpacing(8)
        batch_currency_row.addWidget(QLabel("currency:"))
        batch_currency_row.addWidget(self.batch_currency_combo)
        batch_currency_row.addWidget(self.batch_set_currency_btn)
        batch_currency_row.addWidget(self.batch_clear_currency_btn)
        batch_currency_row.addStretch(1)
        batch_scale_row = QHBoxLayout()
        batch_scale_row.setSpacing(8)
        batch_scale_row.addWidget(QLabel("scale:"))
        batch_scale_row.addWidget(self.batch_scale_combo)
        batch_scale_row.addWidget(self.batch_set_scale_btn)
        batch_scale_row.addWidget(self.batch_clear_scale_btn)
        batch_scale_row.addStretch(1)
        batch_value_type_row = QHBoxLayout()
        batch_value_type_row.setSpacing(8)
        batch_value_type_row.addWidget(QLabel("value_type:"))
        batch_value_type_row.addWidget(self.batch_value_type_combo)
        batch_value_type_row.addWidget(self.batch_set_value_type_btn)
        batch_value_type_row.addWidget(self.batch_clear_value_type_btn)
        batch_value_type_row.addStretch(1)
        batch_value_context_row = QHBoxLayout()
        batch_value_context_row.setSpacing(8)
        batch_value_context_row.addWidget(QLabel("value_context:"))
        batch_value_context_row.addWidget(self.batch_value_context_combo)
        batch_value_context_row.addWidget(self.batch_set_value_context_btn)
        batch_value_context_row.addWidget(self.batch_clear_value_context_btn)
        batch_value_context_row.addStretch(1)
        batch_balance_type_row = QHBoxLayout()
        batch_balance_type_row.setSpacing(8)
        batch_balance_type_row.addWidget(QLabel("balance_type:"))
        batch_balance_type_row.addWidget(self.batch_balance_type_combo)
        batch_balance_type_row.addWidget(self.batch_set_balance_type_btn)
        batch_balance_type_row.addWidget(self.batch_clear_balance_type_btn)
        batch_balance_type_row.addStretch(1)
        batch_path_source_row = QHBoxLayout()
        batch_path_source_row.setSpacing(8)
        batch_path_source_row.addWidget(QLabel("path_source:"))
        batch_path_source_row.addWidget(self.batch_path_source_combo)
        batch_path_source_row.addWidget(self.batch_set_path_source_btn)
        batch_path_source_row.addWidget(self.batch_clear_path_source_btn)
        batch_path_source_row.addStretch(1)
        batch_resize_head = QHBoxLayout()
        batch_resize_head.setSpacing(8)
        batch_resize_head.addWidget(QLabel("Grow step (px):"))
        batch_resize_head.addWidget(self.batch_resize_step_spin)
        batch_resize_head.addStretch(1)
        batch_row_3 = QHBoxLayout()
        batch_row_3.setSpacing(8)
        batch_row_3.addWidget(self.batch_expand_left_btn)
        batch_row_3.addWidget(self.batch_expand_right_btn)
        batch_row_3.addWidget(self.batch_expand_up_btn)
        batch_row_3.addWidget(self.batch_expand_down_btn)
        batch_layout.addWidget(self.batch_selected_label)
        batch_layout.addWidget(self.batch_path_level_edit)
        batch_layout.addLayout(batch_row_1)
        batch_layout.addLayout(batch_row_insert)
        batch_layout.addLayout(batch_row_2)
        batch_layout.addLayout(batch_value_row)
        batch_layout.addLayout(batch_refference_row)
        batch_layout.addLayout(batch_note_row)
        batch_layout.addLayout(batch_note_name_row)
        batch_layout.addLayout(batch_period_type_row)
        batch_layout.addLayout(batch_duration_type_row)
        batch_layout.addLayout(batch_recurring_period_row)
        batch_layout.addLayout(batch_period_start_row)
        batch_layout.addLayout(batch_period_end_row)
        batch_layout.addLayout(batch_is_beur_row)
        batch_layout.addLayout(batch_beur_num_row)
        batch_layout.addLayout(batch_currency_row)
        batch_layout.addLayout(batch_scale_row)
        batch_layout.addLayout(batch_value_type_row)
        batch_layout.addLayout(batch_value_context_row)
        batch_layout.addLayout(batch_path_source_row)
        batch_layout.addLayout(batch_resize_head)
        batch_layout.addLayout(batch_row_3)
        fact_layout.addWidget(self.batch_box)

        self.batch_box.setVisible(False)

        tip = QLabel(
            "Select a box to edit fields here. "
            "Click a box to select it, drag on empty page area for rectangle select, and hold Shift to add to selection. "
            "Use Alt+drag on the page to preview an equation for the selected box, then press Shift (Alt+Shift) or click Apply Equation to save it. "
            "Use Approve + Next or \u2691 Flag from the top toolbar for internal page decisions, then filter/sort from the Pages panel. "
            "Use Batch Edit to change value/note_ref/comment_ref/note_name/period/duration_type/recurring_period fields/note_flag/note_num/currency/scale/value_type/value_context/path_source and path levels across selected boxes. "
            "Use Batch Grow or Alt+Arrow to expand selected boxes in one direction. "
            "Use Arrow keys to move selected box(es), Shift+Arrow for faster nudge. "
            "Use +/Remove and Move Up/Move Down to manage path hierarchy. "
            "Pan page with Ctrl+Arrow keys or right/middle mouse drag."
        )
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        right_layout.addWidget(doc_box)
        right_layout.addWidget(page_meta_box)
        right_layout.addWidget(self.page_issues_box)
        right_layout.addWidget(fact_box, 1)
        right_layout.addWidget(tip)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        self.right_scroll = right_scroll
        right_scroll.setObjectName("inspectorScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        total_w = max(self.width(), 1000)
        left_w = 112
        center_w = max(460, int(total_w * 0.56))
        right_w = max(320, total_w - left_w - center_w - 36)
        splitter.setSizes([left_w, center_w, right_w])

        self.prev_btn.clicked.connect(lambda: self.show_page(self.current_index - 1))
        self.next_btn.clicked.connect(lambda: self.show_page(self.current_index + 1))
        self.view.previous_page_requested.connect(lambda: self.show_page(self.current_index - 1))
        self.view.next_page_requested.connect(lambda: self.show_page(self.current_index + 1))
        self.page_jump_spin.valueChanged.connect(self._on_page_jump_requested)
        self.page_jump_btn.clicked.connect(lambda: self._on_page_jump_requested(self.page_jump_spin.value()))
        self.page_thumb_list.currentRowChanged.connect(self._on_thumbnail_row_changed)
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.lens_btn.toggled.connect(self._on_lens_toggled)
        self.copy_image_btn.clicked.connect(self.copy_displayed_image)
        self.fit_btn.clicked.connect(self._fit_view_height)
        self.page_json_btn.clicked.connect(self.show_current_page_json)
        self.apply_entity_all_btn.clicked.connect(self.apply_entity_name_to_all_missing_pages)
        self.help_btn.clicked.connect(self.show_help_dialog)
        self.exit_btn.clicked.connect(self.request_application_exit)
        self.save_btn.clicked.connect(self._trigger_save_annotations)
        self.import_btn.clicked.connect(self.import_annotations_json)
        self.ai_btn.clicked.connect(self.open_ai_dialog)
        self.delete_nav_btn.clicked.connect(self.delete_selected_fact)
        self.dup_fact_btn.clicked.connect(self.duplicate_selected_fact)
        self.del_fact_btn.clicked.connect(self.delete_selected_fact)
        self.apply_equation_btn.clicked.connect(self.apply_equation_to_selected_fact)
        self.clear_equation_btn.clicked.connect(self.clear_equation_from_selected_fact)
        self.equation_add_variant_btn.clicked.connect(self.add_equation_variant_to_selected_fact)
        self.equation_delete_variant_btn.clicked.connect(self.delete_selected_equation_variants_from_selected_fact)
        self.equation_variant_up_btn.clicked.connect(lambda: self.move_selected_equation_variant(-1))
        self.equation_variant_down_btn.clicked.connect(lambda: self.move_selected_equation_variant(1))
        self.fact_equation_variants_list.delete_requested.connect(self.delete_selected_equation_variants_from_selected_fact)
        self.fact_equation_variants_list.itemSelectionChanged.connect(self._update_equation_variant_controls)
        self.fact_equation_variants_list.itemClicked.connect(self._on_equation_variant_item_clicked)
        self._equation_variant_delete_shortcut = QShortcut(
            QKeySequence(Qt.Key_Delete),
            self.fact_equation_variants_list,
            activated=self.delete_selected_equation_variants_from_selected_fact,
        )
        self._equation_variant_delete_shortcut.setContext(Qt.WidgetShortcut)
        self._equation_variant_backspace_shortcut = QShortcut(
            QKeySequence(Qt.Key_Backspace),
            self.fact_equation_variants_list,
            activated=self.delete_selected_equation_variants_from_selected_fact,
        )
        self._equation_variant_backspace_shortcut.setContext(Qt.WidgetShortcut)
        self.fact_equation_terms_list.itemSelectionChanged.connect(self._update_equation_term_controls)
        self.fact_equation_terms_list.itemDoubleClicked.connect(self._on_equation_term_item_double_clicked)
        self.equation_mark_add_btn.clicked.connect(lambda: self._apply_operator_to_selected_equation_terms("+"))
        self.equation_mark_subtract_btn.clicked.connect(lambda: self._apply_operator_to_selected_equation_terms("-"))
        self.batch_toggle_btn.clicked.connect(self.toggle_batch_panel)
        self.page_issues_list.itemClicked.connect(self._on_page_issue_clicked)
        self.facts_list.itemSelectionChanged.connect(self._on_fact_list_selection_changed)
        self.facts_list.installEventFilter(self)
        self.facts_list.viewport().installEventFilter(self)
        self.entity_name_edit.editingFinished.connect(self._on_meta_edited)
        self.page_num_edit.editingFinished.connect(self._on_meta_edited)
        self.title_edit.editingFinished.connect(self._on_meta_edited)
        self.page_annotation_note_edit.editingFinished.connect(self._on_meta_edited)
        self.page_approve_continue_btn.clicked.connect(self.approve_current_page_and_continue)
        self.page_flag_btn.clicked.connect(self.flag_current_page_for_review)
        self.type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.statement_type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.doc_language_combo.activated.connect(lambda _: self._on_meta_edited())
        self.doc_direction_combo.activated.connect(lambda _: self._on_meta_edited())
        self.company_name_edit.editingFinished.connect(self._on_meta_edited)
        self.company_id_edit.editingFinished.connect(self._on_meta_edited)
        self.report_year_edit.editingFinished.connect(self._on_meta_edited)
        self.entity_type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.fact_value_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("value"))
        self.fact_note_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("comment_ref"))
        self.fact_note_name_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_name"))
        self.fact_is_beur_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("note_flag"))
        self.fact_beur_num_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_num"))
        self.fact_refference_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_ref"))
        self.fact_date_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("date"))
        self.fact_period_type_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("period_type"))
        self.fact_duration_type_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("duration_type"))
        self.fact_recurring_period_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("recurring_period"))
        self.fact_period_start_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("period_start"))
        self.fact_period_end_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("period_end"))
        self.fact_currency_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("currency"))
        self.fact_scale_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("scale"))
        self.fact_value_type_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("value_type"))
        self.fact_value_context_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("value_context"))
        self.fact_row_role_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("row_role"))
        self.fact_path_source_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("path_source"))
        self.fact_path_list.itemChanged.connect(lambda _: self._on_fact_editor_field_edited("path"))
        self.fact_path_list.itemSelectionChanged.connect(self._update_path_controls)
        self.fact_path_list.model().rowsMoved.connect(lambda *_: self._on_path_reordered())
        self.path_add_btn.clicked.connect(self.add_path_level)
        self.path_remove_btn.clicked.connect(self.remove_selected_path_level)
        self.path_up_btn.clicked.connect(self.move_selected_path_up)
        self.path_down_btn.clicked.connect(self.move_selected_path_down)
        self.path_invert_btn.clicked.connect(self.invert_selected_path_levels)
        self.batch_path_level_edit.textChanged.connect(self._update_batch_controls)
        self.batch_value_edit.textChanged.connect(self._update_batch_controls)
        self.batch_refference_edit.textChanged.connect(self._update_batch_controls)
        self.batch_note_edit.textChanged.connect(self._update_batch_controls)
        self.batch_note_name_edit.textChanged.connect(self._update_batch_controls)
        self.batch_date_edit.textChanged.connect(self._update_batch_controls)
        self.batch_period_type_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_duration_type_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_recurring_period_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_period_start_edit.textChanged.connect(self._update_batch_controls)
        self.batch_period_end_edit.textChanged.connect(self._update_batch_controls)
        self.batch_is_beur_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_beur_num_edit.textChanged.connect(self._update_batch_controls)
        self.batch_currency_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_scale_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_value_type_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_value_context_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_path_source_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_prepend_path_btn.clicked.connect(self.batch_prepend_path_level)
        self.batch_append_path_btn.clicked.connect(self.batch_append_path_level)
        self.batch_insert_path_btn.clicked.connect(self.batch_insert_path_level)
        self.batch_remove_first_btn.clicked.connect(self.batch_remove_first_level)
        self.batch_remove_last_btn.clicked.connect(self.batch_remove_last_level)
        self.batch_set_value_btn.clicked.connect(self.batch_set_value)
        self.batch_clear_value_btn.clicked.connect(self.batch_clear_value)
        self.batch_set_refference_btn.clicked.connect(self.batch_set_refference)
        self.batch_clear_refference_btn.clicked.connect(self.batch_clear_refference)
        self.batch_set_note_btn.clicked.connect(self.batch_set_note)
        self.batch_clear_note_btn.clicked.connect(self.batch_clear_note)
        self.batch_set_note_name_btn.clicked.connect(self.batch_set_note_name)
        self.batch_clear_note_name_btn.clicked.connect(self.batch_clear_note_name)
        self.batch_set_date_btn.clicked.connect(self.batch_set_date)
        self.batch_clear_date_btn.clicked.connect(self.batch_clear_date)
        self.batch_set_period_type_btn.clicked.connect(self.batch_set_period_type)
        self.batch_clear_period_type_btn.clicked.connect(self.batch_clear_period_type)
        self.batch_set_duration_type_btn.clicked.connect(self.batch_set_duration_type)
        self.batch_clear_duration_type_btn.clicked.connect(self.batch_clear_duration_type)
        self.batch_set_recurring_period_btn.clicked.connect(self.batch_set_recurring_period)
        self.batch_clear_recurring_period_btn.clicked.connect(self.batch_clear_recurring_period)
        self.batch_set_period_start_btn.clicked.connect(self.batch_set_period_start)
        self.batch_clear_period_start_btn.clicked.connect(self.batch_clear_period_start)
        self.batch_set_period_end_btn.clicked.connect(self.batch_set_period_end)
        self.batch_clear_period_end_btn.clicked.connect(self.batch_clear_period_end)
        self.batch_set_is_beur_btn.clicked.connect(self.batch_set_is_beur)
        self.batch_clear_is_beur_btn.clicked.connect(self.batch_clear_is_beur)
        self.batch_set_beur_num_btn.clicked.connect(self.batch_set_beur_num)
        self.batch_clear_beur_num_btn.clicked.connect(self.batch_clear_beur_num)
        self.batch_set_currency_btn.clicked.connect(self.batch_set_currency)
        self.batch_clear_currency_btn.clicked.connect(self.batch_clear_currency)
        self.batch_set_scale_btn.clicked.connect(self.batch_set_scale)
        self.batch_clear_scale_btn.clicked.connect(self.batch_clear_scale)
        self.batch_set_value_type_btn.clicked.connect(self.batch_set_value_type)
        self.batch_clear_value_type_btn.clicked.connect(self.batch_clear_value_type)
        self.batch_set_value_context_btn.clicked.connect(self.batch_set_value_context)
        self.batch_clear_value_context_btn.clicked.connect(self.batch_clear_value_context)
        self.batch_set_path_source_btn.clicked.connect(self.batch_set_path_source)
        self.batch_clear_path_source_btn.clicked.connect(self.batch_clear_path_source)
        self.batch_expand_left_btn.clicked.connect(lambda: self.batch_expand_selected("left"))
        self.batch_expand_right_btn.clicked.connect(lambda: self.batch_expand_selected("right"))
        self.batch_expand_up_btn.clicked.connect(lambda: self.batch_expand_selected("up"))
        self.batch_expand_down_btn.clicked.connect(lambda: self.batch_expand_selected("down"))

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+I"), self, activated=self.import_annotations_json)
        QShortcut(QKeySequence("Meta+I"), self, activated=self.import_annotations_json)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.open_ai_dialog)
        QShortcut(QKeySequence("Meta+G"), self, activated=self.open_ai_dialog)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.duplicate_selected_fact)
        QShortcut(QKeySequence("Meta+D"), self, activated=self.duplicate_selected_fact)
        self._delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_fact_shortcut)
        self._delete_shortcut.setContext(Qt.ApplicationShortcut)
        self._delete_backspace_shortcut = QShortcut(
            QKeySequence(Qt.Key_Backspace),
            self,
            activated=self._delete_selected_fact_shortcut,
        )
        self._delete_backspace_shortcut.setContext(Qt.ApplicationShortcut)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._fit_view_height)
        QShortcut(QKeySequence("Alt+F"), self, activated=self.focus_fact_annotation_panel)
        QShortcut(QKeySequence("F1"), self, activated=self.show_help_dialog)

        self._save_action = QAction("Save", self)
        self._save_action.triggered.connect(self._trigger_save_annotations)
        self.addAction(self._save_action)
        self._save_shortcuts = [
            QShortcut(QKeySequence.Save, self, activated=self._trigger_save_annotations),
            QShortcut(QKeySequence("Ctrl+S"), self, activated=self._trigger_save_annotations),
            QShortcut(QKeySequence("Meta+S"), self, activated=self._trigger_save_annotations),
        ]
        for shortcut in self._save_shortcuts:
            shortcut.setContext(Qt.ApplicationShortcut)
        self.page_label.setObjectName("monoLabel")
        self._populate_page_thumbnails()
        self._configure_hidden_status_bar()
        self.statusBar().showMessage("Ready")
        self._set_fact_editor_enabled(False)
        self._clear_fact_editor()
        self._clear_gt_activity()
        self._update_path_controls()
        self._update_batch_controls()
        self._update_gemini_fill_button_state()
        self._update_gemini_complete_button_state()
        self._update_history_controls()

    def _configure_hidden_status_bar(self) -> None:
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        status_bar.setVisible(False)
        status_bar.setMaximumHeight(0)
        status_bar.setStyleSheet(
            "QStatusBar { background: transparent; border: none; min-height: 0px; max-height: 0px; }"
        )

    def _apply_ui_sizing(self) -> None:
        base_font = QFont(self.font())
        if base_font.pointSize() < 11:
            base_font.setPointSize(11)
        self.setFont(base_font)

    def _path_tone_brushes(self, tone: Optional[str]) -> tuple[QBrush, QBrush] | None:
        if tone is None:
            return None
        dark_theme = self.palette().base().color().lightness() < 128
        if tone == "shared":
            bg = QColor("#193b2e" if dark_theme else "#dff6e7")
            fg = QColor("#dff7ea" if dark_theme else "#166534")
            return QBrush(bg), QBrush(fg)
        if tone == "variant":
            bg = QColor("#4a3114" if dark_theme else "#fff0d8")
            fg = QColor("#ffe4bf" if dark_theme else "#9a5b00")
            return QBrush(bg), QBrush(fg)
        return None

    def _make_path_item(
        self,
        text: str,
        *,
        tone: Optional[str] = None,
        editable: bool = True,
        tooltip: Optional[str] = None,
        path_index: Optional[int] = None,
        path_operation_index: Optional[int] = None,
    ) -> QListWidgetItem:
        item = QListWidgetItem(text)
        flags = item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if editable:
            flags |= Qt.ItemIsEditable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled
        item.setFlags(flags)
        if tooltip:
            item.setToolTip(tooltip)
        if path_index is not None:
            item.setData(PATH_LEVEL_INDEX_ROLE, int(path_index))
        if path_operation_index is not None:
            item.setData(PATH_OPERATION_INDEX_ROLE, int(path_operation_index))
        elif path_index is not None:
            item.setData(PATH_OPERATION_INDEX_ROLE, int(path_index))
        brushes = self._path_tone_brushes(tone)
        if brushes is not None:
            bg, fg = brushes
            item.setBackground(bg)
            item.setForeground(fg)
        return item

    def _path_operation_index(self, item: Optional[QListWidgetItem]) -> Optional[int]:
        if item is None:
            return None
        value = item.data(PATH_OPERATION_INDEX_ROLE)
        if value in (None, ""):
            return None
        return int(value)

    def _neighbor_path_operation(self, row: int, step: int) -> tuple[Optional[int], Optional[int]]:
        idx = row + step
        while 0 <= idx < self.fact_path_list.count():
            operation_index = self._path_operation_index(self.fact_path_list.item(idx))
            if operation_index is not None:
                return idx, operation_index
            idx += step
        return None, None

    def _set_path_list_editable(self, editable: bool, *, structural: bool = True) -> None:
        self._path_list_editable = editable
        self._path_list_structure_editable = bool(editable and structural)
        if editable:
            self.fact_path_list.setEditTriggers(
                QAbstractItemView.DoubleClicked
                | QAbstractItemView.EditKeyPressed
                | QAbstractItemView.SelectedClicked
            )
            if self._path_list_structure_editable:
                self.fact_path_list.setDragDropMode(QAbstractItemView.InternalMove)
                self.fact_path_list.setDefaultDropAction(Qt.MoveAction)
                self.fact_path_list.setDragEnabled(True)
                self.fact_path_list.viewport().setAcceptDrops(True)
                self.fact_path_list.setDropIndicatorShown(True)
            else:
                self.fact_path_list.setDragDropMode(QAbstractItemView.NoDragDrop)
                self.fact_path_list.setDragEnabled(False)
                self.fact_path_list.viewport().setAcceptDrops(False)
                self.fact_path_list.setDropIndicatorShown(False)
            return
        self.fact_path_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fact_path_list.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.fact_path_list.setDragEnabled(False)
        self.fact_path_list.viewport().setAcceptDrops(False)
        self.fact_path_list.setDropIndicatorShown(False)

    def _update_path_controls(self) -> None:
        list_enabled = self.fact_path_list.isEnabled()
        row = self.fact_path_list.currentRow()
        count = self.fact_path_list.count()
        current_item = self.fact_path_list.item(row) if row >= 0 else None
        removable_shared_level = False
        movable_shared_up = False
        movable_shared_down = False
        if list_enabled and row >= 0 and not self._path_list_structure_editable and current_item is not None:
            current_index_raw = current_item.data(PATH_LEVEL_INDEX_ROLE)
            removable_shared_level = current_index_raw not in (None, "")
            current_operation_index = self._path_operation_index(current_item)
            _prev_row, prev_operation_index = self._neighbor_path_operation(row, -1)
            _next_row, next_operation_index = self._neighbor_path_operation(row, 1)
            movable_shared_up = (
                current_operation_index is not None
                and prev_operation_index is not None
                and prev_operation_index == current_operation_index - 1
            )
            movable_shared_down = (
                current_operation_index is not None
                and next_operation_index is not None
                and next_operation_index == current_operation_index + 1
            )
        structural_enabled = list_enabled and self._path_list_structure_editable
        self.path_add_btn.setEnabled(structural_enabled)
        self.path_remove_btn.setEnabled((structural_enabled and row >= 0) or removable_shared_level)
        self.path_up_btn.setEnabled((structural_enabled and row > 0) or movable_shared_up)
        self.path_down_btn.setEnabled((structural_enabled and 0 <= row < count - 1) or movable_shared_down)
        self.path_invert_btn.setEnabled(list_enabled and bool(self._selected_fact_items()))

    def add_path_level(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_structure_editable:
            return
        target_row = self.fact_path_list.count()
        next_idx = target_row + 1
        item = self._make_path_item(f"path_{next_idx}")
        self._is_loading_fact_editor = True
        try:
            self.fact_path_list.addItem(item)
            self.fact_path_list.setCurrentItem(item)
        finally:
            self._is_loading_fact_editor = False
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")
        if 0 <= target_row < self.fact_path_list.count():
            refreshed_item = self.fact_path_list.item(target_row)
            if refreshed_item is not None:
                self.fact_path_list.setCurrentRow(target_row)
                self.fact_path_list.editItem(refreshed_item)

    def remove_selected_path_level(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_editable:
            return
        row = self.fact_path_list.currentRow()
        if row < 0:
            return
        if not self._path_list_structure_editable:
            selected_items = self._selected_fact_items()
            if not selected_items:
                return
            item = self.fact_path_list.item(row)
            if item is None:
                return
            path_index_raw = item.data(PATH_LEVEL_INDEX_ROLE)
            if path_index_raw in (None, ""):
                return
            path_index = int(path_index_raw)
            removed_level = item.text().strip()
            changed = False
            for selected_item in selected_items:
                current = normalize_fact_data(selected_item.fact_data)
                path_value = [str(part).strip() for part in (current.get("path") or []) if str(part).strip()]
                if not (0 <= path_index < len(path_value)):
                    continue
                del path_value[path_index]
                updated = normalize_fact_data({**current, "path": path_value})
                if updated == current:
                    continue
                selected_item.fact_data = updated
                changed = True
            if not changed:
                return
            self.refresh_facts_list()
            self._record_history_snapshot()
            self._sync_fact_editor_from_selection()
            if removed_level:
                self.statusBar().showMessage(f"Removed shared path level '{removed_level}'.", 3000)
            return
        self.fact_path_list.takeItem(row)
        if self.fact_path_list.count() > 0:
            self.fact_path_list.setCurrentRow(min(row, self.fact_path_list.count() - 1))
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

    def move_selected_path_up(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_editable:
            return
        row = self.fact_path_list.currentRow()
        if row < 0:
            return
        if not self._path_list_structure_editable:
            item = self.fact_path_list.item(row)
            prev_row, prev_index = self._neighbor_path_operation(row, -1)
            prev_item = self.fact_path_list.item(prev_row) if prev_row is not None else None
            if item is None or prev_item is None:
                return
            path_index = self._path_operation_index(item)
            if path_index is None or prev_index is None:
                return
            if prev_index != path_index - 1:
                return
            selected_items = self._selected_fact_items()
            if not selected_items:
                return
            changed = False
            for selected_item in selected_items:
                current = normalize_fact_data(selected_item.fact_data)
                path_value = [str(part).strip() for part in (current.get("path") or []) if str(part).strip()]
                if not (0 <= prev_index < len(path_value) and 0 <= path_index < len(path_value)):
                    continue
                path_value[prev_index], path_value[path_index] = path_value[path_index], path_value[prev_index]
                updated = normalize_fact_data({**current, "path": path_value})
                if updated == current:
                    continue
                selected_item.fact_data = updated
                changed = True
            if not changed:
                return
            self.refresh_facts_list()
            self._record_history_snapshot()
            self._sync_fact_editor_from_selection()
            if prev_row is not None and 0 <= prev_row < self.fact_path_list.count():
                self.fact_path_list.setCurrentRow(prev_row)
            self.statusBar().showMessage("Moved path level up across selected bboxes.", 3000)
            return
        if row <= 0:
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row - 1, item)
        self.fact_path_list.setCurrentRow(row - 1)
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

    def move_selected_path_down(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_editable:
            return
        row = self.fact_path_list.currentRow()
        if row < 0:
            return
        if not self._path_list_structure_editable:
            item = self.fact_path_list.item(row)
            next_row, next_index = self._neighbor_path_operation(row, 1)
            next_item = self.fact_path_list.item(next_row) if next_row is not None else None
            if item is None or next_item is None:
                return
            path_index = self._path_operation_index(item)
            if path_index is None or next_index is None:
                return
            if next_index != path_index + 1:
                return
            selected_items = self._selected_fact_items()
            if not selected_items:
                return
            changed = False
            for selected_item in selected_items:
                current = normalize_fact_data(selected_item.fact_data)
                path_value = [str(part).strip() for part in (current.get("path") or []) if str(part).strip()]
                if not (0 <= path_index < len(path_value) and 0 <= next_index < len(path_value)):
                    continue
                path_value[path_index], path_value[next_index] = path_value[next_index], path_value[path_index]
                updated = normalize_fact_data({**current, "path": path_value})
                if updated == current:
                    continue
                selected_item.fact_data = updated
                changed = True
            if not changed:
                return
            self.refresh_facts_list()
            self._record_history_snapshot()
            self._sync_fact_editor_from_selection()
            if next_row is not None and 0 <= next_row < self.fact_path_list.count():
                self.fact_path_list.setCurrentRow(next_row)
            self.statusBar().showMessage("Moved path level down across selected bboxes.", 3000)
            return
        if row >= (self.fact_path_list.count() - 1):
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row + 1, item)
        self.fact_path_list.setCurrentRow(row + 1)
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

    def invert_selected_path_levels(self) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            self.statusBar().showMessage("No selected bboxes.", 2500)
            return

        changed_count = 0
        for item in selected_items:
            current = normalize_fact_data(item.fact_data)
            path = [str(part).strip() for part in (current.get("path") or []) if str(part).strip()]
            updated = normalize_fact_data({**current, "path": list(reversed(path))})
            if updated == current:
                continue
            item.fact_data = updated
            changed_count += 1

        if changed_count == 0:
            self.statusBar().showMessage("Invert path made no changes.", 2500)
            return

        self.refresh_facts_list()
        self._record_history_snapshot()
        self.statusBar().showMessage(f"Inverted path hierarchy ({changed_count} bbox(es)).", 3500)

    def _on_path_reordered(self) -> None:
        if not self._path_list_structure_editable:
            return
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

    def _update_batch_controls(self) -> None:
        selected_count = len(self._selected_fact_items()) if hasattr(self, "scene") else 0
        has_selection = selected_count > 0
        has_text = bool(self.batch_path_level_edit.text().strip()) if hasattr(self, "batch_path_level_edit") else False
        has_value_text = bool(self.batch_value_edit.text().strip()) if hasattr(self, "batch_value_edit") else False
        has_refference_text = bool(self.batch_refference_edit.text().strip()) if hasattr(self, "batch_refference_edit") else False
        has_note_text = bool(self.batch_note_edit.text().strip()) if hasattr(self, "batch_note_edit") else False
        has_note_name_text = bool(self.batch_note_name_edit.text().strip()) if hasattr(self, "batch_note_name_edit") else False
        has_date_text = bool(self.batch_date_edit.text().strip()) if hasattr(self, "batch_date_edit") else False
        has_period_type_choice = bool(self.batch_period_type_combo.currentText().strip()) if hasattr(self, "batch_period_type_combo") else False
        has_duration_type_choice = bool(self.batch_duration_type_combo.currentText().strip()) if hasattr(self, "batch_duration_type_combo") else False
        has_recurring_period_choice = bool(self.batch_recurring_period_combo.currentText().strip()) if hasattr(self, "batch_recurring_period_combo") else False
        has_period_start_text = bool(self.batch_period_start_edit.text().strip()) if hasattr(self, "batch_period_start_edit") else False
        has_period_end_text = bool(self.batch_period_end_edit.text().strip()) if hasattr(self, "batch_period_end_edit") else False
        has_is_beur_choice = bool(self.batch_is_beur_combo.currentText().strip()) if hasattr(self, "batch_is_beur_combo") else False
        has_beur_num_text = bool(self.batch_beur_num_edit.text().strip()) if hasattr(self, "batch_beur_num_edit") else False
        has_currency_choice = bool(self.batch_currency_combo.currentText().strip()) if hasattr(self, "batch_currency_combo") else False
        has_scale_choice = bool(self.batch_scale_combo.currentText().strip()) if hasattr(self, "batch_scale_combo") else False
        has_value_type_choice = bool(self.batch_value_type_combo.currentText().strip()) if hasattr(self, "batch_value_type_combo") else False
        has_value_context_choice = bool(self.batch_value_context_combo.currentText().strip()) if hasattr(self, "batch_value_context_combo") else False
        has_path_source_choice = bool(self.batch_path_source_combo.currentText().strip()) if hasattr(self, "batch_path_source_combo") else False
        if hasattr(self, "batch_selected_label"):
            self.batch_selected_label.setText(f"Selected: {selected_count}")
        if hasattr(self, "batch_prepend_path_btn"):
            self.batch_prepend_path_btn.setEnabled(has_selection and has_text)
        if hasattr(self, "batch_append_path_btn"):
            self.batch_append_path_btn.setEnabled(has_selection and has_text)
        if hasattr(self, "batch_insert_path_btn"):
            self.batch_insert_path_btn.setEnabled(has_selection and has_text)
        if hasattr(self, "batch_insert_index_spin"):
            self.batch_insert_index_spin.setEnabled(has_selection)
        if hasattr(self, "batch_remove_first_btn"):
            self.batch_remove_first_btn.setEnabled(has_selection)
        if hasattr(self, "batch_remove_last_btn"):
            self.batch_remove_last_btn.setEnabled(has_selection)
        if hasattr(self, "batch_value_edit"):
            self.batch_value_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_value_btn"):
            self.batch_set_value_btn.setEnabled(has_selection and has_value_text)
        if hasattr(self, "batch_clear_value_btn"):
            self.batch_clear_value_btn.setEnabled(has_selection)
        if hasattr(self, "batch_refference_edit"):
            self.batch_refference_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_refference_btn"):
            self.batch_set_refference_btn.setEnabled(has_selection and has_refference_text)
        if hasattr(self, "batch_clear_refference_btn"):
            self.batch_clear_refference_btn.setEnabled(has_selection)
        if hasattr(self, "batch_note_edit"):
            self.batch_note_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_note_btn"):
            self.batch_set_note_btn.setEnabled(has_selection and has_note_text)
        if hasattr(self, "batch_clear_note_btn"):
            self.batch_clear_note_btn.setEnabled(has_selection)
        if hasattr(self, "batch_note_name_edit"):
            self.batch_note_name_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_note_name_btn"):
            self.batch_set_note_name_btn.setEnabled(has_selection and has_note_name_text)
        if hasattr(self, "batch_clear_note_name_btn"):
            self.batch_clear_note_name_btn.setEnabled(has_selection)
        if hasattr(self, "batch_date_edit"):
            self.batch_date_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_date_btn"):
            self.batch_set_date_btn.setEnabled(has_selection and has_date_text)
        if hasattr(self, "batch_clear_date_btn"):
            self.batch_clear_date_btn.setEnabled(has_selection)
        if hasattr(self, "batch_period_type_combo"):
            self.batch_period_type_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_period_type_btn"):
            self.batch_set_period_type_btn.setEnabled(has_selection and has_period_type_choice)
        if hasattr(self, "batch_clear_period_type_btn"):
            self.batch_clear_period_type_btn.setEnabled(has_selection)
        if hasattr(self, "batch_duration_type_combo"):
            self.batch_duration_type_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_duration_type_btn"):
            self.batch_set_duration_type_btn.setEnabled(has_selection and has_duration_type_choice)
        if hasattr(self, "batch_clear_duration_type_btn"):
            self.batch_clear_duration_type_btn.setEnabled(has_selection)
        if hasattr(self, "batch_recurring_period_combo"):
            self.batch_recurring_period_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_recurring_period_btn"):
            self.batch_set_recurring_period_btn.setEnabled(has_selection and has_recurring_period_choice)
        if hasattr(self, "batch_clear_recurring_period_btn"):
            self.batch_clear_recurring_period_btn.setEnabled(has_selection)
        if hasattr(self, "batch_period_start_edit"):
            self.batch_period_start_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_period_start_btn"):
            self.batch_set_period_start_btn.setEnabled(has_selection and has_period_start_text)
        if hasattr(self, "batch_clear_period_start_btn"):
            self.batch_clear_period_start_btn.setEnabled(has_selection)
        if hasattr(self, "batch_period_end_edit"):
            self.batch_period_end_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_period_end_btn"):
            self.batch_set_period_end_btn.setEnabled(has_selection and has_period_end_text)
        if hasattr(self, "batch_clear_period_end_btn"):
            self.batch_clear_period_end_btn.setEnabled(has_selection)
        if hasattr(self, "batch_is_beur_combo"):
            self.batch_is_beur_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_is_beur_btn"):
            self.batch_set_is_beur_btn.setEnabled(has_selection and has_is_beur_choice)
        if hasattr(self, "batch_clear_is_beur_btn"):
            self.batch_clear_is_beur_btn.setEnabled(has_selection)
        if hasattr(self, "batch_beur_num_edit"):
            self.batch_beur_num_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_beur_num_btn"):
            self.batch_set_beur_num_btn.setEnabled(has_selection and has_beur_num_text)
        if hasattr(self, "batch_clear_beur_num_btn"):
            self.batch_clear_beur_num_btn.setEnabled(has_selection)
        if hasattr(self, "batch_currency_combo"):
            self.batch_currency_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_currency_btn"):
            self.batch_set_currency_btn.setEnabled(has_selection and has_currency_choice)
        if hasattr(self, "batch_clear_currency_btn"):
            self.batch_clear_currency_btn.setEnabled(has_selection)
        if hasattr(self, "batch_scale_combo"):
            self.batch_scale_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_scale_btn"):
            self.batch_set_scale_btn.setEnabled(has_selection and has_scale_choice)
        if hasattr(self, "batch_clear_scale_btn"):
            self.batch_clear_scale_btn.setEnabled(has_selection)
        if hasattr(self, "batch_value_type_combo"):
            self.batch_value_type_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_value_type_btn"):
            self.batch_set_value_type_btn.setEnabled(has_selection and has_value_type_choice)
        if hasattr(self, "batch_clear_value_type_btn"):
            self.batch_clear_value_type_btn.setEnabled(has_selection)
        if hasattr(self, "batch_value_context_combo"):
            self.batch_value_context_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_value_context_btn"):
            self.batch_set_value_context_btn.setEnabled(has_selection and has_value_context_choice)
        if hasattr(self, "batch_clear_value_context_btn"):
            self.batch_clear_value_context_btn.setEnabled(has_selection)
        if hasattr(self, "batch_path_source_combo"):
            self.batch_path_source_combo.setEnabled(has_selection)
        if hasattr(self, "batch_set_path_source_btn"):
            self.batch_set_path_source_btn.setEnabled(has_selection and has_path_source_choice)
        if hasattr(self, "batch_clear_path_source_btn"):
            self.batch_clear_path_source_btn.setEnabled(has_selection)
        if hasattr(self, "batch_expand_left_btn"):
            self.batch_expand_left_btn.setEnabled(has_selection)
        if hasattr(self, "batch_expand_right_btn"):
            self.batch_expand_right_btn.setEnabled(has_selection)
        if hasattr(self, "batch_expand_up_btn"):
            self.batch_expand_up_btn.setEnabled(has_selection)
        if hasattr(self, "batch_expand_down_btn"):
            self.batch_expand_down_btn.setEnabled(has_selection)
        if hasattr(self, "batch_resize_step_spin"):
            self.batch_resize_step_spin.setEnabled(has_selection)

    def _batch_update_selected_facts(
        self,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        success_message: str,
    ) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            self.statusBar().showMessage("No selected bboxes for batch update.", 2500)
            return

        changed_count = 0
        for item in selected_items:
            original = normalize_fact_data(item.fact_data)
            updated = normalize_fact_data(transform(deepcopy(original)))
            if updated != original:
                item.fact_data = updated
                changed_count += 1

        if changed_count == 0:
            self.statusBar().showMessage("Batch update made no changes.", 2500)
            return

        self.refresh_facts_list(refresh_issues=False)
        self._record_history_snapshot()
        self.statusBar().showMessage(f"{success_message} ({changed_count} bbox(es)).", 3500)

    def batch_prepend_path_level(self) -> None:
        level = self.batch_path_level_edit.text().strip()
        if not level:
            self.statusBar().showMessage("Enter a path level first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            path = [str(p).strip() for p in (fact.get("path") or []) if str(p).strip()]
            if not path or path[0] != level:
                fact["path"] = [level, *path]
            else:
                fact["path"] = path
            return fact

        self._batch_update_selected_facts(_transform, f"Added parent path '{level}'")

    def batch_append_path_level(self) -> None:
        level = self.batch_path_level_edit.text().strip()
        if not level:
            self.statusBar().showMessage("Enter a path level first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            path = [str(p).strip() for p in (fact.get("path") or []) if str(p).strip()]
            if not path or path[-1] != level:
                path.append(level)
            fact["path"] = path
            return fact

        self._batch_update_selected_facts(_transform, f"Added child path '{level}'")

    def batch_insert_path_level(self) -> None:
        level = self.batch_path_level_edit.text().strip()
        if not level:
            self.statusBar().showMessage("Enter a path level first.", 2500)
            return
        position = int(self.batch_insert_index_spin.value()) if hasattr(self, "batch_insert_index_spin") else 1
        insert_index = max(0, position - 1)

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            path = [str(p).strip() for p in (fact.get("path") or []) if str(p).strip()]
            idx = min(insert_index, len(path))
            path.insert(idx, level)
            fact["path"] = path
            return fact

        self._batch_update_selected_facts(_transform, f"Inserted path '{level}' at position {position}")

    def batch_remove_first_level(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            path = [str(p).strip() for p in (fact.get("path") or []) if str(p).strip()]
            fact["path"] = path[1:] if path else path
            return fact

        self._batch_update_selected_facts(_transform, "Removed first path level")

    def batch_remove_last_level(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            path = [str(p).strip() for p in (fact.get("path") or []) if str(p).strip()]
            fact["path"] = path[:-1] if path else path
            return fact

        self._batch_update_selected_facts(_transform, "Removed last path level")

    def _selected_path_signatures(self, selected_items: List[AnnotRectItem]) -> List[tuple[str, ...]]:
        signatures: set[tuple[str, ...]] = set()
        for item in selected_items:
            fact = normalize_fact_data(item.fact_data)
            path = tuple(str(part).strip() for part in (fact.get("path") or []) if str(part).strip())
            signatures.add(path)
        return sorted(signatures)

    def batch_clear_refference(self) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            self.statusBar().showMessage("No selected bboxes for batch update.", 2500)
            return

        path_signatures = self._selected_path_signatures(selected_items)
        if len(path_signatures) > 1:
            sample_paths = [
                " > ".join(path) if path else "(empty path)"
                for path in path_signatures[:3]
            ]
            more_count = len(path_signatures) - len(sample_paths)
            detail = ", ".join(sample_paths)
            if more_count > 0:
                detail = f"{detail}, +{more_count} more"
            choice = QMessageBox.warning(
                self,
                "Different attributes selected",
                (
                    "Selected bboxes do not all share the same attribute path.\n"
                    f"Found paths: {detail}\n\n"
                    "Clear note_ref for all selected bboxes anyway?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_ref"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared note_ref")

    def batch_set_value(self) -> None:
        value = self.batch_value_edit.text().strip()
        if not value:
            self.statusBar().showMessage("Enter value text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value"] = value
            return fact

        self._batch_update_selected_facts(_transform, "Updated value")

    def batch_clear_value(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value"] = ""
            return fact

        self._batch_update_selected_facts(_transform, "Cleared value")

    def batch_set_refference(self) -> None:
        refference = self.batch_refference_edit.text().strip()
        if not refference:
            self.statusBar().showMessage("Enter note_ref text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_ref"] = refference
            return fact

        self._batch_update_selected_facts(_transform, "Updated note_ref")

    def batch_set_note(self) -> None:
        note = self.batch_note_edit.text().strip()
        if not note:
            self.statusBar().showMessage("Enter comment_ref text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["comment_ref"] = note
            return fact

        self._batch_update_selected_facts(_transform, "Updated comment_ref")

    def batch_clear_note(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["comment_ref"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared comment_ref")

    def batch_set_note_name(self) -> None:
        note_name = self.batch_note_name_edit.text().strip()
        if not note_name:
            self.statusBar().showMessage("Enter note_name text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_name"] = note_name
            return fact

        self._batch_update_selected_facts(_transform, "Updated note_name")

    def batch_clear_note_name(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_name"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared note_name")

    def batch_set_date(self) -> None:
        date = self.batch_date_edit.text().strip()
        if not date:
            self.statusBar().showMessage("Enter date text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["date"] = date
            return fact

        self._batch_update_selected_facts(_transform, "Updated date")

    def batch_clear_date(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["date"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared date")

    def batch_set_period_type(self) -> None:
        period_type = self.batch_period_type_combo.currentText().strip()
        if not period_type:
            self.statusBar().showMessage("Choose period_type first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_type"] = period_type
            return fact

        self._batch_update_selected_facts(_transform, f"Updated period_type to {period_type}")

    def batch_clear_period_type(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_type"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared period_type")

    def batch_set_duration_type(self) -> None:
        duration_type = self.batch_duration_type_combo.currentText().strip()
        if not duration_type:
            self.statusBar().showMessage("Choose duration_type first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["duration_type"] = duration_type
            return fact

        self._batch_update_selected_facts(_transform, f"Updated duration_type to {duration_type}")

    def batch_clear_duration_type(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["duration_type"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared duration_type")

    def batch_set_recurring_period(self) -> None:
        recurring_period = self.batch_recurring_period_combo.currentText().strip()
        if not recurring_period:
            self.statusBar().showMessage("Choose recurring_period first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["recurring_period"] = recurring_period
            return fact

        self._batch_update_selected_facts(_transform, f"Updated recurring_period to {recurring_period}")

    def batch_clear_recurring_period(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["recurring_period"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared recurring_period")

    def batch_set_period_start(self) -> None:
        period_start = self.batch_period_start_edit.text().strip()
        if not period_start:
            self.statusBar().showMessage("Enter period_start text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_start"] = period_start
            return fact

        self._batch_update_selected_facts(_transform, "Updated period_start")

    def batch_clear_period_start(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_start"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared period_start")

    def batch_set_period_end(self) -> None:
        period_end = self.batch_period_end_edit.text().strip()
        if not period_end:
            self.statusBar().showMessage("Enter period_end text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_end"] = period_end
            return fact

        self._batch_update_selected_facts(_transform, "Updated period_end")

    def batch_clear_period_end(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["period_end"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared period_end")

    def batch_set_is_beur(self) -> None:
        selected = self.batch_is_beur_combo.currentText().strip().lower()
        if selected == "true":
            value: Optional[bool] = True
        elif selected == "false":
            value = False
        else:
            self.statusBar().showMessage("Choose note_flag value first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_flag"] = bool(value)
            return fact

        self._batch_update_selected_facts(_transform, f"Updated note_flag to {selected}")

    def batch_clear_is_beur(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_flag"] = False
            return fact

        self._batch_update_selected_facts(_transform, "Set note_flag to false")

    def batch_set_beur_num(self) -> None:
        beur_num = self.batch_beur_num_edit.text().strip()
        if not beur_num:
            self.statusBar().showMessage("Enter note_num first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_num"] = int(beur_num)
            return fact

        self._batch_update_selected_facts(_transform, "Updated note_num")

    def batch_clear_beur_num(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_num"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared note_num")

    def batch_set_currency(self) -> None:
        currency = self.batch_currency_combo.currentText().strip()
        if not currency:
            self.statusBar().showMessage("Choose currency first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["currency"] = currency
            return fact

        self._batch_update_selected_facts(_transform, f"Updated currency to {currency}")

    def batch_clear_currency(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["currency"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared currency")

    def batch_set_scale(self) -> None:
        selected = self.batch_scale_combo.currentText().strip()
        if not selected:
            self.statusBar().showMessage("Choose scale first.", 2500)
            return
        try:
            scale_value = int(selected)
        except ValueError:
            self.statusBar().showMessage("Invalid scale value.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["scale"] = scale_value
            return fact

        self._batch_update_selected_facts(_transform, f"Updated scale to {scale_value}")

    def batch_clear_scale(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["scale"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared scale")

    def batch_set_value_type(self) -> None:
        value_type = self.batch_value_type_combo.currentText().strip()
        if not value_type:
            self.statusBar().showMessage("Choose value_type first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value_type"] = value_type
            return fact

        self._batch_update_selected_facts(_transform, f"Updated value_type to {value_type}")

    def batch_clear_value_type(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value_type"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared value_type")

    def batch_set_value_context(self) -> None:
        value_context = self.batch_value_context_combo.currentText().strip()
        if not value_context:
            self.statusBar().showMessage("Choose value_context first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value_context"] = value_context
            return fact

        self._batch_update_selected_facts(_transform, f"Updated value_context to {value_context}")

    def batch_clear_value_context(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["value_context"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared value_context")

    def batch_set_path_source(self) -> None:
        path_source = self.batch_path_source_combo.currentText().strip()
        if not path_source:
            self.statusBar().showMessage("Choose path_source first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["path_source"] = path_source
            return fact

        self._batch_update_selected_facts(_transform, f"Updated path_source to {path_source}")

    def batch_clear_path_source(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["path_source"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared path_source")

    def _batch_resize_step(self, multiplier: int = 1) -> float:
        base = float(self.batch_resize_step_spin.value()) if hasattr(self, "batch_resize_step_spin") else 1.0
        mult = float(max(1, multiplier))
        return max(1.0, base * mult)

    def batch_expand_selected(self, direction: str, multiplier: int = 1) -> None:
        step = self._batch_resize_step(multiplier=multiplier)
        grow_left = step if direction == "left" else 0.0
        grow_right = step if direction == "right" else 0.0
        grow_up = step if direction == "up" else 0.0
        grow_down = step if direction == "down" else 0.0
        self._batch_resize_selected_boxes(
            grow_left=grow_left,
            grow_right=grow_right,
            grow_up=grow_up,
            grow_down=grow_down,
            success_message=f"Grew selected boxes {direction}",
        )

    def _batch_resize_selected_boxes(
        self,
        *,
        grow_left: float = 0.0,
        grow_right: float = 0.0,
        grow_up: float = 0.0,
        grow_down: float = 0.0,
        success_message: str,
    ) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            self.statusBar().showMessage("No selected bboxes for batch resize.", 2500)
            return
        if self.scene.image_rect.isNull():
            self.statusBar().showMessage("Image bounds are unavailable.", 2500)
            return

        bounds = self.scene.image_rect
        changed_count = 0
        eps = 0.01

        for item in selected_items:
            original = item_scene_rect(item)
            updated = QRectF(original)
            if grow_left:
                updated.setLeft(updated.left() - grow_left)
            if grow_right:
                updated.setRight(updated.right() + grow_right)
            if grow_up:
                updated.setTop(updated.top() - grow_up)
            if grow_down:
                updated.setBottom(updated.bottom() + grow_down)

            updated.setLeft(max(bounds.left(), updated.left()))
            updated.setTop(max(bounds.top(), updated.top()))
            updated.setRight(min(bounds.right(), updated.right()))
            updated.setBottom(min(bounds.bottom(), updated.bottom()))

            if (
                abs(updated.x() - original.x()) <= eps
                and abs(updated.y() - original.y()) <= eps
                and abs(updated.width() - original.width()) <= eps
                and abs(updated.height() - original.height()) <= eps
            ):
                continue

            pos = item.pos()
            item.setRect(
                QRectF(
                    updated.left() - pos.x(),
                    updated.top() - pos.y(),
                    updated.width(),
                    updated.height(),
                )
            )
            changed_count += 1

        if changed_count == 0:
            self.statusBar().showMessage("Batch resize made no changes.", 2500)
            return

        self.refresh_facts_list()
        self._record_history_snapshot()
        self.statusBar().showMessage(f"{success_message} ({changed_count} bbox(es)).", 3500)

    def _set_equation_reference_preview_items(self, items: List[AnnotRectItem]) -> None:
        current_ids = {id(item) for item in items if self._is_alive_fact_item(item)}
        for item in self._equation_reference_preview_items:
            if self._is_alive_fact_item(item):
                item.set_equation_reference_preview(id(item) in current_ids)
        next_items: list[AnnotRectItem] = []
        for item in items:
            if not self._is_alive_fact_item(item):
                continue
            item.set_equation_reference_preview(True)
            next_items.append(item)
        self._equation_reference_preview_items = next_items

    def _clear_equation_candidate(self) -> None:
        self._equation_target_item = None
        self._equation_candidate_text = None
        self._equation_candidate_fact_text = None
        self._equation_candidate_result_text = None
        self._equation_candidate_invalid_values = []
        self._equation_candidate_terms = []
        self._set_equation_reference_preview_items([])
        if hasattr(self, "fact_equation_edit"):
            self._refresh_equation_panel()

    def _rebuild_equation_candidate_from_terms(self) -> None:
        candidate_text, result_text, fact_candidate_text, invalid_values, structured_terms = _build_equation_candidate_from_facts(
            [
                {
                    "fact_num": term.get("fact_num"),
                    "value": term.get("raw_value"),
                    "natural_sign": term.get("natural_sign"),
                    "operator": term.get("operator"),
                }
                for term in self._equation_candidate_terms
                if isinstance(term, dict)
            ]
        )
        self._equation_candidate_text = candidate_text
        self._equation_candidate_fact_text = fact_candidate_text
        self._equation_candidate_result_text = result_text
        self._equation_candidate_invalid_values = invalid_values
        self._equation_candidate_terms = structured_terms

    def _update_equation_term_controls(self) -> None:
        if not hasattr(self, "fact_equation_terms_list"):
            return
        has_selection = bool(self.fact_equation_terms_list.selectedItems())
        visible = self.fact_equation_terms_list.isVisible()
        self.equation_mark_add_btn.setEnabled(visible and has_selection)
        self.equation_mark_subtract_btn.setEnabled(visible and has_selection)

    def _refresh_equation_variants_list(self, equation_bundles: list[dict[str, Any]], *, enabled: bool) -> None:
        if not hasattr(self, "fact_equation_variants_list"):
            return
        self._is_loading_equation_variants = True
        try:
            self.fact_equation_variants_list.clear()
            for index, bundle in enumerate(equation_bundles):
                equation_text = str(bundle.get("equation") or "").strip()
                if not equation_text:
                    continue
                preview = (equation_text[:46] + "...") if len(equation_text) > 49 else equation_text
                item = QListWidgetItem(f"{index + 1}. {preview}")
                item.setToolTip(equation_text)
                item.setData(EQUATION_VARIANT_INDEX_ROLE, index)
                item.setData(EQUATION_VARIANT_SIGNATURE_ROLE, _equation_bundle_signature(bundle))
                self.fact_equation_variants_list.addItem(item)
            if self.fact_equation_variants_list.count() > 0:
                self.fact_equation_variants_list.setCurrentRow(0)
                current_item = self.fact_equation_variants_list.item(0)
                if current_item is not None:
                    current_item.setSelected(True)
        finally:
            self._is_loading_equation_variants = False
        self.fact_equation_variants_list.setEnabled(enabled and self.fact_equation_variants_list.count() > 0)
        self._update_equation_variant_controls()

    def _selected_equation_variant_index(self) -> int | None:
        if not hasattr(self, "fact_equation_variants_list"):
            return None
        item = self.fact_equation_variants_list.currentItem()
        if item is None:
            return None
        index_raw = item.data(EQUATION_VARIANT_INDEX_ROLE)
        if not isinstance(index_raw, int):
            return None
        return index_raw

    def _selected_equation_variant_indices(self) -> list[int]:
        if not hasattr(self, "fact_equation_variants_list"):
            return []
        indices: list[int] = []
        for item in self.fact_equation_variants_list.selectedItems():
            index_raw = item.data(EQUATION_VARIANT_INDEX_ROLE)
            if isinstance(index_raw, int):
                indices.append(index_raw)
        if not indices:
            current_index = self._selected_equation_variant_index()
            if current_index is not None:
                indices.append(current_index)
        return sorted(set(indices))

    def _selected_equation_variant_signatures(self) -> set[tuple[str, str]]:
        if not hasattr(self, "fact_equation_variants_list"):
            return set()
        signatures: set[tuple[str, str]] = set()
        for item in self.fact_equation_variants_list.selectedItems():
            raw_signature = item.data(EQUATION_VARIANT_SIGNATURE_ROLE)
            if isinstance(raw_signature, tuple) and len(raw_signature) == 2:
                signatures.add((str(raw_signature[0] or ""), str(raw_signature[1] or "")))
        if signatures:
            return signatures
        current_item = self.fact_equation_variants_list.currentItem()
        if current_item is None:
            return set()
        raw_signature = current_item.data(EQUATION_VARIANT_SIGNATURE_ROLE)
        if isinstance(raw_signature, tuple) and len(raw_signature) == 2:
            return {(str(raw_signature[0] or ""), str(raw_signature[1] or ""))}
        return set()

    def _update_equation_variant_controls(self) -> None:
        if not hasattr(self, "fact_equation_variants_list"):
            return
        selected_indices = self._selected_equation_variant_indices()
        has_selection = bool(selected_indices)
        has_single_selection = len(selected_indices) == 1
        count = self.fact_equation_variants_list.count()
        self.equation_delete_variant_btn.setEnabled(has_selection)
        selected_row = self.fact_equation_variants_list.currentRow()
        self.equation_variant_up_btn.setEnabled(has_single_selection and selected_row > 0)
        self.equation_variant_down_btn.setEnabled(has_single_selection and 0 <= selected_row < count - 1)

    def _focus_equation_panel(self) -> None:
        if hasattr(self, "fact_equation_variants_list") and self.fact_equation_variants_list.isEnabled():
            self.fact_equation_variants_list.setFocus(Qt.ShortcutFocusReason)
            return
        if hasattr(self, "fact_equation_edit") and self.fact_equation_edit.isEnabled():
            self.fact_equation_edit.setFocus(Qt.ShortcutFocusReason)

    def move_selected_equation_variant(self, direction: int) -> None:
        if direction not in {-1, 1}:
            return
        selected_items = self._selected_fact_items()
        if len(selected_items) != 1:
            return
        target_item = selected_items[0]
        if not self._is_alive_fact_item(target_item):
            return
        selected_index = self._selected_equation_variant_index()
        if selected_index is None:
            return

        current = normalize_fact_data(target_item.fact_data)
        equation_bundles = _equation_bundles_from_fact_payload(current)
        if not equation_bundles:
            return
        next_index = selected_index + direction
        if not (0 <= next_index < len(equation_bundles)):
            return
        reordered = list(equation_bundles)
        reordered[selected_index], reordered[next_index] = reordered[next_index], reordered[selected_index]
        updated = _fact_payload_with_active_equation_bundle(current, reordered, active_index=0)
        if updated == current:
            return
        target_item.fact_data = updated
        self._clear_equation_candidate()
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._focus_equation_panel()
        self.statusBar().showMessage("Reordered saved equations.", 2300)

    def delete_selected_equation_variants_from_selected_fact(self) -> None:
        selected_items = self._selected_fact_items()
        if len(selected_items) != 1:
            self.statusBar().showMessage("Select one fact to delete saved equations.", 2200)
            return
        target_item = selected_items[0]
        if not self._is_alive_fact_item(target_item):
            return
        selected_signatures = self._selected_equation_variant_signatures()
        if not selected_signatures:
            self.statusBar().showMessage("Select saved equation(s) to delete.", 2200)
            return
        current = normalize_fact_data(target_item.fact_data)
        equation_bundles = _equation_bundles_from_fact_payload(current)
        if not equation_bundles:
            self.statusBar().showMessage("No saved equations to delete.", 2200)
            return
        remaining = [bundle for bundle in equation_bundles if _equation_bundle_signature(bundle) not in selected_signatures]
        deleted_count = len(equation_bundles) - len(remaining)
        if deleted_count <= 0:
            self.statusBar().showMessage("No saved equations to delete.", 2200)
            return
        if remaining:
            updated = _fact_payload_with_active_equation_bundle(current, remaining, active_index=0)
        else:
            updated = normalize_fact_data(
                {
                    **current,
                    "equations": None,
                }
            )
        if updated == current:
            self.statusBar().showMessage("Selected equation was not deleted.", 2200)
            return
        target_item.fact_data = updated
        self._clear_equation_candidate()
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._focus_equation_panel()
        if deleted_count == 1:
            self.statusBar().showMessage("Deleted 1 selected saved equation.", 2400)
        else:
            self.statusBar().showMessage(f"Deleted {deleted_count} selected saved equations.", 2400)

    def _on_equation_variant_item_clicked(self, item: QListWidgetItem) -> None:
        if self._is_loading_equation_variants or item is None:
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            self._update_equation_variant_controls()
            return
        selected_items = self._selected_fact_items()
        if len(selected_items) != 1:
            return
        target_item = selected_items[0]
        if not self._is_alive_fact_item(target_item):
            return
        target_index = item.data(EQUATION_VARIANT_INDEX_ROLE)
        if not isinstance(target_index, int):
            return
        current = normalize_fact_data(target_item.fact_data)
        equation_bundles = _equation_bundles_from_fact_payload(current)
        if not equation_bundles:
            return
        updated = _fact_payload_with_active_equation_bundle(current, equation_bundles, active_index=target_index)
        if updated == current:
            return
        target_item.fact_data = updated
        self._clear_equation_candidate()
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._focus_equation_panel()
        self.statusBar().showMessage("Switched active saved equation for selected fact.", 2500)

    def _refresh_equation_term_list(self) -> None:
        if not hasattr(self, "fact_equation_terms_list"):
            return
        pending = self._equation_target_item is not None and bool(self._equation_candidate_terms)
        show_controls = pending and any(
            isinstance(term, dict) and term.get("fact_num") is not None
            for term in self._equation_candidate_terms
        )
        self._is_loading_equation_terms = True
        try:
            self.fact_equation_terms_list.clear()
            for index, term in enumerate(self._equation_candidate_terms):
                if not isinstance(term, dict):
                    continue
                fact_num = term.get("fact_num")
                if fact_num is None:
                    continue
                operator = _normalize_equation_operator(term.get("operator"))
                raw_value = str(term.get("raw_value") or "").strip() or "<empty>"
                status = str(term.get("status") or "")
                if status == "invalid":
                    label = f"{operator} f{fact_num} = {raw_value} (ignored)"
                elif status == "normalized_dash":
                    label = f"{operator} f{fact_num} = {raw_value} (as 0)"
                else:
                    label = f"{operator} f{fact_num} = {raw_value}"
                item = QListWidgetItem(label)
                item.setData(EQUATION_TERM_INDEX_ROLE, index)
                self.fact_equation_terms_list.addItem(item)
        finally:
            self._is_loading_equation_terms = False
        self.fact_equation_terms_list.setVisible(show_controls)
        self.equation_mark_add_btn.setVisible(show_controls)
        self.equation_mark_subtract_btn.setVisible(show_controls)
        self._update_equation_term_controls()

    def add_equation_variant_to_selected_fact(self) -> None:
        target_item = self._equation_target_item
        if not self._is_alive_fact_item(target_item):
            self._clear_equation_candidate()
            self.statusBar().showMessage("Select one target bbox before adding an equation.", 2500)
            return
        if self._equation_candidate_text is None or self._equation_candidate_result_text is None:
            self.statusBar().showMessage("No calculable equation preview to add.", 2500)
            return

        selected_items = self._selected_fact_items()
        if len(selected_items) != 1 or selected_items[0] is not target_item:
            self._clear_equation_candidate()
            self.statusBar().showMessage("Re-select the target bbox before adding the equation.", 3000)
            return

        current = normalize_fact_data(target_item.fact_data)
        has_fact_references = bool(str(self._equation_candidate_fact_text or "").strip())
        new_bundle = _normalize_equation_bundle_payload(
            {
                "equation": self._equation_candidate_text,
                "fact_equation": self._equation_candidate_fact_text,
            }
        )
        if new_bundle is None:
            self.statusBar().showMessage("Equation preview is empty and was not added.", 2500)
            return
        existing_bundles = _equation_bundles_from_fact_payload(current)
        existing_index = next(
            (
                idx
                for idx, bundle in enumerate(existing_bundles)
                if _equation_bundle_signature(bundle) == _equation_bundle_signature(new_bundle)
            ),
            None,
        )
        if existing_index is not None:
            updated = _fact_payload_with_active_equation_bundle(current, existing_bundles, active_index=existing_index)
            message = "Equation already exists; switched to it."
        else:
            updated = _fact_payload_with_active_equation_bundle(
                {**current, "row_role": "total" if has_fact_references else current.get("row_role")},
                [new_bundle, *existing_bundles],
                active_index=0,
            )
            message = "Added equation and set it as active."
        if updated == current:
            self.statusBar().showMessage("Equation is already active for the selected fact.", 2500)
            return

        target_item.fact_data = updated
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._clear_equation_candidate()
        self._focus_equation_panel()
        self.statusBar().showMessage(message, 2600)

    def _apply_operator_to_selected_equation_terms(self, operator: str) -> None:
        if self._is_loading_equation_terms:
            return
        normalized_operator = _normalize_equation_operator(operator)
        selected_items = self.fact_equation_terms_list.selectedItems() if hasattr(self, "fact_equation_terms_list") else []
        if not selected_items:
            return
        changed = False
        for item in selected_items:
            index_raw = item.data(EQUATION_TERM_INDEX_ROLE)
            if not isinstance(index_raw, int) or not (0 <= index_raw < len(self._equation_candidate_terms)):
                continue
            term = self._equation_candidate_terms[index_raw]
            if not isinstance(term, dict):
                continue
            if _normalize_equation_operator(term.get("operator")) == normalized_operator:
                continue
            term["operator"] = normalized_operator
            changed = True
        if not changed:
            return
        self._rebuild_equation_candidate_from_terms()
        self._refresh_equation_panel()

    def _on_equation_term_item_double_clicked(self, item: QListWidgetItem) -> None:
        if self._is_loading_equation_terms or item is None:
            return
        index_raw = item.data(EQUATION_TERM_INDEX_ROLE)
        if not isinstance(index_raw, int) or not (0 <= index_raw < len(self._equation_candidate_terms)):
            return
        term = self._equation_candidate_terms[index_raw]
        if not isinstance(term, dict):
            return
        next_operator = "-" if _normalize_equation_operator(term.get("operator")) == "+" else "+"
        term["operator"] = next_operator
        self._rebuild_equation_candidate_from_terms()
        self._refresh_equation_panel()

    def _set_equation_result_display(self, text: str, *, tone: str = "neutral") -> None:
        self.fact_equation_result_label.setText(text)
        if tone == "danger":
            self.fact_equation_result_label.setStyleSheet("color: #b7791f; font-weight: 600;")
        elif tone == "ok":
            self.fact_equation_result_label.setStyleSheet("color: #027a48; font-weight: 600;")
        else:
            self.fact_equation_result_label.setStyleSheet("")

    @staticmethod
    def _duration_type_requires_recurrence(value: Any) -> bool:
        text = str(value or "").strip().lower()
        return text in {"recurrent", "recurring"}

    def _refresh_recurring_period_visibility(self) -> None:
        if not hasattr(self, "fact_recurring_period_block"):
            return
        selected_count = len(self._selected_fact_items()) if hasattr(self, "scene") else 0
        show_recurrence = (
            selected_count == 1
            and self._duration_type_requires_recurrence(self.fact_duration_type_combo.currentText())
            and self.fact_duration_type_combo.isEnabled()
        )
        self.fact_recurring_period_block.setVisible(show_recurrence)
        self.fact_recurring_period_combo.setEnabled(show_recurrence)

    def _refresh_equation_panel(self) -> None:
        if not hasattr(self, "fact_equation_edit"):
            return

        selected_items = self._selected_fact_items()
        selection_has_saved_equation = any(
            bool(_equation_bundles_from_fact_payload(normalize_fact_data(item.fact_data)))
            for item in selected_items
            if self._is_alive_fact_item(item)
        )
        single_item = selected_items[0] if len(selected_items) == 1 else None
        saved_equation = ""
        saved_fact_equation = ""
        equation_bundles: list[dict[str, Any]] = []
        if single_item is not None:
            saved_fact = normalize_fact_data(single_item.fact_data)
            equation_bundles = _equation_bundles_from_fact_payload(saved_fact)
            active_bundle = equation_bundles[0] if equation_bundles else None
            saved_equation = str((active_bundle or {}).get("equation") or "")
            saved_fact_equation = str((active_bundle or {}).get("fact_equation") or "")
            target_value = saved_fact.get("value")
            target_natural_sign = saved_fact.get("natural_sign")
        else:
            target_value = None
            target_natural_sign = None

        self.fact_equation_edit.setEnabled(single_item is not None)
        self.fact_equation_result_label.setEnabled(single_item is not None)
        self.fact_equation_status_label.setEnabled(single_item is not None)
        self.fact_equation_variants_list.setEnabled(single_item is not None)
        self.equation_add_variant_btn.setEnabled(False)
        self.equation_delete_variant_btn.setEnabled(False)
        self.equation_variant_up_btn.setEnabled(False)
        self.equation_variant_down_btn.setEnabled(False)
        self.apply_equation_btn.setEnabled(False)
        self.clear_equation_btn.setEnabled(False)
        self._refresh_equation_variants_list(equation_bundles, enabled=(single_item is not None))
        self._refresh_equation_term_list()

        if single_item is None:
            self.fact_equation_edit.setText("")
            self.fact_equation_status_label.setText("")
            self._set_equation_result_display("-", tone="neutral")
            self.clear_equation_btn.setEnabled(selection_has_saved_equation)
            return

        pending_for_selection = self._equation_target_item is single_item and self._equation_candidate_text is not None
        has_saved_equation = bool(saved_equation or saved_fact_equation or equation_bundles)
        if pending_for_selection:
            self.fact_equation_edit.setText(str(self._equation_candidate_text or "").strip())
            if self._equation_candidate_result_text is not None:
                result_tone, comparison_message = _equation_result_match_state(
                    self._equation_candidate_result_text,
                    target_value,
                    target_natural_sign,
                )
                self._set_equation_result_display(self._equation_candidate_result_text, tone=result_tone)
            else:
                comparison_message = "Cannot calculate preview."
                self._set_equation_result_display("cannot calculate", tone="danger")

            if self._equation_candidate_invalid_values:
                preview = ", ".join(self._equation_candidate_invalid_values[:3])
                if len(self._equation_candidate_invalid_values) > 3:
                    preview = f"{preview}, +{len(self._equation_candidate_invalid_values) - 3} more"
                self.fact_equation_status_label.setText(
                    f"Ignored {len(self._equation_candidate_invalid_values)} invalid value(s): {preview}. {comparison_message}"
                )
            else:
                self.fact_equation_status_label.setText(
                    "Candidate equation preview. "
                    f"{comparison_message} Selected children default to additive; use Mark - only for exceptional subtractive cases, then click + to add a new equation variant or Apply Equation to overwrite the active one."
                )

            self.apply_equation_btn.setEnabled(
                self._equation_candidate_result_text is not None
                and bool(self._equation_candidate_text)
                and (
                    self._equation_candidate_text != saved_equation
                    or (self._equation_candidate_fact_text or "") != saved_fact_equation
                )
            )
            self.equation_add_variant_btn.setEnabled(
                self._equation_candidate_result_text is not None and bool(self._equation_candidate_text)
            )
            self.clear_equation_btn.setEnabled(has_saved_equation)
            return

        self.fact_equation_edit.setText(saved_equation)
        if not saved_equation:
            self.fact_equation_status_label.setText(
                "Hold Alt and drag on the page to build an equation preview. Click + to add another equation or Apply Equation to overwrite the active one."
            )
            self._set_equation_result_display("-", tone="neutral")
            self.clear_equation_btn.setEnabled(bool(equation_bundles))
            return

        preview = _evaluate_equation_string(saved_equation)
        if preview is None:
            self.fact_equation_status_label.setText("Saved equation cannot be calculated.")
            self._set_equation_result_display("cannot calculate", tone="danger")
            self.clear_equation_btn.setEnabled(bool(equation_bundles))
            return

        result_tone, comparison_message = _equation_result_match_state(
            preview,
            target_value,
            target_natural_sign,
        )
        self.fact_equation_status_label.setText(f"Saved equation preview. {comparison_message}")
        self._set_equation_result_display(preview, tone=result_tone)
        self.clear_equation_btn.setEnabled(True)

    def _set_fact_editor_enabled(self, enabled: bool, *, multi_select: bool = False) -> None:
        self.fact_value_edit.setEnabled(enabled)
        self.fact_equation_edit.setEnabled(enabled and not multi_select)
        self.fact_note_edit.setEnabled(enabled)
        self.fact_note_name_edit.setEnabled(enabled)
        self.fact_is_beur_combo.setEnabled(enabled)
        self.fact_beur_num_edit.setEnabled(enabled)
        self.fact_refference_edit.setEnabled(enabled)
        self.fact_date_edit.setEnabled(enabled)
        self.fact_period_type_combo.setEnabled(enabled)
        self.fact_duration_type_combo.setEnabled(enabled)
        self.fact_recurring_period_combo.setEnabled(enabled)
        self.fact_period_start_edit.setEnabled(enabled)
        self.fact_period_end_edit.setEnabled(enabled)
        self.fact_currency_combo.setEnabled(enabled)
        self.fact_scale_combo.setEnabled(enabled)
        self.fact_value_type_combo.setEnabled(enabled)
        self.fact_value_context_combo.setEnabled(enabled)
        self.fact_row_role_combo.setEnabled(enabled)
        self.fact_path_source_combo.setEnabled(enabled)
        self.fact_path_list.setEnabled(enabled)
        self.fact_natural_sign_label.setEnabled(enabled)
        self.fact_equation_result_label.setEnabled(enabled and not multi_select)
        self.fact_equation_status_label.setEnabled(enabled and not multi_select)
        self.fact_equation_variants_list.setEnabled(enabled and not multi_select)
        self.equation_add_variant_btn.setEnabled(False)
        self.equation_delete_variant_btn.setEnabled(False)
        self.equation_variant_up_btn.setEnabled(False)
        self.equation_variant_down_btn.setEnabled(False)
        self.apply_equation_btn.setEnabled(False)
        self.clear_equation_btn.setEnabled(False)
        self._set_path_list_editable(enabled and not multi_select)
        self.dup_fact_btn.setEnabled(enabled and not multi_select)
        self.del_fact_btn.setEnabled(enabled)
        if not enabled or multi_select:
            self._set_equation_result_display("-", tone="neutral")
            self.fact_equation_status_label.setText("" if not enabled else "Equation preview is available for a single selected fact.")
            self._refresh_equation_variants_list([], enabled=False)
        self._refresh_recurring_period_visibility()
        self._update_path_controls()

    def _reset_fact_editor_placeholders(self) -> None:
        for edit in (
            self.fact_value_edit,
            self.fact_equation_edit,
            self.fact_note_edit,
            self.fact_note_name_edit,
            self.fact_beur_num_edit,
            self.fact_refference_edit,
            self.fact_date_edit,
            self.fact_period_start_edit,
            self.fact_period_end_edit,
        ):
            edit.setPlaceholderText("")
            edit.setModified(False)
        for combo in (
            self.fact_is_beur_combo,
            self.fact_period_type_combo,
            self.fact_duration_type_combo,
            self.fact_recurring_period_combo,
            self.fact_currency_combo,
            self.fact_scale_combo,
            self.fact_value_type_combo,
            self.fact_value_context_combo,
            self.fact_row_role_combo,
            self.fact_path_source_combo,
        ):
            if hasattr(combo, "setPlaceholderText"):
                combo.setPlaceholderText("")
        self.fact_path_list.setToolTip("")

    def _clear_fact_editor(self) -> None:
        self._is_loading_fact_editor = True
        try:
            self.fact_editor_box.setTitle("Selected Fact")
            self._reset_fact_editor_placeholders()
            self.fact_bbox_label.setText("-")
            self.fact_value_edit.setText("")
            self.fact_equation_edit.setText("")
            self.fact_note_edit.setText("")
            self.fact_note_name_edit.setText("")
            self.fact_is_beur_combo.setCurrentIndex(0)
            self.fact_beur_num_edit.setText("")
            self.fact_refference_edit.setText("")
            self.fact_date_edit.setText("")
            self.fact_period_type_combo.setCurrentIndex(0)
            self.fact_duration_type_combo.setCurrentIndex(0)
            self.fact_recurring_period_combo.setCurrentIndex(0)
            self.fact_period_start_edit.setText("")
            self.fact_period_end_edit.setText("")
            self.fact_currency_combo.setCurrentIndex(0)
            self.fact_scale_combo.setCurrentIndex(0)
            self.fact_value_type_combo.setCurrentIndex(0)
            self.fact_value_context_combo.setCurrentIndex(0)
            self.fact_row_role_combo.setCurrentIndex(0)
            self.fact_path_source_combo.setCurrentIndex(0)
            self.fact_path_list.clear()
            self.fact_natural_sign_label.setText("-")
            self.fact_equation_status_label.setText("")
            self._set_equation_result_display("-", tone="neutral")
            self._refresh_equation_variants_list([], enabled=False)
            self.equation_add_variant_btn.setEnabled(False)
            self.equation_delete_variant_btn.setEnabled(False)
            self.equation_variant_up_btn.setEnabled(False)
            self.equation_variant_down_btn.setEnabled(False)
        finally:
            self._is_loading_fact_editor = False
        self._update_path_controls()

    def _populate_fact_editor(self, item: AnnotRectItem) -> None:
        if not self._is_alive_fact_item(item):
            self._clear_fact_editor()
            return
        fact = normalize_fact_data(item.fact_data)
        rect = item_scene_rect(item)
        self._is_loading_fact_editor = True
        try:
            self.fact_editor_box.setTitle("Selected Fact")
            self._reset_fact_editor_placeholders()
            self.fact_bbox_label.setText(
                f"{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}"
            )
            self.fact_value_edit.setText(str(fact.get("value", "")))
            active_bundle = _active_equation_bundle_from_fact_payload(fact) or {}
            self.fact_equation_edit.setText(str(active_bundle.get("equation") or ""))
            self.fact_note_edit.setText(str(fact.get("comment_ref") or ""))
            self.fact_note_name_edit.setText(str(fact.get("note_name") or ""))
            is_beur = bool(fact.get("note_flag"))
            is_beur_text = "true" if is_beur else "false"
            idx_is_beur = self.fact_is_beur_combo.findText(is_beur_text)
            self.fact_is_beur_combo.setCurrentIndex(max(0, idx_is_beur))
            self.fact_beur_num_edit.setText("" if fact.get("note_num") is None else str(fact.get("note_num")))
            self.fact_refference_edit.setText(str(fact.get("note_ref") or ""))
            self.fact_date_edit.setText(str(fact.get("date") or ""))
            period_type = str(fact.get("period_type") or "")
            idx_period_type = self.fact_period_type_combo.findText(period_type)
            self.fact_period_type_combo.setCurrentIndex(max(0, idx_period_type))
            duration_type = str(fact.get("duration_type") or "")
            idx_duration_type = self.fact_duration_type_combo.findText(duration_type)
            self.fact_duration_type_combo.setCurrentIndex(max(0, idx_duration_type))
            recurring_period = str(fact.get("recurring_period") or "")
            idx_recurring_period = self.fact_recurring_period_combo.findText(recurring_period)
            self.fact_recurring_period_combo.setCurrentIndex(max(0, idx_recurring_period))
            self.fact_period_start_edit.setText(str(fact.get("period_start") or ""))
            self.fact_period_end_edit.setText(str(fact.get("period_end") or ""))
            self.fact_path_list.clear()
            for path_index, path_level in enumerate(fact.get("path") or []):
                self.fact_path_list.addItem(self._make_path_item(str(path_level), path_index=path_index))
            if self.fact_path_list.count() > 0:
                self.fact_path_list.setCurrentRow(0)

            currency = str(fact.get("currency") or "")
            idx_currency = self.fact_currency_combo.findText(currency)
            self.fact_currency_combo.setCurrentIndex(max(0, idx_currency))

            scale = "" if fact.get("scale") is None else str(fact.get("scale"))
            idx_scale = self.fact_scale_combo.findText(scale)
            self.fact_scale_combo.setCurrentIndex(max(0, idx_scale))

            value_type = str(fact.get("value_type") or "")
            idx_value_type = self.fact_value_type_combo.findText(value_type)
            self.fact_value_type_combo.setCurrentIndex(max(0, idx_value_type))
            value_context = str(fact.get("value_context") or "")
            idx_value_context = self.fact_value_context_combo.findText(value_context)
            self.fact_value_context_combo.setCurrentIndex(max(0, idx_value_context))
            row_role = str(fact.get("row_role") or "")
            idx_row_role = self.fact_row_role_combo.findText(row_role)
            self.fact_row_role_combo.setCurrentIndex(max(0, idx_row_role))
            self.fact_natural_sign_label.setText(str(fact.get("natural_sign") or "-"))

            path_source = str(fact.get("path_source") or "")
            idx_path_source = self.fact_path_source_combo.findText(path_source)
            self.fact_path_source_combo.setCurrentIndex(max(0, idx_path_source))
        finally:
            self._is_loading_fact_editor = False
        self._update_path_controls()

    def _shared_fact_value(self, items: List[AnnotRectItem], key: str) -> tuple[Any, bool]:
        values = [normalize_fact_data(item.fact_data).get(key) for item in items]
        if not values:
            return None, False
        first = values[0]
        if all(value == first for value in values[1:]):
            return first, False
        return None, True

    def _set_multi_line_edit_value(self, edit: QLineEdit, key: str, items: List[AnnotRectItem]) -> None:
        value, mixed = self._shared_fact_value(items, key)
        if mixed:
            edit.clear()
            edit.setPlaceholderText(MULTI_VALUE_PLACEHOLDER)
        else:
            edit.setText(str(value or ""))
            edit.setPlaceholderText("")
        edit.setModified(False)

    def _set_multi_combo_value(
        self,
        combo: QComboBox,
        key: str,
        items: List[AnnotRectItem],
        *,
        formatter: Optional[Callable[[Any], str]] = None,
    ) -> None:
        value, mixed = self._shared_fact_value(items, key)
        if mixed:
            combo.setCurrentIndex(-1)
            if hasattr(combo, "setPlaceholderText"):
                combo.setPlaceholderText(MULTI_VALUE_PLACEHOLDER)
            return
        text = formatter(value) if formatter is not None else str(value or "")
        idx = combo.findText(text)
        combo.setCurrentIndex(max(-1, idx))
        if idx < 0:
            combo.setCurrentIndex(-1)
        if hasattr(combo, "setPlaceholderText"):
            combo.setPlaceholderText("")

    def _populate_multi_fact_editor(self, items: List[AnnotRectItem]) -> None:
        self._is_loading_fact_editor = True
        try:
            selected_count = len(items)
            self.fact_editor_box.setTitle(f"Selected Facts ({selected_count})")
            self._reset_fact_editor_placeholders()
            self.fact_bbox_label.setText(f"{selected_count} selected")
            self._set_multi_line_edit_value(self.fact_value_edit, "value", items)
            self.fact_equation_edit.clear()
            self._set_multi_line_edit_value(self.fact_note_edit, "comment_ref", items)
            self._set_multi_line_edit_value(self.fact_note_name_edit, "note_name", items)
            self._set_multi_combo_value(
                self.fact_is_beur_combo,
                "note_flag",
                items,
                formatter=lambda value: "true" if bool(value) else "false",
            )
            self._set_multi_line_edit_value(self.fact_beur_num_edit, "note_num", items)
            self._set_multi_line_edit_value(self.fact_refference_edit, "note_ref", items)
            self._set_multi_line_edit_value(self.fact_date_edit, "date", items)
            self._set_multi_combo_value(self.fact_period_type_combo, "period_type", items)
            self._set_multi_combo_value(self.fact_duration_type_combo, "duration_type", items)
            self._set_multi_combo_value(self.fact_recurring_period_combo, "recurring_period", items)
            self._set_multi_line_edit_value(self.fact_period_start_edit, "period_start", items)
            self._set_multi_line_edit_value(self.fact_period_end_edit, "period_end", items)
            self._set_multi_combo_value(self.fact_currency_combo, "currency", items)
            self._set_multi_combo_value(
                self.fact_scale_combo,
                "scale",
                items,
                formatter=lambda value: "" if value is None else str(value),
            )
            self._set_multi_combo_value(self.fact_value_type_combo, "value_type", items)
            self._set_multi_combo_value(self.fact_value_context_combo, "value_context", items)
            self._set_multi_combo_value(self.fact_row_role_combo, "row_role", items)
            self._set_multi_combo_value(self.fact_path_source_combo, "path_source", items)
            natural_sign_value, natural_sign_mixed = self._shared_fact_value(items, "natural_sign")
            self.fact_natural_sign_label.setText(
                MULTI_VALUE_PLACEHOLDER if natural_sign_mixed else str(natural_sign_value or "-")
            )

            path_value, path_mixed = self._shared_fact_value(items, "path")
            self.fact_path_list.clear()
            if not path_mixed:
                for path_index, path_level in enumerate(path_value or []):
                    self.fact_path_list.addItem(self._make_path_item(str(path_level), path_index=path_index))
                self._set_path_list_editable(True, structural=True)
                if self.fact_path_list.count() > 0:
                    self.fact_path_list.setCurrentRow(0)
            else:
                path_signatures = self._selected_path_signatures(items)
                prefix = _shared_path_prefix(path_signatures)
                shared_elements = _shared_path_elements(path_signatures)
                variant_operation_index = len(prefix) if path_signatures and any(len(path) > len(prefix) for path in path_signatures) else None

                if prefix:
                    for level_index, level in enumerate(prefix):
                        self.fact_path_list.addItem(
                            self._make_path_item(
                                str(level),
                                tone="shared",
                                editable=True,
                                tooltip=f"Shared path node across {selected_count} selected bboxes. Edit to rename everywhere.",
                                path_index=level_index,
                            )
                        )

                for level in shared_elements:
                    if level in prefix:
                        continue
                    self.fact_path_list.addItem(
                        self._make_path_item(
                            str(level),
                            tone="shared",
                            editable=False,
                            tooltip=(
                                f"Shared path element across all {selected_count} selected bboxes, "
                                "but not at a shared editable position."
                            ),
                        )
                    )

                if path_signatures:
                    if prefix and all(len(path) == len(prefix) + 1 for path in path_signatures):
                        leaf_values = list(dict.fromkeys(path[len(prefix)] for path in path_signatures))
                        preview = " | ".join(leaf_values[:3])
                        if len(leaf_values) > 3:
                            preview = f"{preview} | +{len(leaf_values) - 3} more"
                        self.fact_path_list.addItem(
                            self._make_path_item(
                                preview,
                                tone="variant",
                                editable=False,
                                tooltip="Differing final path node values:\n" + "\n".join(leaf_values),
                                path_operation_index=variant_operation_index,
                            )
                        )
                    else:
                        rendered_paths = [" > ".join(path) if path else "(empty path)" for path in path_signatures]
                        preview = "Different path tails"
                        self.fact_path_list.addItem(
                            self._make_path_item(
                                preview,
                                tone="variant",
                                editable=False,
                                tooltip="Selected bboxes have different path tails:\n" + "\n".join(rendered_paths),
                                path_operation_index=variant_operation_index,
                            )
                        )
                self._set_path_list_editable(bool(prefix), structural=False)
                if self.fact_path_list.count() > 0:
                    self.fact_path_list.setCurrentRow(0)
            self.fact_path_list.setToolTip(
                (
                    "Shared path nodes and elements are highlighted in green. "
                    "Only shared leading nodes are editable across selected bboxes. "
                    "Diverging path nodes are highlighted in orange."
                )
                if path_mixed and self._path_list_editable
                else (
                    "Selected bboxes do not share an editable path prefix. Shared non-prefix elements are shown for reference; use Batch Edit to change path structure."
                    if path_mixed
                    else "Selected bboxes share the same path. Edits here apply to every selected bbox."
                )
            )
        finally:
            self._is_loading_fact_editor = False
        self._update_path_controls()

    def _fact_data_from_editor(self) -> Dict[str, Any]:
        path_parts: List[str] = []
        for idx in range(self.fact_path_list.count()):
            item = self.fact_path_list.item(idx)
            if item is None:
                continue
            text = item.text().strip()
            if text:
                path_parts.append(text)
        scale_text = self.fact_scale_combo.currentText().strip()
        is_beur_text = self.fact_is_beur_combo.currentText().strip().lower()
        is_beur_value = is_beur_text == "true"
        payload: Dict[str, Any] = {
            "value": self.fact_value_edit.text().strip(),
            "comment_ref": self.fact_note_edit.text().strip() or None,
            "note_name": self.fact_note_name_edit.text().strip() or None,
            "note_flag": is_beur_value,
            "note_num": int(self.fact_beur_num_edit.text().strip()) if self.fact_beur_num_edit.text().strip() else None,
            "note_ref": self.fact_refference_edit.text().strip() or None,
            "date": self.fact_date_edit.text().strip() or None,
            "period_type": self.fact_period_type_combo.currentText().strip() or None,
            "period_start": self.fact_period_start_edit.text().strip() or None,
            "period_end": self.fact_period_end_edit.text().strip() or None,
            "duration_type": self.fact_duration_type_combo.currentText().strip() or None,
            "recurring_period": self.fact_recurring_period_combo.currentText().strip() or None,
            "path": path_parts,
            "path_source": self.fact_path_source_combo.currentText().strip() or None,
            "currency": self.fact_currency_combo.currentText().strip() or None,
            "scale": int(scale_text) if scale_text else None,
            "value_type": self.fact_value_type_combo.currentText().strip() or None,
            "value_context": self.fact_value_context_combo.currentText().strip() or None,
            "row_role": self.fact_row_role_combo.currentText().strip() or None,
        }
        return normalize_fact_data(payload)

    def _is_alive_fact_item(self, item: Optional[AnnotRectItem]) -> bool:
        if not isinstance(item, AnnotRectItem):
            return False
        try:
            if sip.isdeleted(item):
                return False
            return item.scene() is self.scene
        except RuntimeError:
            return False

    def _sync_fact_editor_from_selection(self) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            self.batch_box.setVisible(False)
            self._clear_fact_editor()
            self._set_fact_editor_enabled(False)
            self.batch_toggle_btn.setText("Show Batch Edit")
            self._refresh_equation_panel()
            self._refresh_recurring_period_visibility()
            return
        if len(selected_items) == 1:
            self._set_fact_editor_enabled(True, multi_select=False)
            self._populate_fact_editor(selected_items[0])
            self.batch_toggle_btn.setText("Hide Batch Edit" if self.batch_box.isVisible() else "Show Batch Edit")
            self._refresh_equation_panel()
            self._refresh_recurring_period_visibility()
            return
        self._set_fact_editor_enabled(True, multi_select=True)
        self._populate_multi_fact_editor(selected_items)
        self.batch_toggle_btn.setText("Hide Batch Edit" if self.batch_box.isVisible() else "Show Batch Edit")
        self._refresh_equation_panel()
        self._refresh_recurring_period_visibility()

    def _apply_fact_field_to_selected_items(
        self,
        field_name: str,
        value: Any,
        *,
        widget: Optional[QLineEdit] = None,
    ) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            return
        if widget is not None and not widget.isModified():
            return
        changed = False
        for item in selected_items:
            current = normalize_fact_data(item.fact_data)
            updated = normalize_fact_data({**current, field_name: value})
            if updated == current:
                continue
            item.fact_data = updated
            changed = True
        if widget is not None:
            widget.setModified(False)
        if not changed:
            return
        self.refresh_facts_list()
        self._record_history_snapshot()

    def _apply_fact_path_to_selected_items(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_editable:
            return
        selected_items = self._selected_fact_items()
        if not selected_items:
            return
        if not self._path_list_structure_editable:
            shared_updates: list[tuple[int, str]] = []
            for idx in range(self.fact_path_list.count()):
                item = self.fact_path_list.item(idx)
                if item is None:
                    continue
                path_index_raw = item.data(PATH_LEVEL_INDEX_ROLE)
                if path_index_raw in (None, ""):
                    continue
                text = item.text().strip()
                if not text:
                    self.statusBar().showMessage(
                        "Shared path levels cannot be empty. Use Batch Edit to remove levels.",
                        3500,
                    )
                    self._sync_fact_editor_from_selection()
                    return
                shared_updates.append((int(path_index_raw), text))
            if not shared_updates:
                return
            changed = False
            for item in selected_items:
                current = normalize_fact_data(item.fact_data)
                path_value = [str(part).strip() for part in (current.get("path") or []) if str(part).strip()]
                updated_path = list(path_value)
                for path_index, level_text in shared_updates:
                    if 0 <= path_index < len(updated_path):
                        updated_path[path_index] = level_text
                updated = normalize_fact_data({**current, "path": updated_path})
                if updated == current:
                    continue
                item.fact_data = updated
                changed = True
            if not changed:
                return
            self.refresh_facts_list()
            self._record_history_snapshot()
            return
        updated_fact = self._fact_data_from_editor()
        path_value = updated_fact.get("path") or []
        changed = False
        for item in selected_items:
            current = normalize_fact_data(item.fact_data)
            updated = normalize_fact_data({**current, "path": path_value})
            if updated == current:
                continue
            item.fact_data = updated
            changed = True
        if not changed:
            return
        self.refresh_facts_list()
        self._record_history_snapshot()

    def _on_fact_editor_field_edited(self, field_name: str) -> None:
        if self._is_loading_fact_editor or self._syncing_selection:
            return
        if not self._selected_fact_items():
            return
        if field_name == "value":
            self._apply_fact_field_to_selected_items("value", self.fact_value_edit.text().strip(), widget=self.fact_value_edit)
            return
        if field_name == "comment_ref":
            self._apply_fact_field_to_selected_items(
                "comment_ref",
                self.fact_note_edit.text().strip() or None,
                widget=self.fact_note_edit,
            )
            return
        if field_name == "note_name":
            self._apply_fact_field_to_selected_items("note_name", self.fact_note_name_edit.text().strip() or None, widget=self.fact_note_name_edit)
            return
        if field_name == "note_flag":
            self._apply_fact_field_to_selected_items(
                "note_flag",
                self.fact_is_beur_combo.currentText().strip().lower() == "true",
            )
            return
        if field_name == "note_num":
            note_num_text = self.fact_beur_num_edit.text().strip()
            self._apply_fact_field_to_selected_items(
                "note_num",
                int(note_num_text) if note_num_text else None,
                widget=self.fact_beur_num_edit,
            )
            return
        if field_name == "note_ref":
            self._apply_fact_field_to_selected_items(
                "note_ref",
                self.fact_refference_edit.text().strip() or None,
                widget=self.fact_refference_edit,
            )
            return
        if field_name == "date":
            self._apply_fact_field_to_selected_items("date", self.fact_date_edit.text().strip() or None, widget=self.fact_date_edit)
            self._sync_fact_editor_from_selection()
            return
        if field_name == "period_type":
            self._apply_fact_field_to_selected_items("period_type", self.fact_period_type_combo.currentText().strip() or None)
            self._sync_fact_editor_from_selection()
            return
        if field_name == "duration_type":
            self._apply_fact_field_to_selected_items("duration_type", self.fact_duration_type_combo.currentText().strip() or None)
            self._sync_fact_editor_from_selection()
            return
        if field_name == "recurring_period":
            self._apply_fact_field_to_selected_items("recurring_period", self.fact_recurring_period_combo.currentText().strip() or None)
            return
        if field_name == "period_start":
            self._apply_fact_field_to_selected_items("period_start", self.fact_period_start_edit.text().strip() or None, widget=self.fact_period_start_edit)
            return
        if field_name == "period_end":
            self._apply_fact_field_to_selected_items("period_end", self.fact_period_end_edit.text().strip() or None, widget=self.fact_period_end_edit)
            return
        if field_name == "currency":
            self._apply_fact_field_to_selected_items("currency", self.fact_currency_combo.currentText().strip() or None)
            return
        if field_name == "scale":
            scale_text = self.fact_scale_combo.currentText().strip()
            self._apply_fact_field_to_selected_items("scale", int(scale_text) if scale_text else None)
            return
        if field_name == "value_type":
            self._apply_fact_field_to_selected_items("value_type", self.fact_value_type_combo.currentText().strip() or None)
            return
        if field_name == "value_context":
            self._apply_fact_field_to_selected_items("value_context", self.fact_value_context_combo.currentText().strip() or None)
            return
        if field_name == "row_role":
            self._apply_fact_field_to_selected_items(
                "row_role",
                self.fact_row_role_combo.currentText().strip() or None,
            )
            self._sync_fact_editor_from_selection()
            return
        if field_name == "path_source":
            self._apply_fact_field_to_selected_items("path_source", self.fact_path_source_combo.currentText().strip() or None)
            return
        if field_name == "path":
            self._apply_fact_path_to_selected_items()

    def _fit_view(self) -> None:
        if self._fitting_view:
            return
        scene_rect = self.scene.itemsBoundingRect()
        if scene_rect.isNull():
            return
        self._fitting_view = True
        self.view.resetTransform()
        self.view.fitInView(scene_rect, Qt.KeepAspectRatio)
        self._fitting_view = False

    def _fit_view_height(self) -> None:
        if self._fitting_view:
            return
        image_rect = self.scene.image_rect
        if image_rect.isNull():
            return
        viewport_h = float(self.view.viewport().height())
        if viewport_h <= 2.0:
            return
        scale = max(0.05, min(40.0, viewport_h / image_rect.height()))
        self._fitting_view = True
        try:
            self.view.resetTransform()
            self.view.scale(scale, scale)
            self.view.centerOn(image_rect.center())
        finally:
            self._fitting_view = False

    def _apply_pending_auto_fit(self) -> None:
        if not self._pending_auto_fit:
            return
        if not self.isVisible():
            return
        viewport = self.view.viewport().rect()
        if viewport.width() <= 2 or viewport.height() <= 2:
            return
        self._pending_auto_fit = False
        self._fit_view_height()

    def schedule_auto_fit_current_page(self) -> None:
        self._pending_auto_fit = True
        QTimer.singleShot(0, self._apply_pending_auto_fit)

    def _focus_fact_items_in_view(self, items: List[AnnotRectItem]) -> None:
        _ = items
        # Selection should remain visually explicit, but it should not auto-pan/zoom the page.
        return

    def _on_lens_toggled(self, enabled: bool) -> None:
        self.view.set_lens_enabled(enabled)
        if enabled:
            self.statusBar().showMessage("Zoom lens enabled.", 2000)
        else:
            self.statusBar().showMessage("Zoom lens disabled.", 2000)

    def _set_gt_buttons_enabled(self, enabled: bool) -> None:
        _ = enabled
        self._refresh_ai_dialog_state()

    def _update_gemini_fill_button_state(self, *, force_disable: bool = False) -> None:
        _ = force_disable
        self._refresh_ai_dialog_state()

    def _update_gemini_complete_button_state(self, *, force_disable: bool = False) -> None:
        _ = force_disable
        self._refresh_ai_dialog_state()

    def toggle_batch_panel(self) -> None:
        visible = not self.batch_box.isVisible()
        self.batch_box.setVisible(visible)
        self.batch_toggle_btn.setText("Hide Batch Edit" if visible else "Show Batch Edit")

    def _set_gt_activity(self, provider: str, status: str, *, fact_count: int = 0, running: bool = False) -> None:
        if hasattr(self, "_ai_controller"):
            self._ai_controller._set_status(f"{provider}: {status}", fact_count=int(fact_count), running=running)

    def _clear_gt_activity(self) -> None:
        if hasattr(self, "_ai_controller"):
            self._ai_controller._set_status("Idle.", fact_count=0, running=False)

    def _stop_active_generation(self) -> None:
        if hasattr(self, "_ai_controller"):
            self._ai_controller.stop_active_generation()

    def _apply_zoom(self, factor: float) -> None:
        current_zoom = self.view.transform().m11()
        next_zoom = current_zoom * factor
        if next_zoom < 0.05 or next_zoom > 40.0:
            return
        self.view.scale(factor, factor)

    def zoom_in(self) -> None:
        self._apply_zoom(1.2)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.2)

    def _snapshot_state(self) -> Dict[str, Any]:
        if not self._is_restoring_history:
            self._capture_current_state()
        return {
            "current_index": self.current_index,
            "page_states": deepcopy(self.page_states),
            "metadata": deepcopy(self.document_meta),
        }

    def _init_history(self) -> None:
        self._history = [self._snapshot_state()]
        self._history_index = 0
        self._update_history_controls()

    def _update_history_controls(self) -> None:
        can_undo = self._history_index > 0
        can_redo = 0 <= self._history_index < (len(self._history) - 1)
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(can_undo)
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(can_redo)

    def _record_history_snapshot(self) -> None:
        if self._is_restoring_history or self._is_loading_page:
            return
        snapshot = self._snapshot_state()
        if 0 <= self._history_index < len(self._history) and snapshot == self._history[self._history_index]:
            return

        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
        self._history.append(snapshot)
        self._history_index = len(self._history) - 1

        if len(self._history) > self._history_limit:
            overflow = len(self._history) - self._history_limit
            self._history = self._history[overflow:]
            self._history_index = max(0, self._history_index - overflow)
        self._update_history_controls()

    def _restore_history_snapshot(self, index: int) -> None:
        if index < 0 or index >= len(self._history):
            return

        snapshot = deepcopy(self._history[index])
        target_index = snapshot.get("current_index", 0)
        if not self.page_images:
            return
        target_index = max(0, min(int(target_index), len(self.page_images) - 1))

        self._is_restoring_history = True
        try:
            self.page_states = snapshot.get("page_states", {})
            self.document_meta = normalize_document_meta(snapshot.get("metadata", snapshot.get("document_meta")))
            self._history_index = index
            self._recompute_all_page_issues(emit=False)
            self.show_page(target_index)
        finally:
            self._is_restoring_history = False
        self._update_history_controls()

    def undo(self) -> None:
        self._restore_history_snapshot(self._history_index - 1)

    def redo(self) -> None:
        self._restore_history_snapshot(self._history_index + 1)

    def _default_meta(self, index: int) -> Dict[str, Any]:
        return default_page_meta(index)

    def _default_state(self, index: int) -> PageState:
        return PageState(meta=self._default_meta(index), facts=[])

    def _snapshot_saved_content(self) -> Dict[str, Any]:
        return {
            "page_states": deepcopy(self.page_states),
            "metadata": deepcopy(self.document_meta),
        }

    def _mark_saved_content(self) -> None:
        self._capture_current_state()
        self._last_saved_content = self._snapshot_saved_content()

    def _has_unsaved_changes(self) -> bool:
        self._capture_current_state()
        return self._snapshot_saved_content() != self._last_saved_content

    def cancel_pending_close(self) -> None:
        self._pending_close_approved = False

    def confirm_close(self, *, prepare_for_close: bool = False) -> bool:
        if not self._has_unsaved_changes():
            if prepare_for_close:
                self._pending_close_approved = True
            return True

        action = _prompt_unsaved_close_action(self)
        if action == "save":
            saved = self.save_annotations()
            if saved and prepare_for_close:
                self._pending_close_approved = True
            return saved
        if action == "discard":
            if prepare_for_close:
                self._pending_close_approved = True
            return True
        return False

    def request_application_exit(self) -> None:
        top_level = self.window()
        if top_level is None or top_level is self:
            self.close()
            return
        top_level.close()

    def _document_meta_from_ui(self) -> Dict[str, Any]:
        language_text = str(self.doc_language_combo.currentText() or "").strip().lower()
        if language_text.startswith("hebrew"):
            language = "he"
        elif language_text.startswith("english"):
            language = "en"
        else:
            language = None

        direction_text = str(self.doc_direction_combo.currentText() or "").strip().lower()
        if direction_text == "rtl":
            reading_direction = "rtl"
        elif direction_text == "ltr":
            reading_direction = "ltr"
        else:
            reading_direction = None
        company_name = self.company_name_edit.text().strip() or None
        company_id = self.company_id_edit.text().strip() or None
        report_year_text = self.report_year_edit.text().strip()
        report_year = int(report_year_text) if report_year_text else None
        entity_type = self.entity_type_combo.currentText().strip() or None
        report_scope = self.report_scope_combo.currentText().strip() or None
        return normalize_document_meta(
            {
                "language": language,
                "reading_direction": reading_direction,
                "company_name": company_name,
                "company_id": company_id,
                "report_year": report_year,
                "entity_type": entity_type,
                "report_scope": report_scope,
            }
        )

    def _page_meta_from_ui(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name_edit.text().strip() or None,
            "page_num": self.page_num_edit.text().strip() or None,
            "page_type": self.type_combo.currentText(),
            "statement_type": self.statement_type_combo.currentText().strip() or None,
            "title": self.title_edit.text().strip() or None,
            "annotation_note": self.page_annotation_note_edit.text().strip() or None,
            "annotation_status": _normalize_page_annotation_status(self._page_annotation_status),
        }

    def _live_page_state(self, *, use_current_fact_items: bool = False) -> Optional[PageState]:
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            return None
        fact_items = self._fact_items if use_current_fact_items else self._sorted_fact_items()
        facts = [
            BoxRecord(
                bbox=bbox_to_dict(item_scene_rect(item)),
                fact=normalize_fact_data(item.fact_data),
            )
            for item in fact_items
            if self._is_alive_fact_item(item)
        ]
        return PageState(meta=self._page_meta_from_ui(), facts=facts)

    def _page_issue_summary_for_page(
        self,
        page_name: str,
        *,
        use_live_current: bool = False,
        use_current_fact_items: bool = False,
    ) -> PageIssueSummary:
        summaries = self._document_issue_summaries(
            use_live_current=use_live_current,
            use_current_fact_items=use_current_fact_items,
        ).page_summaries
        return summaries.get(page_name, PageIssueSummary(page_image=page_name))

    def _issue_page_states(
        self,
        *,
        use_live_current: bool = False,
        use_current_fact_items: bool = False,
    ) -> List[tuple[str, PageState]]:
        states: List[tuple[str, PageState]] = []
        current_page_name = None
        if self.current_index >= 0 and self.current_index < len(self.page_images):
            current_page_name = self.page_images[self.current_index].name
        live_state = None
        if use_live_current and current_page_name is not None:
            live_state = self._live_page_state(use_current_fact_items=use_current_fact_items)

        for idx, page_path in enumerate(self.page_images):
            page_name = page_path.name
            if use_live_current and page_name == current_page_name and live_state is not None:
                states.append((page_name, live_state))
                continue
            states.append((page_name, self.page_states.get(page_name, self._default_state(idx))))
        return states

    def _document_issue_summaries(
        self,
        *,
        use_live_current: bool = False,
        use_current_fact_items: bool = False,
    ) -> DocumentIssueSummary:
        return validate_document_issues(
            self._issue_page_states(
                use_live_current=use_live_current,
                use_current_fact_items=use_current_fact_items,
            )
        )

    def _issue_tooltip_text(self, summary: PageIssueSummary) -> str:
        parts = [summary.page_image]
        if summary.reg_flag_count:
            parts.append(f"Reg flags: {summary.reg_flag_count}")
        if summary.warning_count:
            parts.append(f"Warnings: {summary.warning_count}")
        if not summary.reg_flag_count and not summary.warning_count:
            parts.append("No issues")
        return " | ".join(parts)

    def _page_annotation_note_for_index(self, index: int) -> str:
        if index < 0 or index >= len(self.page_images):
            return ""
        if index == self.current_index and hasattr(self, "page_annotation_note_edit"):
            return str(self.page_annotation_note_edit.text() or "").strip()
        page_name = self.page_images[index].name
        state = self.page_states.get(page_name, self._default_state(index))
        meta = state.meta or {}
        return str(meta.get("annotation_note") or "").strip()

    def _page_annotation_status_for_index(self, index: int) -> str | None:
        if index < 0 or index >= len(self.page_images):
            return None
        if index == self.current_index and hasattr(self, "_page_annotation_status"):
            return _normalize_page_annotation_status(self._page_annotation_status)
        page_name = self.page_images[index].name
        state = self.page_states.get(page_name, self._default_state(index))
        meta = state.meta or {}
        return _normalize_page_annotation_status(meta.get("annotation_status"))

    def _page_annotation_status_text(self, status: str | None) -> str:
        if status == "approved":
            return "Approved"
        if status == "flagged":
            return "Flagged"
        return "Unclassified"

    def _refresh_page_annotation_status_ui(self) -> None:
        status = _normalize_page_annotation_status(self._page_annotation_status)
        self._page_annotation_status = status
        text = self._page_annotation_status_text(status)
        tone = "ok" if status == "approved" else ("warn" if status == "flagged" else "accent")
        self.page_annotation_status_label.setText(text)
        self.page_annotation_status_label.setProperty("tone", tone)
        self.page_annotation_status_label.style().unpolish(self.page_annotation_status_label)
        self.page_annotation_status_label.style().polish(self.page_annotation_status_label)

    def _visible_thumbnail_page_indices(self) -> list[int]:
        return list(range(len(self.page_images)))

    def _thumbnail_row_for_page_index(self, page_index: int) -> int:
        return int(self._thumbnail_row_lookup.get(page_index, -1))

    def _page_index_for_thumbnail_row(self, row: int) -> int:
        if row < 0 or row >= len(self._thumbnail_page_indices):
            return -1
        return int(self._thumbnail_page_indices[row])

    def _next_page_index_in_thumbnail_order(self) -> int | None:
        if 0 <= self.current_index + 1 < len(self.page_images):
            return self.current_index + 1
        return None

    def _thumbnail_tooltip_text(self, summary: PageIssueSummary, index: int) -> str:
        base = self._issue_tooltip_text(summary)
        status = self._page_annotation_status_for_index(index)
        if status is not None:
            base = f"{base} | Annotation: {self._page_annotation_status_text(status)}"
        note = self._page_annotation_note_for_index(index)
        if not note:
            return base
        truncated = note if len(note) <= 120 else f"{note[:117]}..."
        return f"{base} | Revisit: {truncated}"

    def _issue_badge_text(self, count: int) -> str:
        if count <= 0:
            return ""
        return "99+" if count > 99 else str(count)

    def _draw_issue_badge(self, painter: QPainter, x: int, y: int, text: str, bg_hex: str) -> None:
        if not text:
            return
        badge_width = 22 if len(text) <= 2 else 28
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(bg_hex))
        painter.drawRoundedRect(x, y, badge_width, 18, 8, 8)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(x, y, badge_width, 18), Qt.AlignCenter, text)

    def _update_page_issue_panel(self, summary: PageIssueSummary) -> None:
        self.page_reg_flags_label.setText(f"Reg Flags: {summary.reg_flag_count}")
        self.page_warnings_label.setText(f"Warnings: {summary.warning_count}")
        self.page_reg_flags_label.setProperty("tone", "danger" if summary.reg_flag_count else "accent")
        self.page_warnings_label.setProperty("tone", "warn" if summary.warning_count else "accent")
        self.page_reg_flags_label.style().unpolish(self.page_reg_flags_label)
        self.page_reg_flags_label.style().polish(self.page_reg_flags_label)
        self.page_warnings_label.style().unpolish(self.page_warnings_label)
        self.page_warnings_label.style().polish(self.page_warnings_label)

        self.page_issues_list.clear()
        if not summary.issues:
            self.page_issues_hint_label.setText("No issues on this page.")
            return

        self.page_issues_hint_label.setText("Click an issue to jump to the related fact or field.")
        for issue in summary.issues:
            prefix = "Reg Flag" if issue.severity == "reg_flag" else "Warning"
            item = QListWidgetItem(f"{prefix}: {issue.message}")
            item.setData(Qt.UserRole, issue)
            self.page_issues_list.addItem(item)

    def _recompute_all_page_issues(self, *, emit: bool = True) -> None:
        document_summary = self._document_issue_summaries()
        self._page_issue_summaries = dict(document_summary.page_summaries)
        if hasattr(self, "page_thumb_list"):
            for idx in range(len(self.page_images)):
                self._refresh_thumbnail_for_index(idx)
        if self.current_index >= 0 and self.current_index < len(self.page_images):
            current_name = self.page_images[self.current_index].name
            self._update_page_issue_panel(self._page_issue_summaries.get(current_name, PageIssueSummary(page_image=current_name)))
        if emit:
            self._emit_document_issues_changed()

    def document_issue_summary(self) -> DocumentIssueSummary:
        return self._document_issue_summaries(use_live_current=True)

    def _emit_document_issues_changed(self) -> None:
        summary = self.document_issue_summary()
        signature = (
            summary.reg_flag_count,
            summary.warning_count,
            summary.pages_with_reg_flags,
            summary.pages_with_warnings,
        )
        if signature == self._last_document_issue_signature:
            return
        self._last_document_issue_signature = signature
        self.document_issues_changed.emit(summary)

    def _refresh_current_page_issues(self, *, use_current_fact_items: bool = False) -> None:
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            return
        page_name = self.page_images[self.current_index].name
        document_summary = self._document_issue_summaries(
            use_live_current=True,
            use_current_fact_items=use_current_fact_items,
        )
        self._page_issue_summaries = dict(document_summary.page_summaries)
        summary = self._page_issue_summaries.get(page_name, PageIssueSummary(page_image=page_name))
        self._update_page_issue_panel(summary)
        if hasattr(self, "page_thumb_list"):
            for idx in range(len(self.page_images)):
                self._refresh_thumbnail_for_index(idx)
        else:
            self._refresh_thumbnail_for_index(self.current_index)
        self._emit_document_issues_changed()

    def _focus_issue_field(self, field_name: Optional[str], *, fact_scoped: bool = False) -> None:
        if not field_name:
            return
        if fact_scoped:
            widget_map = {
                "value": self.fact_value_edit,
                "note_name": self.fact_note_name_edit,
                "note_num": self.fact_beur_num_edit,
                "note_flag": self.fact_is_beur_combo,
                "note_ref": self.fact_refference_edit,
                "comment_ref": self.fact_note_edit,
                "period_type": self.fact_period_type_combo,
                "period_start": self.fact_period_start_edit,
                "period_end": self.fact_period_end_edit,
                "duration_type": self.fact_duration_type_combo,
                "recurring_period": self.fact_recurring_period_combo,
                "value_type": self.fact_value_type_combo,
                "value_context": self.fact_value_context_combo,
                "row_role": self.fact_row_role_combo,
                "path_source": self.fact_path_source_combo,
            }
        else:
            widget_map = {
                "page_type": self.type_combo,
                "statement_type": self.statement_type_combo,
                "currency": self.facts_list,
                "scale": self.facts_list,
                "value_type": self.facts_list,
                "value_context": self.facts_list,
            }
        widget = widget_map.get(field_name)
        if widget is not None:
            widget.setFocus()

    def _on_page_issue_clicked(self, item: QListWidgetItem) -> None:
        issue = item.data(Qt.UserRole)
        if not isinstance(issue, PageIssue):
            return
        if issue.fact_index is not None and 0 <= issue.fact_index < len(self._fact_items):
            target_item = self._fact_items[issue.fact_index]
            if self._is_alive_fact_item(target_item):
                self.scene.clearSelection()
                target_item.setSelected(True)
                self.facts_list.setCurrentRow(issue.fact_index)
                self._focus_fact_items_in_view([target_item])
                self._sync_fact_editor_from_selection()
                self._focus_issue_field(issue.field_name, fact_scoped=True)
                return
        self._focus_issue_field(issue.field_name, fact_scoped=False)

    def _set_document_meta_ui(self, document_meta: Dict[str, Any]) -> None:
        normalized = normalize_document_meta(document_meta)
        language = normalized.get("language")
        reading_direction = normalized.get("reading_direction")
        company_name = normalized.get("company_name")
        company_id = normalized.get("company_id")
        report_year = normalized.get("report_year")
        entity_type = str(normalized.get("entity_type") or "")
        if language == "he":
            self.doc_language_combo.setCurrentIndex(1)
        elif language == "en":
            self.doc_language_combo.setCurrentIndex(2)
        else:
            self.doc_language_combo.setCurrentIndex(0)

        if reading_direction == "rtl":
            self.doc_direction_combo.setCurrentIndex(1)
        elif reading_direction == "ltr":
            self.doc_direction_combo.setCurrentIndex(2)
        else:
            self.doc_direction_combo.setCurrentIndex(0)
        self.company_name_edit.setText(str(company_name or ""))
        self.company_id_edit.setText(str(company_id or ""))
        self.report_year_edit.setText("" if report_year is None else str(report_year))
        report_scope = str(normalized.get("report_scope") or "")
        report_scope_idx = self.report_scope_combo.findText(report_scope)
        self.report_scope_combo.setCurrentIndex(max(0, report_scope_idx))
        entity_type_idx = self.entity_type_combo.findText(entity_type)
        self.entity_type_combo.setCurrentIndex(max(0, entity_type_idx))

    def _direction_payload_for_current_page(self) -> Dict[str, Any]:
        page_payload: Dict[str, Any] = {}
        if self.current_index >= 0 and self.current_index < len(self.page_images):
            page_name = self.page_images[self.current_index].name
            state = self.page_states.get(page_name)
            if state is not None:
                page_payload = {
                    "pages": [
                        {
                            "image": page_name,
                            "meta": dict(state.meta or {}),
                            "facts": [
                                {
                                    "bbox": box.bbox,
                                    **normalize_fact_data(box.fact),
                                }
                                for box in state.facts
                            ],
                        }
                    ]
                }
        return page_payload

    def _active_reading_direction(self) -> str:
        info = resolve_reading_direction(
            self.document_meta,
            payload=self._direction_payload_for_current_page(),
            default_direction="rtl",
        )
        return "rtl" if str(info.get("direction") or "rtl") == "rtl" else "ltr"

    def _page_index_by_name(self, page_name: str) -> int:
        for idx, image_path in enumerate(self.page_images):
            if image_path.name == page_name:
                return idx
        return -1

    def _image_dimensions_for_page(self, page_name: str) -> Optional[tuple[float, float]]:
        image_path = self.images_dir / page_name
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return None
        return float(pixmap.width()), float(pixmap.height())

    @staticmethod
    def _bbox_looks_normalized_1000(bbox: Dict[str, Any]) -> bool:
        return bbox_looks_normalized_1000(bbox)

    def _load_gemini_few_shot_examples(
        self,
        *,
        preset: str = FEW_SHOT_PRESET_CLASSIC,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        repo_roots = build_repo_roots(cwd=Path.cwd(), anchor_file=Path(__file__).resolve())
        if preset == FEW_SHOT_PRESET_EXTENDED:
            return load_complex_few_shot_examples(
                repo_roots=repo_roots,
                selections=DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
            )
        if preset == FEW_SHOT_PRESET_2015_TWO_SHOT:
            return load_complex_few_shot_examples(
                repo_roots=repo_roots,
                selections=DEFAULT_2015_TWO_SHOT_SELECTIONS,
            )
        if preset == FEW_SHOT_PRESET_ONE_SHOT:
            return load_test_pdf_few_shot_examples(
                repo_roots=repo_roots,
                page_names=(DEFAULT_TEST_ONE_SHOT_PAGE,),
            )
        return load_test_pdf_few_shot_examples(
            repo_roots=repo_roots,
            page_names=DEFAULT_TEST_FEW_SHOT_PAGES,
        )

    def _resolve_qwen_config_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        env_path = os.getenv("FINETREE_QWEN_CONFIG")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        candidates.append(Path.cwd() / "configs/qwen_ui_runpod_queue.yaml")
        candidates.append(Path.cwd() / "configs/finetune_qwen35a3_vl.yaml")
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / "configs/qwen_ui_runpod_queue.yaml")
            candidates.append(parent / "configs/finetune_qwen35a3_vl.yaml")

        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    def _initial_qwen_model_name(self) -> str:
        env_model = os.getenv("FINETREE_QWEN_MODEL")
        if isinstance(env_model, str) and env_model.strip():
            return env_model.strip()

        cfg_path = self._resolve_qwen_config_path()
        if cfg_path is None:
            return "local-qwen"

        try:
            cfg = load_finetune_config(cfg_path)
        except Exception:
            return "local-qwen"

        model_name = (
            cfg.inference.endpoint_model
            or cfg.inference.model_path
            or cfg.model.base_model
            or "local-qwen"
        )
        return str(model_name).strip() or "local-qwen"

    def _initial_qwen_enable_thinking(self) -> bool:
        cfg_path = self._resolve_qwen_config_path()
        if cfg_path is None:
            return False

        try:
            cfg = load_finetune_config(cfg_path)
        except Exception:
            return False
        return bool(cfg.inference.enable_thinking)

    def _build_page_prompt_payload(
        self,
        *,
        page_name: str,
        page_meta: Dict[str, Any],
        facts: list[dict[str, Any]],
    ) -> Dict[str, Any]:
        return build_ai_page_prompt_payload(
            page_name=page_name,
            page_meta=page_meta,
            facts=facts,
        )

    def _build_gemini_fill_prompt_from_template(
        self,
        template: str,
        *,
        request_payload: Dict[str, Any],
        selected_fact_fields: set[str],
        include_statement_type: bool,
    ) -> str:
        return build_ai_gemini_fill_prompt(
            template,
            request_payload=request_payload,
            selected_fact_fields=selected_fact_fields,
            include_statement_type=include_statement_type,
        )

    def _build_gemini_autocomplete_prompt_from_template(
        self,
        template: str,
        *,
        request_payload: Dict[str, Any],
    ) -> str:
        pages = request_payload.get("pages")
        page_name = ""
        if isinstance(pages, list) and pages and isinstance(pages[0], dict):
            page_name = str(pages[0].get("image") or "")
        return build_ai_gemini_autocomplete_prompt(
            template,
            request_payload=request_payload,
            image_dimensions=self._image_dimensions_for_page(page_name) if page_name else None,
        )

    def _fact_payload_from_item(self, item: AnnotRectItem) -> Dict[str, Any]:
        return {
            "bbox": bbox_to_dict(item_scene_rect(item)),
            **normalize_fact_data(item.fact_data),
        }

    def _fact_uniqueness_key(self, fact_payload: Dict[str, Any]) -> tuple[Any, ...]:
        normalized_fact = normalize_fact_data(fact_payload)
        bbox = normalize_bbox_data(fact_payload.get("bbox"))
        path = tuple(str(p) for p in (normalized_fact.get("path") or []))
        active_bundle = _active_equation_bundle_from_fact_payload(normalized_fact) or {}
        return (
            round(float(bbox["x"]), 2),
            round(float(bbox["y"]), 2),
            round(float(bbox["w"]), 2),
            round(float(bbox["h"]), 2),
            str(normalized_fact.get("value") or ""),
            str(normalized_fact.get("comment_ref") or ""),
            str(normalized_fact.get("note_flag") if normalized_fact.get("note_flag") is not None else ""),
            str(normalized_fact.get("note_name") or ""),
            str(normalized_fact.get("note_num") if normalized_fact.get("note_num") is not None else ""),
            str(normalized_fact.get("note_ref") or ""),
            str(normalized_fact.get("date") or ""),
            str(active_bundle.get("equation") or ""),
            str(active_bundle.get("fact_equation") or ""),
            str(normalized_fact.get("period_type") or ""),
            str(normalized_fact.get("period_start") or ""),
            str(normalized_fact.get("period_end") or ""),
            str(normalized_fact.get("duration_type") or ""),
            str(normalized_fact.get("recurring_period") or ""),
            str(normalized_fact.get("currency") or ""),
            str(normalized_fact.get("scale") if normalized_fact.get("scale") is not None else ""),
            str(normalized_fact.get("value_type") or ""),
            str(normalized_fact.get("value_context") or ""),
            str(normalized_fact.get("natural_sign") or ""),
            str(normalized_fact.get("row_role") or ""),
            str(normalized_fact.get("path_source") or ""),
            path,
        )

    def _current_page_fact_snapshot_signature(
        self,
        ordered_items: List[AnnotRectItem],
    ) -> List[tuple[Any, ...]]:
        signature: List[tuple[Any, ...]] = []
        for item in ordered_items:
            signature.append(self._fact_uniqueness_key(self._fact_payload_from_item(item)))
        return signature

    def _ai_page_context(self) -> AIPageContext | None:
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            return None
        page_path = self.page_images[self.current_index]
        page_name = page_path.name
        ordered_items = self._sorted_fact_items()
        state = self.page_states.get(page_name, self._default_state(self.current_index))
        page_meta = PageMeta.model_validate(
            {**self._default_meta(self.current_index), **(state.meta or {})}
        ).model_dump(mode="json")
        selected_ids = {id(item) for item in self._selected_fact_items()}
        selected_fact_nums = [
            fact_num
            for item in ordered_items
            if id(item) in selected_ids
            for fact_num in [self._fact_num_for_item(item)]
            if fact_num is not None
        ]
        ordered_fact_payloads = [deepcopy(self._fact_payload_from_item(item)) for item in ordered_items]
        return AIPageContext(
            page_path=page_path,
            page_name=page_name,
            page_index=self.current_index,
            page_meta=page_meta,
            ordered_fact_payloads=ordered_fact_payloads,
            ordered_fact_signature=self._current_page_fact_snapshot_signature(ordered_items),
            selected_fact_nums=selected_fact_nums,
            existing_fact_count=len(ordered_fact_payloads),
            selected_fact_count=len(selected_fact_nums),
            image_dimensions=self._image_dimensions_for_page(page_name),
        )

    def open_ai_dialog(
        self,
        _checked: bool = False,
        *,
        provider: AIProvider | None = None,
        action: AIActionKind | None = None,
    ) -> None:
        self._ai_controller.open_dialog(provider=provider, action=action)

    def _refresh_ai_dialog_state(self) -> None:
        self._ai_controller.refresh_dialog_state()

    def _normalized_stream_fact_payload(self, fact_payload: Dict[str, Any]) -> Dict[str, Any] | None:
        return normalize_ai_fact_payload(fact_payload)

    def _normalized_autocomplete_generated_fact_payload(self, fact_payload: Dict[str, Any]) -> Dict[str, Any] | None:
        return normalize_ai_fact_payload(fact_payload, clear_fact_num=True)

    def _autocomplete_candidate_payloads_for_bbox_mode(
        self,
        *,
        fact_payloads: List[Dict[str, Any]],
        mode: str,
        image_width: float,
        image_height: float,
    ) -> List[Dict[str, Any]]:
        return payloads_for_bbox_mode(
            fact_payloads,
            mode=mode,
            image_width=image_width,
            image_height=image_height,
        )

    @staticmethod
    def _score_bbox_payload_ink(image: QImage, bbox_payload: Dict[str, Any]) -> tuple[float, float]:
        return score_bbox_payload_ink(image, bbox_payload)

    def _score_autocomplete_bbox_candidate_payloads(self, page_name: str, fact_payloads: List[Dict[str, Any]]) -> float:
        return score_bbox_candidate_payloads(self.images_dir / page_name, fact_payloads)

    def _gemini_generated_bbox_policy(self) -> str:
        raw_policy = str(
            os.getenv("FINETREE_GEMINI_GENERATED_BBOX_POLICY") or GEMINI_GENERATED_BBOX_POLICY_DEFAULT
        ).strip().lower()
        return "auto" if raw_policy == "auto" else "pixel"

    def _resolve_autocomplete_bbox_mode(
        self,
        page_name: str,
        fact_payloads: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], str]:
        result = resolve_bbox_mode(
            image_path=self.images_dir / page_name,
            image_dimensions=self._image_dimensions_for_page(page_name),
            fact_payloads=fact_payloads,
        )
        self._gemini_autocomplete_last_bbox_scores = dict(result.scores)
        if self._gemini_generated_bbox_policy() != "auto":
            image_dims = self._image_dimensions_for_page(page_name)
            if image_dims is None:
                return [deepcopy(payload) for payload in fact_payloads], BBOX_MODE_PIXEL_AS_IS
            pixel_payloads = self._autocomplete_candidate_payloads_for_bbox_mode(
                fact_payloads=fact_payloads,
                mode=BBOX_MODE_PIXEL_AS_IS,
                image_width=image_dims[0],
                image_height=image_dims[1],
            )
            return pixel_payloads, BBOX_MODE_PIXEL_AS_IS
        return result.payloads, result.mode

    def _ordered_fact_payloads_by_geometry(self, fact_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return ordered_fact_payloads_by_geometry(
            fact_payloads,
            reading_direction=self._active_reading_direction(),
        )

    def _apply_page_state_to_scene(self, page_name: str) -> None:
        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return
        if self.current_index >= 0 and self.page_images[self.current_index].name == page_name:
            was_restoring = self._is_restoring_history
            self._is_restoring_history = True
            try:
                self.show_page(page_idx)
            finally:
                self._is_restoring_history = was_restoring
        else:
            self._recompute_all_page_issues()
            self._refresh_thumbnail_for_index(page_idx)

    def _merge_autocomplete_buffered_facts(self, page_name: str) -> tuple[bool, str | None]:
        snapshot = self._gemini_autocomplete_snapshot or {}
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            return False, "Current page is unavailable."
        if self.page_images[self.current_index].name != page_name:
            return False, "Current page changed during Gemini Auto Complete. Please rerun."

        ordered_items = self._sorted_fact_items()
        current_signature = self._current_page_fact_snapshot_signature(ordered_items)
        expected_signature = list(snapshot.get("ordered_fact_signature") or [])
        if current_signature != expected_signature:
            return False, "Current page facts changed during Gemini Auto Complete. Please rerun."

        locked_fact_payloads = [deepcopy(payload) for payload in (snapshot.get("locked_fact_payloads") or [])]
        if not self._gemini_autocomplete_buffered_facts:
            self._gemini_stream_fact_count = 0
            self._gemini_autocomplete_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
            return True, None

        resolved_payloads, bbox_mode = self._resolve_autocomplete_bbox_mode(
            page_name,
            self._gemini_autocomplete_buffered_facts,
        )
        self._gemini_autocomplete_last_bbox_mode = bbox_mode
        locked_keys = {self._fact_uniqueness_key(payload) for payload in locked_fact_payloads}
        deduped_new_payloads: list[dict[str, Any]] = []
        seen_new_keys: set[tuple[Any, ...]] = set()
        for payload in resolved_payloads:
            payload_key = self._fact_uniqueness_key(payload)
            if payload_key in locked_keys or payload_key in seen_new_keys:
                continue
            seen_new_keys.add(payload_key)
            deduped_new_payloads.append(deepcopy(payload))
        self._gemini_stream_fact_count = len(deduped_new_payloads)
        if not deduped_new_payloads:
            return True, None

        merged_payloads = self._ordered_fact_payloads_by_geometry(
            locked_fact_payloads + deduped_new_payloads
        )
        resequenced_facts = resequence_fact_numbers_and_remap_fact_equations(
            [normalize_fact_data(payload) for payload in merged_payloads]
        )
        merged_records = [
            BoxRecord(
                bbox=normalize_bbox_data(payload.get("bbox")),
                fact=resequenced_fact,
            )
            for payload, resequenced_fact in zip(merged_payloads, resequenced_facts)
        ]

        page_idx = self._page_index_by_name(page_name)
        state = self.page_states.get(page_name, self._default_state(page_idx))
        self.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=merged_records)
        self._apply_page_state_to_scene(page_name)
        return True, None

    def _merge_gemini_gt_buffered_facts(self, page_name: str) -> tuple[bool, str | None]:
        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return False, "Target page is unavailable."

        if not self._gemini_gt_buffered_facts:
            self._gemini_stream_fact_count = 0
            self._gemini_gt_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
            self._gemini_gt_last_bbox_scores = {
                BBOX_MODE_PIXEL_AS_IS: 0.0,
                BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
            }
            self._gemini_gt_live_applied = False
            state = self.page_states.get(page_name, self._default_state(page_idx))
            self.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=[])
            self._apply_page_state_to_scene(page_name)
            return True, None

        resolved_payloads, bbox_mode = self._resolve_autocomplete_bbox_mode(
            page_name,
            self._gemini_gt_buffered_facts,
        )
        self._gemini_gt_last_bbox_mode = bbox_mode
        self._gemini_gt_last_bbox_scores = dict(
            self._gemini_autocomplete_last_bbox_scores
            or {
                BBOX_MODE_PIXEL_AS_IS: 0.0,
                BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
            }
        )
        merged_records = self._build_box_records_from_fact_payloads(resolved_payloads)

        state = self.page_states.get(page_name, self._default_state(page_idx))
        self.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=merged_records)
        self._gemini_stream_fact_count = len(merged_records)
        self._gemini_gt_live_applied = bool(merged_records)
        self._apply_page_state_to_scene(page_name)
        return True, None

    def _gemini_gt_payloads_for_bbox_mode(
        self,
        *,
        page_name: str,
        fact_payloads: list[dict[str, Any]],
        mode: str,
    ) -> list[dict[str, Any]]:
        image_dims = self._image_dimensions_for_page(page_name)
        if image_dims is None:
            return [deepcopy(payload) for payload in fact_payloads]
        return self._autocomplete_candidate_payloads_for_bbox_mode(
            fact_payloads=fact_payloads,
            mode=mode,
            image_width=image_dims[0],
            image_height=image_dims[1],
        )

    def _build_box_records_from_fact_payloads(self, fact_payloads: list[dict[str, Any]]) -> list[BoxRecord]:
        deduped_payloads: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for payload in fact_payloads:
            payload_key = self._fact_uniqueness_key(payload)
            if payload_key in seen_keys:
                continue
            seen_keys.add(payload_key)
            deduped_payloads.append(deepcopy(payload))

        ordered_payloads = self._ordered_fact_payloads_by_geometry(deduped_payloads)
        resequenced_facts = resequence_fact_numbers_and_remap_fact_equations(
            [normalize_fact_data(payload) for payload in ordered_payloads]
        )
        return [
            BoxRecord(
                bbox=normalize_bbox_data(payload.get("bbox")),
                fact=resequenced_fact,
            )
            for payload, resequenced_fact in zip(ordered_payloads, resequenced_facts)
        ]

    def _render_gemini_gt_live_buffer(
        self,
        page_name: str,
        *,
        resolved_payloads: list[dict[str, Any]] | None = None,
    ) -> bool:
        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return False
        if not self._gemini_gt_buffered_facts:
            return False

        if resolved_payloads is None:
            resolved_payloads = self._gemini_gt_payloads_for_bbox_mode(
                page_name=page_name,
                fact_payloads=self._gemini_gt_buffered_facts,
                mode=self._gemini_gt_live_bbox_mode,
            )
        merged_records = self._build_box_records_from_fact_payloads(resolved_payloads)
        state = self.page_states.get(page_name, self._default_state(page_idx))
        self.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=merged_records)
        self._gemini_gt_live_applied = bool(merged_records)
        self._apply_page_state_to_scene(page_name)
        return True

    def _update_gemini_gt_live_stream(self, page_name: str) -> None:
        if self._gemini_stream_mode != "gt":
            return
        if not self._gemini_gt_buffered_facts:
            return
        resolved_payloads, bbox_mode = self._resolve_autocomplete_bbox_mode(
            page_name,
            self._gemini_gt_buffered_facts,
        )
        self._gemini_gt_live_bbox_mode = bbox_mode
        self._gemini_gt_last_bbox_mode = bbox_mode
        self._gemini_gt_last_bbox_scores = dict(
            self._gemini_autocomplete_last_bbox_scores
            or {
                BBOX_MODE_PIXEL_AS_IS: 0.0,
                BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
            }
        )
        if len(self._gemini_gt_buffered_facts) >= max(1, int(self._gemini_gt_live_lock_min_facts)):
            self._gemini_gt_live_bbox_mode_locked = True
        self._render_gemini_gt_live_buffer(page_name, resolved_payloads=resolved_payloads)

    def _ai_request_fact_payloads(self, ordered_items: List[AnnotRectItem]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for item in ordered_items:
            payload = self._fact_payload_from_item(item)
            payloads.append(
                {
                    "fact_num": self._fact_num_for_item(item),
                    "bbox": bbox_to_list(payload.get("bbox")),
                    **normalize_fact_data(payload),
                }
            )
        return payloads

    def _build_gemini_fill_request_payload(
        self,
        *,
        page_name: str,
        selected_fact_nums: List[int],
        selected_fact_fields: set[str],
        include_statement_type: bool,
        ordered_items: List[AnnotRectItem],
    ) -> Dict[str, Any]:
        page_idx = self._page_index_by_name(page_name)
        state = self.page_states.get(page_name, self._default_state(max(page_idx, 0)))
        page_meta = PageMeta.model_validate({**self._default_meta(max(page_idx, 0)), **(state.meta or {})}).model_dump(mode="json")
        return build_ai_gemini_fill_request_payload(
            page_name=page_name,
            page_meta=page_meta,
            ordered_fact_payloads=self._ai_request_fact_payloads(ordered_items),
            selected_fact_nums=selected_fact_nums,
            selected_fact_fields=selected_fact_fields,
            include_statement_type=include_statement_type,
        )

    def _build_gemini_autocomplete_request_payload(
        self,
        *,
        page_name: str,
        ordered_items: List[AnnotRectItem],
    ) -> Dict[str, Any]:
        page_idx = self._page_index_by_name(page_name)
        state = self.page_states.get(page_name, self._default_state(max(page_idx, 0)))
        page_meta = PageMeta.model_validate({**self._default_meta(max(page_idx, 0)), **(state.meta or {})}).model_dump(mode="json")
        return build_ai_gemini_autocomplete_request_payload(
            page_name=page_name,
            page_meta=page_meta,
            ordered_fact_payloads=self._ai_request_fact_payloads(ordered_items),
        )

    def _apply_stream_meta(self, page_name: str, meta_payload: Dict[str, Any]) -> None:
        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return
        state = self.page_states.get(page_name, self._default_state(page_idx))
        normalized_meta = PageMeta.model_validate(
            {**self._default_meta(page_idx), **(state.meta or {}), **(meta_payload or {})}
        ).model_dump(mode="json")
        self.page_states[page_name] = PageState(meta=normalized_meta, facts=list(state.facts))

        if self.current_index >= 0 and self.page_images[self.current_index].name == page_name:
            self._is_loading_page = True
            try:
                self.entity_name_edit.setText(normalized_meta.get("entity_name") or "")
                self.page_num_edit.setText(normalized_meta.get("page_num") or "")
                type_value = normalized_meta.get("page_type") or PageType.other.value
                type_idx = self.type_combo.findText(type_value)
                self.type_combo.setCurrentIndex(type_idx if type_idx >= 0 else 0)
                statement_type_value = str(normalized_meta.get("statement_type") or "")
                statement_type_idx = self.statement_type_combo.findText(statement_type_value)
                self.statement_type_combo.setCurrentIndex(max(0, statement_type_idx))
                self.title_edit.setText(normalized_meta.get("title") or "")
                self.page_annotation_note_edit.setText(normalized_meta.get("annotation_note") or "")
                self._page_annotation_status = _normalize_page_annotation_status(normalized_meta.get("annotation_status"))
                self._refresh_page_annotation_status_ui()
            finally:
                self._is_loading_page = False
            self._refresh_current_page_issues()
        else:
            self._recompute_all_page_issues()
        self._refresh_thumbnail_for_index(page_idx)

    def _apply_stream_fact(
        self,
        page_name: str,
        fact_payload: Dict[str, Any],
        seen_facts: Optional[set[tuple[Any, ...]]] = None,
        *,
        stream_source: str = "gemini",
    ) -> bool:
        active_seen = seen_facts if seen_facts is not None else self._gemini_stream_seen_facts
        fact_key = self._fact_uniqueness_key(fact_payload)
        if fact_key in active_seen:
            return False
        active_seen.add(fact_key)

        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return False

        if stream_source == "gemini" and self._gemini_stream_mode == "autocomplete":
            normalized_payload = self._normalized_autocomplete_generated_fact_payload(fact_payload)
            if normalized_payload is None:
                return False
            self._gemini_autocomplete_buffered_facts.append(normalized_payload)
            return True

        if stream_source == "gemini" and self._gemini_stream_mode == "gt":
            normalized_payload = self._normalized_stream_fact_payload(fact_payload)
            if normalized_payload is None:
                return False
            self._gemini_gt_buffered_facts.append(normalized_payload)
            return True

        normalized_payload = self._normalized_stream_fact_payload(fact_payload)
        if normalized_payload is None:
            return False
        bbox = normalize_bbox_data(normalized_payload.get("bbox"))
        fact_data = normalize_fact_data(normalized_payload)

        state = self.page_states.get(page_name, self._default_state(page_idx))
        state_facts = list(state.facts)
        state_facts.append(BoxRecord(bbox=bbox, fact=fact_data))
        self.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=state_facts)

        if self.current_index >= 0 and self.page_images[self.current_index].name == page_name:
            rect = dict_to_rect(bbox).intersected(self.scene.image_rect)
            if rect.width() >= 1 and rect.height() >= 1:
                self.scene.addItem(AnnotRectItem(rect, fact_data))
                self.refresh_facts_list()
                self._capture_current_state()
        else:
            self._recompute_all_page_issues()
        self._refresh_thumbnail_for_index(page_idx)
        return True

    def import_annotations_json(self) -> None:
        if not self.page_images:
            QMessageBox.warning(self, "Import error", "No pages are loaded.")
            return

        default_page_name = self.page_images[self.current_index if self.current_index >= 0 else 0].name
        dialog = JsonImportDialog(default_page_name=default_page_name, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        raw_text = dialog.json_text().strip()
        if not raw_text:
            QMessageBox.information(self, "Import canceled", "No JSON text was provided.")
            return

        try:
            payload = json.loads(raw_text)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", str(exc))
            return
        imported_document_meta = extract_document_meta(payload)

        import_normalized_1000 = dialog.import_normalized_1000_enabled()
        if isinstance(payload, dict):
            bbox_space = str(payload.get("bbox_space") or "").strip().lower()
            if bbox_space in {"normalized_1000", "norm1000", "1000"}:
                import_normalized_1000 = True
            elif bbox_space in {"pixel", "pixels", "absolute"}:
                import_normalized_1000 = False

        imported_states = parse_import_payload(
            payload,
            [p.name for p in self.page_images],
            default_page_image_name=default_page_name,
        )
        if not imported_states:
            QMessageBox.warning(
                self,
                "Import did not match any pages",
                "No importable pages were found. Check 'image' names or use a current-page shape.",
            )
            return

        self._capture_current_state()
        imported_count = 0
        converted_bbox_count = 0
        for page_name, imported_state in imported_states.items():
            page_idx = self._page_index_by_name(page_name)
            if page_idx < 0:
                continue
            normalized_meta = {**self._default_meta(page_idx), **(imported_state.meta or {})}
            image_dims = self._image_dimensions_for_page(page_name)
            normalized_facts: List[BoxRecord] = []
            for record in imported_state.facts:
                bbox = normalize_bbox_data(record.bbox)
                if import_normalized_1000 and image_dims and self._bbox_looks_normalized_1000(bbox):
                    bbox = denormalize_bbox_from_1000(bbox, image_dims[0], image_dims[1])
                    converted_bbox_count += 1
                normalized_facts.append(
                    BoxRecord(
                        bbox=bbox,
                        fact=normalize_fact_data(record.fact),
                    )
                )
            self.page_states[page_name] = PageState(meta=normalized_meta, facts=normalized_facts)
            imported_count += 1

        if imported_count == 0:
            QMessageBox.warning(self, "Import skipped", "Imported JSON did not match any existing page images.")
            return

        if compact_document_meta(imported_document_meta):
            self.document_meta = imported_document_meta
        self._recompute_all_page_issues(emit=False)

        target_index = self.current_index if self.current_index >= 0 else 0
        if imported_count == 1:
            single_page = next(iter(imported_states.keys()))
            single_idx = self._page_index_by_name(single_page)
            if single_idx >= 0:
                target_index = single_idx

        was_restoring = self._is_restoring_history
        self._is_restoring_history = True
        try:
            self.show_page(target_index)
        finally:
            self._is_restoring_history = was_restoring

        self._populate_page_thumbnails()
        self._record_history_snapshot()
        if converted_bbox_count:
            self.statusBar().showMessage(
                f"Imported {imported_count} page(s); converted {converted_bbox_count} normalized bbox(es).",
                5500,
            )
        else:
            self.statusBar().showMessage(f"Imported annotations for {imported_count} page(s).", 4500)

    def generate_gemini_fill_selected_fields(self) -> None:
        self.open_ai_dialog(provider=AIProvider.GEMINI, action=AIActionKind.FIX_SELECTED)

    def _cancel_gemini_fill(self) -> None:
        if self._gemini_fill_worker is not None:
            self._gemini_fill_cancel_requested = True
            self._gemini_fill_worker.request_cancel()
            self._set_gt_activity("Gemini Auto-Fix", "Stopping auto-fix task...", fact_count=0, running=True)

    def _on_gemini_fill_completed(self, patch_payload: Dict[str, Any]) -> None:
        page_name = self._gemini_fill_target_page
        snapshot = self._gemini_fill_snapshot or {}
        if page_name is None:
            return
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            self._on_gemini_fill_failed("Current page is unavailable.")
            return
        if self.page_images[self.current_index].name != page_name:
            self._on_gemini_fill_failed("Current page changed during Gemini fill. Please rerun.")
            return

        ordered_items = self._sorted_fact_items()
        current_signature = self._current_page_fact_snapshot_signature(ordered_items)
        expected_signature = list(snapshot.get("ordered_fact_signature") or [])
        if current_signature != expected_signature:
            self._on_gemini_fill_failed("Selected facts changed during Gemini fill. Please rerun.")
            return

        selected_fact_nums = list(snapshot.get("selected_fact_nums") or [])
        selected_lookup = set(int(num) for num in selected_fact_nums)
        ordered_items_by_fact_num = {
            fact_num: item
            for item in ordered_items
            for fact_num in [self._fact_num_for_item(item)]
            if fact_num is not None
        }
        fact_updates = patch_payload.get("fact_updates")
        if not isinstance(fact_updates, list):
            self._on_gemini_fill_failed("Patch payload is missing fact_updates.")
            return

        staged_fact_updates: list[tuple[AnnotRectItem, Dict[str, Any]]] = []
        try:
            for entry in fact_updates:
                if not isinstance(entry, dict):
                    raise ValueError("Invalid fact_updates entry.")
                fact_num = entry.get("fact_num")
                if isinstance(fact_num, bool) or not isinstance(fact_num, int):
                    raise ValueError("fact_num must be an integer.")
                if fact_num not in selected_lookup:
                    raise ValueError(f"Patch referenced non-selected fact_num: {fact_num}.")
                item = ordered_items_by_fact_num.get(fact_num)
                if item is None:
                    raise ValueError(f"fact_num out of range: {fact_num}.")
                updates = entry.get("updates")
                if not isinstance(updates, dict):
                    raise ValueError(f"updates for fact_num {fact_num} must be an object.")
                current_fact = normalize_fact_data(item.fact_data)
                staged_fact_updates.append((item, normalize_fact_data({**current_fact, **updates})))

            staged_meta_updates: Dict[str, Any] = {}
            meta_updates = patch_payload.get("meta_updates")
            if isinstance(meta_updates, dict) and "statement_type" in meta_updates:
                if not self._gemini_fill_include_statement_type:
                    raise ValueError("Patch included statement_type update but statement_type was not requested.")
                page_idx = self.current_index
                state = self.page_states.get(page_name, self._default_state(page_idx))
                normalized_meta = PageMeta.model_validate(
                    {
                        **self._default_meta(page_idx),
                        **(state.meta or {}),
                        "statement_type": meta_updates.get("statement_type"),
                    }
                ).model_dump(mode="json")
                staged_meta_updates["statement_type"] = normalized_meta.get("statement_type")
        except Exception as exc:
            self._on_gemini_fill_failed(str(exc))
            return

        for item, updated_fact in staged_fact_updates:
            item.fact_data = updated_fact
        if staged_meta_updates:
            self._apply_stream_meta(page_name, staged_meta_updates)
        if staged_fact_updates or staged_meta_updates:
            updated_count = len(staged_fact_updates)
            self.refresh_facts_list()
            self._record_history_snapshot()
            self.statusBar().showMessage(
                f"Gemini Auto-Fix complete ({updated_count} fact(s) updated).",
                6000,
            )
            self._set_gt_activity(
                "Gemini Auto-Fix",
                f"Gemini Auto-Fix complete. Updated {updated_count} fact(s).",
                fact_count=updated_count,
                running=False,
            )
            QMessageBox.information(
                self,
                "Gemini Auto-Fix",
                f"Gemini Auto-Fix finished.\nUpdated {updated_count} fact(s).",
            )
        else:
            self._set_gt_activity(
                "Gemini Auto-Fix",
                "Gemini Auto-Fix complete. No changes returned.",
                fact_count=0,
                running=False,
            )
            self.statusBar().showMessage("Gemini Auto-Fix complete (no changes).", 5000)
            QMessageBox.information(
                self,
                "Gemini Auto-Fix",
                "Gemini Auto-Fix finished.\nNo changes were applied.",
            )

    def _on_gemini_fill_failed(self, message: str) -> None:
        self._set_gt_activity("Gemini Auto-Fix", f"Error: {message}", fact_count=0, running=False)
        QMessageBox.warning(self, "Gemini Auto-Fix failed", message)

    def _on_gemini_fill_finished(self) -> None:
        if self._gemini_fill_cancel_requested:
            self._set_gt_activity("Gemini Auto-Fix", "Gemini Auto-Fix stopped.", fact_count=0, running=False)
            self.statusBar().showMessage("Gemini Auto-Fix stopped.", 4000)
        self._gemini_fill_thread = None
        self._gemini_fill_worker = None
        self._gemini_fill_target_page = None
        self._gemini_fill_snapshot = None
        self._gemini_fill_selected_fact_fields = set()
        self._gemini_fill_include_statement_type = False
        if self._gemini_stream_thread is None and self._qwen_stream_thread is None:
            self._set_gt_buttons_enabled(True)

    def generate_gemini_ground_truth(self) -> None:
        self.open_ai_dialog(provider=AIProvider.GEMINI, action=AIActionKind.GROUND_TRUTH)

    def generate_gemini_auto_complete(self) -> None:
        self.open_ai_dialog(provider=AIProvider.GEMINI, action=AIActionKind.AUTO_COMPLETE)

    def _cancel_gemini_stream(self) -> None:
        self._ai_controller.stop_active_generation()

    def _on_gemini_stream_chunk(self, text: str) -> None:
        self._ai_controller._on_gemini_stream_chunk(text)

    def _on_gemini_stream_limit_reached(self) -> None:
        self._ai_controller._on_gemini_stream_limit_reached()

    def _on_gemini_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        self._ai_controller._on_gemini_stream_meta(meta_payload)

    def _on_gemini_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        self._ai_controller._on_gemini_stream_fact(fact_payload)

    def _on_gemini_stream_completed(self, extraction_obj: Any) -> None:
        self._ai_controller._on_gemini_stream_completed(extraction_obj)

    def _on_gemini_stream_failed(self, message: str) -> None:
        self._ai_controller._on_gemini_stream_failed(message)

    def _on_gemini_stream_finished(self) -> None:
        self._ai_controller._on_gemini_stream_finished()

    def generate_qwen_ground_truth(self) -> None:
        self.open_ai_dialog(provider=AIProvider.QWEN, action=AIActionKind.GROUND_TRUTH)

    def _cancel_qwen_stream(self) -> None:
        self._ai_controller.stop_active_generation()

    def _on_qwen_stream_chunk(self, text: str) -> None:
        self._ai_controller._on_qwen_stream_chunk(text)

    def _on_qwen_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        self._ai_controller._on_qwen_stream_meta(meta_payload)

    def _on_qwen_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        self._ai_controller._on_qwen_stream_fact(fact_payload)

    def _on_qwen_stream_completed(self, extraction_obj: Any) -> None:
        self._ai_controller._on_qwen_stream_completed(extraction_obj)

    def _on_qwen_stream_failed(self, message: str) -> None:
        self._ai_controller._on_qwen_stream_failed(message)

    def _on_qwen_stream_finished(self) -> None:
        self._ai_controller._on_qwen_stream_finished()

    def _capture_current_state(self) -> None:
        if self.current_index < 0 or self._is_restoring_history:
            return
        self.document_meta = self._document_meta_from_ui()
        page_name = self.page_images[self.current_index].name
        meta = self._page_meta_from_ui()
        facts: List[BoxRecord] = []
        for item in self._sorted_fact_items():
            facts.append(BoxRecord(bbox=bbox_to_dict(item_scene_rect(item)), fact=normalize_fact_data(item.fact_data)))
        self.page_states[page_name] = PageState(meta=meta, facts=facts)

    def apply_entity_name_to_all_missing_pages(self) -> None:
        if not self.page_images:
            return
        entity_name = self.entity_name_edit.text().strip()
        if not entity_name:
            QMessageBox.information(
                self,
                "Entity name required",
                "Enter an entity_name first, then click 'Apply Entity'.",
            )
            return

        apply_mode = _prompt_entity_apply_mode(self, entity_name)
        if apply_mode is None:
            return

        self._capture_current_state()
        overwrite_existing = apply_mode == "force_all"
        updated = apply_entity_name_to_pages(
            self.page_states,
            self.page_images,
            entity_name,
            overwrite_existing=overwrite_existing,
        )
        if hasattr(self, "_populate_page_thumbnails"):
            self._populate_page_thumbnails()
        self.show_page(self.current_index)
        self._record_history_snapshot()
        if overwrite_existing:
            message = f"Applied entity_name to {updated} page(s) across the PDF."
        else:
            message = f"Applied entity_name to {updated} page(s) with empty entity_name."
        self.statusBar().showMessage(message, 5000)

    def _page_is_annotated(self, index: int) -> bool:
        if index < 0 or index >= len(self.page_images):
            return False
        page_name = self.page_images[index].name
        state = self.page_states.get(page_name, self._default_state(index))
        return page_has_annotation(state, index)

    def _thumbnail_icon_for_page(self, page_path: Path, index: int) -> QIcon:
        pixmap = QPixmap(str(page_path))
        icon_size = self.page_thumb_list.iconSize()
        page_name = page_path.name
        issue_summary = self._page_issue_summaries.get(page_name)
        if issue_summary is None:
            issue_summary = self._page_issue_summary_for_page(page_name)
            self._page_issue_summaries[page_name] = issue_summary
        if pixmap.isNull():
            fallback = QPixmap(icon_size)
            fallback.fill(QColor(215, 225, 236))
            return QIcon(fallback)
        thumb = pixmap.scaled(icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(icon_size)
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        x = (icon_size.width() - thumb.width()) // 2
        y = (icon_size.height() - thumb.height()) // 2
        painter.drawPixmap(x, y, thumb)
        if issue_summary.reg_flag_count > 0:
            self._draw_issue_badge(
                painter,
                6,
                6,
                self._issue_badge_text(issue_summary.reg_flag_count),
                "#dc2626",
            )
        if issue_summary.warning_count > 0:
            badge_width = 22 if len(self._issue_badge_text(issue_summary.warning_count)) <= 2 else 28
            self._draw_issue_badge(
                painter,
                max(6, icon_size.width() - badge_width - 6),
                6,
                self._issue_badge_text(issue_summary.warning_count),
                "#d97706",
            )
        if (
            issue_summary.reg_flag_count == 0
            and issue_summary.warning_count == 0
            and self._page_is_annotated(index)
        ):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#0f9fb6"))
            painter.drawRoundedRect(6, 6, 24, 18, 8, 8)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(6, 6, 24, 18), Qt.AlignCenter, "OK")
        if self._page_annotation_note_for_index(index):
            self._draw_issue_badge(
                painter,
                max(6, icon_size.width() - 28),
                max(6, icon_size.height() - 24),
                "RV",
                "#2563eb",
            )
        page_status = self._page_annotation_status_for_index(index)
        if page_status == "approved":
            self._draw_issue_badge(
                painter,
                6,
                max(6, icon_size.height() - 24),
                "AP",
                "#14804a",
            )
        elif page_status == "flagged":
            self._draw_issue_badge(
                painter,
                6,
                max(6, icon_size.height() - 24),
                "FG",
                "#b54708",
            )
        painter.end()
        return QIcon(canvas)

    def _refresh_thumbnail_for_index(self, index: int) -> None:
        if not hasattr(self, "page_thumb_list"):
            return
        if index < 0 or index >= len(self.page_images):
            return
        row = self._thumbnail_row_for_page_index(index)
        if row < 0:
            return
        item = self.page_thumb_list.item(row)
        if item is None:
            return
        page_path = self.page_images[index]
        page_summary = self._page_issue_summaries.get(page_path.name, PageIssueSummary(page_image=page_path.name))
        item.setIcon(self._thumbnail_icon_for_page(page_path, index))
        item.setToolTip(self._thumbnail_tooltip_text(page_summary, index))

    def _load_existing_annotations(self) -> None:
        if not self.annotations_path.exists():
            return
        try:
            payload = load_any_schema(json.loads(self.annotations_path.read_text(encoding="utf-8")))
        except Exception as exc:
            QMessageBox.warning(self, "Failed to load annotations", str(exc))
            return
        self.page_states = load_page_states(payload, [p.name for p in self.page_images])
        self.document_meta = extract_document_meta(payload)
        self._recompute_all_page_issues(emit=False)
        self._populate_page_thumbnails()

    def _populate_page_thumbnails(self) -> None:
        if not hasattr(self, "page_thumb_list"):
            return
        visible_indices = self._visible_thumbnail_page_indices()
        self._thumbnail_page_indices = list(visible_indices)
        self._thumbnail_row_lookup = {page_index: row for row, page_index in enumerate(visible_indices)}
        self.page_thumb_list.blockSignals(True)
        self.page_thumb_list.clear()
        icon_size = self.page_thumb_list.iconSize()
        for page_index in visible_indices:
            page_path = self.page_images[page_index]
            item = QListWidgetItem(str(page_index + 1))
            page_summary = self._page_issue_summaries.get(page_path.name, PageIssueSummary(page_image=page_path.name))
            item.setData(Qt.UserRole, page_index)
            item.setToolTip(self._thumbnail_tooltip_text(page_summary, page_index))
            item.setTextAlignment(Qt.AlignCenter)
            item.setIcon(self._thumbnail_icon_for_page(page_path, page_index))
            item.setSizeHint(QSize(icon_size.width() + 20, icon_size.height() + 30))
            self.page_thumb_list.addItem(item)
        row = self._thumbnail_row_for_page_index(self.current_index)
        self.page_thumb_list.setCurrentRow(row)
        self.page_thumb_list.blockSignals(False)

    def _on_page_jump_requested(self, page_number: int) -> None:
        if not self.page_images:
            return
        index = max(0, min(int(page_number) - 1, len(self.page_images) - 1))
        if index == self.current_index:
            return
        self.show_page(index)

    def _on_thumbnail_row_changed(self, row: int) -> None:
        index = self._page_index_for_thumbnail_row(row)
        if index < 0 or index >= len(self.page_images):
            return
        if index == self.current_index:
            return
        self.show_page(index)

    def show_page(self, index: int) -> None:
        if index < 0 or index >= len(self.page_images):
            return

        if not self._is_restoring_history:
            self._capture_current_state()
        self._clear_equation_candidate()
        self.current_index = index
        page_path = self.page_images[index]
        page_name = page_path.name
        state = self.page_states.get(page_name, self._default_state(index))
        self.page_states[page_name] = state

        pixmap = QPixmap(str(page_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Image load failed", f"Could not open {page_path}")
            return

        self.scene.clear()
        pix = self.scene.addPixmap(pixmap)
        pix.setTransformationMode(Qt.SmoothTransformation)
        pix.setZValue(-10)
        image_rect = QRectF(pixmap.rect())
        pan_margin = max(400.0, max(image_rect.width(), image_rect.height()) * 0.75)
        self.scene.setSceneRect(image_rect.adjusted(-pan_margin, -pan_margin, pan_margin, pan_margin))
        self.scene.set_image_rect(image_rect)

        for record in state.facts:
            rect = dict_to_rect(record.bbox).intersected(self.scene.image_rect)
            if rect.width() < 1 or rect.height() < 1:
                continue
            self.scene.addItem(AnnotRectItem(rect, record.fact))

        meta = PageMeta.model_validate({**self._default_meta(index), **(state.meta or {})}).model_dump(mode="json")
        self._is_loading_page = True
        try:
            self.entity_name_edit.setText(meta.get("entity_name") or "")
            self.page_num_edit.setText(meta.get("page_num") or "")
            type_value = meta.get("page_type") or PageType.other.value
            type_idx = self.type_combo.findText(type_value)
            self.type_combo.setCurrentIndex(type_idx if type_idx >= 0 else 0)
            statement_type_value = str(meta.get("statement_type") or "")
            statement_type_idx = self.statement_type_combo.findText(statement_type_value)
            self.statement_type_combo.setCurrentIndex(max(0, statement_type_idx))
            self.title_edit.setText(meta.get("title") or "")
            self.page_annotation_note_edit.setText(meta.get("annotation_note") or "")
            self._page_annotation_status = _normalize_page_annotation_status(meta.get("annotation_status"))
            self._refresh_page_annotation_status_ui()
            self._set_document_meta_ui(self.document_meta)
        finally:
            self._is_loading_page = False

        self.page_label.setText(f"Page {index + 1}/{len(self.page_images)} - {page_name}")
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.page_images) - 1)
        self.page_jump_spin.blockSignals(True)
        self.page_jump_spin.setValue(index + 1)
        self.page_jump_spin.blockSignals(False)
        self.page_thumb_list.blockSignals(True)
        thumb_row = self._thumbnail_row_for_page_index(index)
        self.page_thumb_list.setCurrentRow(thumb_row)
        current_thumb = self.page_thumb_list.item(thumb_row) if thumb_row >= 0 else None
        if current_thumb is not None:
            self.page_thumb_list.scrollToItem(current_thumb, QAbstractItemView.PositionAtCenter)
        self.page_thumb_list.blockSignals(False)
        self.refresh_facts_list()
        self._refresh_page_json_dialog()
        self.schedule_auto_fit_current_page()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._apply_pending_auto_fit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_pending_auto_fit()

    def _scene_fact_items(self) -> List[AnnotRectItem]:
        return [item for item in self.scene.items() if isinstance(item, AnnotRectItem) and self._is_alive_fact_item(item)]

    def _fact_num_for_item(self, item: AnnotRectItem) -> int | None:
        if not self._is_alive_fact_item(item):
            return None
        return _coerce_positive_int(normalize_fact_data(item.fact_data).get("fact_num"))

    def _ensure_fact_numbers_for_items(self, ordered_items: List[AnnotRectItem]) -> None:
        alive_items = [item for item in ordered_items if self._is_alive_fact_item(item)]
        if not alive_items:
            return
        current_facts = [normalize_fact_data(item.fact_data) for item in alive_items]
        current_fact_nums = [_coerce_positive_int(fact.get("fact_num")) for fact in current_facts]
        assigned_fact_nums = [int(fact_num) for fact_num in current_fact_nums if fact_num is not None]

        if (
            len(set(assigned_fact_nums)) == len(assigned_fact_nums)
            and assigned_fact_nums == list(range(1, len(assigned_fact_nums) + 1))
        ):
            next_fact_num = len(assigned_fact_nums) + 1
            resequenced_facts = []
            for current_fact, fact_num in zip(current_facts, current_fact_nums):
                if fact_num is None:
                    resequenced_facts.append(normalize_fact_data({**current_fact, "fact_num": next_fact_num}))
                    next_fact_num += 1
                else:
                    resequenced_facts.append(current_fact)
        else:
            resequenced_facts = resequence_fact_numbers_and_remap_fact_equations(current_facts)
        for item, current_fact, resequenced_fact in zip(alive_items, current_facts, resequenced_facts):
            if resequenced_fact != current_fact:
                item.fact_data = resequenced_fact

    def _sorted_fact_items(self) -> List[AnnotRectItem]:
        items = self._scene_fact_items()
        if len(items) <= 1:
            self._ensure_fact_numbers_for_items(items)
            return items
        facts_for_order: list[dict[str, Any]] = []
        for item in items:
            facts_for_order.append({"bbox": bbox_to_dict(item_scene_rect(item))})
        direction = self._active_reading_direction()
        ordered_indices = canonical_fact_order_indices(
            facts_for_order,
            direction="rtl" if direction == "rtl" else "ltr",
            row_tolerance_ratio=0.35,
            row_tolerance_min_px=6.0,
        )
        ordered_items = [items[idx] for idx in ordered_indices if 0 <= idx < len(items)]
        self._ensure_fact_numbers_for_items(ordered_items)
        return ordered_items

    def _apply_fact_order_labels(self) -> None:
        indexed_item_ids = {id(item) for item in self._fact_items}
        for item in self._fact_items:
            if self._is_alive_fact_item(item):
                item.set_order_label(None, visible=False)
        for scene_item in self.scene.items():
            if (
                isinstance(scene_item, AnnotRectItem)
                and self._is_alive_fact_item(scene_item)
                and id(scene_item) not in indexed_item_ids
            ):
                scene_item.set_order_label(None, visible=False)

    @staticmethod
    def _fact_has_matching_saved_equation(fact: Dict[str, Any]) -> bool:
        active_bundle = _active_equation_bundle_from_fact_payload(fact) or {}
        saved_equation = str(active_bundle.get("equation") or "").strip()
        if not saved_equation:
            return False
        preview = _evaluate_equation_string(saved_equation)
        if preview is None:
            return False
        tone, _message = _equation_result_match_state(
            preview,
            fact.get("value"),
            fact.get("natural_sign"),
        )
        return tone == "ok"

    def _apply_equation_match_visuals(self) -> None:
        indexed_item_ids = {id(item) for item in self._fact_items}
        for item in self._fact_items:
            if not self._is_alive_fact_item(item):
                continue
            fact = normalize_fact_data(item.fact_data)
            item.set_equation_match_ok(self._fact_has_matching_saved_equation(fact))
        for scene_item in self.scene.items():
            if (
                isinstance(scene_item, AnnotRectItem)
                and self._is_alive_fact_item(scene_item)
                and id(scene_item) not in indexed_item_ids
            ):
                scene_item.set_equation_match_ok(False)

    def refresh_facts_list(self, *, refresh_issues: bool = True, preserve_sequence: bool = False) -> None:
        selected = self._selected_fact_item()
        selected_items = set(self._selected_fact_items())
        self._fact_items = self._sorted_fact_items() if not preserve_sequence else self._fact_items_in_display_sequence()
        if self.current_index >= 0 and self.current_index < len(self.page_images):
            current_page_name = self.page_images[self.current_index].name
            current_state = self.page_states.get(current_page_name)
            statement_type = None
            if current_state is not None and isinstance(current_state.meta, dict):
                statement_type = str(current_state.meta.get("statement_type") or "").strip() or None
            rebuilt_facts, _equation_findings = audit_and_rebuild_financial_facts(
                [normalize_fact_data(item.fact_data) for item in self._fact_items],
                statement_type=statement_type,
                apply_repairs=True,
            )
            for item, rebuilt_fact in zip(self._fact_items, rebuilt_facts):
                if rebuilt_fact != normalize_fact_data(item.fact_data):
                    item.fact_data = rebuilt_fact
        self._apply_fact_order_labels()
        self._apply_equation_match_visuals()
        self._syncing_selection = True
        self.facts_list.clear()
        facts_count = len(self._fact_items)
        self.facts_count_label.setText(f"{facts_count} fact{'s' if facts_count != 1 else ''}" if facts_count else "No facts")
        self.facts_count_label.setProperty("tone", "accent" if facts_count else "warn")
        self.facts_count_label.style().unpolish(self.facts_count_label)
        self.facts_count_label.style().polish(self.facts_count_label)
        selected_row = -1
        for idx, item in enumerate(self._fact_items, start=1):
            rect = item_scene_rect(item)
            normalized_fact = normalize_fact_data(item.fact_data)
            value = str(normalized_fact.get("value") or "")
            path = " > ".join(normalized_fact.get("path") or [])
            comment_ref = str(normalized_fact.get("comment_ref") or "")
            note_num = "" if normalized_fact.get("note_num") is None else str(normalized_fact.get("note_num"))
            note_name = str(normalized_fact.get("note_name") or "")
            fact_num = self._fact_num_for_item(item) or idx
            summary = f"#{fact_num} [{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}] {value}"
            if path:
                summary = f"{summary} | {path}"
            if comment_ref:
                trimmed_comment_ref = (comment_ref[:32] + "...") if len(comment_ref) > 35 else comment_ref
                summary = f"{summary} | comment_ref: {trimmed_comment_ref}"
            if note_num:
                trimmed_note_num = (note_num[:32] + "...") if len(note_num) > 35 else note_num
                summary = f"{summary} | note_num: {trimmed_note_num}"
            if note_name:
                trimmed_note_name = (note_name[:32] + "...") if len(note_name) > 35 else note_name
                summary = f"{summary} | note_name: {trimmed_note_name}"
            summary = f"{summary} | note_flag: {bool(normalized_fact.get('note_flag'))}"
            if normalized_fact.get("note_ref"):
                summary = f"{summary} | note_ref: {normalized_fact.get('note_ref')}"
            if normalized_fact.get("period_type"):
                summary = f"{summary} | period_type: {normalized_fact.get('period_type')}"
            if normalized_fact.get("duration_type"):
                summary = f"{summary} | duration_type: {normalized_fact.get('duration_type')}"
            if normalized_fact.get("recurring_period"):
                summary = f"{summary} | recurring_period: {normalized_fact.get('recurring_period')}"
            if normalized_fact.get("value_context"):
                summary = f"{summary} | value_context: {normalized_fact.get('value_context')}"
            if normalized_fact.get("natural_sign"):
                summary = f"{summary} | natural_sign: {normalized_fact.get('natural_sign')}"
            if normalized_fact.get("row_role"):
                summary = f"{summary} | row_role: {normalized_fact.get('row_role')}"
            if normalized_fact.get("path_source"):
                summary = f"{summary} | path_source: {normalized_fact.get('path_source')}"
            active_bundle = _active_equation_bundle_from_fact_payload(normalized_fact) or {}
            if active_bundle.get("equation"):
                equation = str(active_bundle.get("equation") or "")
                trimmed_equation = (equation[:40] + "...") if len(equation) > 43 else equation
                summary = f"{summary} | equation: {trimmed_equation}"
            if active_bundle.get("fact_equation"):
                fact_equation = str(active_bundle.get("fact_equation") or "")
                trimmed_fact_equation = (fact_equation[:40] + "...") if len(fact_equation) > 43 else fact_equation
                summary = f"{summary} | fact_equation: {trimmed_fact_equation}"
            self.facts_list.addItem(QListWidgetItem(summary))
            list_item = self.facts_list.item(idx - 1)
            if list_item is not None and item in selected_items:
                list_item.setSelected(True)
            if item is selected and selected_row < 0:
                selected_row = idx - 1
        if selected_row >= 0 and self.facts_list.selectionModel() is not None:
            index = self.facts_list.model().index(selected_row, 0)
            self.facts_list.selectionModel().setCurrentIndex(index, QItemSelectionModel.NoUpdate)
        self._syncing_selection = False
        self._sync_fact_editor_from_selection()
        if refresh_issues:
            self._refresh_current_page_issues(use_current_fact_items=True)
        self._update_batch_controls()
        self._update_gemini_fill_button_state()
        self._update_gemini_complete_button_state()

    def _fact_items_in_display_sequence(self) -> List[AnnotRectItem]:
        ordered: list[AnnotRectItem] = []
        seen_ids: set[int] = set()
        for item in self._fact_items:
            if not self._is_alive_fact_item(item):
                continue
            ordered.append(item)
            seen_ids.add(id(item))
        for item in self.scene.items():
            if not isinstance(item, AnnotRectItem) or not self._is_alive_fact_item(item):
                continue
            if id(item) in seen_ids:
                continue
            ordered.append(item)
            seen_ids.add(id(item))
        return ordered

    def _on_equation_reference_selection_started(self) -> None:
        selected_items = self._selected_fact_items()
        self._clear_equation_candidate()
        if len(selected_items) != 1:
            self.statusBar().showMessage("Select one target bbox before using Alt-drag.", 2500)
            return
        self._equation_target_item = selected_items[0]
        self._refresh_equation_panel()

    def _sort_items_for_equation(self, items: List[AnnotRectItem]) -> List[AnnotRectItem]:
        ordered: list[AnnotRectItem] = []
        seen_ids: set[int] = set()
        for item in items:
            if not self._is_alive_fact_item(item):
                continue
            item_id = id(item)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            ordered.append(item)
        return ordered

    def _on_equation_reference_selection_changed(self, items: object) -> None:
        if not self._is_alive_fact_item(self._equation_target_item):
            self._clear_equation_candidate()
            return

        target_item = self._equation_target_item
        assert target_item is not None
        selected_items = self._selected_fact_items()
        if len(selected_items) != 1 or selected_items[0] is not target_item:
            self._clear_equation_candidate()
            return

        raw_items = items if isinstance(items, list) else []
        reference_items = [
            item
            for item in raw_items
            if isinstance(item, AnnotRectItem) and self._is_alive_fact_item(item) and item is not target_item
        ]
        ordered_reference_items = self._sort_items_for_equation(reference_items)
        self._set_equation_reference_preview_items(ordered_reference_items)
        saved_target_fact = normalize_fact_data(target_item.fact_data)
        active_bundle = _active_equation_bundle_from_fact_payload(saved_target_fact) or {}
        operator_by_fact_num = _fact_equation_operator_map(active_bundle.get("fact_equation"))

        candidate_text, result_text, fact_candidate_text, invalid_values, structured_terms = _build_equation_candidate_from_facts(
            [
                {
                    "fact_num": self._fact_num_for_item(item),
                    "value": normalize_fact_data(item.fact_data).get("value"),
                    "natural_sign": normalize_fact_data(item.fact_data).get("natural_sign"),
                    "operator": operator_by_fact_num.get(self._fact_num_for_item(item), "+"),
                }
                for item in ordered_reference_items
            ]
        )
        self._equation_candidate_text = candidate_text
        self._equation_candidate_fact_text = fact_candidate_text
        self._equation_candidate_result_text = result_text
        self._equation_candidate_invalid_values = invalid_values
        self._equation_candidate_terms = structured_terms
        self._refresh_equation_panel()

    def _on_equation_approval_requested(self) -> None:
        if not self.apply_equation_btn.isEnabled():
            return
        self.apply_equation_to_selected_fact()

    def clear_equation_from_selected_fact(self) -> None:
        selected_items = [item for item in self._selected_fact_items() if self._is_alive_fact_item(item)]
        if not selected_items:
            self._clear_equation_candidate()
            self.statusBar().showMessage("Select at least one bbox to clear equation.", 2200)
            return

        cleared_count = 0
        for item in selected_items:
            current = normalize_fact_data(item.fact_data)
            if not _equation_bundles_from_fact_payload(current):
                continue
            updated = normalize_fact_data(
                {
                    **current,
                    "equations": None,
                }
            )
            if updated == current:
                continue
            item.fact_data = updated
            cleared_count += 1

        if cleared_count == 0:
            self.statusBar().showMessage("Selected facts have no saved equation.", 2200)
            return
        self._clear_equation_candidate()
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._focus_equation_panel()
        if cleared_count == 1:
            self.statusBar().showMessage("Cleared equation from 1 selected fact.", 2500)
        else:
            self.statusBar().showMessage(f"Cleared equation from {cleared_count} selected facts.", 2500)

    def apply_equation_to_selected_fact(self) -> None:
        target_item = self._equation_target_item
        if not self._is_alive_fact_item(target_item):
            self._clear_equation_candidate()
            return
        if self._equation_candidate_text is None or self._equation_candidate_result_text is None:
            self.statusBar().showMessage("No calculable equation preview to apply.", 2500)
            return

        selected_items = self._selected_fact_items()
        if len(selected_items) != 1 or selected_items[0] is not target_item:
            self._clear_equation_candidate()
            self.statusBar().showMessage("Re-select the target bbox before applying the equation.", 3000)
            return

        current = normalize_fact_data(target_item.fact_data)
        has_fact_references = bool(str(self._equation_candidate_fact_text or "").strip())
        new_bundle = _normalize_equation_bundle_payload(
            {
                "equation": self._equation_candidate_text,
                "fact_equation": self._equation_candidate_fact_text,
            }
        )
        if new_bundle is None:
            self.statusBar().showMessage("No valid equation preview to apply.", 2500)
            return
        existing_bundles = _equation_bundles_from_fact_payload(current)
        updated = _fact_payload_with_active_equation_bundle(
            {**current, "row_role": "total" if has_fact_references else current.get("row_role")},
            [new_bundle, *existing_bundles[1:]] if existing_bundles else [new_bundle],
            active_index=0,
        )
        if updated == current:
            self.view.disable_calculate_drag_mode()
            self.scene.reset_equation_reference_session()
            self._clear_equation_candidate()
            self.statusBar().showMessage("Equation is already applied to the selected fact.", 2500)
            return

        target_item.fact_data = updated
        self.view.disable_calculate_drag_mode()
        self.scene.reset_equation_reference_session()
        self.refresh_facts_list()
        self._record_history_snapshot()
        self._clear_equation_candidate()
        self._focus_equation_panel()
        self.statusBar().showMessage("Applied equation to selected fact.", 2500)

    def _selected_fact_item(self) -> Optional[AnnotRectItem]:
        for item in self.scene.selectedItems():
            if isinstance(item, AnnotRectItem) and self._is_alive_fact_item(item):
                return item
        row = self.facts_list.currentRow()
        if 0 <= row < len(self._fact_items):
            item = self._fact_items[row]
            if self._is_alive_fact_item(item):
                return item
        return None

    def _on_box_created(self, item: AnnotRectItem) -> None:
        self.scene.clearSelection()
        item.setSelected(True)
        self.refresh_facts_list()
        self._refresh_thumbnail_for_index(self.current_index)
        self.fact_value_edit.setFocus()
        self.fact_value_edit.selectAll()
        self._record_history_snapshot()

    def _on_box_duplicated(self, item: AnnotRectItem) -> None:
        if not self._is_alive_fact_item(item):
            return
        self.refresh_facts_list()
        self._refresh_thumbnail_for_index(self.current_index)
        self._record_history_snapshot()
        self.statusBar().showMessage("Duplicated bbox. Drag to place it.", 2500)

    def _on_box_double_clicked(self, item: AnnotRectItem) -> None:
        self.scene.clearSelection()
        item.setSelected(True)
        self.refresh_facts_list()
        self.fact_value_edit.setFocus()
        self.fact_value_edit.selectAll()

    def _on_box_geometry_changed(self, _item: AnnotRectItem) -> None:
        self.refresh_facts_list(refresh_issues=False, preserve_sequence=True)
        self._record_history_snapshot()

    def _on_meta_edited(self) -> None:
        if self._is_loading_page or self._is_restoring_history:
            return
        previous = normalize_document_meta(self.document_meta)
        self.document_meta = self._document_meta_from_ui()
        self._refresh_page_annotation_status_ui()
        if self.document_meta != previous:
            self.refresh_facts_list()
        else:
            self._refresh_current_page_issues()
        self._populate_page_thumbnails()
        self._record_history_snapshot()

    def _set_current_page_annotation_status(self, status: str) -> bool:
        normalized_status = _normalize_page_annotation_status(status)
        if normalized_status is None:
            return False
        if normalized_status == _normalize_page_annotation_status(self._page_annotation_status):
            self.statusBar().showMessage(
                f"Page already marked as {self._page_annotation_status_text(normalized_status).lower()}.",
                2200,
            )
            return False
        self._page_annotation_status = normalized_status
        self._refresh_page_annotation_status_ui()
        self._on_meta_edited()
        self.statusBar().showMessage(
            f"Page marked as {self._page_annotation_status_text(normalized_status).lower()}.",
            2200,
        )
        return True

    def approve_current_page_and_continue(self) -> None:
        changed = self._set_current_page_annotation_status("approved")
        next_index = self._next_page_index_in_thumbnail_order()
        if next_index is not None and next_index != self.current_index:
            self.show_page(next_index)
            return
        if changed:
            self.statusBar().showMessage("Page approved. No additional page to continue to.", 2600)

    def flag_current_page_for_review(self) -> None:
        self._set_current_page_annotation_status("flagged")

    def _on_scene_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        self._clear_equation_candidate()
        selected_items = set(self._selected_fact_items())
        selected = self._selected_fact_item()
        self._syncing_selection = True
        self.facts_list.clearSelection()
        selected_row = -1
        for row, item in enumerate(self._fact_items):
            if item in selected_items:
                list_item = self.facts_list.item(row)
                if list_item is not None:
                    list_item.setSelected(True)
                if item is selected and selected_row < 0:
                    selected_row = row
        if selected_row >= 0 and self.facts_list.selectionModel() is not None:
            index = self.facts_list.model().index(selected_row, 0)
            self.facts_list.selectionModel().setCurrentIndex(index, QItemSelectionModel.NoUpdate)
        self._syncing_selection = False
        self._sync_fact_editor_from_selection()
        self._update_batch_controls()
        self._update_gemini_fill_button_state()
        self._update_gemini_complete_button_state()
        self._refresh_page_json_dialog()

    def _on_fact_list_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        self._clear_equation_candidate()
        selected_rows = [self.facts_list.row(list_item) for list_item in self.facts_list.selectedItems()]
        self._syncing_selection = True
        self.scene.clearSelection()
        for row in selected_rows:
            if 0 <= row < len(self._fact_items):
                item = self._fact_items[row]
                if self._is_alive_fact_item(item):
                    item.setSelected(True)
        self._syncing_selection = False
        selected_items = self._selected_fact_items()
        if selected_items:
            self._focus_fact_items_in_view(selected_items)
        self._sync_fact_editor_from_selection()
        self._update_batch_controls()
        self._update_gemini_fill_button_state()
        self._update_gemini_complete_button_state()
        self._refresh_page_json_dialog()

    def eventFilter(self, watched: QObject, event: QObject) -> bool:
        return super().eventFilter(watched, event)

    def _selected_fact_items(self) -> List[AnnotRectItem]:
        return [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, AnnotRectItem) and self._is_alive_fact_item(item)
        ]

    def _nudge_selected_facts(self, dx: int, dy: int) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            return
        if dx == 0 and dy == 0:
            return

        bounds = self.scene.image_rect
        min_left = min(item_scene_rect(item).left() for item in selected_items)
        max_right = max(item_scene_rect(item).right() for item in selected_items)
        min_top = min(item_scene_rect(item).top() for item in selected_items)
        max_bottom = max(item_scene_rect(item).bottom() for item in selected_items)

        min_dx = bounds.left() - min_left
        max_dx = bounds.right() - max_right
        min_dy = bounds.top() - min_top
        max_dy = bounds.bottom() - max_bottom
        bounded_dx = max(min(float(dx), max_dx), min_dx)
        bounded_dy = max(min(float(dy), max_dy), min_dy)

        if abs(bounded_dx) < 0.001 and abs(bounded_dy) < 0.001:
            return

        for item in selected_items:
            pos = item.pos()
            item.setPos(QPointF(pos.x() + bounded_dx, pos.y() + bounded_dy))

        self.refresh_facts_list(refresh_issues=False, preserve_sequence=True)
        self._refresh_current_page_issues(use_current_fact_items=True)
        self._record_history_snapshot()

    def select_all_bboxes(self) -> None:
        alive_items = [item for item in self._fact_items if self._is_alive_fact_item(item)]
        if not alive_items:
            self.statusBar().showMessage("No bboxes on this page.", 2000)
            return
        self._syncing_selection = True
        self.scene.clearSelection()
        for item in alive_items:
            item.setSelected(True)
        self._syncing_selection = False
        self._on_scene_selection_changed()
        self.statusBar().showMessage(f"Selected {len(alive_items)} bboxes on this page.", 2000)

    def _shift_rect_inside_image(self, rect: QRectF) -> QRectF:
        bounds = self.scene.image_rect
        shifted = QRectF(rect)
        if shifted.right() > bounds.right():
            shifted.moveRight(bounds.right())
        if shifted.bottom() > bounds.bottom():
            shifted.moveBottom(bounds.bottom())
        if shifted.left() < bounds.left():
            shifted.moveLeft(bounds.left())
        if shifted.top() < bounds.top():
            shifted.moveTop(bounds.top())
        return shifted

    def _suggest_duplicate_rect(self, source_rect: QRectF) -> QRectF:
        offsets = [
            (12, 12),
            (-12, -12),
            (12, -12),
            (-12, 12),
            (24, 0),
            (0, 24),
            (-24, 0),
            (0, -24),
        ]
        for dx, dy in offsets:
            candidate = QRectF(source_rect)
            candidate.translate(dx, dy)
            candidate = self._shift_rect_inside_image(candidate)
            if abs(candidate.x() - source_rect.x()) > 0.01 or abs(candidate.y() - source_rect.y()) > 0.01:
                return candidate
        return QRectF(source_rect)

    def duplicate_selected_fact(self) -> None:
        item = self._selected_fact_item()
        if item is None:
            QMessageBox.information(self, "No selection", "Select a bounding box to duplicate.")
            return

        source_rect = QRectF(item_scene_rect(item))
        new_rect = self._suggest_duplicate_rect(source_rect)
        copy_fact = normalize_fact_data(deepcopy(item.fact_data))
        new_item = AnnotRectItem(new_rect, copy_fact)
        self.scene.addItem(new_item)
        self.scene.clearSelection()
        new_item.setSelected(True)
        self.refresh_facts_list()
        self._refresh_thumbnail_for_index(self.current_index)
        self._record_history_snapshot()
        self.statusBar().showMessage("Duplicated selected bbox.", 2500)

    def focus_fact_annotation_panel(self) -> None:
        selected_count = len(self._selected_fact_items())
        anchor_widget = self.fact_editor_box if selected_count == 1 else self.facts_list
        self.right_scroll.ensureWidgetVisible(anchor_widget, 0, 18)
        focus_widget: QWidget = self.fact_value_edit if selected_count == 1 else self.facts_list
        focus_widget.setFocus(Qt.ShortcutFocusReason)
        if focus_widget is self.fact_value_edit:
            self.fact_value_edit.selectAll()
        self.statusBar().showMessage("Focused fact annotation panel.", 1500)

    def _delete_shortcut_blocked_by_focus(self) -> bool:
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        if isinstance(focus_widget, (QLineEdit, QPlainTextEdit, QComboBox, QSpinBox)):
            return True
        if isinstance(focus_widget, QListWidget):
            if focus_widget in {self.fact_path_list, self.fact_equation_variants_list, self.fact_equation_terms_list}:
                return True
        return False

    def _delete_selected_fact_shortcut(self) -> None:
        focus_widget = QApplication.focusWidget()
        if focus_widget in {self.fact_equation_variants_list, self.fact_equation_variants_list.viewport()}:
            self.delete_selected_equation_variants_from_selected_fact()
            return
        if self._delete_shortcut_blocked_by_focus():
            self.statusBar().showMessage("Delete shortcut is disabled while editing fields.", 2200)
            return
        self.delete_selected_fact()

    def delete_selected_fact(self) -> None:
        selected_items = self._selected_fact_items()
        if not selected_items:
            item = self._selected_fact_item()
            if item is not None:
                selected_items = [item]
        if not selected_items:
            self.statusBar().showMessage("No bbox selected to delete.", 2000)
            return

        for item in selected_items:
            self.scene.removeItem(item)
        self.refresh_facts_list()
        self._refresh_thumbnail_for_index(self.current_index)
        self._record_history_snapshot()
        deleted_count = len(selected_items)
        if deleted_count == 1:
            self.statusBar().showMessage("Deleted selected bbox.", 2000)
        else:
            self.statusBar().showMessage(f"Deleted {deleted_count} selected bboxes.", 2500)

    @staticmethod
    def _payload_uses_legacy_schema(payload: Any) -> bool:
        return payload_uses_legacy_aliases(payload)

    def _backup_legacy_annotations_if_needed(self) -> Optional[dict[str, Any]]:
        if not self.annotations_path.exists():
            return None
        try:
            existing_payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not payload_requires_migration(existing_payload):
            return None
        direction_info = resolve_reading_direction(
            self.document_meta,
            payload=existing_payload,
            default_direction="rtl",
        )
        return create_annotation_backup(
            REPO_ROOT,
            self.annotations_path,
            reason="schema_migration",
            algo_version="annotator_schema_migration_v1",
            direction_source=str(direction_info.get("source") or "annotator_save"),
        )

    def _build_live_annotations_payload(
        self,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        self._capture_current_state()
        payload, equation_findings = build_annotations_payload_with_findings(
            self.images_dir,
            self.page_images,
            self.page_states,
            document_meta=self.document_meta,
        )
        _normalized_payload, format_findings = normalize_annotation_payload(payload)
        format_warning_findings = [
            finding
            for finding in format_findings
            if any(code in {"noncanonical_date", "placeholder_value", "noncanonical_value"} for code in finding.get("issue_codes", []))
        ]
        return payload, equation_findings, format_warning_findings

    def _page_json_title_and_text(self) -> tuple[str, str, dict[int, int]]:
        payload, _equation_findings, _format_warning_findings = self._build_live_annotations_payload()
        page_payload = payload["pages"][self.current_index]
        page_text = json.dumps(page_payload, indent=2, ensure_ascii=False)
        fact_start_positions: dict[int, int] = {}
        for match in re.finditer(r'"fact_num":\s*(\d+)', page_text):
            fact_num = int(match.group(1))
            start_pos = match.start()
            while start_pos > 0 and page_text[start_pos] != "{":
                start_pos -= 1
            fact_start_positions[fact_num] = start_pos
        return f"Page JSON - {page_payload.get('image', '')}", page_text, fact_start_positions

    def _refresh_page_json_dialog(self) -> None:
        dialog = self._page_json_dialog
        if dialog is None or not dialog.isVisible():
            return
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            return
        try:
            title, page_text, fact_start_positions = self._page_json_title_and_text()
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            dialog.close()
            return
        selected = self._selected_fact_item()
        fact_start_pos: Optional[int] = None
        if selected is not None:
            fact_num = normalize_fact_data(selected.fact_data).get("fact_num")
            if isinstance(fact_num, int):
                fact_start_pos = fact_start_positions.get(fact_num)
        dialog.set_page_json(title=title, page_text=page_text, fact_start_pos=fact_start_pos)

    def _trigger_save_annotations(self) -> bool:
        if self._save_shortcut_in_flight:
            return False
        self._save_shortcut_in_flight = True
        try:
            return self.save_annotations()
        finally:
            QTimer.singleShot(0, lambda: setattr(self, "_save_shortcut_in_flight", False))

    def keyPressEvent(self, event) -> None:
        if _is_save_key_event(event):
            if self._trigger_save_annotations():
                event.accept()
                return
        super().keyPressEvent(event)

    def event(self, event: QObject) -> bool:
        if isinstance(event, QKeyEvent) and event.type() == QEvent.KeyPress and _is_save_key_event(event):
            if self._trigger_save_annotations():
                event.accept()
                return True
        return super().event(event)

    def save_annotations(self) -> bool:
        try:
            payload, equation_findings, format_warning_findings = self._build_live_annotations_payload()
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return False
        warning_count = len(format_warning_findings) + len(equation_findings)
        serialized = serialize_annotations_json(payload)
        no_changes = False
        if self.annotations_path.exists():
            try:
                no_changes = self.annotations_path.read_text(encoding="utf-8") == serialized
            except OSError:
                no_changes = False
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        backup_info: Optional[dict[str, Any]] = None
        try:
            if not no_changes:
                backup_info = self._backup_legacy_annotations_if_needed()
            atomic_write_text(self.annotations_path, serialized, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        if no_changes:
            warning_suffix = f" | warnings={warning_count}" if warning_count else ""
            self.statusBar().showMessage(
                f"No changes detected. File is up to date: {self.annotations_path}{warning_suffix}",
                6000,
            )
            self.annotations_saved.emit(self.annotations_path)
            self.annotations_save_status.emit(
                {
                    "annotations_path": self.annotations_path,
                    "no_changes": True,
                    "warning_count": warning_count,
                    "format_warning_count": len(format_warning_findings),
                    "equation_warning_count": len(equation_findings),
                    "equation_findings": equation_findings,
                    "backup_path": None,
                }
            )
            self._mark_saved_content()
            return True
        warning_suffix = f" | warnings={warning_count}" if warning_count else ""
        backup_path: Optional[Path] = None
        if backup_info is not None:
            backup_path = Path(str(backup_info["backup_path"]))
        self.statusBar().showMessage(f"Saved: {self.annotations_path}{warning_suffix}", 6000)
        self.annotations_saved.emit(self.annotations_path)
        self.annotations_save_status.emit(
                {
                    "annotations_path": self.annotations_path,
                    "no_changes": False,
                    "warning_count": warning_count,
                    "format_warning_count": len(format_warning_findings),
                    "equation_warning_count": len(equation_findings),
                    "equation_findings": equation_findings,
                    "backup_path": backup_path,
                }
            )
        self._mark_saved_content()
        self._refresh_page_json_dialog()
        return True

    def has_unsaved_changes(self) -> bool:
        return self._has_unsaved_changes()

    def reset_all_approved_pages(self, *, mark_saved_state: bool = False) -> int:
        self._capture_current_state()
        changed = 0
        for idx, page_path in enumerate(self.page_images):
            page_name = page_path.name
            state = self.page_states.get(page_name, self._default_state(idx))
            meta = {**default_page_meta(idx), **(state.meta or {})}
            if _normalize_page_annotation_status(meta.get("annotation_status")) != "approved":
                self.page_states[page_name] = PageState(meta=meta, facts=list(state.facts))
                continue
            meta["annotation_status"] = None
            self.page_states[page_name] = PageState(meta=meta, facts=list(state.facts))
            changed += 1
        if changed and 0 <= self.current_index < len(self.page_images):
            current_name = self.page_images[self.current_index].name
            current_state = self.page_states.get(current_name, self._default_state(self.current_index))
            self._page_annotation_status = _normalize_page_annotation_status((current_state.meta or {}).get("annotation_status"))
            self._refresh_page_annotation_status_ui()
        if changed:
            self._populate_page_thumbnails()
            self._refresh_current_page_issues(use_current_fact_items=True)
            self._refresh_page_json_dialog()
        if mark_saved_state:
            self._mark_saved_content()
        return changed

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._pending_close_approved:
            self._pending_close_approved = False
            event.accept()
            return

        if self.confirm_close(prepare_for_close=True):
            event.accept()
            return
        event.ignore()

    def copy_displayed_image(self) -> None:
        if self.scene.image_rect.isNull():
            self.statusBar().showMessage("No image to copy.", 2000)
            return

        source_rect = QRectF(self.scene.image_rect)
        width = max(1, int(round(source_rect.width())))
        height = max(1, int(round(source_rect.height())))
        target_pixmap = QPixmap(width, height)
        target_pixmap.fill(Qt.white)

        painter = QPainter(target_pixmap)
        self.scene.render(painter, QRectF(0, 0, width, height), source_rect)
        painter.end()

        QApplication.clipboard().setPixmap(target_pixmap)
        self.statusBar().showMessage("Copied displayed page image to clipboard.", 2500)

    def show_current_page_json(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            QMessageBox.warning(self, "Page JSON", "No current page is loaded.")
            return
        try:
            title, page_text, fact_start_positions = self._page_json_title_and_text()
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        if self._page_json_dialog is None:
            dialog = PageJsonDialog(self)
            dialog.copy_btn.clicked.connect(
                lambda: (
                    QApplication.clipboard().setText(dialog.text_view.toPlainText()),
                    self.statusBar().showMessage("Copied current page JSON.", 2500),
                )
            )
            self._page_json_dialog = dialog
        selected = self._selected_fact_item()
        fact_start_pos: Optional[int] = None
        if selected is not None:
            fact_num = normalize_fact_data(selected.fact_data).get("fact_num")
            if isinstance(fact_num, int):
                fact_start_pos = fact_start_positions.get(fact_num)
        self._page_json_dialog.set_page_json(title=title, page_text=page_text, fact_start_pos=fact_start_pos)
        self._page_json_dialog.show()
        self._page_json_dialog.raise_()
        self._page_json_dialog.activateWindow()

    def show_help_dialog(self) -> None:
        help_text = (
            "Keyboard shortcuts\n"
            "- Ctrl/Cmd+S: Save annotations\n"
            "- Ctrl/Cmd+Z: Undo\n"
            "- Ctrl/Cmd+Y or Ctrl/Cmd+Shift+Z: Redo\n"
            "- Ctrl/Cmd+I: Import JSON\n"
            "- Ctrl/Cmd+G: Open AI window\n"
            "- Ctrl/Cmd+D: Duplicate selected bbox\n"
            "- Delete or Backspace: Delete selected bbox(es)\n"
            "- +: Zoom in when the page view is focused\n"
            "- -: Zoom out when the page view is focused\n"
            "- Ctrl+=: Zoom in\n"
            "- Ctrl+-: Zoom out\n"
            "- Ctrl+0: Fit page height to panel\n"
            "- Alt+F: Focus fact annotation panel\n"
            "- F1: Open this help\n"
            "\n"
            "Page navigation\n"
            "- Use Prev/Next buttons\n"
            "- Press A for previous page and D for next page when the page view is focused\n"
            "- Use the Go to page spinner in the top bar\n"
            "- Scroll and click the Pages thumbnail strip on the left\n"
            "- Toggle Lens in the top bar for magnified cursor inspection\n"
            "\n"
            "Selection and editing\n"
            "- Ctrl/Cmd+A: Select all bboxes on the current page when the page view is focused\n"
            "- Click on a bbox: Select only that bbox\n"
            "- Left-drag on empty page area: Rectangle-select bboxes\n"
            "- Shift + click or Shift + left-drag: Add to the current selection\n"
            "- Arrow keys: Move selected bbox(es) by 1 px\n"
            "- Shift+Arrow: Move selected bbox(es) by 10 px\n"
            "- Alt+Arrow: Grow selected bbox(es) in one direction by batch step\n"
            "- Alt+Shift+Arrow: Grow selected bbox(es) by 10x batch step\n"
            "- Hold Alt + drag: Build equation preview for the selected target bbox\n"
            "- While holding Alt, press Shift: Approve/apply current equation preview\n"
            "- Ctrl+Arrow: Pan page\n"
            "\n"
            "Mouse interactions\n"
            "- Cmd-drag on empty page area: Draw one new bbox (Ctrl-drag on non-macOS)\n"
            "- Drag selected bbox: Move it\n"
            "- Drag bbox edge/corner: Resize the active bbox\n"
            "- Use Alt+Arrow batch resize for multi-selected bboxes\n"
            "- Right-drag or middle-drag: Pan page\n"
            "- Cmd/Ctrl + drag on bbox: Duplicate and drag\n"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Help - Shortcuts")
        dialog.resize(760, 620)
        root = QVBoxLayout(dialog)

        text_view = QPlainTextEdit()
        text_view.setReadOnly(True)
        text_view.setPlainText(help_text)
        root.addWidget(text_view, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_btn)
        root.addLayout(actions)

        dialog.exec_()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate PDF page images with schema-aligned facts and metadata.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="PDF file path (auto-extracts images) or images directory path.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing page images (png/jpg/webp).",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Output annotations JSON path (default: data/annotations/<images-dir-name>.json).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF conversion DPI in PDF mode.",
    )
    return parser.parse_args(argv)


StartupContext = _StartupContext


def _default_annotations_path(images_dir: Path) -> Path:
    return _default_annotations_path_impl(images_dir)


def _resolve_startup_context(
    input_path_arg: Optional[str],
    images_dir_arg: Optional[str],
    annotations_arg: Optional[str],
) -> StartupContext:
    return _resolve_startup_context_impl(input_path_arg, images_dir_arg, annotations_arg)


def _missing_page_numbers(images_dir: Path, page_count: int) -> List[int]:
    return workspace_mod.missing_page_numbers(images_dir, page_count)


def _build_page_ranges(page_numbers: List[int]) -> List[tuple[int, int]]:
    return workspace_mod.build_page_ranges(page_numbers)


def _ensure_pdf_images(pdf_path: Path, images_dir: Path, dpi: int = 200) -> Dict[str, Any]:
    return workspace_mod.ensure_pdf_images(pdf_path, images_dir, dpi=dpi)


def _ensure_annotations_path_writable(annotations_path: Path) -> None:
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    if annotations_path.exists():
        if not os.access(annotations_path, os.W_OK):
            raise PermissionError(f"Annotations file is not writable: {annotations_path}")
    elif not os.access(annotations_path.parent, os.W_OK):
        raise PermissionError(f"Annotations directory is not writable: {annotations_path.parent}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        context = _resolve_startup_context(args.input_path, args.images_dir, args.annotations)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Startup mode: {context.mode}")
    if context.images_dir is not None:
        print(f"Images dir: {context.images_dir}")
    if context.annotations_path is not None:
        print(f"Annotations: {context.annotations_path}")
    if context.pdf_path is not None:
        print(f"PDF: {context.pdf_path}")

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    from .dashboard import DashboardWindow

    window = DashboardWindow(context, dpi=args.dpi)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
