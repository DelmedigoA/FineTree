"""Simple PyQt5 app for annotating PDF page images with schema-based facts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pdf2image import convert_from_path, pdfinfo_from_path
from pydantic import ValidationError
from PyQt5 import sip
from PyQt5.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, QItemSelectionModel, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QCloseEvent, QFont, QIcon, QIntValidator, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QResizeEvent, QShowEvent, QTextCursor, QTransform
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

from .annotation_core import (
    BoxRecord,
    CURRENCY_OPTIONS,
    PageState,
    SCALE_OPTIONS,
    apply_entity_name_to_pages,
    build_annotations_payload,
    denormalize_bbox_from_1000,
    default_page_meta,
    extract_document_meta,
    load_page_states,
    parse_import_payload,
    normalize_bbox_data,
    normalize_fact_data,
    serialize_annotations_json,
)
from .fact_ordering import canonical_fact_order_indices, compact_document_meta, normalize_document_meta, resolve_reading_direction
from .fact_normalization import normalize_annotation_payload
from .finetune.config import load_finetune_config
from .gemini_few_shot import (
    DEFAULT_COMPLEX_FEW_SHOT_SELECTIONS,
    DEFAULT_TEST_FEW_SHOT_PAGES,
    build_repo_roots,
    load_complex_few_shot_examples,
    load_test_pdf_few_shot_examples,
)
from .page_issues import DocumentIssueSummary, PageIssue, PageIssueSummary, validate_document_issues
from .schema_contract import default_extraction_prompt_template
from .schemas import PageType
from .workspace import page_has_annotation

FEW_SHOT_PRESET_CLASSIC = "classic_4"
FEW_SHOT_PRESET_EXTENDED = "extended_7"

FEW_SHOT_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    (FEW_SHOT_PRESET_CLASSIC, "Classic 4-shot"),
    (FEW_SHOT_PRESET_EXTENDED, "Extended 7-shot"),
)

FEW_SHOT_PRESET_SUMMARY: dict[str, str] = {
    FEW_SHOT_PRESET_CLASSIC: "classic(4): test 1,4,9,2",
    FEW_SHOT_PRESET_EXTENDED: "extended(7): test 9,4,5,10 | pdf_3 18,23 | pdf_2 8",
}
FEW_SHOT_PRESET_HELP_TEXT = " | ".join(
    FEW_SHOT_PRESET_SUMMARY.get(preset_id, preset_id)
    for preset_id, _ in FEW_SHOT_PRESET_CHOICES
)
QWEN_GT_MAX_NEW_TOKENS = 10_000

QWEN_GT_MODEL_CHOICES: tuple[str, ...] = (
    "qwen-flash-gt",
    "qwen3.5-plus-2026-02-15",
    "qwen3.5-27b",
    "Qwen/Qwen3.5-27B",
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_ROOT = PACKAGE_ROOT / "assets"
ICONS_ROOT = ASSETS_ROOT / "icons"
PROMPTS_ROOT = REPO_ROOT / "prompts"
DEFAULT_EXTRACTION_PROMPT_PATH = PROMPTS_ROOT / "extraction_prompt.txt"
LEGACY_EXTRACTION_PROMPT_PATH = REPO_ROOT / "prompt.txt"
GEMINI_BUTTON_ICON = ICONS_ROOT / "gemini.png"
QWEN_BUTTON_ICON = ICONS_ROOT / "qwen.png"
MULTI_VALUE_PLACEHOLDER = "Multiple values"
PATH_LEVEL_INDEX_ROLE = Qt.UserRole + 17


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


class AnnotationView(QGraphicsView):
    zoom_requested = pyqtSignal(float)
    nudge_selected_requested = pyqtSignal(int, int)
    resize_selected_requested = pyqtSignal(str, int)
    select_all_requested = pyqtSignal()
    previous_page_requested = pyqtSignal()
    next_page_requested = pyqtSignal()
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
        # Grouped bbox drags generate many small repaints; bounding-rect updates are cheaper here.
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_lens_enabled(self, enabled: bool) -> None:
        self._lens_enabled = bool(enabled)
        if not self._lens_enabled:
            self._lens_view_pos = None
        self.viewport().update()

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
            else:
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
        pen = QPen(Qt.red)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.setPen(pen)
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

    def set_order_label(self, order: Optional[int], *, visible: bool = True) -> None:
        next_order = int(order) if order is not None else None
        next_visible = bool(visible)
        if next_order == self._order_label and next_visible == self._show_order_label:
            return
        self._order_label = next_order
        self._show_order_label = next_visible
        self.update()

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
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
                self._active_handle = handle
                self._resize_start_rect = item_scene_rect(self)
                self._resize_start_scene_pos = event.scenePos()
                self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
                self.setCursor(self._cursor_for_handle(handle))
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
            if isinstance(scene, AnnotationScene) and self._geometry_changed_since_press():
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_rect = QRectF()
        self._drawing = False
        self._draw_start = None
        self._temp_rect_item: Optional[QGraphicsRectItem] = None
        self._selecting = False
        self._select_start = None
        self._temp_select_item: Optional[QGraphicsRectItem] = None
        self._pending_toggle_selection: Optional[tuple[AnnotRectItem, ...]] = None

    def set_image_rect(self, rect: QRectF) -> None:
        self.image_rect = rect

    def _selected_annot_items(self) -> list[AnnotRectItem]:
        return [item for item in self.selectedItems() if isinstance(item, AnnotRectItem)]

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.image_rect.contains(event.scenePos()):
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, AnnotRectItem) and (event.modifiers() & Qt.ShiftModifier):
                next_selection = list(self._selected_annot_items())
                if item in next_selection:
                    next_selection = [selected_item for selected_item in next_selection if selected_item is not item]
                else:
                    next_selection.append(item)
                self._pending_toggle_selection = tuple(next_selection)
                event.accept()
                return
            if (event.modifiers() & Qt.ShiftModifier) and (item is None or isinstance(item, QGraphicsPixmapItem)):
                self._selecting = True
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
                self.clearSelection()
                for item in self.items(rect, Qt.IntersectsItemShape):
                    if isinstance(item, AnnotRectItem):
                        item.setSelected(True)
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
            '{"meta": {...}, "facts": [...]} or {"image": "...", "meta": {...}, "facts": [...]} '
            'or full {"pages": [...]} document.\n'
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


class GeminiPromptDialog(QDialog):
    def __init__(
        self,
        prompt_text: str,
        model_name: str,
        parent: Optional[QWidget] = None,
        *,
        show_few_shot_controls: bool = False,
        few_shot_enabled_default: bool = True,
        few_shot_presets: tuple[tuple[str, str], ...] = FEW_SHOT_PRESET_CHOICES,
        few_shot_preset_default: str = FEW_SHOT_PRESET_CLASSIC,
        few_shot_summary: str = "",
        show_thinking_control: bool = True,
        thinking_enabled_default: bool = True,
        thinking_tooltip: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gemini Prompt")
        self.resize(980, 760)

        root = QVBoxLayout(self)
        hint = QLabel(
            "Edit the prompt before sending to Gemini.\n"
            "This prompt is used for the initial extraction run."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        form = QFormLayout()
        self.form_layout = form
        self.model_edit = QLineEdit(model_name)
        form.addRow("model", self.model_edit)
        self.thinking_check = QCheckBox("Enable thinking")
        self.thinking_check.setChecked(bool(thinking_enabled_default))
        if thinking_tooltip:
            self.thinking_check.setToolTip(thinking_tooltip)
        if show_thinking_control:
            form.addRow("thinking", self.thinking_check)
        else:
            self.thinking_check.setVisible(False)
        self.few_shot_check = QCheckBox("Use few-shot examples")
        self.few_shot_check.setChecked(bool(few_shot_enabled_default))
        self.few_shot_preset_combo = QComboBox()
        for preset_id, preset_label in few_shot_presets:
            self.few_shot_preset_combo.addItem(preset_label, preset_id)
        default_idx = self.few_shot_preset_combo.findData(few_shot_preset_default)
        if default_idx >= 0:
            self.few_shot_preset_combo.setCurrentIndex(default_idx)
        self.few_shot_summary_label = QLabel(few_shot_summary)
        self.few_shot_summary_label.setWordWrap(True)
        self.few_shot_summary_label.setObjectName("hintText")
        if show_few_shot_controls:
            form.addRow("few_shot", self.few_shot_check)
            form.addRow("few_shot mode", self.few_shot_preset_combo)
            form.addRow("few_shot preset", self.few_shot_summary_label)
        else:
            self.few_shot_check.setVisible(False)
            self.few_shot_preset_combo.setVisible(False)
            self.few_shot_summary_label.setVisible(False)
        root.addLayout(form)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(prompt_text)
        root.addWidget(self.prompt_edit, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Start Gemini GT")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    def prompt(self) -> str:
        return self.prompt_edit.toPlainText()

    def model(self) -> str:
        return self.model_edit.text().strip()

    def enable_thinking(self) -> bool:
        return self.thinking_check.isVisible() and self.thinking_check.isChecked()

    def use_few_shot(self) -> bool:
        return self.few_shot_check.isVisible() and self.few_shot_check.isChecked()

    def few_shot_preset(self) -> str:
        if not self.few_shot_preset_combo.isVisible():
            return FEW_SHOT_PRESET_CLASSIC
        value = self.few_shot_preset_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        return FEW_SHOT_PRESET_CLASSIC


class GeminiStreamDialog(QDialog):
    stop_requested = pyqtSignal()

    def __init__(self, page_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Gemini GT Stream - {page_name}")
        self.resize(900, 620)
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        root = QVBoxLayout(self)
        self.status_label = QLabel("Connecting to Gemini...")
        root.addWidget(self.status_label)

        self.text_view = QPlainTextEdit()
        self.text_view.setReadOnly(True)
        root.addWidget(self.text_view, 1)

        actions = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.close_btn = QPushButton("Close")
        actions.addWidget(self.stop_btn)
        actions.addStretch(1)
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.close_btn.clicked.connect(self.close)

    def append_text(self, text: str) -> None:
        if not text:
            return
        cursor = self.text_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.text_view.setTextCursor(cursor)
        self.text_view.ensureCursorVisible()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def mark_done(self, text: str) -> None:
        self.set_status(text)
        self.stop_btn.setEnabled(False)

    def mark_error(self, text: str) -> None:
        self.set_status(text)
        self.stop_btn.setEnabled(False)


class GeminiStreamWorker(QObject):
    chunk_received = pyqtSignal(str)
    meta_received = pyqtSignal(dict)
    fact_received = pyqtSignal(dict)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        image_path: Path,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        few_shot_examples: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.few_shot_examples = few_shot_examples
        self.enable_thinking = bool(enable_thinking)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            from .gemini_vlm import StreamingPageExtractionParser, stream_content_from_image

            parser = StreamingPageExtractionParser()
            for chunk in stream_content_from_image(
                image_path=self.image_path,
                prompt=self.prompt,
                model=self.model,
                api_key=self.api_key,
                few_shot_examples=self.few_shot_examples,
                enable_thinking=self.enable_thinking,
            ):
                if self._cancel_requested:
                    break
                if not chunk:
                    continue
                self.chunk_received.emit(chunk)
                meta, facts = parser.feed(chunk)
                if meta is not None:
                    self.meta_received.emit(meta)
                for fact in facts:
                    self.fact_received.emit(fact)

            if not self._cancel_requested:
                extraction = parser.finalize()
                self.completed.emit(extraction)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class QwenPromptDialog(GeminiPromptDialog):
    def __init__(
        self,
        prompt_text: str,
        model_name: str,
        parent: Optional[QWidget] = None,
        *,
        show_few_shot_controls: bool = True,
        few_shot_enabled_default: bool = True,
        few_shot_presets: tuple[tuple[str, str], ...] = FEW_SHOT_PRESET_CHOICES,
        few_shot_preset_default: str = FEW_SHOT_PRESET_CLASSIC,
        few_shot_summary: str = "",
    ) -> None:
        super().__init__(
            prompt_text=prompt_text,
            model_name=model_name,
            parent=parent,
            show_few_shot_controls=show_few_shot_controls,
            few_shot_enabled_default=few_shot_enabled_default,
            few_shot_presets=few_shot_presets,
            few_shot_preset_default=few_shot_preset_default,
            few_shot_summary=few_shot_summary,
        )
        self.setWindowTitle("Qwen Prompt")
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        for option in QWEN_GT_MODEL_CHOICES:
            self.model_combo.addItem(option)
        self.model_combo.setCurrentText(model_name)
        self.model_edit.setVisible(False)
        model_row, _ = self.form_layout.getWidgetPosition(self.model_edit)
        if model_row >= 0:
            self.form_layout.removeRow(model_row)
        self.form_layout.insertRow(0, "model", self.model_combo)
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Start Qwen GT")

    def model(self) -> str:
        return self.model_combo.currentText().strip()


class QwenStreamDialog(GeminiStreamDialog):
    def __init__(self, page_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(page_name=page_name, parent=parent)
        self.setWindowTitle(f"Qwen GT Stream - {page_name}")
        self.set_status("Connecting to Qwen...")


class QwenStreamWorker(QObject):
    chunk_received = pyqtSignal(str)
    meta_received = pyqtSignal(dict)
    fact_received = pyqtSignal(dict)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        image_path: Path,
        prompt: str,
        model: str,
        config_path: Optional[str] = None,
        max_new_tokens: int = QWEN_GT_MAX_NEW_TOKENS,
        few_shot_examples: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.config_path = config_path
        self.max_new_tokens = int(max_new_tokens)
        self.few_shot_examples = few_shot_examples
        self.enable_thinking = bool(enable_thinking)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            from .gemini_vlm import StreamingPageExtractionParser
            from .qwen_vlm import stream_content_from_image

            parser = StreamingPageExtractionParser()
            for chunk in stream_content_from_image(
                image_path=self.image_path,
                prompt=self.prompt,
                model=self.model,
                config_path=self.config_path,
                max_new_tokens=self.max_new_tokens,
                few_shot_examples=self.few_shot_examples,
                enable_thinking=self.enable_thinking,
            ):
                if self._cancel_requested:
                    break
                if not chunk:
                    continue
                self.chunk_received.emit(chunk)
                meta, facts = parser.feed(chunk)
                if meta is not None:
                    self.meta_received.emit(meta)
                for fact in facts:
                    self.fact_received.emit(fact)

            if not self._cancel_requested:
                extraction = parser.finalize()
                self.completed.emit(extraction)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class AnnotationWindow(QMainWindow):
    annotations_saved = pyqtSignal(object)
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
        self._gemini_stream_thread: Optional[QThread] = None
        self._gemini_stream_worker: Optional[GeminiStreamWorker] = None
        self._gemini_stream_target_page: Optional[str] = None
        self._gemini_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._gemini_stream_fact_count = 0
        self._gemini_stream_cancel_requested = False
        self._gemini_model_name = os.getenv("FINETREE_GEMINI_MODEL", "gemini-3-flash-preview")
        self._gemini_enable_thinking = True
        self._qwen_stream_thread: Optional[QThread] = None
        self._qwen_stream_worker: Optional[QwenStreamWorker] = None
        self._qwen_stream_target_page: Optional[str] = None
        self._qwen_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._qwen_stream_fact_count = 0
        self._qwen_stream_cancel_requested = False
        self._qwen_model_name = self._initial_qwen_model_name()
        self._qwen_enable_thinking = self._initial_qwen_enable_thinking()
        self._gt_activity_provider: Optional[str] = None
        self._page_issue_summaries: Dict[str, PageIssueSummary] = {}
        self._last_document_issue_signature: Optional[tuple[int, int, int, int]] = None
        self._last_saved_content: Dict[str, Any] = {"page_states": {}, "document_meta": {}}
        self._pending_close_approved = False
        self._pending_auto_fit = False

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
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setContentsMargins(18, 20, 18, 18)

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
            (self.gemini_gt_btn, "Gemini", "Gemini GT", self._load_repo_icon(GEMINI_BUTTON_ICON)),
            (self.qwen_gt_btn, "Qwen", "Qwen GT", self._load_repo_icon(QWEN_BUTTON_ICON)),
            (self.delete_nav_btn, "Delete", "Delete selected bounding box", None),
            (self.zoom_out_btn, "Zoom -", "Zoom out", None),
            (self.zoom_in_btn, "Zoom +", "Zoom in", None),
            (self.lens_btn, "Lens", "Toggle zoom lens", None),
            (self.copy_image_btn, "Copy Image", "Copy current page image", None),
            (self.fit_btn, "Fit", "Fit page to view height", None),
            (self.page_json_btn, "JSON", "Show current page JSON", None),
            (self.apply_entity_all_btn, "Apply Entity", "Apply current entity across pages", None),
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
        self.gemini_gt_btn = QPushButton("Gemini GT")
        self.qwen_gt_btn = QPushButton("Qwen GT")
        self.delete_nav_btn = QPushButton("Delete BBox")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.lens_btn = QPushButton("Lens")
        self.lens_btn.setCheckable(True)
        self.copy_image_btn = QPushButton("Copy Image")
        self.fit_btn = QPushButton("Fit")
        self.page_json_btn = QPushButton("Page JSON")
        self.apply_entity_all_btn = QPushButton("Apply Entity To Missing")
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
            self.gemini_gt_btn,
            self.qwen_gt_btn,
            self.delete_nav_btn,
            self.zoom_out_btn,
            self.zoom_in_btn,
            self.lens_btn,
            self.copy_image_btn,
            self.fit_btn,
            self.page_json_btn,
            self.apply_entity_all_btn,
            self.help_btn,
            self.save_btn,
            self.exit_btn,
        ):
            btn.setObjectName("toolbarActionBtn")
        self._apply_top_nav_icons()
        self.gemini_gt_btn.setProperty("variant", "primary")
        self.qwen_gt_btn.setProperty("variant", "primary")
        self.save_btn.setProperty("variant", "primary")
        for button in (self.import_btn, self.page_json_btn, self.help_btn, self.apply_entity_all_btn, self.exit_btn):
            button.setProperty("variant", "ghost")

        doc_layout = QHBoxLayout()
        doc_layout.setSpacing(6)
        doc_layout.addWidget(self.save_btn)
        doc_layout.addWidget(self.import_btn)
        doc_layout.addWidget(self.page_json_btn)
        doc_layout.addWidget(self.help_btn)
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
        gt_layout.addWidget(self.gemini_gt_btn)
        gt_layout.addWidget(self.qwen_gt_btn)
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
        right.setMinimumWidth(344)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

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
        self.title_edit = QLineEdit()
        self.doc_language_combo = QComboBox()
        self.doc_language_combo.addItems(["Auto", "Hebrew (he)", "English (en)"])
        self.doc_direction_combo = QComboBox()
        self.doc_direction_combo.addItems(["Auto", "RTL", "LTR"])
        self.company_name_edit = QLineEdit()
        self.company_id_edit = QLineEdit()
        self.report_year_edit = QLineEdit()
        self.report_year_edit.setValidator(QIntValidator(0, 9999, self))
        field_min_width = 248
        self.entity_name_edit.setMinimumWidth(field_min_width)
        self.page_num_edit.setMinimumWidth(field_min_width)
        self.type_combo.setMinimumWidth(field_min_width)
        self.title_edit.setMinimumWidth(field_min_width)
        self.doc_language_combo.setMinimumWidth(field_min_width)
        self.doc_direction_combo.setMinimumWidth(field_min_width)
        self.company_name_edit.setMinimumWidth(field_min_width)
        self.company_id_edit.setMinimumWidth(field_min_width)
        self.report_year_edit.setMinimumWidth(field_min_width)
        doc_form.addRow(self._inspector_label("Language"), self.doc_language_combo)
        doc_form.addRow(self._inspector_label("Direction"), self.doc_direction_combo)
        doc_form.addRow(self._inspector_label("Company Name"), self.company_name_edit)
        doc_form.addRow(self._inspector_label("Company ID"), self.company_id_edit)
        doc_form.addRow(self._inspector_label("Report Year"), self.report_year_edit)
        right_layout.addWidget(doc_box)
        meta_form.addRow(self._inspector_label("Entity"), self.entity_name_edit)
        meta_form.addRow(self._inspector_label("Page"), self.page_num_edit)
        meta_form.addRow(self._inspector_label("Type", required=True), self.type_combo)
        meta_form.addRow(self._inspector_label("Title"), self.title_edit)
        right_layout.addWidget(page_meta_box)

        self.page_issues_box = QGroupBox("Page Issues")
        self.page_issues_box.setObjectName("inspectorSection")
        page_issues_layout = QVBoxLayout(self.page_issues_box)
        page_issues_layout.setContentsMargins(18, 20, 18, 18)
        page_issues_layout.setSpacing(10)
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
        right_layout.addWidget(self.page_issues_box)

        fact_box = QGroupBox("Facts (Bounding Boxes)")
        fact_box.setObjectName("inspectorSection")
        fact_layout = QVBoxLayout(fact_box)
        fact_layout.setContentsMargins(18, 20, 18, 18)
        fact_layout.setSpacing(10)
        self.show_order_labels_check = QCheckBox("Show order labels on bboxes")
        self.show_order_labels_check.setObjectName("inspectorOption")
        self.show_order_labels_check.setChecked(True)
        self.facts_count_label = QLabel("No facts")
        self.facts_count_label.setObjectName("statusPill")
        self.facts_count_label.setProperty("tone", "accent")
        self.facts_list = QListWidget()
        self.facts_list.setObjectName("factsList")
        self.facts_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.facts_list.setSpacing(4)
        self.facts_list.setMinimumHeight(120)
        self.facts_list.setMaximumHeight(190)
        self.fact_bbox_label = QLabel("-")
        self.fact_value_edit = QLineEdit()
        self.fact_note_edit = QLineEdit()
        self.fact_note_name_edit = QLineEdit()
        self.fact_is_beur_combo = QComboBox()
        self.fact_is_beur_combo.addItems(["false", "true"])
        self.fact_beur_num_edit = QLineEdit()
        self.fact_refference_edit = QLineEdit()
        self.fact_date_edit = QLineEdit()
        self.fact_currency_combo = QComboBox()
        self.fact_currency_combo.addItems(["", *CURRENCY_OPTIONS])
        self.fact_scale_combo = QComboBox()
        self.fact_scale_combo.addItems(["", *[str(s) for s in SCALE_OPTIONS]])
        self.fact_value_type_combo = QComboBox()
        self.fact_value_type_combo.addItems(["", "amount", "%"])
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
        self.fact_path_list.setMinimumHeight(104)
        self.fact_path_list.setMaximumHeight(140)

        self.path_add_btn = QPushButton("+ Add Level")
        self.path_remove_btn = QPushButton("Remove")
        self.path_up_btn = QPushButton("Move Up")
        self.path_down_btn = QPushButton("Move Down")
        self.path_add_btn.setObjectName("smallActionBtn")
        self.path_remove_btn.setObjectName("smallActionBtn")
        self.path_up_btn.setObjectName("smallActionBtn")
        self.path_down_btn.setObjectName("smallActionBtn")

        path_actions = QHBoxLayout()
        path_actions.setSpacing(8)
        path_actions.addWidget(self.path_add_btn)
        path_actions.addWidget(self.path_remove_btn)
        path_actions.addWidget(self.path_up_btn)
        path_actions.addWidget(self.path_down_btn)
        path_actions.addStretch(1)

        path_panel = QWidget()
        path_panel_layout = QVBoxLayout(path_panel)
        path_panel_layout.setContentsMargins(0, 0, 0, 0)
        path_panel_layout.setSpacing(8)
        path_panel_layout.addWidget(self.fact_path_list)
        path_panel_layout.addLayout(path_actions)

        self.fact_editor_box = QGroupBox("Selected Fact")
        self.fact_editor_box.setObjectName("inspectorSubsection")
        fact_editor_layout = QVBoxLayout(self.fact_editor_box)
        fact_editor_layout.setContentsMargins(18, 18, 18, 16)
        fact_editor_layout.setSpacing(10)

        def add_fact_editor_row(
            left_block: QWidget,
            right_block: QWidget,
            *,
            left_stretch: int = 1,
            right_stretch: int = 1,
        ) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(left_block, left_stretch)
            row.addWidget(right_block, right_stretch)
            fact_editor_layout.addLayout(row)

        self.fact_bbox_label.setObjectName("factBboxLabel")
        fact_field_min_width = 132
        self.fact_bbox_label.setMinimumWidth(0)
        self.fact_value_edit.setMinimumWidth(fact_field_min_width)
        self.fact_note_edit.setMinimumWidth(fact_field_min_width)
        self.fact_note_name_edit.setMinimumWidth(fact_field_min_width)
        self.fact_is_beur_combo.setMinimumWidth(112)
        self.fact_beur_num_edit.setMinimumWidth(fact_field_min_width)
        self.fact_beur_num_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.fact_refference_edit.setMinimumWidth(fact_field_min_width)
        self.fact_date_edit.setMinimumWidth(fact_field_min_width)
        self.fact_currency_combo.setMinimumWidth(112)
        self.fact_scale_combo.setMinimumWidth(112)
        self.fact_value_type_combo.setMinimumWidth(112)
        self.fact_path_list.setMinimumWidth(0)
        self.fact_is_beur_combo.setMaximumWidth(220)
        self.fact_currency_combo.setMaximumWidth(220)
        self.fact_scale_combo.setMaximumWidth(220)
        self.fact_value_type_combo.setMaximumWidth(220)

        bbox_block = self._inspector_field_block("BBox", self.fact_bbox_label)
        bbox_block.setMaximumWidth(132)
        note_flag_block = self._inspector_field_block("Note Flag", self.fact_is_beur_combo)
        note_flag_block.setMaximumWidth(220)
        currency_block = self._inspector_field_block("Currency", self.fact_currency_combo)
        currency_block.setMaximumWidth(220)
        scale_block = self._inspector_field_block("Scale", self.fact_scale_combo)
        scale_block.setMaximumWidth(220)
        value_type_block = self._inspector_field_block("Value Type", self.fact_value_type_combo)
        value_type_block.setMaximumWidth(220)

        add_fact_editor_row(
            bbox_block,
            self._inspector_field_block("Value", self.fact_value_edit, required=True),
            left_stretch=0,
            right_stretch=1,
        )
        add_fact_editor_row(
            self._inspector_field_block("Comment", self.fact_note_edit),
            note_flag_block,
            left_stretch=2,
            right_stretch=1,
        )
        add_fact_editor_row(
            self._inspector_field_block("Note Name", self.fact_note_name_edit),
            self._inspector_field_block("Note Num", self.fact_beur_num_edit),
        )
        add_fact_editor_row(
            self._inspector_field_block("Note Ref", self.fact_refference_edit, required=True),
            self._inspector_field_block("Date", self.fact_date_edit),
        )
        add_fact_editor_row(currency_block, scale_block)
        add_fact_editor_row(value_type_block, QWidget())
        fact_editor_layout.addWidget(self._inspector_field_block("Path", path_panel))

        self.dup_fact_btn = QPushButton("Duplicate")
        self.del_fact_btn = QPushButton("Delete")
        self.dup_fact_btn.setObjectName("smallActionBtn")
        self.del_fact_btn.setObjectName("smallActionBtn")
        self.del_fact_btn.setProperty("variant", "danger")
        facts_header = QHBoxLayout()
        facts_header.setSpacing(10)
        facts_header.addWidget(self.show_order_labels_check)
        facts_header.addStretch(1)
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
        self.batch_toggle_btn.setProperty("variant", "ghost")
        fact_action_row.addWidget(self.batch_toggle_btn)
        fact_action_row.addStretch(1)
        fact_layout.addLayout(fact_action_row)
        self.batch_box = QGroupBox("Batch Edit Selected BBoxes")
        self.batch_box.setObjectName("inspectorSubsection")
        batch_layout = QVBoxLayout(self.batch_box)
        batch_layout.setContentsMargins(18, 20, 18, 18)
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
        self.batch_refference_edit.setPlaceholderText("Note reference for selected bboxes")
        self.batch_set_refference_btn = QPushButton("Set Note Reference")
        self.batch_clear_refference_btn = QPushButton("Clear Note Reference")
        self.batch_note_edit = QLineEdit()
        self.batch_note_edit.setPlaceholderText("Comment text for selected bboxes")
        self.batch_set_note_btn = QPushButton("Set Comment")
        self.batch_clear_note_btn = QPushButton("Clear Comment")
        self.batch_note_name_edit = QLineEdit()
        self.batch_note_name_edit.setPlaceholderText("Note name for selected bboxes")
        self.batch_set_note_name_btn = QPushButton("Set Note Name")
        self.batch_clear_note_name_btn = QPushButton("Clear Note Name")
        self.batch_date_edit = QLineEdit()
        self.batch_date_edit.setPlaceholderText("Date for selected bboxes (e.g. 30.09.2021)")
        self.batch_set_date_btn = QPushButton("Set Date")
        self.batch_clear_date_btn = QPushButton("Clear Date")
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
        self.batch_value_type_combo.addItems(["", "amount", "%"])
        self.batch_set_value_type_btn = QPushButton("Set value_type")
        self.batch_clear_value_type_btn = QPushButton("Clear value_type")
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
        batch_date_row = QHBoxLayout()
        batch_date_row.setSpacing(8)
        batch_date_row.addWidget(self.batch_date_edit)
        batch_date_row.addWidget(self.batch_set_date_btn)
        batch_date_row.addWidget(self.batch_clear_date_btn)
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
        batch_layout.addLayout(batch_date_row)
        batch_layout.addLayout(batch_is_beur_row)
        batch_layout.addLayout(batch_beur_num_row)
        batch_layout.addLayout(batch_currency_row)
        batch_layout.addLayout(batch_scale_row)
        batch_layout.addLayout(batch_value_type_row)
        batch_layout.addLayout(batch_resize_head)
        batch_layout.addLayout(batch_row_3)
        fact_layout.addWidget(self.batch_box)
        right_layout.addWidget(fact_box, 1)

        self.batch_box.setVisible(False)

        self.gt_activity_box = QGroupBox("Generation Activity")
        self.gt_activity_box.setObjectName("inspectorSubsection")
        gt_activity_layout = QVBoxLayout(self.gt_activity_box)
        gt_activity_layout.setContentsMargins(18, 20, 18, 18)
        gt_activity_layout.setSpacing(8)
        self.gt_activity_status_label = QLabel("No active generation.")
        self.gt_activity_status_label.setObjectName("subtitleLabel")
        self.gt_activity_provider_label = QLabel("Idle")
        self.gt_activity_provider_label.setObjectName("statusPill")
        self.gt_activity_provider_label.setProperty("tone", "accent")
        self.gt_activity_count_label = QLabel("Parsed facts: 0")
        self.gt_activity_count_label.setObjectName("monoLabel")
        self.gt_activity_stop_btn = QPushButton("Stop Generation")
        self.gt_activity_stop_btn.setProperty("variant", "danger")
        self.gt_activity_stop_btn.setEnabled(False)
        gt_activity_layout.addWidget(self.gt_activity_provider_label, 0, Qt.AlignLeft)
        gt_activity_layout.addWidget(self.gt_activity_status_label)
        gt_activity_layout.addWidget(self.gt_activity_count_label)
        gt_activity_layout.addWidget(self.gt_activity_stop_btn, 0, Qt.AlignLeft)
        right_layout.addWidget(self.gt_activity_box)

        tip = QLabel(
            "Select a box to edit fields here. "
            "Use Shift+click on boxes or Shift+drag on empty page area to select multiple boxes. "
            "Use Batch Edit to change value/note_reference/comment/note_name/date/note_flag/note_num/currency/scale/value_type and path levels across selected boxes. "
            "Use Batch Grow or Alt+Arrow to expand selected boxes in one direction. "
            "Use Arrow keys to move selected box(es), Shift+Arrow for faster nudge. "
            "Use +/Remove and Move Up/Move Down to manage path hierarchy. "
            "Pan page with Ctrl+Arrow keys or right/middle mouse drag."
        )
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        right_layout.addWidget(tip)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
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
        self.save_btn.clicked.connect(self.save_annotations)
        self.import_btn.clicked.connect(self.import_annotations_json)
        self.gemini_gt_btn.clicked.connect(self.generate_gemini_ground_truth)
        self.qwen_gt_btn.clicked.connect(self.generate_qwen_ground_truth)
        self.delete_nav_btn.clicked.connect(self.delete_selected_fact)
        self.dup_fact_btn.clicked.connect(self.duplicate_selected_fact)
        self.del_fact_btn.clicked.connect(self.delete_selected_fact)
        self.batch_toggle_btn.clicked.connect(self.toggle_batch_panel)
        self.gt_activity_stop_btn.clicked.connect(self._stop_active_generation)
        self.page_issues_list.itemClicked.connect(self._on_page_issue_clicked)
        self.facts_list.itemSelectionChanged.connect(self._on_fact_list_selection_changed)
        self.show_order_labels_check.toggled.connect(self._on_show_order_labels_toggled)
        self.entity_name_edit.editingFinished.connect(self._on_meta_edited)
        self.page_num_edit.editingFinished.connect(self._on_meta_edited)
        self.title_edit.editingFinished.connect(self._on_meta_edited)
        self.type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.doc_language_combo.activated.connect(lambda _: self._on_meta_edited())
        self.doc_direction_combo.activated.connect(lambda _: self._on_meta_edited())
        self.company_name_edit.editingFinished.connect(self._on_meta_edited)
        self.company_id_edit.editingFinished.connect(self._on_meta_edited)
        self.report_year_edit.editingFinished.connect(self._on_meta_edited)
        self.fact_value_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("value"))
        self.fact_note_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("comment"))
        self.fact_note_name_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_name"))
        self.fact_is_beur_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("note_flag"))
        self.fact_beur_num_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_num"))
        self.fact_refference_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("note_reference"))
        self.fact_date_edit.editingFinished.connect(lambda: self._on_fact_editor_field_edited("date"))
        self.fact_currency_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("currency"))
        self.fact_scale_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("scale"))
        self.fact_value_type_combo.activated.connect(lambda _: self._on_fact_editor_field_edited("value_type"))
        self.fact_path_list.itemChanged.connect(lambda _: self._on_fact_editor_field_edited("path"))
        self.fact_path_list.itemSelectionChanged.connect(self._update_path_controls)
        self.fact_path_list.model().rowsMoved.connect(lambda *_: self._on_path_reordered())
        self.path_add_btn.clicked.connect(self.add_path_level)
        self.path_remove_btn.clicked.connect(self.remove_selected_path_level)
        self.path_up_btn.clicked.connect(self.move_selected_path_up)
        self.path_down_btn.clicked.connect(self.move_selected_path_down)
        self.batch_path_level_edit.textChanged.connect(self._update_batch_controls)
        self.batch_value_edit.textChanged.connect(self._update_batch_controls)
        self.batch_refference_edit.textChanged.connect(self._update_batch_controls)
        self.batch_note_edit.textChanged.connect(self._update_batch_controls)
        self.batch_note_name_edit.textChanged.connect(self._update_batch_controls)
        self.batch_date_edit.textChanged.connect(self._update_batch_controls)
        self.batch_is_beur_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_beur_num_edit.textChanged.connect(self._update_batch_controls)
        self.batch_currency_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_scale_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_value_type_combo.activated.connect(lambda _: self._update_batch_controls())
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
        self.batch_expand_left_btn.clicked.connect(lambda: self.batch_expand_selected("left"))
        self.batch_expand_right_btn.clicked.connect(lambda: self.batch_expand_selected("right"))
        self.batch_expand_up_btn.clicked.connect(lambda: self.batch_expand_selected("up"))
        self.batch_expand_down_btn.clicked.connect(lambda: self.batch_expand_selected("down"))

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_annotations)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+I"), self, activated=self.import_annotations_json)
        QShortcut(QKeySequence("Meta+I"), self, activated=self.import_annotations_json)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.generate_gemini_ground_truth)
        QShortcut(QKeySequence("Meta+G"), self, activated=self.generate_gemini_ground_truth)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, activated=self.generate_qwen_ground_truth)
        QShortcut(QKeySequence("Meta+Shift+G"), self, activated=self.generate_qwen_ground_truth)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.duplicate_selected_fact)
        QShortcut(QKeySequence("Meta+D"), self, activated=self.duplicate_selected_fact)
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self.delete_selected_fact)
        delete_shortcut.setContext(Qt.ApplicationShortcut)
        backspace_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self.delete_selected_fact)
        backspace_shortcut.setContext(Qt.ApplicationShortcut)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._fit_view_height)
        QShortcut(QKeySequence("F1"), self, activated=self.show_help_dialog)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_annotations)
        self.addAction(save_action)
        self.page_label.setObjectName("monoLabel")
        self._populate_page_thumbnails()
        self._configure_hidden_status_bar()
        self.statusBar().showMessage("Ready")
        self._set_fact_editor_enabled(False)
        self._clear_fact_editor()
        self._clear_gt_activity()
        self._update_path_controls()
        self._update_batch_controls()
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
        brushes = self._path_tone_brushes(tone)
        if brushes is not None:
            bg, fg = brushes
            item.setBackground(bg)
            item.setForeground(fg)
        return item

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
        if list_enabled and row >= 0 and not self._path_list_structure_editable and current_item is not None:
            removable_shared_level = current_item.data(PATH_LEVEL_INDEX_ROLE) not in (None, "")
        structural_enabled = list_enabled and self._path_list_structure_editable
        self.path_add_btn.setEnabled(structural_enabled)
        self.path_remove_btn.setEnabled((structural_enabled and row >= 0) or removable_shared_level)
        self.path_up_btn.setEnabled(structural_enabled and row > 0)
        self.path_down_btn.setEnabled(structural_enabled and 0 <= row < count - 1)

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
        if not self.fact_path_list.isEnabled() or not self._path_list_structure_editable:
            return
        row = self.fact_path_list.currentRow()
        if row <= 0:
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row - 1, item)
        self.fact_path_list.setCurrentRow(row - 1)
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

    def move_selected_path_down(self) -> None:
        if not self.fact_path_list.isEnabled() or not self._path_list_structure_editable:
            return
        row = self.fact_path_list.currentRow()
        if row < 0 or row >= (self.fact_path_list.count() - 1):
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row + 1, item)
        self.fact_path_list.setCurrentRow(row + 1)
        self._update_path_controls()
        self._on_fact_editor_field_edited("path")

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
        has_is_beur_choice = bool(self.batch_is_beur_combo.currentText().strip()) if hasattr(self, "batch_is_beur_combo") else False
        has_beur_num_text = bool(self.batch_beur_num_edit.text().strip()) if hasattr(self, "batch_beur_num_edit") else False
        has_currency_choice = bool(self.batch_currency_combo.currentText().strip()) if hasattr(self, "batch_currency_combo") else False
        has_scale_choice = bool(self.batch_scale_combo.currentText().strip()) if hasattr(self, "batch_scale_combo") else False
        has_value_type_choice = bool(self.batch_value_type_combo.currentText().strip()) if hasattr(self, "batch_value_type_combo") else False
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
                    "Clear note_reference for all selected bboxes anyway?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_reference"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared note_reference")

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
            self.statusBar().showMessage("Enter note_reference text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note_reference"] = refference
            return fact

        self._batch_update_selected_facts(_transform, "Updated note_reference")

    def batch_set_note(self) -> None:
        note = self.batch_note_edit.text().strip()
        if not note:
            self.statusBar().showMessage("Enter comment text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["comment"] = note
            return fact

        self._batch_update_selected_facts(_transform, "Updated comment")

    def batch_clear_note(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["comment"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared comment")

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

    def _set_fact_editor_enabled(self, enabled: bool, *, multi_select: bool = False) -> None:
        self.fact_value_edit.setEnabled(enabled)
        self.fact_note_edit.setEnabled(enabled)
        self.fact_note_name_edit.setEnabled(enabled)
        self.fact_is_beur_combo.setEnabled(enabled)
        self.fact_beur_num_edit.setEnabled(enabled)
        self.fact_refference_edit.setEnabled(enabled)
        self.fact_date_edit.setEnabled(enabled)
        self.fact_currency_combo.setEnabled(enabled)
        self.fact_scale_combo.setEnabled(enabled)
        self.fact_value_type_combo.setEnabled(enabled)
        self.fact_path_list.setEnabled(enabled)
        self._set_path_list_editable(enabled and not multi_select)
        self.dup_fact_btn.setEnabled(enabled and not multi_select)
        self.del_fact_btn.setEnabled(enabled)
        self._update_path_controls()

    def _reset_fact_editor_placeholders(self) -> None:
        for edit in (
            self.fact_value_edit,
            self.fact_note_edit,
            self.fact_note_name_edit,
            self.fact_beur_num_edit,
            self.fact_refference_edit,
            self.fact_date_edit,
        ):
            edit.setPlaceholderText("")
            edit.setModified(False)
        for combo in (
            self.fact_is_beur_combo,
            self.fact_currency_combo,
            self.fact_scale_combo,
            self.fact_value_type_combo,
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
            self.fact_note_edit.setText("")
            self.fact_note_name_edit.setText("")
            self.fact_is_beur_combo.setCurrentIndex(0)
            self.fact_beur_num_edit.setText("")
            self.fact_refference_edit.setText("")
            self.fact_date_edit.setText("")
            self.fact_currency_combo.setCurrentIndex(0)
            self.fact_scale_combo.setCurrentIndex(0)
            self.fact_value_type_combo.setCurrentIndex(0)
            self.fact_path_list.clear()
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
            self.fact_note_edit.setText(str(fact.get("comment") or ""))
            self.fact_note_name_edit.setText(str(fact.get("note_name") or ""))
            is_beur = bool(fact.get("note_flag"))
            is_beur_text = "true" if is_beur else "false"
            idx_is_beur = self.fact_is_beur_combo.findText(is_beur_text)
            self.fact_is_beur_combo.setCurrentIndex(max(0, idx_is_beur))
            self.fact_beur_num_edit.setText("" if fact.get("note_num") is None else str(fact.get("note_num")))
            self.fact_refference_edit.setText(str(fact.get("note_reference") or ""))
            self.fact_date_edit.setText(str(fact.get("date") or ""))
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
            self._set_multi_line_edit_value(self.fact_note_edit, "comment", items)
            self._set_multi_line_edit_value(self.fact_note_name_edit, "note_name", items)
            self._set_multi_combo_value(
                self.fact_is_beur_combo,
                "note_flag",
                items,
                formatter=lambda value: "true" if bool(value) else "false",
            )
            self._set_multi_line_edit_value(self.fact_beur_num_edit, "note_num", items)
            self._set_multi_line_edit_value(self.fact_refference_edit, "note_reference", items)
            self._set_multi_line_edit_value(self.fact_date_edit, "date", items)
            self._set_multi_combo_value(self.fact_currency_combo, "currency", items)
            self._set_multi_combo_value(
                self.fact_scale_combo,
                "scale",
                items,
                formatter=lambda value: "" if value is None else str(value),
            )
            self._set_multi_combo_value(self.fact_value_type_combo, "value_type", items)

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
                prefix: list[str] = list(path_signatures[0]) if path_signatures else []
                for path in path_signatures[1:]:
                    common = 0
                    while common < len(prefix) and common < len(path) and prefix[common] == path[common]:
                        common += 1
                    prefix = prefix[:common]

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
                            )
                        )
                self._set_path_list_editable(bool(prefix), structural=False)
                if self.fact_path_list.count() > 0:
                    self.fact_path_list.setCurrentRow(0)
            self.fact_path_list.setToolTip(
                (
                    "Shared path nodes are highlighted in green and can be renamed across selected bboxes. "
                    "Diverging path nodes are highlighted in orange."
                )
                if path_mixed and self._path_list_editable
                else (
                    "Selected bboxes do not share an editable path prefix. Use Batch Edit to change path structure."
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
        return normalize_fact_data(
            {
                "value": self.fact_value_edit.text().strip(),
                "comment": self.fact_note_edit.text().strip() or None,
                "note_name": self.fact_note_name_edit.text().strip() or None,
                "note_flag": is_beur_value,
                "note_num": int(self.fact_beur_num_edit.text().strip()) if self.fact_beur_num_edit.text().strip() else None,
                "note_reference": self.fact_refference_edit.text().strip() or None,
                "date": self.fact_date_edit.text().strip() or None,
                "path": path_parts,
                "currency": self.fact_currency_combo.currentText().strip() or None,
                "scale": int(scale_text) if scale_text else None,
                "value_type": self.fact_value_type_combo.currentText().strip() or None,
            }
        )

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
            self._clear_fact_editor()
            self._set_fact_editor_enabled(False)
            self.batch_toggle_btn.setText("Show Batch Edit")
            return
        if len(selected_items) == 1:
            self._set_fact_editor_enabled(True, multi_select=False)
            self._populate_fact_editor(selected_items[0])
            self.batch_toggle_btn.setText("Hide Batch Edit" if self.batch_box.isVisible() else "Show Batch Edit")
            return
        if not self.batch_box.isVisible():
            self.batch_box.setVisible(True)
        self._set_fact_editor_enabled(True, multi_select=True)
        self._populate_multi_fact_editor(selected_items)
        self.batch_toggle_btn.setText("Hide Batch Edit")

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
        if field_name == "comment":
            self._apply_fact_field_to_selected_items("comment", self.fact_note_edit.text().strip() or None, widget=self.fact_note_edit)
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
        if field_name == "note_reference":
            self._apply_fact_field_to_selected_items(
                "note_reference",
                self.fact_refference_edit.text().strip() or None,
                widget=self.fact_refference_edit,
            )
            return
        if field_name == "date":
            self._apply_fact_field_to_selected_items("date", self.fact_date_edit.text().strip() or None, widget=self.fact_date_edit)
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
        alive_items = [item for item in items if self._is_alive_fact_item(item)]
        if not alive_items:
            return
        rects = [item_scene_rect(item) for item in alive_items]
        rects = [rect for rect in rects if not rect.isNull()]
        if not rects:
            return

        target_rect = QRectF(rects[0])
        for rect in rects[1:]:
            target_rect = target_rect.united(rect)

        image_rect = self.scene.image_rect
        if not image_rect.isNull():
            target_rect = target_rect.intersected(image_rect)
        if target_rect.isNull():
            return

        pad_x = max(44.0, target_rect.width() * 1.35)
        pad_y = max(44.0, target_rect.height() * 1.35)
        focus_rect = target_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        if not image_rect.isNull():
            focus_rect = focus_rect.intersected(image_rect)
        if focus_rect.isNull():
            focus_rect = target_rect

        viewport = self.view.viewport().rect()
        if viewport.width() <= 2 or viewport.height() <= 2:
            self.view.centerOn(target_rect.center())
            return

        self._fitting_view = True
        try:
            self.view.resetTransform()
            self.view.fitInView(focus_rect, Qt.KeepAspectRatio)
            current_zoom = self.view.transform().m11()
            if current_zoom < 0.05:
                self.view.resetTransform()
                self.view.scale(0.05, 0.05)
            elif current_zoom > 2.4:
                self.view.resetTransform()
                self.view.scale(2.4, 2.4)
            self.view.centerOn(target_rect.center())
        finally:
            self._fitting_view = False

    def _on_lens_toggled(self, enabled: bool) -> None:
        self.view.set_lens_enabled(enabled)
        if enabled:
            self.statusBar().showMessage("Zoom lens enabled.", 2000)
        else:
            self.statusBar().showMessage("Zoom lens disabled.", 2000)

    def _set_gt_buttons_enabled(self, enabled: bool) -> None:
        if hasattr(self, "gemini_gt_btn"):
            self.gemini_gt_btn.setEnabled(enabled)
        if hasattr(self, "qwen_gt_btn"):
            self.qwen_gt_btn.setEnabled(enabled)

    def toggle_batch_panel(self) -> None:
        visible = not self.batch_box.isVisible()
        self.batch_box.setVisible(visible)
        self.batch_toggle_btn.setText("Hide Batch Edit" if visible else "Show Batch Edit")

    def _set_gt_activity(self, provider: str, status: str, *, fact_count: int = 0, running: bool = False) -> None:
        self._gt_activity_provider = provider
        self.gt_activity_provider_label.setText(provider)
        tone = "accent" if running else ("ok" if "complete" in status.lower() else ("warn" if "stop" in status.lower() else "accent"))
        self.gt_activity_provider_label.setProperty("tone", tone)
        self.gt_activity_provider_label.style().unpolish(self.gt_activity_provider_label)
        self.gt_activity_provider_label.style().polish(self.gt_activity_provider_label)
        self.gt_activity_status_label.setText(status)
        self.gt_activity_count_label.setText(f"Parsed facts: {int(fact_count)}")
        self.gt_activity_stop_btn.setEnabled(running)

    def _clear_gt_activity(self) -> None:
        self._gt_activity_provider = None
        self._set_gt_activity("Idle", "No active generation.", fact_count=0, running=False)

    def _stop_active_generation(self) -> None:
        if self._gemini_stream_thread is not None:
            self._cancel_gemini_stream()
            return
        if self._qwen_stream_thread is not None:
            self._cancel_qwen_stream()
            return

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
            "document_meta": deepcopy(self.document_meta),
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
            self.document_meta = normalize_document_meta(snapshot.get("document_meta"))
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
            "document_meta": deepcopy(self.document_meta),
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
        return normalize_document_meta(
            {
                "language": language,
                "reading_direction": reading_direction,
                "company_name": company_name,
                "company_id": company_id,
                "report_year": report_year,
            }
        )

    def _page_meta_from_ui(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name_edit.text().strip() or None,
            "page_num": self.page_num_edit.text().strip() or None,
            "type": self.type_combo.currentText(),
            "title": self.title_edit.text().strip() or None,
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
        if hasattr(self, "page_thumb_list") and self.page_thumb_list.count() == len(self.page_images):
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
        if hasattr(self, "page_thumb_list") and self.page_thumb_list.count() == len(self.page_images):
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
                "note_reference": self.fact_refference_edit,
            }
        else:
            widget_map = {
                "type": self.type_combo,
                "currency": self.facts_list,
                "scale": self.facts_list,
                "value_type": self.facts_list,
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
        try:
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            w = float(bbox.get("w", 0.0))
            h = float(bbox.get("h", 0.0))
        except Exception:
            return False
        limit = 1000.0 + 1e-6
        return (
            0.0 <= x <= limit
            and 0.0 <= y <= limit
            and 0.0 <= w <= limit
            and 0.0 <= h <= limit
            and (x + w) <= limit
            and (y + h) <= limit
        )

    def _resolve_prompt_txt_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        env_path = os.getenv("FINETREE_PROMPT_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        candidates.append(Path.cwd() / "prompts" / "extraction_prompt.txt")
        candidates.append(DEFAULT_EXTRACTION_PROMPT_PATH)
        candidates.append(Path.cwd() / "prompt.txt")
        candidates.append(LEGACY_EXTRACTION_PROMPT_PATH)
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / "prompts" / "extraction_prompt.txt")
            candidates.append(parent / "prompt.txt")

        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

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

    def _build_prompt_from_template(self, template: str, page_image_path: Path) -> str:
        prompt = template.replace("{{PAGE_IMAGE}}", str(page_image_path))
        prompt = prompt.replace("{{IMAGE_NAME}}", page_image_path.name)
        return prompt

    def _fact_uniqueness_key(self, fact_payload: Dict[str, Any]) -> tuple[Any, ...]:
        normalized_fact = normalize_fact_data(fact_payload)
        bbox = normalize_bbox_data(fact_payload.get("bbox"))
        path = tuple(str(p) for p in (normalized_fact.get("path") or []))
        return (
            round(float(bbox["x"]), 2),
            round(float(bbox["y"]), 2),
            round(float(bbox["w"]), 2),
            round(float(bbox["h"]), 2),
            str(normalized_fact.get("value") or ""),
            str(normalized_fact.get("comment") or ""),
            str(normalized_fact.get("note_flag") if normalized_fact.get("note_flag") is not None else ""),
            str(normalized_fact.get("note_name") or ""),
            str(normalized_fact.get("note_num") if normalized_fact.get("note_num") is not None else ""),
            str(normalized_fact.get("note_reference") or ""),
            str(normalized_fact.get("date") or ""),
            path,
        )

    def _apply_stream_meta(self, page_name: str, meta_payload: Dict[str, Any]) -> None:
        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return
        state = self.page_states.get(page_name, self._default_state(page_idx))
        normalized_meta = {**self._default_meta(page_idx), **(state.meta or {}), **(meta_payload or {})}
        self.page_states[page_name] = PageState(meta=normalized_meta, facts=list(state.facts))

        if self.current_index >= 0 and self.page_images[self.current_index].name == page_name:
            self._is_loading_page = True
            try:
                self.entity_name_edit.setText(normalized_meta.get("entity_name") or "")
                self.page_num_edit.setText(normalized_meta.get("page_num") or "")
                type_value = normalized_meta.get("type") or PageType.other.value
                type_idx = self.type_combo.findText(type_value)
                self.type_combo.setCurrentIndex(type_idx if type_idx >= 0 else 0)
                self.title_edit.setText(normalized_meta.get("title") or "")
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
    ) -> bool:
        active_seen = seen_facts if seen_facts is not None else self._gemini_stream_seen_facts
        fact_key = self._fact_uniqueness_key(fact_payload)
        if fact_key in active_seen:
            return False
        active_seen.add(fact_key)

        page_idx = self._page_index_by_name(page_name)
        if page_idx < 0:
            return False

        raw_bbox = fact_payload.get("bbox")
        if not isinstance(raw_bbox, dict) and not (isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4):
            return False
        bbox = normalize_bbox_data(raw_bbox)
        image_dims = self._image_dimensions_for_page(page_name)
        if image_dims and self._bbox_looks_normalized_1000(bbox):
            bbox = denormalize_bbox_from_1000(bbox, image_dims[0], image_dims[1])
        fact_data = normalize_fact_data(fact_payload)

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

    def generate_gemini_ground_truth(self) -> None:
        if self._qwen_stream_thread is not None:
            QMessageBox.information(self, "Gemini GT", "A Qwen stream is already running.")
            return
        if self._gemini_stream_thread is not None:
            QMessageBox.information(self, "Gemini GT", "A Gemini stream is already running.")
            return
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            QMessageBox.warning(self, "Gemini GT", "No current page is loaded.")
            return

        page_path = self.page_images[self.current_index]
        page_name = page_path.name
        self._capture_current_state()

        current_state = self.page_states.get(page_name, self._default_state(self.current_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    "Generate Gemini ground truth and replace this page annotations?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        prompt_path = self._resolve_prompt_txt_path()
        if prompt_path is None:
            prompt_template = default_extraction_prompt_template()
        else:
            try:
                prompt_template = prompt_path.read_text(encoding="utf-8")
            except Exception as exc:
                QMessageBox.warning(self, "Gemini GT", f"Failed to read prompt template:\n{exc}")
                return

        prompt_text = self._build_prompt_from_template(prompt_template, page_path)
        prompt_dialog = GeminiPromptDialog(
            prompt_text=prompt_text,
            model_name=self._gemini_model_name,
            parent=self,
            show_few_shot_controls=True,
            show_thinking_control=True,
            thinking_enabled_default=self._gemini_enable_thinking,
            thinking_tooltip="Checked uses Gemini thinking mode. Unchecked requests minimal/non-thinking mode.",
            few_shot_enabled_default=True,
            few_shot_preset_default=FEW_SHOT_PRESET_CLASSIC,
            few_shot_summary=FEW_SHOT_PRESET_HELP_TEXT,
        )
        if prompt_dialog.exec_() != QDialog.Accepted:
            return
        prompt_text = prompt_dialog.prompt().strip()
        model_name = prompt_dialog.model().strip() or self._gemini_model_name
        enable_thinking = prompt_dialog.enable_thinking()
        use_few_shot = prompt_dialog.use_few_shot()
        few_shot_preset = prompt_dialog.few_shot_preset()
        if not prompt_text:
            QMessageBox.warning(self, "Gemini GT", "Prompt cannot be empty.")
            return

        try:
            from .gemini_vlm import resolve_api_key
        except Exception as exc:
            QMessageBox.warning(self, "Gemini GT", f"Gemini backend is unavailable:\n{exc}")
            return

        gemini_api_key = resolve_api_key()
        if not gemini_api_key:
            QMessageBox.warning(
                self,
                "Gemini GT",
                (
                    "Gemini API key not found.\n\n"
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY, or run through Doppler "
                    "with a secret named GOOGLE_API_KEY / GEMINI_API_KEY."
                ),
            )
            return

        self._gemini_model_name = model_name
        self._gemini_enable_thinking = enable_thinking
        few_shot_examples: Optional[list[dict[str, Any]]] = None
        if use_few_shot:
            loaded_examples, warnings = self._load_gemini_few_shot_examples(preset=few_shot_preset)
            preset_summary = FEW_SHOT_PRESET_SUMMARY.get(few_shot_preset, few_shot_preset)
            if loaded_examples:
                few_shot_examples = loaded_examples
                if warnings:
                    self.statusBar().showMessage(
                        f"Gemini few-shot ({preset_summary}) loaded with warnings: {'; '.join(warnings[:2])}",
                        6500,
                    )
            else:
                warning_text = "; ".join(warnings[:2]) if warnings else "Few-shot preset unavailable."
                self.statusBar().showMessage(
                    f"Gemini few-shot ({preset_summary}) fallback to standard mode: {warning_text}",
                    7000,
                )

        page_idx = self.current_index
        existing_meta = self.page_states.get(page_name, self._default_state(page_idx)).meta or {}
        self.page_states[page_name] = PageState(
            meta={**self._default_meta(page_idx), **existing_meta},
            facts=[],
        )
        was_restoring = self._is_restoring_history
        self._is_restoring_history = True
        try:
            self.show_page(page_idx)
        finally:
            self._is_restoring_history = was_restoring

        self._gemini_stream_target_page = page_name
        self._gemini_stream_seen_facts = set()
        self._gemini_stream_fact_count = 0
        self._gemini_stream_cancel_requested = False

        thinking_mode = "thinking" if enable_thinking else "non-thinking"
        if few_shot_examples:
            self._set_gt_activity(
                "Gemini",
                f"Streaming from {model_name} ({thinking_mode}) with few-shot ({len(few_shot_examples)} examples) ...",
                fact_count=0,
                running=True,
            )
        else:
            self._set_gt_activity(
                "Gemini",
                f"Streaming from {model_name} ({thinking_mode}) ...",
                fact_count=0,
                running=True,
            )

        worker = GeminiStreamWorker(
            image_path=page_path,
            prompt=prompt_text,
            model=model_name,
            api_key=gemini_api_key,
            few_shot_examples=few_shot_examples,
            enable_thinking=enable_thinking,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk_received.connect(self._on_gemini_stream_chunk)
        worker.meta_received.connect(self._on_gemini_stream_meta)
        worker.fact_received.connect(self._on_gemini_stream_fact)
        worker.completed.connect(self._on_gemini_stream_completed)
        worker.failed.connect(self._on_gemini_stream_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_gemini_stream_finished)

        self._gemini_stream_worker = worker
        self._gemini_stream_thread = thread
        self._set_gt_buttons_enabled(False)
        self.statusBar().showMessage(f"Streaming Gemini GT for {page_name}...", 3000)
        thread.start()

    def _cancel_gemini_stream(self) -> None:
        if self._gemini_stream_worker is not None:
            self._gemini_stream_cancel_requested = True
            self._gemini_stream_worker.request_cancel()
            self._set_gt_activity("Gemini", "Stopping stream...", fact_count=self._gemini_stream_fact_count, running=True)

    def _on_gemini_stream_chunk(self, text: str) -> None:
        _ = text

    def _on_gemini_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        page_name = self._gemini_stream_target_page
        if page_name is None:
            return
        self._apply_stream_meta(page_name, meta_payload)
        self._set_gt_activity("Gemini", "Meta received. Streaming facts...", fact_count=self._gemini_stream_fact_count, running=True)

    def _on_gemini_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        page_name = self._gemini_stream_target_page
        if page_name is None:
            return
        added = self._apply_stream_fact(page_name, fact_payload, seen_facts=self._gemini_stream_seen_facts)
        if added:
            self._gemini_stream_fact_count += 1
            self._set_gt_activity(
                "Gemini",
                f"Streaming facts... {self._gemini_stream_fact_count} parsed",
                fact_count=self._gemini_stream_fact_count,
                running=True,
            )

    def _on_gemini_stream_completed(self, extraction_obj: Any) -> None:
        page_name = self._gemini_stream_target_page
        if page_name is None:
            return
        try:
            meta_payload = extraction_obj.meta.model_dump(mode="json")
            self._apply_stream_meta(page_name, meta_payload)
            for fact in extraction_obj.facts:
                added = self._apply_stream_fact(
                    page_name,
                    fact.model_dump(mode="json"),
                    seen_facts=self._gemini_stream_seen_facts,
                )
                if added:
                    self._gemini_stream_fact_count += 1
        except Exception as exc:
            self._on_gemini_stream_failed(f"Final parse failed: {exc}")
            return

        self._record_history_snapshot()
        self._set_gt_activity(
            "Gemini",
            f"Gemini GT complete. Parsed {self._gemini_stream_fact_count} fact(s).",
            fact_count=self._gemini_stream_fact_count,
            running=False,
        )
        self.statusBar().showMessage(f"Gemini GT complete ({self._gemini_stream_fact_count} fact(s)).", 6000)

    def _on_gemini_stream_failed(self, message: str) -> None:
        self._set_gt_activity("Gemini", f"Error: {message}", fact_count=self._gemini_stream_fact_count, running=False)
        QMessageBox.warning(
            self,
            "Gemini GT failed",
            f"{message}\n\nAny facts already streamed remain on the page.",
        )

    def _on_gemini_stream_finished(self) -> None:
        if self._gemini_stream_cancel_requested:
            self._set_gt_activity(
                "Gemini",
                f"Gemini GT stopped. Parsed {self._gemini_stream_fact_count} fact(s) before stop.",
                fact_count=self._gemini_stream_fact_count,
                running=False,
            )
            self.statusBar().showMessage(
                f"Gemini GT stopped ({self._gemini_stream_fact_count} fact(s) parsed).",
                5000,
            )
        self._gemini_stream_thread = None
        self._gemini_stream_worker = None
        self._gemini_stream_target_page = None
        if self._qwen_stream_thread is None:
            self._set_gt_buttons_enabled(True)

    def generate_qwen_ground_truth(self) -> None:
        if self._gemini_stream_thread is not None:
            QMessageBox.information(self, "Qwen GT", "A Gemini stream is already running.")
            return
        if self._qwen_stream_thread is not None:
            QMessageBox.information(self, "Qwen GT", "A Qwen stream is already running.")
            return
        if self.current_index < 0 or self.current_index >= len(self.page_images):
            QMessageBox.warning(self, "Qwen GT", "No current page is loaded.")
            return

        page_path = self.page_images[self.current_index]
        page_name = page_path.name
        self._capture_current_state()

        current_state = self.page_states.get(page_name, self._default_state(self.current_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    "Generate Qwen ground truth and replace this page annotations?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        prompt_path = self._resolve_prompt_txt_path()
        if prompt_path is None:
            prompt_template = default_extraction_prompt_template()
        else:
            try:
                prompt_template = prompt_path.read_text(encoding="utf-8")
            except Exception as exc:
                QMessageBox.warning(self, "Qwen GT", f"Failed to read prompt template:\n{exc}")
                return

        prompt_text = self._build_prompt_from_template(prompt_template, page_path)
        prompt_dialog = QwenPromptDialog(
            prompt_text=prompt_text,
            model_name=self._qwen_model_name,
            parent=self,
            show_few_shot_controls=True,
            show_thinking_control=True,
            thinking_enabled_default=self._qwen_enable_thinking,
            thinking_tooltip="Checked enables Qwen thinking where the selected backend supports it.",
            few_shot_enabled_default=True,
            few_shot_preset_default=FEW_SHOT_PRESET_CLASSIC,
            few_shot_summary=FEW_SHOT_PRESET_HELP_TEXT,
        )
        if prompt_dialog.exec_() != QDialog.Accepted:
            return
        prompt_text = prompt_dialog.prompt().strip()
        model_name = prompt_dialog.model().strip() or self._qwen_model_name
        enable_thinking = prompt_dialog.enable_thinking()
        use_few_shot = prompt_dialog.use_few_shot()
        few_shot_preset = prompt_dialog.few_shot_preset()
        if not prompt_text:
            QMessageBox.warning(self, "Qwen GT", "Prompt cannot be empty.")
            return

        try:
            from .qwen_vlm import is_qwen_flash_model_requested, resolve_qwen_flash_api_key
        except Exception as exc:
            QMessageBox.warning(self, "Qwen GT", f"Qwen backend is unavailable:\n{exc}")
            return

        is_qwen_flash = is_qwen_flash_model_requested(model_name)
        qwen_config_path: Optional[Path] = None
        if is_qwen_flash:
            if not resolve_qwen_flash_api_key():
                QMessageBox.warning(
                    self,
                    "Qwen GT",
                    (
                        "Qwen hosted API key not found.\n\n"
                        "Set FINETREE_QWEN_FLASH_API_KEY, FINETREE_QWEN_API_KEY, "
                        "QWEN_API_KEY, or DASHSCOPE_API_KEY."
                    ),
                )
                return
        else:
            qwen_config_path = self._resolve_qwen_config_path()
            if qwen_config_path is None:
                QMessageBox.warning(
                    self,
                    "Qwen GT",
                    (
                        "Could not find Qwen fine-tune config.\n"
                        "Expected configs/qwen_ui_runpod_queue.yaml, "
                        "configs/finetune_qwen35a3_vl.yaml, or FINETREE_QWEN_CONFIG."
                    ),
                )
                return

        self._qwen_model_name = model_name
        self._qwen_enable_thinking = enable_thinking
        few_shot_examples: Optional[list[dict[str, Any]]] = None
        if use_few_shot and is_qwen_flash:
            loaded_examples, warnings = self._load_gemini_few_shot_examples(preset=few_shot_preset)
            preset_summary = FEW_SHOT_PRESET_SUMMARY.get(few_shot_preset, few_shot_preset)
            if loaded_examples:
                few_shot_examples = loaded_examples
                if warnings:
                    self.statusBar().showMessage(
                        f"Qwen few-shot ({preset_summary}) loaded with warnings: {'; '.join(warnings[:2])}",
                        6500,
                    )
            else:
                warning_text = "; ".join(warnings[:2]) if warnings else "Few-shot preset unavailable."
                self.statusBar().showMessage(
                    f"Qwen few-shot ({preset_summary}) fallback to standard mode: {warning_text}",
                    7000,
                )
        elif use_few_shot:
            self.statusBar().showMessage(
                "Qwen few-shot is currently enabled for qwen-flash-gt only; running standard mode.",
                7000,
            )

        page_idx = self.current_index
        existing_meta = self.page_states.get(page_name, self._default_state(page_idx)).meta or {}
        self.page_states[page_name] = PageState(
            meta={**self._default_meta(page_idx), **existing_meta},
            facts=[],
        )
        was_restoring = self._is_restoring_history
        self._is_restoring_history = True
        try:
            self.show_page(page_idx)
        finally:
            self._is_restoring_history = was_restoring

        self._qwen_stream_target_page = page_name
        self._qwen_stream_seen_facts = set()
        self._qwen_stream_fact_count = 0
        self._qwen_stream_cancel_requested = False

        thinking_mode = "thinking" if enable_thinking else "non-thinking"
        if few_shot_examples:
            self._set_gt_activity(
                "Qwen",
                f"Streaming from {model_name} ({thinking_mode}) with few-shot ({len(few_shot_examples)} examples) ...",
                fact_count=0,
                running=True,
            )
        else:
            self._set_gt_activity(
                "Qwen",
                f"Streaming from {model_name} ({thinking_mode}) ...",
                fact_count=0,
                running=True,
            )

        worker = QwenStreamWorker(
            image_path=page_path,
            prompt=prompt_text,
            model=model_name,
            config_path=str(qwen_config_path) if qwen_config_path is not None else None,
            few_shot_examples=few_shot_examples,
            enable_thinking=enable_thinking,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk_received.connect(self._on_qwen_stream_chunk)
        worker.meta_received.connect(self._on_qwen_stream_meta)
        worker.fact_received.connect(self._on_qwen_stream_fact)
        worker.completed.connect(self._on_qwen_stream_completed)
        worker.failed.connect(self._on_qwen_stream_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_qwen_stream_finished)

        self._qwen_stream_worker = worker
        self._qwen_stream_thread = thread
        self._set_gt_buttons_enabled(False)
        self.statusBar().showMessage(f"Streaming Qwen GT for {page_name}...", 3000)
        thread.start()

    def _cancel_qwen_stream(self) -> None:
        if self._qwen_stream_worker is not None:
            self._qwen_stream_cancel_requested = True
            self._qwen_stream_worker.request_cancel()
            self._set_gt_activity("Qwen", "Stopping stream...", fact_count=self._qwen_stream_fact_count, running=True)

    def _on_qwen_stream_chunk(self, text: str) -> None:
        _ = text

    def _on_qwen_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        page_name = self._qwen_stream_target_page
        if page_name is None:
            return
        self._apply_stream_meta(page_name, meta_payload)
        self._set_gt_activity("Qwen", "Meta received. Streaming facts...", fact_count=self._qwen_stream_fact_count, running=True)

    def _on_qwen_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        page_name = self._qwen_stream_target_page
        if page_name is None:
            return
        added = self._apply_stream_fact(page_name, fact_payload, seen_facts=self._qwen_stream_seen_facts)
        if added:
            self._qwen_stream_fact_count += 1
            self._set_gt_activity(
                "Qwen",
                f"Streaming facts... {self._qwen_stream_fact_count} parsed",
                fact_count=self._qwen_stream_fact_count,
                running=True,
            )

    def _on_qwen_stream_completed(self, extraction_obj: Any) -> None:
        page_name = self._qwen_stream_target_page
        if page_name is None:
            return
        try:
            meta_payload = extraction_obj.meta.model_dump(mode="json")
            self._apply_stream_meta(page_name, meta_payload)
            for fact in extraction_obj.facts:
                added = self._apply_stream_fact(
                    page_name,
                    fact.model_dump(mode="json"),
                    seen_facts=self._qwen_stream_seen_facts,
                )
                if added:
                    self._qwen_stream_fact_count += 1
        except Exception as exc:
            self._on_qwen_stream_failed(f"Final parse failed: {exc}")
            return

        self._record_history_snapshot()
        self._set_gt_activity(
            "Qwen",
            f"Qwen GT complete. Parsed {self._qwen_stream_fact_count} fact(s).",
            fact_count=self._qwen_stream_fact_count,
            running=False,
        )
        self.statusBar().showMessage(f"Qwen GT complete ({self._qwen_stream_fact_count} fact(s)).", 6000)

    def _on_qwen_stream_failed(self, message: str) -> None:
        self._set_gt_activity("Qwen", f"Error: {message}", fact_count=self._qwen_stream_fact_count, running=False)
        QMessageBox.warning(
            self,
            "Qwen GT failed",
            f"{message}\n\nAny facts already streamed remain on the page.",
        )

    def _on_qwen_stream_finished(self) -> None:
        if self._qwen_stream_cancel_requested:
            self._set_gt_activity(
                "Qwen",
                f"Qwen GT stopped. Parsed {self._qwen_stream_fact_count} fact(s) before stop.",
                fact_count=self._qwen_stream_fact_count,
                running=False,
            )
            self.statusBar().showMessage(
                f"Qwen GT stopped ({self._qwen_stream_fact_count} fact(s) parsed).",
                5000,
            )
        self._qwen_stream_thread = None
        self._qwen_stream_worker = None
        self._qwen_stream_target_page = None
        if self._gemini_stream_thread is None:
            self._set_gt_buttons_enabled(True)

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
        painter.end()
        return QIcon(canvas)

    def _refresh_thumbnail_for_index(self, index: int) -> None:
        if not hasattr(self, "page_thumb_list"):
            return
        if index < 0 or index >= self.page_thumb_list.count() or index >= len(self.page_images):
            return
        item = self.page_thumb_list.item(index)
        if item is None:
            return
        page_path = self.page_images[index]
        page_summary = self._page_issue_summaries.get(page_path.name, PageIssueSummary(page_image=page_path.name))
        item.setIcon(self._thumbnail_icon_for_page(page_path, index))
        item.setToolTip(self._issue_tooltip_text(page_summary))

    def _load_existing_annotations(self) -> None:
        if not self.annotations_path.exists():
            return
        try:
            payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
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
        self.page_thumb_list.blockSignals(True)
        self.page_thumb_list.clear()
        icon_size = self.page_thumb_list.iconSize()
        for idx, page_path in enumerate(self.page_images):
            item = QListWidgetItem(str(idx + 1))
            page_summary = self._page_issue_summaries.get(page_path.name, PageIssueSummary(page_image=page_path.name))
            item.setToolTip(self._issue_tooltip_text(page_summary))
            item.setTextAlignment(Qt.AlignCenter)
            item.setIcon(self._thumbnail_icon_for_page(page_path, idx))
            item.setSizeHint(QSize(icon_size.width() + 20, icon_size.height() + 30))
            self.page_thumb_list.addItem(item)
        self.page_thumb_list.blockSignals(False)

    def _on_page_jump_requested(self, page_number: int) -> None:
        if not self.page_images:
            return
        index = max(0, min(int(page_number) - 1, len(self.page_images) - 1))
        if index == self.current_index:
            return
        self.show_page(index)

    def _on_thumbnail_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.page_images):
            return
        if row == self.current_index:
            return
        self.show_page(row)

    def show_page(self, index: int) -> None:
        if index < 0 or index >= len(self.page_images):
            return

        if not self._is_restoring_history:
            self._capture_current_state()
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

        meta = {**self._default_meta(index), **(state.meta or {})}
        self._is_loading_page = True
        try:
            self.entity_name_edit.setText(meta.get("entity_name") or "")
            self.page_num_edit.setText(meta.get("page_num") or "")
            type_value = meta.get("type") or PageType.other.value
            type_idx = self.type_combo.findText(type_value)
            self.type_combo.setCurrentIndex(type_idx if type_idx >= 0 else 0)
            self.title_edit.setText(meta.get("title") or "")
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
        self.page_thumb_list.setCurrentRow(index)
        current_thumb = self.page_thumb_list.item(index)
        if current_thumb is not None:
            self.page_thumb_list.scrollToItem(current_thumb, QAbstractItemView.PositionAtCenter)
        self.page_thumb_list.blockSignals(False)
        self.refresh_facts_list()
        self.schedule_auto_fit_current_page()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._apply_pending_auto_fit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_pending_auto_fit()

    def _sorted_fact_items(self) -> List[AnnotRectItem]:
        items = [i for i in self.scene.items() if isinstance(i, AnnotRectItem) and self._is_alive_fact_item(i)]
        if len(items) <= 1:
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
        return [items[idx] for idx in ordered_indices if 0 <= idx < len(items)]

    def _apply_fact_order_labels(self) -> None:
        show_labels = self.show_order_labels_check.isChecked()
        indexed_item_ids = {id(item) for item in self._fact_items}
        for idx, item in enumerate(self._fact_items, start=1):
            if self._is_alive_fact_item(item):
                item.set_order_label(idx, visible=show_labels)
        for scene_item in self.scene.items():
            if (
                isinstance(scene_item, AnnotRectItem)
                and self._is_alive_fact_item(scene_item)
                and id(scene_item) not in indexed_item_ids
            ):
                scene_item.set_order_label(None, visible=False)

    def refresh_facts_list(self, *, refresh_issues: bool = True) -> None:
        selected = self._selected_fact_item()
        selected_items = set(self._selected_fact_items())
        self._fact_items = self._sorted_fact_items()
        self._apply_fact_order_labels()
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
            value = str(item.fact_data.get("value") or "")
            path = " > ".join(item.fact_data.get("path") or [])
            comment = str(item.fact_data.get("comment") or "")
            note_num = "" if item.fact_data.get("note_num") is None else str(item.fact_data.get("note_num"))
            note_name = str(item.fact_data.get("note_name") or "")
            summary = f"#{idx} [{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}] {value}"
            if path:
                summary = f"{summary} | {path}"
            if comment:
                trimmed_comment = (comment[:32] + "...") if len(comment) > 35 else comment
                summary = f"{summary} | comment: {trimmed_comment}"
            if note_num:
                trimmed_note_num = (note_num[:32] + "...") if len(note_num) > 35 else note_num
                summary = f"{summary} | note_num: {trimmed_note_num}"
            if note_name:
                trimmed_note_name = (note_name[:32] + "...") if len(note_name) > 35 else note_name
                summary = f"{summary} | note_name: {trimmed_note_name}"
            summary = f"{summary} | note_flag: {bool(item.fact_data.get('note_flag'))}"
            if item.fact_data.get("note_reference"):
                summary = f"{summary} | note_reference: {item.fact_data.get('note_reference')}"
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

    def _on_show_order_labels_toggled(self, _checked: bool) -> None:
        self._apply_fact_order_labels()

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
        self.refresh_facts_list(refresh_issues=False)
        self._record_history_snapshot()

    def _on_meta_edited(self) -> None:
        if self._is_loading_page or self._is_restoring_history:
            return
        previous = normalize_document_meta(self.document_meta)
        self.document_meta = self._document_meta_from_ui()
        if self.document_meta != previous:
            self.refresh_facts_list()
        else:
            self._refresh_current_page_issues()
        self._refresh_thumbnail_for_index(self.current_index)
        self._record_history_snapshot()

    def _on_scene_selection_changed(self) -> None:
        if self._syncing_selection:
            return
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

    def _on_fact_list_selection_changed(self) -> None:
        if self._syncing_selection:
            return
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

        self.refresh_facts_list()
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

    def save_annotations(self) -> bool:
        self._capture_current_state()
        try:
            payload = build_annotations_payload(
                self.images_dir,
                self.page_images,
                self.page_states,
                document_meta=self.document_meta,
            )
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return False
        _normalized_payload, format_findings = normalize_annotation_payload(payload)
        warning_findings = [
            finding
            for finding in format_findings
            if any(code in {"noncanonical_date", "placeholder_value", "noncanonical_value"} for code in finding.get("issue_codes", []))
        ]
        serialized = serialize_annotations_json(payload)
        no_changes = False
        if self.annotations_path.exists():
            try:
                no_changes = self.annotations_path.read_text(encoding="utf-8") == serialized
            except OSError:
                no_changes = False
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.annotations_path.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        if no_changes:
            warning_suffix = f" | format_warnings={len(warning_findings)}" if warning_findings else ""
            self.statusBar().showMessage(
                f"No changes detected. File is up to date: {self.annotations_path}{warning_suffix}",
                6000,
            )
            QMessageBox.information(
                self,
                "Saved",
                f"Annotations saved to:\n{self.annotations_path}\n\nNo changes detected.",
            )
            self.annotations_saved.emit(self.annotations_path)
            self._mark_saved_content()
            if warning_findings:
                preview = "\n".join(
                    f"- {f.get('page')} fact#{int(f.get('fact_index', 0)) + 1}: {', '.join(f.get('issue_codes', []))}"
                    for f in warning_findings[:8]
                )
                QMessageBox.warning(
                    self,
                    "Format warnings",
                    (
                        "Saved successfully, but some facts have non-canonical date/value format.\n\n"
                        f"{preview}\n\n"
                        "Use scripts/check_fact_schema_format.py for full details."
                    ),
                )
            return True
        warning_suffix = f" | format_warnings={len(warning_findings)}" if warning_findings else ""
        self.statusBar().showMessage(f"Saved: {self.annotations_path}{warning_suffix}", 6000)
        QMessageBox.information(self, "Saved", f"Annotations saved to:\n{self.annotations_path}")
        self.annotations_saved.emit(self.annotations_path)
        self._mark_saved_content()
        if warning_findings:
            preview = "\n".join(
                f"- {f.get('page')} fact#{int(f.get('fact_index', 0)) + 1}: {', '.join(f.get('issue_codes', []))}"
                for f in warning_findings[:8]
            )
            QMessageBox.warning(
                self,
                "Format warnings",
                (
                    "Saved successfully, but some facts have non-canonical date/value format.\n\n"
                    f"{preview}\n\n"
                    "Use scripts/check_fact_schema_format.py for full details."
                ),
            )
        return True

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

        self._capture_current_state()
        try:
            payload = build_annotations_payload(
                self.images_dir,
                self.page_images,
                self.page_states,
                document_meta=self.document_meta,
            )
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return

        page_payload = payload["pages"][self.current_index]
        page_text = json.dumps(page_payload, indent=2, ensure_ascii=False)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Page JSON - {page_payload.get('image', '')}")
        dialog.resize(920, 640)
        root = QVBoxLayout(dialog)

        hint = QLabel("Current page representation in the annotations JSON schema.")
        hint.setWordWrap(True)
        root.addWidget(hint)

        text_view = QPlainTextEdit()
        text_view.setReadOnly(True)
        text_view.setPlainText(page_text)
        root.addWidget(text_view, 1)

        actions = QHBoxLayout()
        copy_btn = QPushButton("Copy JSON")
        close_btn = QPushButton("Close")
        actions.addWidget(copy_btn)
        actions.addStretch(1)
        actions.addWidget(close_btn)
        root.addLayout(actions)

        def _copy_json() -> None:
            QApplication.clipboard().setText(page_text)
            self.statusBar().showMessage("Copied current page JSON.", 2500)

        copy_btn.clicked.connect(_copy_json)
        close_btn.clicked.connect(dialog.accept)
        dialog.exec_()

    def show_help_dialog(self) -> None:
        help_text = (
            "Keyboard shortcuts\n"
            "- Ctrl/Cmd+S: Save annotations\n"
            "- Ctrl/Cmd+Z: Undo\n"
            "- Ctrl/Cmd+Y or Ctrl/Cmd+Shift+Z: Redo\n"
            "- Ctrl/Cmd+I: Import JSON\n"
            "- Ctrl/Cmd+G: Gemini GT (current page)\n"
            "- Ctrl/Cmd+Shift+G: Qwen GT (current page)\n"
            "- Ctrl/Cmd+D: Duplicate selected bbox\n"
            "- Delete or Backspace: Delete selected bbox(es)\n"
            "- +: Zoom in when the page view is focused\n"
            "- -: Zoom out when the page view is focused\n"
            "- Ctrl+=: Zoom in\n"
            "- Ctrl+-: Zoom out\n"
            "- Ctrl+0: Fit page height to panel\n"
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
            "- Shift + click on bbox: Add/remove it from the selection\n"
            "- Shift + left-drag on empty page area: Rectangle-select multiple bboxes\n"
            "- Arrow keys: Move selected bbox(es) by 1 px\n"
            "- Shift+Arrow: Move selected bbox(es) by 10 px\n"
            "- Alt+Arrow: Grow selected bbox(es) in one direction by batch step\n"
            "- Alt+Shift+Arrow: Grow selected bbox(es) by 10x batch step\n"
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


@dataclass(frozen=True)
class StartupContext:
    mode: str
    images_dir: Optional[Path] = None
    annotations_path: Optional[Path] = None
    pdf_path: Optional[Path] = None


def _default_annotations_path(images_dir: Path) -> Path:
    return Path("data/annotations") / f"{images_dir.name}.json"


def _resolve_startup_context(
    input_path_arg: Optional[str],
    images_dir_arg: Optional[str],
    annotations_arg: Optional[str],
) -> StartupContext:
    if input_path_arg and images_dir_arg:
        raise ValueError("Cannot provide both positional input_path and --images-dir.")

    if input_path_arg:
        input_path = Path(input_path_arg).expanduser().resolve()
        if input_path.is_dir():
            annotations_path = Path(annotations_arg) if annotations_arg else _default_annotations_path(input_path)
            return StartupContext(mode="images-dir", images_dir=input_path, annotations_path=annotations_path)
        if not input_path.exists():
            raise ValueError(f"Input path not found: {input_path}")
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            images_dir = (Path("data/pdf_images") / input_path.stem).resolve()
            annotations_path = Path(annotations_arg) if annotations_arg else _default_annotations_path(images_dir)
            return StartupContext(mode="pdf", images_dir=images_dir, annotations_path=annotations_path, pdf_path=input_path)
        raise ValueError(f"Unsupported input path: {input_path}. Provide a PDF file or an image directory.")

    if images_dir_arg:
        images_dir = Path(images_dir_arg).expanduser().resolve()
        annotations_path = Path(annotations_arg) if annotations_arg else _default_annotations_path(images_dir)
        return StartupContext(mode="images-dir", images_dir=images_dir, annotations_path=annotations_path)

    return StartupContext(mode="home")


def _missing_page_numbers(images_dir: Path, page_count: int) -> List[int]:
    missing: List[int] = []
    for idx in range(1, page_count + 1):
        target = images_dir / f"page_{idx:04d}.png"
        if not target.is_file():
            missing.append(idx)
    return missing


def _build_page_ranges(page_numbers: List[int]) -> List[tuple[int, int]]:
    if not page_numbers:
        return []
    ranges: List[tuple[int, int]] = []
    start = page_numbers[0]
    prev = page_numbers[0]
    for page_num in page_numbers[1:]:
        if page_num == prev + 1:
            prev = page_num
            continue
        ranges.append((start, prev))
        start = page_num
        prev = page_num
    ranges.append((start, prev))
    return ranges


def _ensure_pdf_images(pdf_path: Path, images_dir: Path, dpi: int = 200) -> Dict[str, Any]:
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF not found: {pdf_path}")

    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf_info = pdfinfo_from_path(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF metadata for {pdf_path}: {exc}") from exc

    try:
        page_count = int((pdf_info or {}).get("Pages"))
    except Exception as exc:
        raise RuntimeError(f"Failed to determine page count for {pdf_path}.") from exc
    if page_count <= 0:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    missing_pages = _missing_page_numbers(images_dir, page_count)
    if not missing_pages:
        return {
            "action": "reused",
            "page_count": page_count,
            "created_pages": 0,
            "reused_pages": page_count,
            "missing_before": 0,
        }

    existing_before = page_count - len(missing_pages)
    created_pages = 0
    for first_page, last_page in _build_page_ranges(missing_pages):
        try:
            rendered_pages = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                use_pdftocairo=True,
                first_page=first_page,
                last_page=last_page,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed converting pages {first_page}-{last_page} for {pdf_path}: {exc}"
            ) from exc

        expected = last_page - first_page + 1
        if len(rendered_pages) != expected:
            raise RuntimeError(
                f"Unexpected rendered page count for range {first_page}-{last_page}: "
                f"expected {expected}, got {len(rendered_pages)}"
            )

        for offset, page in enumerate(rendered_pages):
            page_num = first_page + offset
            target = images_dir / f"page_{page_num:04d}.png"
            page.save(target, format="PNG")
            created_pages += 1

    action = "created" if existing_before == 0 else "healed"
    return {
        "action": action,
        "page_count": page_count,
        "created_pages": created_pages,
        "reused_pages": page_count - created_pages,
        "missing_before": len(missing_pages),
    }


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
