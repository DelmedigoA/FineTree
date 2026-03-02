"""Simple PyQt5 app for annotating PDF page images with schema-based facts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError
from PyQt5 import sip
from PyQt5.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QTextCursor, QTransform
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
    QVBoxLayout,
    QWidget,
)

from .annotation_core import (
    BoxRecord,
    CURRENCY_OPTIONS,
    PageState,
    SCALE_OPTIONS,
    build_annotations_payload,
    denormalize_bbox_from_1000,
    default_page_meta,
    load_page_states,
    parse_import_payload,
    normalize_bbox_data,
    normalize_fact_data,
    propagate_entity_to_next_page,
    serialize_annotations_json,
)
from .finetune.config import load_finetune_config
from .schemas import PageType


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
        pen.setWidth(2)
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
                current_scene = item_scene_rect(self)
                delta = new_pos - self.pos()
                moved = current_scene.translated(delta)
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
            self._active_handle = None
            self._resize_start_rect = None
            self._resize_start_scene_pos = None
            self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
            self.setCursor(Qt.OpenHandCursor)
            scene = self.scene()
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

    def set_image_rect(self, rect: QRectF) -> None:
        self.image_rect = rect

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.image_rect.contains(event.scenePos()):
            if event.modifiers() & Qt.ShiftModifier:
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
            item = self.itemAt(event.scenePos(), QTransform())
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
            if item is None or isinstance(item, QGraphicsPixmapItem):
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
    def __init__(self, prompt_text: str, model_name: str, parent: Optional[QWidget] = None) -> None:
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
        self.model_edit = QLineEdit(model_name)
        form.addRow("model", self.model_edit)
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
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
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
    def __init__(self, prompt_text: str, model_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(prompt_text=prompt_text, model_name=model_name, parent=parent)
        self.setWindowTitle("Qwen Prompt")
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Start Qwen GT")


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
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.config_path = config_path
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
    def __init__(self, images_dir: Path, annotations_path: Path) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.page_images: List[Path] = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        )
        self.page_states: Dict[str, PageState] = {}
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
        self._gemini_stream_dialog: Optional[GeminiStreamDialog] = None
        self._gemini_stream_target_page: Optional[str] = None
        self._gemini_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._gemini_stream_fact_count = 0
        self._gemini_stream_cancel_requested = False
        self._gemini_model_name = os.getenv("FINETREE_GEMINI_MODEL", "gemini-3-flash-preview")
        self._qwen_stream_thread: Optional[QThread] = None
        self._qwen_stream_worker: Optional[QwenStreamWorker] = None
        self._qwen_stream_dialog: Optional[QwenStreamDialog] = None
        self._qwen_stream_target_page: Optional[str] = None
        self._qwen_stream_seen_facts: set[tuple[Any, ...]] = set()
        self._qwen_stream_fact_count = 0
        self._qwen_stream_cancel_requested = False
        self._qwen_model_name = self._initial_qwen_model_name()

        if not self.page_images:
            raise RuntimeError(f"No page images found under: {self.images_dir}")

        self.setWindowTitle("FineTree PDF Annotator")
        self.resize(1320, 860)
        self._build_ui()
        self._load_existing_annotations()
        self.show_page(0)
        self._init_history()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        self._apply_ui_sizing()
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(10, 10, 10, 10)

        nav = QHBoxLayout()
        nav.setSpacing(10)
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.page_label = QLabel("-")
        self.page_jump_spin = QSpinBox()
        self.page_jump_spin.setRange(1, len(self.page_images))
        self.page_jump_spin.setKeyboardTracking(False)
        self.page_jump_spin.setValue(1)
        self.page_jump_spin.setMinimumWidth(88)
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
        self.help_btn = QPushButton("Help")
        self.save_btn = QPushButton("Save (Ctrl+S)")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.page_label)
        nav.addWidget(QLabel("Go to page"))
        nav.addWidget(self.page_jump_spin)
        nav.addWidget(self.page_jump_btn)
        nav.addWidget(self.undo_btn)
        nav.addWidget(self.redo_btn)
        nav.addWidget(self.import_btn)
        nav.addWidget(self.gemini_gt_btn)
        nav.addWidget(self.qwen_gt_btn)
        nav.addWidget(self.delete_nav_btn)
        nav.addWidget(self.zoom_out_btn)
        nav.addWidget(self.zoom_in_btn)
        nav.addWidget(self.lens_btn)
        nav.addWidget(self.copy_image_btn)
        nav.addWidget(self.fit_btn)
        nav.addWidget(self.page_json_btn)
        nav.addWidget(self.help_btn)
        nav.addStretch(1)
        self.output_path_label = QLabel(str(self.annotations_path))
        self.output_path_label.setObjectName("outputPathLabel")
        nav.addWidget(self.output_path_label)
        nav.addWidget(self.save_btn)
        root.addLayout(nav)

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
        self.page_thumb_list.setIconSize(QSize(96, 132))
        self.page_thumb_list.setSpacing(8)

        thumb_panel = QWidget()
        thumb_panel.setMinimumWidth(130)
        thumb_panel.setMaximumWidth(220)
        thumb_layout = QVBoxLayout(thumb_panel)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(6)
        thumb_title = QLabel("Pages")
        thumb_title.setObjectName("hintText")
        thumb_layout.addWidget(thumb_title)
        thumb_layout.addWidget(self.page_thumb_list, 1)

        splitter.addWidget(thumb_panel)
        splitter.addWidget(self.view)

        right = QWidget()
        right.setMinimumWidth(560)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        meta_box = QGroupBox("Page Metadata")
        meta_form = QFormLayout(meta_box)
        meta_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        meta_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        meta_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        meta_form.setHorizontalSpacing(16)
        meta_form.setVerticalSpacing(12)
        self.entity_name_edit = QLineEdit()
        self.page_num_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems([p.value for p in PageType])
        self.title_edit = QLineEdit()
        field_min_width = 500
        self.entity_name_edit.setMinimumWidth(field_min_width)
        self.page_num_edit.setMinimumWidth(field_min_width)
        self.type_combo.setMinimumWidth(field_min_width)
        self.title_edit.setMinimumWidth(field_min_width)
        meta_form.addRow("entity_name", self.entity_name_edit)
        meta_form.addRow("page_num", self.page_num_edit)
        meta_form.addRow("type*", self.type_combo)
        meta_form.addRow("title", self.title_edit)
        right_layout.addWidget(meta_box)

        fact_box = QGroupBox("Facts (Bounding Boxes)")
        fact_layout = QVBoxLayout(fact_box)
        fact_layout.setSpacing(10)
        self.facts_list = QListWidget()
        self.facts_list.setSpacing(4)
        self.fact_bbox_label = QLabel("bbox: -")
        self.fact_value_edit = QLineEdit()
        self.fact_note_edit = QLineEdit()
        self.fact_is_beur_combo = QComboBox()
        self.fact_is_beur_combo.addItems(["", "true", "false"])
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
        self.fact_path_list.setMinimumHeight(150)

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
        fact_editor_form = QFormLayout(self.fact_editor_box)
        fact_editor_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fact_editor_form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        fact_editor_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        fact_editor_form.setHorizontalSpacing(14)
        fact_editor_form.setVerticalSpacing(10)
        fact_editor_form.addRow("bbox", self.fact_bbox_label)
        fact_editor_form.addRow("value*", self.fact_value_edit)
        fact_editor_form.addRow("note", self.fact_note_edit)
        fact_editor_form.addRow("is_beur", self.fact_is_beur_combo)
        fact_editor_form.addRow("beur_num", self.fact_beur_num_edit)
        fact_editor_form.addRow("refference*", self.fact_refference_edit)
        fact_editor_form.addRow("date", self.fact_date_edit)
        fact_editor_form.addRow("currency", self.fact_currency_combo)
        fact_editor_form.addRow("scale", self.fact_scale_combo)
        fact_editor_form.addRow("value_type", self.fact_value_type_combo)
        fact_editor_form.addRow("path", path_panel)
        self.fact_bbox_label.setObjectName("factBboxLabel")
        self.fact_value_edit.setMinimumWidth(field_min_width)
        self.fact_note_edit.setMinimumWidth(field_min_width)
        self.fact_is_beur_combo.setMinimumWidth(field_min_width)
        self.fact_beur_num_edit.setMinimumWidth(field_min_width)
        self.fact_refference_edit.setMinimumWidth(field_min_width)
        self.fact_date_edit.setMinimumWidth(field_min_width)
        self.fact_currency_combo.setMinimumWidth(field_min_width)
        self.fact_scale_combo.setMinimumWidth(field_min_width)
        self.fact_value_type_combo.setMinimumWidth(field_min_width)
        self.fact_path_list.setMinimumWidth(field_min_width)

        self.dup_fact_btn = QPushButton("Duplicate BBox (Cmd/Ctrl+D)")
        self.del_fact_btn = QPushButton("Delete Selected (Del)")
        fact_layout.addWidget(self.facts_list, 1)
        fact_layout.addWidget(self.fact_editor_box)
        fact_layout.addWidget(self.dup_fact_btn)
        fact_layout.addWidget(self.del_fact_btn)
        self.batch_box = QGroupBox("Batch Edit Selected BBoxes")
        batch_layout = QVBoxLayout(self.batch_box)
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
        self.batch_clear_refference_btn = QPushButton("Clear Refference")
        self.batch_note_edit = QLineEdit()
        self.batch_note_edit.setPlaceholderText("Note text for selected bboxes")
        self.batch_set_note_btn = QPushButton("Set Note")
        self.batch_clear_note_btn = QPushButton("Clear Note")
        self.batch_is_beur_combo = QComboBox()
        self.batch_is_beur_combo.addItems(["", "true", "false"])
        self.batch_set_is_beur_btn = QPushButton("Apply is_beur")
        self.batch_clear_is_beur_btn = QPushButton("Clear is_beur")
        self.batch_beur_num_edit = QLineEdit()
        self.batch_beur_num_edit.setPlaceholderText("beur_num for selected bboxes")
        self.batch_set_beur_num_btn = QPushButton("Set beur_num")
        self.batch_clear_beur_num_btn = QPushButton("Clear beur_num")
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
        batch_row_2.addWidget(self.batch_clear_refference_btn)
        batch_note_row = QHBoxLayout()
        batch_note_row.setSpacing(8)
        batch_note_row.addWidget(self.batch_note_edit)
        batch_note_row.addWidget(self.batch_set_note_btn)
        batch_note_row.addWidget(self.batch_clear_note_btn)
        batch_is_beur_row = QHBoxLayout()
        batch_is_beur_row.setSpacing(8)
        batch_is_beur_row.addWidget(QLabel("is_beur:"))
        batch_is_beur_row.addWidget(self.batch_is_beur_combo)
        batch_is_beur_row.addWidget(self.batch_set_is_beur_btn)
        batch_is_beur_row.addWidget(self.batch_clear_is_beur_btn)
        batch_is_beur_row.addStretch(1)
        batch_beur_num_row = QHBoxLayout()
        batch_beur_num_row.setSpacing(8)
        batch_beur_num_row.addWidget(self.batch_beur_num_edit)
        batch_beur_num_row.addWidget(self.batch_set_beur_num_btn)
        batch_beur_num_row.addWidget(self.batch_clear_beur_num_btn)
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
        batch_layout.addLayout(batch_note_row)
        batch_layout.addLayout(batch_is_beur_row)
        batch_layout.addLayout(batch_beur_num_row)
        batch_layout.addLayout(batch_resize_head)
        batch_layout.addLayout(batch_row_3)
        fact_layout.addWidget(self.batch_box)
        right_layout.addWidget(fact_box, 1)

        tip = QLabel(
            "Select a box to edit fields here. "
            "Use Shift+drag on the page to select multiple boxes. "
            "Use Batch Edit to add/remove path levels, set/clear note, apply is_beur, set/clear beur_num, and clear references across selected boxes. "
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
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([170, 860, 560])

        self.prev_btn.clicked.connect(lambda: self.show_page(self.current_index - 1))
        self.next_btn.clicked.connect(lambda: self.show_page(self.current_index + 1))
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
        self.help_btn.clicked.connect(self.show_help_dialog)
        self.save_btn.clicked.connect(self.save_annotations)
        self.import_btn.clicked.connect(self.import_annotations_json)
        self.gemini_gt_btn.clicked.connect(self.generate_gemini_ground_truth)
        self.qwen_gt_btn.clicked.connect(self.generate_qwen_ground_truth)
        self.delete_nav_btn.clicked.connect(self.delete_selected_fact)
        self.dup_fact_btn.clicked.connect(self.duplicate_selected_fact)
        self.del_fact_btn.clicked.connect(self.delete_selected_fact)
        self.facts_list.currentRowChanged.connect(self._on_fact_row_changed)
        self.entity_name_edit.editingFinished.connect(self._on_meta_edited)
        self.page_num_edit.editingFinished.connect(self._on_meta_edited)
        self.title_edit.editingFinished.connect(self._on_meta_edited)
        self.type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.fact_value_edit.editingFinished.connect(self._on_fact_editor_edited)
        self.fact_note_edit.editingFinished.connect(self._on_fact_editor_edited)
        self.fact_is_beur_combo.activated.connect(lambda _: self._on_fact_editor_edited())
        self.fact_beur_num_edit.editingFinished.connect(self._on_fact_editor_edited)
        self.fact_refference_edit.editingFinished.connect(self._on_fact_editor_edited)
        self.fact_date_edit.editingFinished.connect(self._on_fact_editor_edited)
        self.fact_currency_combo.activated.connect(lambda _: self._on_fact_editor_edited())
        self.fact_scale_combo.activated.connect(lambda _: self._on_fact_editor_edited())
        self.fact_value_type_combo.activated.connect(lambda _: self._on_fact_editor_edited())
        self.fact_path_list.itemChanged.connect(lambda _: self._on_fact_editor_edited())
        self.fact_path_list.itemSelectionChanged.connect(self._update_path_controls)
        self.fact_path_list.model().rowsMoved.connect(lambda *_: self._on_path_reordered())
        self.path_add_btn.clicked.connect(self.add_path_level)
        self.path_remove_btn.clicked.connect(self.remove_selected_path_level)
        self.path_up_btn.clicked.connect(self.move_selected_path_up)
        self.path_down_btn.clicked.connect(self.move_selected_path_down)
        self.batch_path_level_edit.textChanged.connect(self._update_batch_controls)
        self.batch_note_edit.textChanged.connect(self._update_batch_controls)
        self.batch_is_beur_combo.activated.connect(lambda _: self._update_batch_controls())
        self.batch_beur_num_edit.textChanged.connect(self._update_batch_controls)
        self.batch_prepend_path_btn.clicked.connect(self.batch_prepend_path_level)
        self.batch_append_path_btn.clicked.connect(self.batch_append_path_level)
        self.batch_insert_path_btn.clicked.connect(self.batch_insert_path_level)
        self.batch_remove_first_btn.clicked.connect(self.batch_remove_first_level)
        self.batch_remove_last_btn.clicked.connect(self.batch_remove_last_level)
        self.batch_clear_refference_btn.clicked.connect(self.batch_clear_refference)
        self.batch_set_note_btn.clicked.connect(self.batch_set_note)
        self.batch_clear_note_btn.clicked.connect(self.batch_clear_note)
        self.batch_set_is_beur_btn.clicked.connect(self.batch_set_is_beur)
        self.batch_clear_is_beur_btn.clicked.connect(self.batch_clear_is_beur)
        self.batch_set_beur_num_btn.clicked.connect(self.batch_set_beur_num)
        self.batch_clear_beur_num_btn.clicked.connect(self.batch_clear_beur_num)
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
        self.page_label.setObjectName("pageIndicator")
        self._populate_page_thumbnails()
        self.statusBar().showMessage("Ready")
        self._set_fact_editor_enabled(False)
        self._clear_fact_editor()
        self._update_path_controls()
        self._update_batch_controls()
        self._update_history_controls()

    def _apply_ui_sizing(self) -> None:
        base_font = QFont(self.font())
        if base_font.pointSize() < 12:
            base_font.setPointSize(12)
        self.setFont(base_font)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #eef2f8;
                color: #1f2a3d;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7e1ef;
                border-radius: 12px;
                font-size: 14pt;
                font-weight: 600;
                margin-top: 14px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1f2a3d;
            }
            QLabel, QStatusBar {
                background: transparent;
                color: #1f2a3d;
                font-size: 12pt;
            }
            QLabel#pageIndicator {
                color: #37537d;
                font-size: 12pt;
                font-weight: 600;
                padding: 0 8px;
            }
            QLabel#outputPathLabel {
                color: #4e6183;
                font-size: 10pt;
            }
            QLabel#hintText {
                color: #3d5379;
                font-size: 11pt;
                padding-top: 2px;
            }
            QLabel#factBboxLabel {
                color: #4b5f82;
                font-weight: 600;
            }
            QLineEdit,
            QComboBox {
                background: #f8fbff;
                border: 1px solid #c6d3e8;
                border-radius: 10px;
                font-size: 13pt;
                min-height: 36px;
                padding: 2px 10px;
            }
            QLineEdit:focus,
            QComboBox:focus {
                border: 2px solid #5b8ff9;
                background: #ffffff;
            }
            QPushButton {
                background: #e7efff;
                border: 1px solid #c4d4f3;
                border-radius: 10px;
                color: #253a5f;
                font-size: 12pt;
                min-height: 38px;
                padding: 4px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #d8e6ff;
            }
            QPushButton:pressed {
                background: #c8dcff;
            }
            QPushButton:disabled {
                color: #8a99b6;
                background: #eef2f9;
                border: 1px solid #dde5f2;
            }
            QPushButton#smallActionBtn {
                min-height: 32px;
                font-size: 11pt;
                padding: 2px 10px;
            }
            QListWidget {
                background: #f8fbff;
                border: 1px solid #c6d3e8;
                border-radius: 10px;
                font-size: 12pt;
                padding: 4px;
            }
            QListWidget::item {
                min-height: 26px;
                border-radius: 6px;
                padding: 2px 6px;
            }
            QListWidget::item:selected {
                background: #cfe0ff;
                color: #1f2a3d;
            }
            QListWidget#thumbList {
                background: #f3f7ff;
                border: 1px solid #c6d3e8;
                border-radius: 10px;
                padding: 6px;
            }
            QListWidget#thumbList::item {
                min-height: 126px;
                border: 1px solid #d6e0ef;
                background: #ffffff;
                margin: 2px;
                padding: 4px;
            }
            QListWidget#thumbList::item:selected {
                border: 2px solid #5b8ff9;
                background: #dfeaff;
                color: #1f2a3d;
            }
            QGraphicsView {
                background: #dfe8f6;
                border: 1px solid #c3d2e7;
                border-radius: 12px;
            }
            QStatusBar {
                background: #e7edf7;
            }
            """
        )

    def _make_path_item(self, text: str) -> QListWidgetItem:
        item = QListWidgetItem(text)
        item.setFlags(
            item.flags()
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
        )
        return item

    def _update_path_controls(self) -> None:
        enabled = self.fact_path_list.isEnabled()
        row = self.fact_path_list.currentRow()
        count = self.fact_path_list.count()
        self.path_add_btn.setEnabled(enabled)
        self.path_remove_btn.setEnabled(enabled and row >= 0)
        self.path_up_btn.setEnabled(enabled and row > 0)
        self.path_down_btn.setEnabled(enabled and 0 <= row < count - 1)

    def add_path_level(self) -> None:
        if not self.fact_path_list.isEnabled():
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
        self._on_fact_editor_edited()
        if 0 <= target_row < self.fact_path_list.count():
            refreshed_item = self.fact_path_list.item(target_row)
            if refreshed_item is not None:
                self.fact_path_list.setCurrentRow(target_row)
                self.fact_path_list.editItem(refreshed_item)

    def remove_selected_path_level(self) -> None:
        if not self.fact_path_list.isEnabled():
            return
        row = self.fact_path_list.currentRow()
        if row < 0:
            return
        self.fact_path_list.takeItem(row)
        if self.fact_path_list.count() > 0:
            self.fact_path_list.setCurrentRow(min(row, self.fact_path_list.count() - 1))
        self._update_path_controls()
        self._on_fact_editor_edited()

    def move_selected_path_up(self) -> None:
        if not self.fact_path_list.isEnabled():
            return
        row = self.fact_path_list.currentRow()
        if row <= 0:
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row - 1, item)
        self.fact_path_list.setCurrentRow(row - 1)
        self._update_path_controls()
        self._on_fact_editor_edited()

    def move_selected_path_down(self) -> None:
        if not self.fact_path_list.isEnabled():
            return
        row = self.fact_path_list.currentRow()
        if row < 0 or row >= (self.fact_path_list.count() - 1):
            return
        item = self.fact_path_list.takeItem(row)
        self.fact_path_list.insertItem(row + 1, item)
        self.fact_path_list.setCurrentRow(row + 1)
        self._update_path_controls()
        self._on_fact_editor_edited()

    def _on_path_reordered(self) -> None:
        self._update_path_controls()
        self._on_fact_editor_edited()

    def _update_batch_controls(self) -> None:
        selected_count = len(self._selected_fact_items()) if hasattr(self, "scene") else 0
        has_selection = selected_count > 0
        has_text = bool(self.batch_path_level_edit.text().strip()) if hasattr(self, "batch_path_level_edit") else False
        has_note_text = bool(self.batch_note_edit.text().strip()) if hasattr(self, "batch_note_edit") else False
        has_is_beur_choice = bool(self.batch_is_beur_combo.currentText().strip()) if hasattr(self, "batch_is_beur_combo") else False
        has_beur_num_text = bool(self.batch_beur_num_edit.text().strip()) if hasattr(self, "batch_beur_num_edit") else False
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
        if hasattr(self, "batch_clear_refference_btn"):
            self.batch_clear_refference_btn.setEnabled(has_selection)
        if hasattr(self, "batch_note_edit"):
            self.batch_note_edit.setEnabled(has_selection)
        if hasattr(self, "batch_set_note_btn"):
            self.batch_set_note_btn.setEnabled(has_selection and has_note_text)
        if hasattr(self, "batch_clear_note_btn"):
            self.batch_clear_note_btn.setEnabled(has_selection)
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

        self.refresh_facts_list()
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
                    "Clear refference for all selected bboxes anyway?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["refference"] = ""
            return fact

        self._batch_update_selected_facts(_transform, "Cleared refference")

    def batch_set_note(self) -> None:
        note = self.batch_note_edit.text().strip()
        if not note:
            self.statusBar().showMessage("Enter note text first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note"] = note
            return fact

        self._batch_update_selected_facts(_transform, "Updated note")

    def batch_clear_note(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["note"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared note")

    def batch_set_is_beur(self) -> None:
        selected = self.batch_is_beur_combo.currentText().strip().lower()
        if selected == "true":
            value: Optional[bool] = True
        elif selected == "false":
            value = False
        else:
            self.statusBar().showMessage("Choose is_beur value first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["is_beur"] = value
            return fact

        self._batch_update_selected_facts(_transform, f"Updated is_beur to {selected}")

    def batch_clear_is_beur(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["is_beur"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared is_beur")

    def batch_set_beur_num(self) -> None:
        beur_num = self.batch_beur_num_edit.text().strip()
        if not beur_num:
            self.statusBar().showMessage("Enter beur_num first.", 2500)
            return

        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["beur_num"] = beur_num
            return fact

        self._batch_update_selected_facts(_transform, "Updated beur_num")

    def batch_clear_beur_num(self) -> None:
        def _transform(fact: Dict[str, Any]) -> Dict[str, Any]:
            fact["beur_num"] = None
            return fact

        self._batch_update_selected_facts(_transform, "Cleared beur_num")

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

    def _set_fact_editor_enabled(self, enabled: bool) -> None:
        self.fact_value_edit.setEnabled(enabled)
        self.fact_note_edit.setEnabled(enabled)
        self.fact_is_beur_combo.setEnabled(enabled)
        self.fact_beur_num_edit.setEnabled(enabled)
        self.fact_refference_edit.setEnabled(enabled)
        self.fact_date_edit.setEnabled(enabled)
        self.fact_currency_combo.setEnabled(enabled)
        self.fact_scale_combo.setEnabled(enabled)
        self.fact_value_type_combo.setEnabled(enabled)
        self.fact_path_list.setEnabled(enabled)
        self.dup_fact_btn.setEnabled(enabled)
        self.del_fact_btn.setEnabled(enabled)
        self._update_path_controls()

    def _clear_fact_editor(self) -> None:
        self._is_loading_fact_editor = True
        try:
            self.fact_bbox_label.setText("-")
            self.fact_value_edit.setText("")
            self.fact_note_edit.setText("")
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
            self.fact_bbox_label.setText(
                f"{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}"
            )
            self.fact_value_edit.setText(str(fact.get("value", "")))
            self.fact_note_edit.setText(str(fact.get("note") or ""))
            is_beur = fact.get("is_beur")
            is_beur_text = "true" if is_beur is True else ("false" if is_beur is False else "")
            idx_is_beur = self.fact_is_beur_combo.findText(is_beur_text)
            self.fact_is_beur_combo.setCurrentIndex(max(0, idx_is_beur))
            self.fact_beur_num_edit.setText(str(fact.get("beur_num") or ""))
            self.fact_refference_edit.setText(str(fact.get("refference") or ""))
            self.fact_date_edit.setText(str(fact.get("date") or ""))
            self.fact_path_list.clear()
            for path_level in fact.get("path") or []:
                self.fact_path_list.addItem(self._make_path_item(str(path_level)))
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
        if is_beur_text == "true":
            is_beur_value: Optional[bool] = True
        elif is_beur_text == "false":
            is_beur_value = False
        else:
            is_beur_value = None
        return normalize_fact_data(
            {
                "value": self.fact_value_edit.text().strip(),
                "note": self.fact_note_edit.text().strip() or None,
                "is_beur": is_beur_value,
                "beur_num": self.fact_beur_num_edit.text().strip() or None,
                "refference": self.fact_refference_edit.text().strip(),
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
        item = self._selected_fact_item()
        if item is None or not self._is_alive_fact_item(item):
            self._clear_fact_editor()
            self._set_fact_editor_enabled(False)
            return
        self._set_fact_editor_enabled(True)
        self._populate_fact_editor(item)

    def _on_fact_editor_edited(self) -> None:
        if self._is_loading_fact_editor or self._syncing_selection:
            return
        item = self._selected_fact_item()
        if item is None or not self._is_alive_fact_item(item):
            return
        updated_fact = self._fact_data_from_editor()
        if updated_fact == normalize_fact_data(item.fact_data):
            return
        item.fact_data = updated_fact
        self.refresh_facts_list()
        self._record_history_snapshot()

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
            self._history_index = index
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

        candidates.append(Path.cwd() / "prompt.txt")
        for parent in Path(__file__).resolve().parents:
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

    def _build_prompt_from_template(self, template: str, page_image_path: Path) -> str:
        prompt = template.replace("{{PAGE_IMAGE}}", str(page_image_path))
        prompt = prompt.replace("{{IMAGE_NAME}}", page_image_path.name)
        return prompt

    def _fact_uniqueness_key(self, fact_payload: Dict[str, Any]) -> tuple[Any, ...]:
        bbox = fact_payload.get("bbox") or {}
        path = tuple(str(p) for p in (fact_payload.get("path") or []))
        return (
            round(float(bbox.get("x", 0.0)), 2),
            round(float(bbox.get("y", 0.0)), 2),
            round(float(bbox.get("w", 0.0)), 2),
            round(float(bbox.get("h", 0.0)), 2),
            str(fact_payload.get("value") or ""),
            str(fact_payload.get("note") or ""),
            str(fact_payload.get("is_beur") if fact_payload.get("is_beur") is not None else ""),
            str(fact_payload.get("beur_num") or ""),
            str(fact_payload.get("refference") or ""),
            str(fact_payload.get("date") or ""),
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
        if not isinstance(raw_bbox, dict):
            return False
        bbox = normalize_bbox_data(raw_bbox)
        image_dims = self._image_dimensions_for_page(page_name)
        if image_dims and self._bbox_looks_normalized_1000(bbox):
            bbox = denormalize_bbox_from_1000(bbox, image_dims[0], image_dims[1])
        fact_data = normalize_fact_data(
            {
                "value": fact_payload.get("value"),
                "note": fact_payload.get("note"),
                "is_beur": fact_payload.get("is_beur"),
                "beur_num": fact_payload.get("beur_num"),
                "refference": fact_payload.get("refference"),
                "date": fact_payload.get("date"),
                "path": fact_payload.get("path") or [],
                "currency": fact_payload.get("currency"),
                "scale": fact_payload.get("scale"),
                "value_type": fact_payload.get("value_type"),
            }
        )

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
            QMessageBox.warning(
                self,
                "Gemini GT",
                "Could not find prompt.txt. Place it in the current working directory or set FINETREE_PROMPT_PATH.",
            )
            return

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
        )
        if prompt_dialog.exec_() != QDialog.Accepted:
            return
        prompt_text = prompt_dialog.prompt().strip()
        model_name = prompt_dialog.model().strip() or self._gemini_model_name
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

        # Keep this dialog modeless so users can continue annotating while streaming.
        self._gemini_stream_dialog = GeminiStreamDialog(page_name=page_name, parent=None)
        self._gemini_stream_dialog.set_status(f"Streaming from {model_name} ...")
        self._gemini_stream_dialog.stop_requested.connect(self._cancel_gemini_stream)
        self._gemini_stream_dialog.show()

        worker = GeminiStreamWorker(
            image_path=page_path,
            prompt=prompt_text,
            model=model_name,
            api_key=gemini_api_key,
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
            if self._gemini_stream_dialog is not None:
                self._gemini_stream_dialog.set_status("Stopping stream...")

    def _on_gemini_stream_chunk(self, text: str) -> None:
        if self._gemini_stream_dialog is not None:
            self._gemini_stream_dialog.append_text(text)

    def _on_gemini_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        page_name = self._gemini_stream_target_page
        if page_name is None:
            return
        self._apply_stream_meta(page_name, meta_payload)
        if self._gemini_stream_dialog is not None:
            self._gemini_stream_dialog.set_status("Meta received. Streaming facts...")

    def _on_gemini_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        page_name = self._gemini_stream_target_page
        if page_name is None:
            return
        added = self._apply_stream_fact(page_name, fact_payload, seen_facts=self._gemini_stream_seen_facts)
        if added and self._gemini_stream_dialog is not None:
            self._gemini_stream_fact_count += 1
            self._gemini_stream_dialog.set_status(f"Streaming facts... {self._gemini_stream_fact_count} parsed")

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
        if self._gemini_stream_dialog is not None:
            self._gemini_stream_dialog.mark_done(
                f"Gemini GT complete. Parsed {self._gemini_stream_fact_count} fact(s)."
            )
        self.statusBar().showMessage(f"Gemini GT complete ({self._gemini_stream_fact_count} fact(s)).", 6000)

    def _on_gemini_stream_failed(self, message: str) -> None:
        if self._gemini_stream_dialog is not None:
            self._gemini_stream_dialog.mark_error(f"Error: {message}")
        QMessageBox.warning(
            self,
            "Gemini GT failed",
            f"{message}\n\nAny facts already streamed remain on the page.",
        )

    def _on_gemini_stream_finished(self) -> None:
        if self._gemini_stream_cancel_requested and self._gemini_stream_dialog is not None:
            self._gemini_stream_dialog.mark_done(
                f"Gemini GT stopped. Parsed {self._gemini_stream_fact_count} fact(s) before stop."
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
            QMessageBox.warning(
                self,
                "Qwen GT",
                "Could not find prompt.txt. Place it in the current working directory or set FINETREE_PROMPT_PATH.",
            )
            return

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
        )
        if prompt_dialog.exec_() != QDialog.Accepted:
            return
        prompt_text = prompt_dialog.prompt().strip()
        model_name = prompt_dialog.model().strip() or self._qwen_model_name
        if not prompt_text:
            QMessageBox.warning(self, "Qwen GT", "Prompt cannot be empty.")
            return

        self._qwen_model_name = model_name

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

        self._qwen_stream_dialog = QwenStreamDialog(page_name=page_name, parent=None)
        self._qwen_stream_dialog.set_status(f"Streaming from {model_name} ...")
        self._qwen_stream_dialog.stop_requested.connect(self._cancel_qwen_stream)
        self._qwen_stream_dialog.show()

        worker = QwenStreamWorker(
            image_path=page_path,
            prompt=prompt_text,
            model=model_name,
            config_path=str(qwen_config_path),
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
            if self._qwen_stream_dialog is not None:
                self._qwen_stream_dialog.set_status("Stopping stream...")

    def _on_qwen_stream_chunk(self, text: str) -> None:
        if self._qwen_stream_dialog is not None:
            self._qwen_stream_dialog.append_text(text)

    def _on_qwen_stream_meta(self, meta_payload: Dict[str, Any]) -> None:
        page_name = self._qwen_stream_target_page
        if page_name is None:
            return
        self._apply_stream_meta(page_name, meta_payload)
        if self._qwen_stream_dialog is not None:
            self._qwen_stream_dialog.set_status("Meta received. Streaming facts...")

    def _on_qwen_stream_fact(self, fact_payload: Dict[str, Any]) -> None:
        page_name = self._qwen_stream_target_page
        if page_name is None:
            return
        added = self._apply_stream_fact(page_name, fact_payload, seen_facts=self._qwen_stream_seen_facts)
        if added:
            self._qwen_stream_fact_count += 1
            if self._qwen_stream_dialog is not None:
                self._qwen_stream_dialog.set_status(f"Streaming facts... {self._qwen_stream_fact_count} parsed")

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
        if self._qwen_stream_dialog is not None:
            self._qwen_stream_dialog.mark_done(
                f"Qwen GT complete. Parsed {self._qwen_stream_fact_count} fact(s)."
            )
        self.statusBar().showMessage(f"Qwen GT complete ({self._qwen_stream_fact_count} fact(s)).", 6000)

    def _on_qwen_stream_failed(self, message: str) -> None:
        if self._qwen_stream_dialog is not None:
            self._qwen_stream_dialog.mark_error(f"Error: {message}")
        QMessageBox.warning(
            self,
            "Qwen GT failed",
            f"{message}\n\nAny facts already streamed remain on the page.",
        )

    def _on_qwen_stream_finished(self) -> None:
        if self._qwen_stream_cancel_requested and self._qwen_stream_dialog is not None:
            self._qwen_stream_dialog.mark_done(
                f"Qwen GT stopped. Parsed {self._qwen_stream_fact_count} fact(s) before stop."
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
        page_name = self.page_images[self.current_index].name
        meta = {
            "entity_name": self.entity_name_edit.text().strip() or None,
            "page_num": self.page_num_edit.text().strip() or None,
            "type": self.type_combo.currentText(),
            "title": self.title_edit.text().strip() or None,
        }
        facts: List[BoxRecord] = []
        for item in self._sorted_fact_items():
            facts.append(BoxRecord(bbox=bbox_to_dict(item_scene_rect(item)), fact=normalize_fact_data(item.fact_data)))
        self.page_states[page_name] = PageState(meta=meta, facts=facts)
        propagate_entity_to_next_page(self.page_states, self.page_images, self.current_index, meta["entity_name"])

    def _load_existing_annotations(self) -> None:
        if not self.annotations_path.exists():
            return
        try:
            payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Failed to load annotations", str(exc))
            return
        self.page_states = load_page_states(payload, [p.name for p in self.page_images])

    def _populate_page_thumbnails(self) -> None:
        if not hasattr(self, "page_thumb_list"):
            return
        self.page_thumb_list.blockSignals(True)
        self.page_thumb_list.clear()
        icon_size = self.page_thumb_list.iconSize()
        for idx, page_path in enumerate(self.page_images):
            item = QListWidgetItem(str(idx + 1))
            item.setToolTip(page_path.name)
            item.setTextAlignment(Qt.AlignCenter)
            pixmap = QPixmap(str(page_path))
            if not pixmap.isNull():
                thumb = pixmap.scaled(icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item.setIcon(QIcon(thumb))
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
        self._fit_view_height()

    def _sorted_fact_items(self) -> List[AnnotRectItem]:
        items = [i for i in self.scene.items() if isinstance(i, AnnotRectItem) and self._is_alive_fact_item(i)]
        def sort_key(item: AnnotRectItem):
            rect = item_scene_rect(item)
            return (rect.y(), rect.x(), rect.width(), rect.height())

        items.sort(key=sort_key)
        return items

    def refresh_facts_list(self) -> None:
        selected = self._selected_fact_item()
        self._fact_items = self._sorted_fact_items()
        self._syncing_selection = True
        self.facts_list.clear()
        selected_row = -1
        for idx, item in enumerate(self._fact_items, start=1):
            rect = item_scene_rect(item)
            value = str(item.fact_data.get("value") or "")
            path = " > ".join(item.fact_data.get("path") or [])
            note = str(item.fact_data.get("note") or "")
            summary = f"#{idx} [{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}] {value}"
            if path:
                summary = f"{summary} | {path}"
            if note:
                trimmed_note = (note[:32] + "...") if len(note) > 35 else note
                summary = f"{summary} | note: {trimmed_note}"
            if item.fact_data.get("is_beur") is not None:
                summary = f"{summary} | is_beur: {item.fact_data.get('is_beur')}"
            if item.fact_data.get("beur_num"):
                summary = f"{summary} | beur_num: {item.fact_data.get('beur_num')}"
            self.facts_list.addItem(QListWidgetItem(summary))
            if item is selected:
                selected_row = idx - 1
        if selected_row >= 0:
            self.facts_list.setCurrentRow(selected_row)
        self._syncing_selection = False
        self._sync_fact_editor_from_selection()
        self._update_batch_controls()

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
        self.fact_value_edit.setFocus()
        self.fact_value_edit.selectAll()
        self._record_history_snapshot()

    def _on_box_duplicated(self, item: AnnotRectItem) -> None:
        if not self._is_alive_fact_item(item):
            return
        self.refresh_facts_list()
        self._record_history_snapshot()
        self.statusBar().showMessage("Duplicated bbox. Drag to place it.", 2500)

    def _on_box_double_clicked(self, item: AnnotRectItem) -> None:
        self.scene.clearSelection()
        item.setSelected(True)
        self.refresh_facts_list()
        self.fact_value_edit.setFocus()
        self.fact_value_edit.selectAll()

    def _on_box_geometry_changed(self, _item: AnnotRectItem) -> None:
        self.refresh_facts_list()
        self._record_history_snapshot()

    def _on_meta_edited(self) -> None:
        if self._is_loading_page or self._is_restoring_history:
            return
        self._record_history_snapshot()

    def _on_scene_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        selected = self._selected_fact_item()
        self._syncing_selection = True
        if selected is None:
            self.facts_list.clearSelection()
        else:
            for row, item in enumerate(self._fact_items):
                if item is selected:
                    self.facts_list.setCurrentRow(row)
                    break
        self._syncing_selection = False
        self._sync_fact_editor_from_selection()
        self._update_batch_controls()

    def _on_fact_row_changed(self, row: int) -> None:
        if self._syncing_selection:
            return
        self.scene.clearSelection()
        if 0 <= row < len(self._fact_items):
            item = self._fact_items[row]
            if self._is_alive_fact_item(item):
                item.setSelected(True)
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
        self._record_history_snapshot()
        deleted_count = len(selected_items)
        if deleted_count == 1:
            self.statusBar().showMessage("Deleted selected bbox.", 2000)
        else:
            self.statusBar().showMessage(f"Deleted {deleted_count} selected bboxes.", 2500)

    def save_annotations(self) -> None:
        self._capture_current_state()
        try:
            payload = build_annotations_payload(self.images_dir, self.page_images, self.page_states)
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path.write_text(serialize_annotations_json(payload), encoding="utf-8")
        self.statusBar().showMessage(f"Saved: {self.annotations_path}", 6000)

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
            payload = build_annotations_payload(self.images_dir, self.page_images, self.page_states)
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
            "- Ctrl+=: Zoom in\n"
            "- Ctrl+-: Zoom out\n"
            "- Ctrl+0: Fit page height to panel\n"
            "- F1: Open this help\n"
            "\n"
            "Page navigation\n"
            "- Use Prev/Next buttons\n"
            "- Use the Go to page spinner in the top bar\n"
            "- Scroll and click the Pages thumbnail strip on the left\n"
            "- Toggle Lens in the top bar for magnified cursor inspection\n"
            "\n"
            "Selection and editing\n"
            "- Shift + left-drag on page: Rectangle-select multiple bboxes\n"
            "- Arrow keys: Move selected bbox(es) by 1 px\n"
            "- Shift+Arrow: Move selected bbox(es) by 10 px\n"
            "- Alt+Arrow: Grow selected bbox(es) in one direction by batch step\n"
            "- Alt+Shift+Arrow: Grow selected bbox(es) by 10x batch step\n"
            "- Ctrl+Arrow: Pan page\n"
            "\n"
            "Mouse interactions\n"
            "- Left-drag on empty page area: Draw a new bbox\n"
            "- Drag selected bbox: Move it\n"
            "- Drag bbox edge/corner: Resize it\n"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate PDF page images with schema-aligned facts and metadata.")
    parser.add_argument(
        "--images-dir",
        default="data/pdf_images/test",
        help="Directory containing page images (png/jpg/webp).",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Output annotations JSON path (default: data/annotations/<images-dir-name>.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    if args.annotations:
        annotations_path = Path(args.annotations)
    else:
        annotations_path = Path("data/annotations") / f"{images_dir.name}.json"

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    try:
        window = AnnotationWindow(images_dir=images_dir, annotations_path=annotations_path)
    except RuntimeError as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
