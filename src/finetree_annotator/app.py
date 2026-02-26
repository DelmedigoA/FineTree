"""Simple PyQt5 app for annotating PDF page images with schema-based facts."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from PyQt5 import sip
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QPainter, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QApplication,
    QComboBox,
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
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
    default_page_meta,
    load_page_states,
    normalize_bbox_data,
    normalize_fact_data,
    propagate_entity_to_next_page,
    serialize_annotations_json,
)
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
    _PAN_STEP = 60

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
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            step = self._PAN_STEP * (3 if event.modifiers() & Qt.ShiftModifier else 1)
            if key == Qt.Key_Left:
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - step)
            elif key == Qt.Key_Right:
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + step)
            elif key == Qt.Key_Up:
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - step)
            elif key == Qt.Key_Down:
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + step)
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

    def set_image_rect(self, rect: QRectF) -> None:
        self.image_rect = rect

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.image_rect.contains(event.scenePos()):
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
                temp_pen = QPen(Qt.yellow)
                temp_pen.setWidth(1)
                temp_pen.setStyle(Qt.DashLine)
                self._temp_rect_item.setPen(temp_pen)
                self._temp_rect_item.setBrush(Qt.transparent)
                self.addItem(self._temp_rect_item)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing and self._temp_rect_item and self._draw_start:
            rect = QRectF(self._draw_start, event.scenePos()).normalized()
            rect = rect.intersected(self.image_rect)
            self._temp_rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
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
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.delete_nav_btn = QPushButton("Delete BBox")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.fit_btn = QPushButton("Fit")
        self.save_btn = QPushButton("Save (Ctrl+S)")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.page_label)
        nav.addWidget(self.undo_btn)
        nav.addWidget(self.redo_btn)
        nav.addWidget(self.delete_nav_btn)
        nav.addWidget(self.zoom_out_btn)
        nav.addWidget(self.zoom_in_btn)
        nav.addWidget(self.fit_btn)
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
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.zoom_requested.connect(self._apply_zoom)
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
        fact_editor_form.addRow("refference*", self.fact_refference_edit)
        fact_editor_form.addRow("date", self.fact_date_edit)
        fact_editor_form.addRow("currency", self.fact_currency_combo)
        fact_editor_form.addRow("scale", self.fact_scale_combo)
        fact_editor_form.addRow("value_type", self.fact_value_type_combo)
        fact_editor_form.addRow("path", path_panel)
        self.fact_bbox_label.setObjectName("factBboxLabel")
        self.fact_value_edit.setMinimumWidth(field_min_width)
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
        right_layout.addWidget(fact_box, 1)

        tip = QLabel(
            "Select a box to edit fields here. "
            "Use +/Remove and Move Up/Move Down to manage path hierarchy."
        )
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        right_layout.addWidget(tip)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([860, 560])

        self.prev_btn.clicked.connect(lambda: self.show_page(self.current_index - 1))
        self.next_btn.clicked.connect(lambda: self.show_page(self.current_index + 1))
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.fit_btn.clicked.connect(self._fit_view)
        self.save_btn.clicked.connect(self.save_annotations)
        self.delete_nav_btn.clicked.connect(self.delete_selected_fact)
        self.dup_fact_btn.clicked.connect(self.duplicate_selected_fact)
        self.del_fact_btn.clicked.connect(self.delete_selected_fact)
        self.facts_list.currentRowChanged.connect(self._on_fact_row_changed)
        self.entity_name_edit.editingFinished.connect(self._on_meta_edited)
        self.page_num_edit.editingFinished.connect(self._on_meta_edited)
        self.title_edit.editingFinished.connect(self._on_meta_edited)
        self.type_combo.activated.connect(lambda _: self._on_meta_edited())
        self.fact_value_edit.editingFinished.connect(self._on_fact_editor_edited)
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

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_annotations)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.duplicate_selected_fact)
        QShortcut(QKeySequence("Meta+D"), self, activated=self.duplicate_selected_fact)
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self.delete_selected_fact)
        delete_shortcut.setContext(Qt.ApplicationShortcut)
        backspace_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self.delete_selected_fact)
        backspace_shortcut.setContext(Qt.ApplicationShortcut)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._fit_view)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_annotations)
        self.addAction(save_action)
        self.page_label.setObjectName("pageIndicator")
        self.statusBar().showMessage("Ready")
        self._set_fact_editor_enabled(False)
        self._clear_fact_editor()
        self._update_path_controls()
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

    def _set_fact_editor_enabled(self, enabled: bool) -> None:
        self.fact_value_edit.setEnabled(enabled)
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
        return normalize_fact_data(
            {
                "value": self.fact_value_edit.text().strip(),
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
        pix.setZValue(-10)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.scene.set_image_rect(QRectF(pixmap.rect()))

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
        self.refresh_facts_list()
        self._fit_view()

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
            summary = f"#{idx} [{int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}] {value}"
            if path:
                summary = f"{summary} | {path}"
            self.facts_list.addItem(QListWidgetItem(summary))
            if item is selected:
                selected_row = idx - 1
        if selected_row >= 0:
            self.facts_list.setCurrentRow(selected_row)
        self._syncing_selection = False
        self._sync_fact_editor_from_selection()

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

    def _on_fact_row_changed(self, row: int) -> None:
        if self._syncing_selection:
            return
        self.scene.clearSelection()
        if 0 <= row < len(self._fact_items):
            item = self._fact_items[row]
            if self._is_alive_fact_item(item):
                item.setSelected(True)
        self._sync_fact_editor_from_selection()

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
        item = self._selected_fact_item()
        if item is None:
            self.statusBar().showMessage("No bbox selected to delete.", 2000)
            return
        self.scene.removeItem(item)
        self.refresh_facts_list()
        self._record_history_snapshot()
        self.statusBar().showMessage("Deleted selected bbox.", 2000)

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
