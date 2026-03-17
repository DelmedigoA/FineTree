from __future__ import annotations

import io
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QColor, QFont, QIcon, QPainter, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from . import app as app_mod
from .finetune import push_dataset_hub
from .schema_contract import (
    PROMPT_FACT_KEYS,
    PROMPT_PAGE_META_KEYS,
    build_custom_extraction_prompt_template,
    build_custom_extraction_schema_preview,
)
from .ui_theme import app_settings, apply_theme, load_theme_choice, save_theme_choice
from .workspace import (
    DEFAULT_DATA_ROOT,
    WorkspaceDocumentSummary,
    WorkspaceImportResult,
    annotations_root,
    build_document_summary,
    discover_workspace_documents,
    delete_workspace_document as delete_workspace_document_files,
    import_pdf_to_workspace,
    reset_document_approved_pages,
    set_document_checked as set_workspace_document_checked,
    set_document_reviewed as set_workspace_document_reviewed,
)


def _set_button_variant(button: QPushButton, variant: str) -> QPushButton:
    button.setProperty("variant", variant)
    button.style().unpolish(button)
    button.style().polish(button)
    return button


NAV_VISIBLE_SETTING_KEY = "ui/nav_visible"
TOKEN_TARGET = 1_000_000
_SUSPICIOUS_RESIZE_BUDGET = 100_000


def _status_tone(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "complete":
        return "complete"
    if normalized in {"in progress", "ready"}:
        return "progress"
    return "attention"


def _format_timestamp(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "No activity yet"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"


def _format_metric_int(value: int) -> str:
    return f"{int(value):,}"


def _make_badge_icon(label: str, bg_hex: str, fg_hex: str = "#ffffff", *, size: int = 22) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(bg_hex))
    inner = max(2, size - 2)
    radius = max(6, size // 3)
    painter.drawRoundedRect(1, 1, inner, inner, radius, radius)
    font = QFont()
    font.setBold(True)
    font.setPointSize(max(8, int(size * (0.3 if len(label) > 1 else 0.42))))
    painter.setFont(font)
    painter.setPen(QColor(fg_hex))
    painter.drawText(pixmap.rect().adjusted(0, -1, 0, 0), Qt.AlignCenter, label)
    painter.end()
    return QIcon(pixmap)


@dataclass(frozen=True)
class DocumentContext:
    doc_id: str
    images_dir: Path
    annotations_path: Path
    pdf_path: Optional[Path] = None
    managed: bool = True

    @property
    def cache_key(self) -> str:
        return f"{self.images_dir.resolve()}::{self.annotations_path.resolve()}"

    @property
    def title(self) -> str:
        if self.pdf_path is not None:
            pdf_stem = str(self.pdf_path.stem or "").strip()
            if pdf_stem:
                return pdf_stem
        return self.doc_id or self.images_dir.name

    @classmethod
    def from_summary(cls, summary: WorkspaceDocumentSummary) -> "DocumentContext":
        return cls(
            doc_id=summary.doc_id,
            images_dir=summary.images_dir,
            annotations_path=summary.annotations_path,
            pdf_path=summary.source_pdf,
            managed=True,
        )


class SignalTextStream(io.TextIOBase):
    def __init__(self, emit_fn) -> None:
        super().__init__()
        self._emit_fn = emit_fn

    def write(self, text: str) -> int:
        if text:
            self._emit_fn(str(text))
        return len(text or "")

    def flush(self) -> None:
        return None


class WorkspaceImportWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, source_pdf: Path, *, data_root: Path = DEFAULT_DATA_ROOT, dpi: int = 200) -> None:
        super().__init__()
        self.source_pdf = Path(source_pdf)
        self.data_root = Path(data_root)
        self.dpi = int(dpi)

    def run(self) -> None:
        try:
            result = import_pdf_to_workspace(self.source_pdf, data_root=self.data_root, dpi=self.dpi)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class PushPipelineWorker(QObject):
    log_emitted = pyqtSignal(str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, argv: list[str]) -> None:
        super().__init__()
        self.argv = list(argv)

    def run(self) -> None:
        stream = SignalTextStream(self.log_emitted.emit)
        captured = io.StringIO()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                exit_code = push_dataset_hub.main(self.argv)
            summary = {"exit_code": int(exit_code), "lines": []}
            self.completed.emit(summary)
        except Exception as exc:
            captured.write(traceback.format_exc())
            self.failed.emit(f"{exc}\n{captured.getvalue()}".strip())
        finally:
            self.finished.emit()


class HomeDocumentCard(QFrame):
    open_requested = pyqtSignal(str)
    prepare_requested = pyqtSignal(str)
    checked_toggled = pyqtSignal(str, bool)
    review_toggled = pyqtSignal(str, bool)
    delete_requested = pyqtSignal(str)

    def __init__(self, summary: WorkspaceDocumentSummary, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setObjectName("docCard")
        self.setProperty("statusTone", _status_tone(summary.status))

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(120, 152)
        self.thumb_label.setScaledContents(False)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setObjectName("surfaceCard")
        root.addWidget(self.thumb_label)

        body = QVBoxLayout()
        body.setSpacing(8)
        root.addLayout(body, 1)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        self.title_label = QLabel(summary.doc_id)
        self.title_label.setObjectName("sectionTitle")
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.status_label = QLabel(summary.status)
        self.status_label.setObjectName("statusPill")
        self.status_label.setProperty("tone", "ok" if summary.status == "Complete" else ("accent" if summary.status in {"Ready", "In progress"} else "warn"))
        title_row.addWidget(self.title_label, 1)
        title_row.addWidget(self.status_label, 0, Qt.AlignRight)
        body.addLayout(title_row)

        self.meta_label = QLabel()
        self.meta_label.setObjectName("subtitleLabel")
        body.addWidget(self.meta_label)
        issues_row = QHBoxLayout()
        issues_row.setContentsMargins(0, 0, 0, 0)
        issues_row.setSpacing(8)
        self.reg_flags_label = QLabel()
        self.reg_flags_label.setObjectName("statusPill")
        self.reg_flags_label.setProperty("tone", "danger")
        self.warnings_label = QLabel()
        self.warnings_label.setObjectName("statusPill")
        self.warnings_label.setProperty("tone", "accent")
        self.no_issues_label = QLabel("No issues")
        self.no_issues_label.setObjectName("statusPill")
        self.no_issues_label.setProperty("tone", "ok")
        issues_row.addWidget(self.reg_flags_label)
        issues_row.addWidget(self.warnings_label)
        issues_row.addWidget(self.no_issues_label)
        issues_row.addStretch(1)
        body.addLayout(issues_row)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.pages_label = QLabel()
        self.pages_label.setObjectName("monoLabel")
        self.facts_label = QLabel()
        self.facts_label.setObjectName("monoLabel")
        self.updated_label = QLabel()
        self.updated_label.setObjectName("monoLabel")
        stats_row.addWidget(self.pages_label)
        stats_row.addWidget(self.facts_label)
        stats_row.addWidget(self.updated_label)
        stats_row.addStretch(1)
        body.addLayout(stats_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        body.addWidget(self.progress)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.action_btn = _set_button_variant(QPushButton("Open"), "primary")
        self.checked_btn = _set_button_variant(QPushButton("Mark Checked"), "ghost")
        self.checked_btn.setCheckable(True)
        self.review_btn = _set_button_variant(QPushButton("Mark Reviewed"), "ghost")
        self.review_btn.setCheckable(True)
        self.delete_btn = _set_button_variant(QPushButton("Remove"), "danger")
        actions.addWidget(self.action_btn)
        actions.addWidget(self.checked_btn)
        actions.addWidget(self.review_btn)
        actions.addWidget(self.delete_btn)
        actions.addStretch(1)
        body.addLayout(actions)

        self.action_btn.clicked.connect(lambda: self.open_requested.emit(self.summary.doc_id))
        self.checked_btn.toggled.connect(self._emit_checked_toggled)
        self.review_btn.toggled.connect(self._emit_review_toggled)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.summary.doc_id))

        self.refresh(summary)

    def _emit_checked_toggled(self, checked: bool) -> None:
        self.checked_toggled.emit(self.summary.doc_id, bool(checked))

    def _emit_review_toggled(self, checked: bool) -> None:
        self.review_toggled.emit(self.summary.doc_id, bool(checked))

    def refresh(self, summary: WorkspaceDocumentSummary) -> None:
        self.summary = summary
        self.setProperty("statusTone", _status_tone(summary.status))
        self.style().unpolish(self)
        self.style().polish(self)
        display_title = summary.source_pdf.stem if summary.source_pdf is not None else summary.doc_id
        self.title_label.setText(display_title)
        pdf_label = summary.source_pdf.name if summary.source_pdf else "No managed PDF"
        self.meta_label.setText(f"{pdf_label}  |  {summary.annotations_path.name}")
        reg_flag_count = int(summary.reg_flag_count or 0)
        warning_count = int(summary.warning_count or 0)
        self.reg_flags_label.setVisible(reg_flag_count > 0)
        self.warnings_label.setVisible(warning_count > 0)
        self.no_issues_label.setVisible(reg_flag_count == 0 and warning_count == 0)
        if reg_flag_count > 0:
            reg_flag_text = "Reg Flag" if reg_flag_count == 1 else "Reg Flags"
            self.reg_flags_label.setText(f"⚑ {reg_flag_count} {reg_flag_text}")
        else:
            self.reg_flags_label.setText("")
        if warning_count > 0:
            warning_text = "Warning" if warning_count == 1 else "Warnings"
            self.warnings_label.setText(f"⚠ {warning_count} {warning_text}")
        else:
            self.warnings_label.setText("")
        self.pages_label.setText(
            f"Approved {summary.approved_page_count}/{summary.page_count or 0} pages"
        )
        self.facts_label.setText(f"Facts {_format_metric_int(int(summary.fact_count or 0))}")
        self.updated_label.setText(f"Updated {_format_timestamp(summary.updated_at)}")
        self.progress.setValue(summary.progress_pct)
        self.progress.setFormat(f"{summary.progress_pct}%")
        self.status_label.setText(summary.status)
        self.status_label.setProperty(
            "tone",
            "ok" if summary.status == "Complete" else ("accent" if summary.status in {"Ready", "In progress"} else "warn"),
        )
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        needs_prepare = summary.status == "Needs extraction" and summary.source_pdf is not None
        can_open = summary.page_count > 0
        if needs_prepare:
            action_text = "Prepare"
            enabled = True
            target_signal = self.prepare_requested
        elif summary.status == "Missing pages":
            action_text = "Unavailable"
            enabled = False
            target_signal = self.open_requested
        elif summary.progress_pct >= 100 and can_open:
            action_text = "Review"
            enabled = True
            target_signal = self.open_requested
        elif summary.annotated_page_count > 0 and can_open:
            action_text = "Resume"
            enabled = True
            target_signal = self.open_requested
        else:
            action_text = "Open"
            enabled = can_open
            target_signal = self.open_requested

        self.action_btn.setText(action_text)
        self.action_btn.setEnabled(enabled)
        try:
            self.action_btn.clicked.disconnect()
        except TypeError:
            pass
        self.action_btn.clicked.connect(lambda: target_signal.emit(self.summary.doc_id))
        checked_allowed = summary.page_count > 0 and summary.annotated_page_count >= summary.page_count
        checked_enabled = bool(summary.checked) or checked_allowed
        self.checked_btn.blockSignals(True)
        self.checked_btn.setChecked(bool(summary.checked))
        self.checked_btn.blockSignals(False)
        self.checked_btn.setEnabled(checked_enabled)
        self.checked_btn.setText("Checked" if summary.checked else "Mark Checked")
        if summary.checked:
            self.checked_btn.setToolTip("Unmark this PDF as checked.")
        elif not checked_allowed:
            self.checked_btn.setToolTip("Finish annotating every page before marking this PDF as checked.")
        else:
            self.checked_btn.setToolTip("Mark this PDF as checked in the first review pass.")
        _set_button_variant(self.checked_btn, "primary" if summary.checked else "ghost")
        review_enabled = summary.page_count > 0 and summary.reg_flag_count == 0
        if not summary.checked:
            review_enabled = False
        self.review_btn.blockSignals(True)
        self.review_btn.setChecked(bool(summary.reviewed))
        self.review_btn.blockSignals(False)
        self.review_btn.setEnabled(review_enabled)
        self.review_btn.setText("Reviewed" if summary.reviewed else "Mark Reviewed")
        if not summary.checked:
            self.review_btn.setToolTip("Mark this PDF as checked before review.")
        elif summary.reg_flag_count > 0:
            self.review_btn.setToolTip(
                f"Fix {summary.reg_flag_count} reg flag(s) across {summary.pages_with_reg_flags} page(s) before review."
            )
        else:
            self.review_btn.setToolTip("Mark this PDF as reviewed and eligible for push.")
        _set_button_variant(self.review_btn, "primary" if summary.reviewed else "ghost")
        if summary.thumbnail_path and summary.thumbnail_path.is_file():
            pixmap = QPixmap(str(summary.thumbnail_path))
            if not pixmap.isNull():
                thumb = pixmap.scaled(QSize(116, 148), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(thumb)
                return
        fallback = QPixmap(116, 148)
        fallback.fill(QColor("#cad6e3"))
        painter = QPainter(fallback)
        painter.setPen(QColor("#405266"))
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(fallback.rect(), Qt.AlignCenter, summary.doc_id[:18])
        painter.end()
        self.thumb_label.setPixmap(fallback)


class HomeView(QWidget):
    import_pdf_requested = pyqtSignal()
    reset_approved_requested = pyqtSignal()
    open_document_requested = pyqtSignal(str)
    prepare_document_requested = pyqtSignal(str)
    checked_document_requested = pyqtSignal(str, bool)
    review_document_requested = pyqtSignal(str, bool)
    delete_document_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("homeView")
        self._documents: list[WorkspaceDocumentSummary] = []
        self._cards: dict[str, HomeDocumentCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        header_card = QFrame()
        header_card.setObjectName("surfaceCard")
        header_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(18, 18, 18, 18)
        header_layout.setSpacing(8)
        eyebrow = QLabel("Workspace")
        eyebrow.setObjectName("eyebrowLabel")
        header_layout.addWidget(eyebrow)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.title_label = QLabel("FineTree document dashboard")
        self.title_label.setObjectName("homeTitle")
        self.subtitle_label = QLabel("Browse managed PDFs, track annotation progress, and jump straight back into a document.")
        self.subtitle_label.setObjectName("homeSubtitle")
        self.subtitle_label.setWordWrap(True)
        header_text = QVBoxLayout()
        header_text.addWidget(self.title_label)
        header_text.addWidget(self.subtitle_label)
        title_row.addLayout(header_text, 1)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.reset_approved_btn = _set_button_variant(QPushButton("Reset all approved"), "ghost")
        self.reset_approved_btn.setEnabled(False)
        self.import_btn = _set_button_variant(QPushButton("Import PDF"), "primary")
        actions.addWidget(self.reset_approved_btn, 0, Qt.AlignTop)
        actions.addWidget(self.import_btn, 0, Qt.AlignTop)
        title_row.addLayout(actions, 0)
        header_layout.addLayout(title_row)
        root.addWidget(header_card)

        self.stats_grid = QGridLayout()
        self.stats_grid.setHorizontalSpacing(8)
        self.stats_grid.setVerticalSpacing(8)
        root.addLayout(self.stats_grid)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search document id or filename")
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Ready", "In progress", "Complete", "Needs extraction", "Missing pages"])
        self.sort_filter = QComboBox()
        self.sort_filter.addItems(["Recent", "Issues Desc", "Approved Asc"])
        filter_row.addWidget(self.search_edit, 1)
        filter_row.addWidget(self.status_filter)
        filter_row.addWidget(self.sort_filter)
        root.addLayout(filter_row)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("homeCardsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        scroll_body = QWidget()
        self.cards_layout = QVBoxLayout(scroll_body)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(scroll_body)
        root.addWidget(self.scroll, 3)

        self.import_btn.clicked.connect(self.import_pdf_requested.emit)
        self.reset_approved_btn.clicked.connect(self.reset_approved_requested.emit)
        self.search_edit.textChanged.connect(self._render_documents)
        self.status_filter.currentTextChanged.connect(self._render_documents)
        self.sort_filter.currentTextChanged.connect(self._render_documents)

    def set_documents(self, documents: list[WorkspaceDocumentSummary]) -> None:
        self._documents = list(documents)
        self._render_documents()

    def _clear_layout_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _stat_card(self, title: str, value: str, caption: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("eyebrowLabel")
        value_label = QLabel(value)
        value_label.setObjectName("subtitleLabel")
        value_font = QFont(value_label.font())
        value_font.setBold(True)
        value_font.setPointSize(max(12, value_font.pointSize()))
        value_label.setFont(value_font)
        caption_label = QLabel(caption)
        caption_label.setObjectName("subtitleLabel")
        caption_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return card

    def _refresh_stats(self, visible_docs: list[WorkspaceDocumentSummary]) -> None:
        while self.stats_grid.count():
            item = self.stats_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        total_docs = len(self._documents)
        total_pages = sum(doc.page_count for doc in self._documents)
        total_approved_pages = sum(int(doc.approved_page_count or 0) for doc in self._documents)
        visible_pages = sum(doc.page_count for doc in visible_docs)
        annotated_pages = sum(doc.annotated_page_count for doc in visible_docs)
        complete_docs = sum(1 for doc in visible_docs if doc.progress_pct >= 100 and doc.page_count > 0)
        total_annotated_images = sum(doc.annotated_page_count for doc in self._documents)
        visible_annotated_images = sum(doc.annotated_page_count for doc in visible_docs)
        total_facts = sum(int(doc.fact_count or 0) for doc in self._documents)
        visible_facts = sum(int(doc.fact_count or 0) for doc in visible_docs)
        total_annotated_tokens = sum(int(doc.annotated_token_count or 0) for doc in self._documents)
        token_progress_pct = int(round((total_annotated_tokens / TOKEN_TARGET) * 100)) if total_annotated_tokens else 0
        avg_progress = int(round((annotated_pages / visible_pages) * 100)) if visible_pages else 0
        approved_pct = int(round((total_approved_pages / total_pages) * 100)) if total_pages else 0
        self.reset_approved_btn.setEnabled(total_approved_pages > 0)
        cards = (
            self._stat_card("Documents", str(total_docs), f"{len(visible_docs)} visible"),
            self._stat_card("Coverage", f"{avg_progress}%", f"{annotated_pages}/{visible_pages or 0} visible pages annotated"),
            self._stat_card("% Approved Pages", f"{approved_pct}%", f"{total_approved_pages}/{total_pages or 0} across workspace"),
            self._stat_card("Completed", str(complete_docs), "Fully annotated document sets"),
            self._stat_card(
                "Annotated Images",
                _format_metric_int(total_annotated_images),
                (
                    f"{visible_annotated_images} visible in current filter"
                    if len(visible_docs) != total_docs
                    else "Across workspace"
                ),
            ),
            self._stat_card(
                "Facts",
                _format_metric_int(total_facts),
                (
                    f"{_format_metric_int(visible_facts)} visible in current filter"
                    if len(visible_docs) != total_docs
                    else "Across workspace"
                ),
            ),
            self._stat_card(
                "Annotated Tokens",
                _format_metric_int(total_annotated_tokens),
                f"{token_progress_pct}% of {_format_metric_int(TOKEN_TARGET)} target",
            ),
        )
        column_count = 4
        for index, card in enumerate(cards):
            row, col = divmod(index, column_count)
            self.stats_grid.addWidget(card, row, col)
        for col in range(column_count):
            self.stats_grid.setColumnStretch(col, 1)

    def _sorted_visible_documents(
        self,
        documents: list[WorkspaceDocumentSummary],
    ) -> list[WorkspaceDocumentSummary]:
        sort_mode = self.sort_filter.currentText()
        if sort_mode == "Approved Asc":
            return sorted(
                documents,
                key=lambda doc: (
                    int(doc.approved_page_count or 0),
                    int(doc.page_count or 0),
                    str(doc.doc_id or "").lower(),
                ),
            )
        if sort_mode != "Issues Desc":
            return list(documents)
        return sorted(
            documents,
            key=lambda doc: (
                bool(doc.checked),
                -int(doc.reg_flag_count or 0),
                -int(doc.warning_count or 0),
                -(float(doc.updated_at or 0.0)),
                str(doc.doc_id or "").lower(),
            ),
        )

    def _render_documents(self) -> None:
        query = self.search_edit.text().strip().lower()
        wanted_status = self.status_filter.currentText()
        visible_docs: list[WorkspaceDocumentSummary] = []
        self._clear_layout_cards()
        self._cards = {}
        for document in self._documents:
            haystack = f"{document.doc_id} {document.source_pdf.name if document.source_pdf else ''}".lower()
            if query and query not in haystack:
                continue
            if wanted_status != "All" and document.status != wanted_status:
                continue
            visible_docs.append(document)
        visible_docs = self._sorted_visible_documents(visible_docs)
        for document in visible_docs:
            card = HomeDocumentCard(document)
            card.open_requested.connect(self.open_document_requested.emit)
            card.prepare_requested.connect(self.prepare_document_requested.emit)
            card.checked_toggled.connect(self.checked_document_requested.emit)
            card.review_toggled.connect(self.review_document_requested.emit)
            card.delete_requested.connect(self.delete_document_requested.emit)
            self.cards_layout.addWidget(card)
            self._cards[document.doc_id] = card
        if not visible_docs:
            empty = QFrame()
            empty.setObjectName("emptyStateCard")
            layout = QVBoxLayout(empty)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(8)
            title = QLabel("No workspace documents match this view")
            title.setObjectName("sectionTitle")
            caption = QLabel("Import a PDF to create a managed document, or clear the filters.")
            caption.setObjectName("subtitleLabel")
            caption.setWordWrap(True)
            button = _set_button_variant(QPushButton("Import PDF"), "primary")
            button.clicked.connect(self.import_pdf_requested.emit)
            layout.addWidget(title)
            layout.addWidget(caption)
            layout.addWidget(button, 0, Qt.AlignLeft)
            self.cards_layout.addWidget(empty)
        self.cards_layout.addStretch(1)
        self._refresh_stats(visible_docs)


class AnnotatorHost(QWidget):
    document_saved = pyqtSignal(object)
    current_document_changed = pyqtSignal(object)
    document_issues_changed = pyqtSignal(object, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("annotatorHost")
        self._contexts: dict[str, DocumentContext] = {}
        self._windows: dict[str, app_mod.AnnotationWindow] = {}
        self._current_key: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        placeholder = QFrame()
        placeholder.setObjectName("emptyStateCard")
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)
        title = QLabel("No document is open")
        title.setObjectName("annotatorTitle")
        subtitle = QLabel("Open a workspace card from Home or launch the app with a PDF/images directory.")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch(1)
        self.placeholder = placeholder
        self.stack.addWidget(self.placeholder)

    def current_context(self) -> Optional[DocumentContext]:
        if self._current_key is None:
            return None
        return self._contexts.get(self._current_key)

    def current_window(self) -> Optional[app_mod.AnnotationWindow]:
        if self._current_key is None:
            return None
        return self._windows.get(self._current_key)

    def open_document(self, context: DocumentContext) -> app_mod.AnnotationWindow:
        key = context.cache_key
        if key not in self._windows:
            window = app_mod.AnnotationWindow(images_dir=context.images_dir, annotations_path=context.annotations_path)
            window.setWindowFlags(Qt.Widget)
            if hasattr(window, "annotations_save_status"):
                window.annotations_save_status.connect(self.document_saved.emit)
            else:
                window.annotations_saved.connect(self.document_saved.emit)
            if hasattr(window, "document_issues_changed"):
                window.document_issues_changed.connect(
                    lambda summary, ctx=context: self.document_issues_changed.emit(ctx, summary)
                )
            self._windows[key] = window
            self._contexts[key] = context
            self.stack.addWidget(window)
        self._current_key = key
        self.stack.setCurrentWidget(self._windows[key])
        self.current_document_changed.emit(context)
        return self._windows[key]

    def close_document(self, doc_id: str) -> bool:
        normalized = str(doc_id or "").strip()
        if not normalized:
            return False

        removed = False
        current_removed = False
        for key, context in list(self._contexts.items()):
            if context.doc_id != normalized:
                continue
            window = self._windows.pop(key, None)
            self._contexts.pop(key, None)
            if window is not None:
                self.stack.removeWidget(window)
                window.setParent(None)
                window.deleteLater()
            if self._current_key == key:
                current_removed = True
            removed = True

        if current_removed:
            if self._windows:
                next_key = next(iter(self._windows))
                self._current_key = next_key
                self.stack.setCurrentWidget(self._windows[next_key])
            else:
                self.show_placeholder()
        return removed

    def show_placeholder(self) -> None:
        self._current_key = None
        self.stack.setCurrentWidget(self.placeholder)
        self.current_document_changed.emit(None)

    def live_document_issue_summaries(self) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for key, window in self._windows.items():
            context = self._contexts.get(key)
            if context is None or not context.managed:
                continue
            issue_method = getattr(window, "document_issue_summary", None)
            if callable(issue_method):
                try:
                    summaries[context.doc_id] = issue_method()
                except Exception:
                    continue
        return summaries

    def managed_windows(self) -> list[tuple[DocumentContext, app_mod.AnnotationWindow]]:
        managed: list[tuple[DocumentContext, app_mod.AnnotationWindow]] = []
        for key, window in self._windows.items():
            context = self._contexts.get(key)
            if context is None or not context.managed:
                continue
            managed.append((context, window))
        return managed

    def confirm_close_document(self, doc_id: str) -> bool:
        normalized = str(doc_id or "").strip()
        if not normalized:
            return True

        confirmed_windows: list[app_mod.AnnotationWindow] = []
        for context, window in self.managed_windows():
            if context.doc_id != normalized:
                continue
            confirm_method = getattr(window, "confirm_close", None)
            if not callable(confirm_method):
                continue
            if confirm_method(prepare_for_close=True):
                confirmed_windows.append(window)
                continue
            for confirmed_window in confirmed_windows:
                cancel_method = getattr(confirmed_window, "cancel_pending_close", None)
                if callable(cancel_method):
                    cancel_method()
            return False
        return True

    def confirm_close_all_documents(self) -> bool:
        confirmed_windows: list[app_mod.AnnotationWindow] = []
        for window in self._windows.values():
            confirm_method = getattr(window, "confirm_close", None)
            if not callable(confirm_method):
                continue
            if confirm_method(prepare_for_close=True):
                confirmed_windows.append(window)
                continue
            for confirmed_window in confirmed_windows:
                cancel_method = getattr(confirmed_window, "cancel_pending_close", None)
                if callable(cancel_method):
                    cancel_method()
            return False
        return True


class PushView(QWidget):
    _DEFAULT_FACT_KEYS: tuple[str, ...] = tuple(key for key in PROMPT_FACT_KEYS if key != "date")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("pushView")
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[PushPipelineWorker] = None
        self._documents: list[WorkspaceDocumentSummary] = []
        self._selected_train_doc_ids: set[str] = set()
        self._split_selection_initialized = False
        self._selected_validation_doc_ids: set[str] = set()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        header = QFrame()
        header.setObjectName("surfaceCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 24, 24, 24)
        eyebrow = QLabel("Dataset Push")
        eyebrow.setObjectName("eyebrowLabel")
        self.title_label = QLabel("Push workspace dataset to Hugging Face")
        self.title_label.setObjectName("pushTitle")
        self.subtitle_label = QLabel("This screen wraps the existing dataset-push pipeline and exposes the current CLI options in a dashboard form.")
        self.subtitle_label.setObjectName("pushSubtitle")
        self.subtitle_label.setWordWrap(True)
        header_layout.addWidget(eyebrow)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        root.addWidget(header)

        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setObjectName("pushContentSplitter")
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(10)
        root.addWidget(self.content_splitter, 1)

        form_scroll = QScrollArea()
        self.form_scroll = form_scroll
        form_scroll.setObjectName("pushFormScroll")
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.NoFrame)
        form_scroll.setMinimumWidth(560)
        form_scroll.setMaximumWidth(1100)
        form_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        form_shell = QWidget()
        form_shell_layout = QHBoxLayout(form_shell)
        form_shell_layout.setContentsMargins(0, 0, 0, 0)
        form_shell_layout.setSpacing(0)
        form_widget = QWidget()
        self.form_content = form_widget
        form_widget.setMaximumWidth(720)
        form_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        self.config_edit = QLineEdit("configs/finetune_qwen35a3_vl.yaml")
        self.repo_id_edit = QLineEdit()
        self.repo_id_minimal_edit = QLineEdit()
        self.repo_id_no_bbox_edit = QLineEdit()
        self.repo_id_no_bbox_min_edit = QLineEdit()
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("Optional; falls back to env or Doppler")
        self.export_dir_edit = QLineEdit("artifacts/hf_dataset_export")
        self.public_check = QCheckBox("Create public dataset repos")
        self.instruction_mode_combo = QComboBox()
        self.instruction_mode_combo.addItems(["source", "minimal"])
        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(0, 10_000_000)
        self.max_pixels_spin = QSpinBox()
        self.max_pixels_spin.setRange(0, 10_000_000)
        self.exclude_doc_ids_edit = QLineEdit()
        self.compact_tokens_check = QCheckBox("Compact assistant JSON")
        self.aggressive_compact_check = QCheckBox("Aggressive compact tokens")
        self.push_all_variants_check = QCheckBox("Push all dataset variants")
        self.push_split_repos_check = QCheckBox("Push train/validation as separate repos")
        self.repo_id_train_edit = QLineEdit()
        self.repo_id_validation_edit = QLineEdit()
        self.allow_duplicate_check = QCheckBox("Allow duplicate facts")
        self.allow_ordering_check = QCheckBox("Allow ordering issues")
        self.allow_format_check = QCheckBox("Allow format issues")
        self.approved_only_check = QCheckBox("Push approved pages only")
        self.approved_only_check.setChecked(True)
        self.approved_only_check.setEnabled(False)
        self.token_status_label = QLabel()
        self.token_status_label.setObjectName("statusPill")
        self.instruction_mode_combo.setMaximumWidth(260)
        self.min_pixels_spin.setMaximumWidth(180)
        self.max_pixels_spin.setMaximumWidth(180)
        self.min_pixels_spin.setSpecialValueText("Unset")
        self.max_pixels_spin.setSpecialValueText("Unset")
        pixel_budget_tip = (
            "Total image area budget, not width/height. "
            "Example: 1200000 is about a 1.2MP target. "
            "Very small values like 1200 will shrink pages to tiny thumbnails."
        )
        self.min_pixels_spin.setToolTip(pixel_budget_tip)
        self.max_pixels_spin.setToolTip(pixel_budget_tip)
        self.token_status_label.setMaximumWidth(220)

        config_row = QWidget()
        config_row_layout = QHBoxLayout(config_row)
        config_row_layout.setContentsMargins(0, 0, 0, 0)
        config_row_layout.setSpacing(8)
        config_row_layout.addWidget(self.config_edit, 1)
        self.config_browse_btn = QPushButton("Browse")
        config_row_layout.addWidget(self.config_browse_btn)

        export_row = QWidget()
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        export_row_layout.setSpacing(8)
        export_row_layout.addWidget(self.export_dir_edit, 1)
        self.export_browse_btn = QPushButton("Browse")
        export_row_layout.addWidget(self.export_browse_btn)

        self.review_scope_card, review_scope_layout = self._push_section_card(
            "Eligible PDFs",
            "Any PDF with approved pages and no regulation flags is pushable. Choose which eligible PDFs belong in training and which belong in validation.",
        )
        self.reviewed_docs_summary = QLabel(
            "No eligible PDFs yet. Approve pages and clear regulation flags first."
        )
        self.reviewed_docs_summary.setObjectName("subtitleLabel")
        self.reviewed_docs_summary.setWordWrap(True)
        self.train_docs_list = QListWidget()
        self.train_docs_list.setObjectName("pushTrainDocList")
        self.train_docs_list.setAlternatingRowColors(False)
        self.train_docs_list.setMinimumHeight(180)
        self.train_docs_list.itemChanged.connect(self._remember_train_selection)
        self.validation_docs_list = QListWidget()
        self.validation_docs_list.setObjectName("pushValidationDocList")
        self.validation_docs_list.setAlternatingRowColors(False)
        self.validation_docs_list.setMinimumHeight(168)
        self.validation_docs_list.itemChanged.connect(self._remember_validation_selection)
        self.select_all_train_btn = _set_button_variant(QPushButton("Train all"), "ghost")
        self.clear_all_train_btn = _set_button_variant(QPushButton("Clear train"), "ghost")
        self.select_all_validation_btn = _set_button_variant(QPushButton("Validate all"), "ghost")
        self.clear_all_validation_btn = _set_button_variant(QPushButton("Clear validation"), "ghost")
        split_actions = QHBoxLayout()
        split_actions.setContentsMargins(0, 0, 0, 0)
        split_actions.setSpacing(8)
        split_actions.addWidget(self.select_all_train_btn)
        split_actions.addWidget(self.clear_all_train_btn)
        split_actions.addWidget(self.select_all_validation_btn)
        split_actions.addWidget(self.clear_all_validation_btn)
        split_actions.addStretch(1)
        review_scope_layout.addWidget(self.reviewed_docs_summary)
        split_lists = QWidget()
        split_lists_layout = QHBoxLayout(split_lists)
        split_lists_layout.setContentsMargins(0, 0, 0, 0)
        split_lists_layout.setSpacing(12)
        split_lists_layout.addWidget(self._labeled_panel("Training PDFs", self.train_docs_list), 1)
        split_lists_layout.addWidget(self._labeled_panel("Validation PDFs", self.validation_docs_list), 1)
        review_scope_layout.addWidget(split_lists)
        review_scope_layout.addLayout(split_actions)
        form_layout.addWidget(self.review_scope_card)

        self.basics_card, basics_layout = self._push_section_card(
            "Basics",
            "Core dataset target and export settings.",
        )
        basics_form = self._push_form_layout()
        basics_form.addRow("Config", config_row)
        basics_form.addRow("Primary repo", self.repo_id_edit)
        basics_form.addRow("Export folder", export_row)
        basics_form.addRow("Instruction mode", self.instruction_mode_combo)
        basics_layout.addLayout(basics_form)
        form_layout.addWidget(self.basics_card)

        self.variants_card, variants_layout = self._push_section_card(
            "Variants",
            "Optional repo targets for alternate dataset variants.",
        )
        variants_form = self._push_form_layout()
        variants_form.addRow("Minimal repo", self.repo_id_minimal_edit)
        variants_form.addRow("No bbox repo", self.repo_id_no_bbox_edit)
        variants_form.addRow("No bbox minimal", self.repo_id_no_bbox_min_edit)
        variants_form.addRow("Train repo", self.repo_id_train_edit)
        variants_form.addRow("Validation repo", self.repo_id_validation_edit)
        variants_layout.addLayout(variants_form)
        variants_checks = self._checkbox_grid(
            self.push_all_variants_check,
            self.push_split_repos_check,
        )
        variants_layout.addWidget(variants_checks)
        form_layout.addWidget(self.variants_card)

        self.options_card, options_layout = self._push_section_card(
            "Access And Validation",
            "Token handling, filters, and validation overrides.",
        )
        options_form = self._push_form_layout()
        options_form.addRow("Token", self.token_edit)
        options_form.addRow("Token status", self.token_status_label)
        options_form.addRow("Pixel budget", self._compact_row(self.min_pixels_spin, self.max_pixels_spin))
        options_form.addRow("Exclude docs", self.exclude_doc_ids_edit)
        options_layout.addLayout(options_form)
        options_checks = self._checkbox_grid(
            self.public_check,
            self.approved_only_check,
            self.compact_tokens_check,
            self.aggressive_compact_check,
            self.allow_duplicate_check,
            self.allow_ordering_check,
            self.allow_format_check,
        )
        options_layout.addWidget(options_checks)
        form_layout.addWidget(self.options_card)

        self.schema_card, schema_layout = self._push_section_card(
            "Schema",
            "Choose exported page-meta and fact fields. `bbox` is always included; `value` stays required.",
        )
        schema_intro = QLabel("The live preview on the right updates immediately and uses the same selection that will be pushed.")
        schema_intro.setObjectName("subtitleLabel")
        schema_intro.setWordWrap(True)
        schema_layout.addWidget(schema_intro)
        self.schema_selection_summary = QLabel()
        self.schema_selection_summary.setObjectName("monoLabel")
        self.schema_selection_summary.setWordWrap(True)
        schema_layout.addWidget(self.schema_selection_summary)
        schema_lists = QWidget()
        schema_lists_layout = QHBoxLayout(schema_lists)
        schema_lists_layout.setContentsMargins(0, 0, 0, 0)
        schema_lists_layout.setSpacing(12)
        self.page_meta_keys_list = QListWidget()
        self.page_meta_keys_list.setObjectName("pushPageMetaKeyList")
        self.page_meta_keys_list.setMinimumHeight(150)
        self.fact_keys_list = QListWidget()
        self.fact_keys_list.setObjectName("pushFactKeyList")
        self.fact_keys_list.setMinimumHeight(240)
        schema_lists_layout.addWidget(self._labeled_panel("Page Meta Fields", self.page_meta_keys_list), 1)
        schema_lists_layout.addWidget(self._labeled_panel("Fact Fields", self.fact_keys_list), 1)
        schema_layout.addWidget(schema_lists)
        form_layout.addWidget(self.schema_card)

        controls_card = QFrame()
        controls_card.setObjectName("surfaceCard")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(12)
        controls_title = QLabel("Run")
        controls_title.setObjectName("sectionTitle")
        controls_caption = QLabel("Start the export and inspect results on the right.")
        controls_caption.setObjectName("subtitleLabel")
        controls_caption.setWordWrap(True)
        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(controls_caption)
        self.run_btn = _set_button_variant(QPushButton("Run push pipeline"), "primary")
        self.clear_log_btn = _set_button_variant(QPushButton("Clear log"), "ghost")
        self.open_export_btn = _set_button_variant(QPushButton("Open export folder"), "ghost")
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.clear_log_btn)
        action_row.addWidget(self.open_export_btn)
        action_row.addStretch(1)
        controls_layout.addLayout(action_row)
        form_layout.addWidget(controls_card)
        form_layout.addStretch(1)
        form_shell_layout.addWidget(form_widget, 0, Qt.AlignTop)
        form_shell_layout.addStretch(1)
        form_scroll.setWidget(form_shell)
        self.content_splitter.addWidget(form_scroll)

        log_card = QFrame()
        self.results_card = log_card
        self.log_card = log_card
        log_card.setObjectName("surfaceCard")
        log_card.setMinimumWidth(320)
        log_card.setMaximumWidth(560)
        log_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(20, 20, 20, 20)
        log_layout.setSpacing(10)
        results_title_row = QHBoxLayout()
        results_title_row.setContentsMargins(0, 0, 0, 0)
        results_title_row.setSpacing(8)
        results_title = QLabel("Results")
        results_title.setObjectName("sectionTitle")
        self.stage_label = QLabel("Idle")
        self.stage_label.setObjectName("statusPill")
        self.stage_label.setProperty("tone", "accent")
        results_title_row.addWidget(results_title)
        results_title_row.addStretch(1)
        results_title_row.addWidget(self.stage_label, 0, Qt.AlignRight)
        log_layout.addLayout(results_title_row)
        self.preview_splitter = QSplitter(Qt.Vertical)
        self.preview_splitter.setChildrenCollapsible(False)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.preview_summary_label = QLabel("Preview updates when you change eligible PDFs or schema fields.")
        self.preview_summary_label.setObjectName("subtitleLabel")
        self.preview_summary_label.setWordWrap(True)
        self.schema_preview_label = QLabel("Schema Preview")
        self.schema_preview_label.setObjectName("sectionTitle")
        self.schema_preview_view = QPlainTextEdit()
        self.schema_preview_view.setReadOnly(True)
        self.prompt_preview_label = QLabel("Instruction Preview")
        self.prompt_preview_label.setObjectName("sectionTitle")
        self.prompt_preview_view = QPlainTextEdit()
        self.prompt_preview_view.setReadOnly(True)
        preview_layout.addWidget(self.preview_summary_label)
        preview_layout.addWidget(self.schema_preview_label)
        preview_layout.addWidget(self.schema_preview_view, 1)
        preview_layout.addWidget(self.prompt_preview_label)
        preview_layout.addWidget(self.prompt_preview_view, 1)
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        self.results_label = QLabel("No push results yet.")
        self.results_label.setObjectName("subtitleLabel")
        self.results_label.setWordWrap(True)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        results_layout.addWidget(self.results_label)
        results_layout.addWidget(self.log_view, 1)
        self.preview_splitter.addWidget(preview_widget)
        self.preview_splitter.addWidget(results_widget)
        self.preview_splitter.setStretchFactor(0, 3)
        self.preview_splitter.setStretchFactor(1, 2)
        log_layout.addWidget(self.preview_splitter, 1)
        self.content_splitter.addWidget(log_card)
        self.content_splitter.setStretchFactor(0, 5)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([920, 380])

        self.config_browse_btn.clicked.connect(self._choose_config)
        self.export_browse_btn.clicked.connect(self._choose_export_dir)
        self.run_btn.clicked.connect(self.run_pipeline)
        self.clear_log_btn.clicked.connect(self.log_view.clear)
        self.open_export_btn.clicked.connect(self._show_export_dir)
        self.select_all_train_btn.clicked.connect(self._select_all_train_docs)
        self.clear_all_train_btn.clicked.connect(self._clear_all_train_docs)
        self.select_all_validation_btn.clicked.connect(self._select_all_validation_docs)
        self.clear_all_validation_btn.clicked.connect(self._clear_all_validation_docs)
        self.token_edit.textChanged.connect(self._refresh_token_status)
        self.page_meta_keys_list.itemChanged.connect(self._update_preview)
        self.fact_keys_list.itemChanged.connect(self._update_preview)
        self._populate_schema_lists()
        self._refresh_token_status()
        self.set_documents([])

    def _push_section_card(self, title: str, caption: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("surfaceCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        caption_label = QLabel(caption)
        caption_label.setObjectName("subtitleLabel")
        caption_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(caption_label)
        return card, layout

    def _push_form_layout(self) -> QFormLayout:
        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        return layout

    def _compact_row(self, *widgets: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return row

    def _checkbox_grid(self, *checks: QCheckBox) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        for index, check in enumerate(checks):
            row = index // 2
            col = index % 2
            layout.addWidget(check, row, col)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        return widget

    def _labeled_panel(self, title: str, widget: QWidget) -> QWidget:
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("subtitleLabel")
        layout.addWidget(title_label)
        layout.addWidget(widget, 1)
        return shell

    def _set_check_list_items(
        self,
        list_widget: QListWidget,
        *,
        keys: tuple[str, ...],
        selected: set[str],
        locked: set[str] | None = None,
    ) -> None:
        locked_keys = set(locked or set())
        list_widget.blockSignals(True)
        list_widget.clear()
        for key in keys:
            item = QListWidgetItem(key)
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if key in selected else Qt.Unchecked)
            if key in locked_keys:
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setText(f"{key} (required)")
            list_widget.addItem(item)
        list_widget.blockSignals(False)

    def _populate_schema_lists(self) -> None:
        self._set_check_list_items(
            self.page_meta_keys_list,
            keys=tuple(PROMPT_PAGE_META_KEYS),
            selected=set(PROMPT_PAGE_META_KEYS),
        )
        self._set_check_list_items(
            self.fact_keys_list,
            keys=tuple(PROMPT_FACT_KEYS),
            selected=set(self._DEFAULT_FACT_KEYS),
            locked={"value"},
        )
        self._update_preview()

    def selected_page_meta_keys(self) -> list[str]:
        selected: list[str] = []
        for index in range(self.page_meta_keys_list.count()):
            item = self.page_meta_keys_list.item(index)
            if item is None or item.checkState() != Qt.Checked:
                continue
            key = str(item.data(Qt.UserRole) or "").strip()
            if key:
                selected.append(key)
        return selected

    def selected_fact_keys(self) -> list[str]:
        selected: list[str] = []
        for index in range(self.fact_keys_list.count()):
            item = self.fact_keys_list.item(index)
            if item is None:
                continue
            key = str(item.data(Qt.UserRole) or "").strip()
            if not key:
                continue
            if item.checkState() == Qt.Checked or key == "value":
                selected.append(key)
        return selected

    def _choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose config", str(Path.cwd()), "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.config_edit.setText(path)

    def _choose_export_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose export directory", str(Path.cwd()))
        if path:
            self.export_dir_edit.setText(path)

    def _show_export_dir(self) -> None:
        export_dir = Path(self.export_dir_edit.text().strip() or "artifacts/hf_dataset_export").expanduser()
        QMessageBox.information(self, "Export directory", str(export_dir.resolve()))

    def _refresh_token_status(self) -> None:
        token = (
            self.token_edit.text().strip()
            or (os.getenv("FINETREE_HF_TOKEN") or "").strip()
            or (os.getenv("HF_TOKEN") or "").strip()
            or (os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
            or (os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
        )
        if token:
            self.token_status_label.setText("Token available")
            self.token_status_label.setProperty("tone", "ok")
        else:
            self.token_status_label.setText("Token missing")
            self.token_status_label.setProperty("tone", "warn")
        self.token_status_label.style().unpolish(self.token_status_label)
        self.token_status_label.style().polish(self.token_status_label)

    def set_documents(self, documents: list[WorkspaceDocumentSummary]) -> None:
        self._documents = list(documents)
        eligible_docs = self.eligible_push_documents(documents)
        eligible_doc_ids = {doc.doc_id for doc in eligible_docs}
        if self._split_selection_initialized:
            self._selected_train_doc_ids &= eligible_doc_ids
            self._selected_validation_doc_ids &= eligible_doc_ids
        elif eligible_doc_ids:
            self._selected_train_doc_ids = set(eligible_doc_ids)
            self._selected_validation_doc_ids.clear()
            self._split_selection_initialized = True
        overlap_doc_ids = self._selected_train_doc_ids & self._selected_validation_doc_ids
        if overlap_doc_ids:
            self._selected_validation_doc_ids -= overlap_doc_ids

        self.train_docs_list.blockSignals(True)
        self.train_docs_list.clear()
        for doc in eligible_docs:
            label = (
                f"{doc.doc_id}  |  approved {doc.approved_page_count}/{doc.page_count} pages  |  "
                f"{doc.progress_pct}%  |  {doc.status}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, doc.doc_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if doc.doc_id in self._selected_train_doc_ids else Qt.Unchecked)
            self.train_docs_list.addItem(item)
        self.train_docs_list.blockSignals(False)

        self.validation_docs_list.blockSignals(True)
        self.validation_docs_list.clear()
        for doc in eligible_docs:
            label = (
                f"{doc.doc_id}  |  approved {doc.approved_page_count}/{doc.page_count} pages  |  "
                f"{doc.progress_pct}%  |  {doc.status}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, doc.doc_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if doc.doc_id in self._selected_validation_doc_ids else Qt.Unchecked)
            self.validation_docs_list.addItem(item)
        self.validation_docs_list.blockSignals(False)

        if len(eligible_docs) == 0:
            self.reviewed_docs_summary.setText(
                "No eligible PDFs yet. A PDF must have at least one approved page and zero regulation flags."
            )
            self.train_docs_list.setEnabled(False)
            self.validation_docs_list.setEnabled(False)
            self.select_all_train_btn.setEnabled(False)
            self.clear_all_train_btn.setEnabled(False)
            self.select_all_validation_btn.setEnabled(False)
            self.clear_all_validation_btn.setEnabled(False)
        else:
            self.train_docs_list.setEnabled(True)
            self.validation_docs_list.setEnabled(True)
            self.select_all_train_btn.setEnabled(True)
            self.clear_all_train_btn.setEnabled(True)
            self.select_all_validation_btn.setEnabled(True)
            self.clear_all_validation_btn.setEnabled(True)
        self._refresh_selection_summary()
        self._update_preview()

    def eligible_push_documents(
        self,
        documents: list[WorkspaceDocumentSummary] | None = None,
    ) -> list[WorkspaceDocumentSummary]:
        source = self._documents if documents is None else documents
        return [
            doc
            for doc in source
            if doc.page_count > 0
            and int(doc.approved_page_count or 0) > 0
            and int(doc.reg_flag_count or 0) == 0
        ]

    def _selected_doc_ids_from_list(self, list_widget: QListWidget) -> list[str]:
        selected: list[str] = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                doc_id = str(item.data(Qt.UserRole) or "").strip()
                if doc_id:
                    selected.append(doc_id)
        return selected

    def selected_train_doc_ids(self) -> list[str]:
        return self._selected_doc_ids_from_list(self.train_docs_list)

    def selected_train_documents(self) -> list[WorkspaceDocumentSummary]:
        selected_ids = set(self._selected_train_doc_ids)
        return [doc for doc in self.eligible_push_documents() if doc.doc_id in selected_ids]

    def selected_validation_doc_ids(self) -> list[str]:
        return self._selected_doc_ids_from_list(self.validation_docs_list)

    def selected_validation_documents(self) -> list[WorkspaceDocumentSummary]:
        selected_ids = set(self._selected_validation_doc_ids)
        return [doc for doc in self.eligible_push_documents() if doc.doc_id in selected_ids]

    def selected_included_doc_ids(self) -> list[str]:
        selected_doc_ids = self._selected_train_doc_ids | self._selected_validation_doc_ids
        return [doc.doc_id for doc in self.eligible_push_documents() if doc.doc_id in selected_doc_ids]

    def selected_included_documents(self) -> list[WorkspaceDocumentSummary]:
        selected_ids = set(self.selected_included_doc_ids())
        return [doc for doc in self.eligible_push_documents() if doc.doc_id in selected_ids]

    def _set_list_selection(self, list_widget: QListWidget, doc_ids: set[str]) -> None:
        list_widget.blockSignals(True)
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item is None:
                continue
            item_doc_id = str(item.data(Qt.UserRole) or "").strip()
            item.setCheckState(Qt.Checked if item_doc_id in doc_ids else Qt.Unchecked)
        list_widget.blockSignals(False)

    def _refresh_selection_summary(self) -> None:
        eligible_docs = self.eligible_push_documents()
        selected_docs = self.selected_included_documents()
        train_count = len(self._selected_train_doc_ids)
        validation_count = len(self._selected_validation_doc_ids)
        total_pages = sum(int(doc.page_count or 0) for doc in selected_docs)
        approved_pages = sum(int(doc.approved_page_count or 0) for doc in selected_docs)
        skipped_pages = max(total_pages - approved_pages, 0)
        self.reviewed_docs_summary.setText(
            f"{len(selected_docs)}/{len(eligible_docs)} eligible PDF(s) selected. "
            f"Train: {train_count}. Validation: {validation_count}. "
            f"Selected pages: {approved_pages} approved of {total_pages} total. "
            f"Non-approved pages skipped: {skipped_pages}."
        )

    def _remember_validation_selection(self, _item: QListWidgetItem) -> None:
        self._selected_validation_doc_ids = set(self.selected_validation_doc_ids())
        self._split_selection_initialized = True
        overlap_doc_ids = self._selected_validation_doc_ids & self._selected_train_doc_ids
        if overlap_doc_ids:
            self._selected_train_doc_ids -= overlap_doc_ids
            self._set_list_selection(self.train_docs_list, self._selected_train_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _remember_train_selection(self, _item: QListWidgetItem) -> None:
        self._selected_train_doc_ids = set(self.selected_train_doc_ids())
        self._split_selection_initialized = True
        overlap_doc_ids = self._selected_train_doc_ids & self._selected_validation_doc_ids
        if overlap_doc_ids:
            self._selected_validation_doc_ids -= overlap_doc_ids
            self._set_list_selection(self.validation_docs_list, self._selected_validation_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _select_all_train_docs(self) -> None:
        eligible_doc_ids = {doc.doc_id for doc in self.eligible_push_documents()}
        self._selected_train_doc_ids = set(eligible_doc_ids)
        self._selected_validation_doc_ids -= eligible_doc_ids
        self._split_selection_initialized = True
        self._set_list_selection(self.train_docs_list, self._selected_train_doc_ids)
        self._set_list_selection(self.validation_docs_list, self._selected_validation_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _clear_all_train_docs(self) -> None:
        self._selected_train_doc_ids.clear()
        self._split_selection_initialized = True
        self._set_list_selection(self.train_docs_list, self._selected_train_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _select_all_validation_docs(self) -> None:
        eligible_doc_ids = {doc.doc_id for doc in self.eligible_push_documents()}
        self._selected_validation_doc_ids = set(eligible_doc_ids)
        self._selected_train_doc_ids -= eligible_doc_ids
        self._split_selection_initialized = True
        self._set_list_selection(self.validation_docs_list, self._selected_validation_doc_ids)
        self._set_list_selection(self.train_docs_list, self._selected_train_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _clear_all_validation_docs(self) -> None:
        self._selected_validation_doc_ids.clear()
        self._split_selection_initialized = True
        self._set_list_selection(self.validation_docs_list, self._selected_validation_doc_ids)
        self._refresh_selection_summary()
        self._update_preview()

    def _build_argv(self) -> list[str]:
        argv = [
            "--config",
            self.config_edit.text().strip() or "configs/finetune_qwen35a3_vl.yaml",
            "--export-dir",
            self.export_dir_edit.text().strip() or "artifacts/hf_dataset_export",
            "--instruction-mode",
            self.instruction_mode_combo.currentText().strip() or "source",
            "--approved-pages-only",
        ]
        included_doc_ids = self.selected_included_doc_ids()
        validation_doc_ids = self.selected_validation_doc_ids()
        page_meta_keys = self.selected_page_meta_keys()
        fact_keys = self.selected_fact_keys()
        if included_doc_ids:
            argv.extend(["--include-doc-ids", ",".join(included_doc_ids)])
        if validation_doc_ids:
            argv.extend(["--validation-doc-ids", ",".join(validation_doc_ids)])
        argv.extend(["--page-meta-keys", ",".join(page_meta_keys)])
        argv.extend(["--fact-keys", ",".join(fact_keys)])
        if self.repo_id_edit.text().strip():
            argv.extend(["--repo-id", self.repo_id_edit.text().strip()])
        if self.repo_id_minimal_edit.text().strip():
            argv.extend(["--repo-id-minimal-instruction", self.repo_id_minimal_edit.text().strip()])
        if self.repo_id_no_bbox_edit.text().strip():
            argv.extend(["--repo-id-no-bbox", self.repo_id_no_bbox_edit.text().strip()])
        if self.repo_id_no_bbox_min_edit.text().strip():
            argv.extend(["--repo-id-no-bbox-minimal-instruction", self.repo_id_no_bbox_min_edit.text().strip()])
        if self.token_edit.text().strip():
            argv.extend(["--token", self.token_edit.text().strip()])
        if self.public_check.isChecked():
            argv.append("--public")
        if self.min_pixels_spin.value() > 0:
            argv.extend(["--min-pixels", str(self.min_pixels_spin.value())])
        if self.max_pixels_spin.value() > 0:
            argv.extend(["--max-pixels", str(self.max_pixels_spin.value())])
        if self.exclude_doc_ids_edit.text().strip():
            argv.extend(["--exclude-doc-ids", self.exclude_doc_ids_edit.text().strip()])
        if self.compact_tokens_check.isChecked():
            argv.append("--compact_tokens")
        if self.aggressive_compact_check.isChecked():
            argv.append("--aggressive-compact-tokens")
        if self.push_all_variants_check.isChecked():
            argv.append("--push-all-variants")
        if self.push_split_repos_check.isChecked():
            argv.append("--push-train-val-separately")
        if self.repo_id_train_edit.text().strip():
            argv.extend(["--repo-id-train", self.repo_id_train_edit.text().strip()])
        if self.repo_id_validation_edit.text().strip():
            argv.extend(["--repo-id-validation", self.repo_id_validation_edit.text().strip()])
        if self.allow_duplicate_check.isChecked():
            argv.append("--allow-duplicate-facts")
        if self.allow_ordering_check.isChecked():
            argv.append("--allow-ordering-issues")
        if self.allow_format_check.isChecked():
            argv.append("--allow-format-issues")
        return argv

    def _resize_budget_warning(self) -> str | None:
        budgets = [
            value
            for value in (self.min_pixels_spin.value(), self.max_pixels_spin.value())
            if int(value) > 0
        ]
        if not budgets:
            return None
        smallest = min(int(value) for value in budgets)
        if smallest >= _SUSPICIOUS_RESIZE_BUDGET:
            return None
        return (
            "Pixel budget uses total image area, not width/height. "
            f"The current value ({smallest}) is too small and will collapse pages to tiny images "
            "(for example, 1200 becomes about 28x28). Use a value like 1200000 for ~1.2MP, or leave it unset."
        )

    def _update_preview(self) -> None:
        page_meta_keys = self.selected_page_meta_keys()
        fact_keys = self.selected_fact_keys()
        eligible_docs = self.eligible_push_documents()
        train_docs = self.selected_train_documents()
        validation_docs = self.selected_validation_documents()
        selected_docs = self.selected_included_documents()
        approved_pages = sum(int(doc.approved_page_count or 0) for doc in selected_docs)
        total_pages = sum(int(doc.page_count or 0) for doc in selected_docs)
        skipped_pages = max(total_pages - approved_pages, 0)
        self.schema_selection_summary.setText(
            f"Page meta: {len(page_meta_keys)} selected  |  Facts: bbox + {len(fact_keys)} key(s)"
        )
        self.preview_summary_label.setText(
            f"Eligible PDFs: {len(eligible_docs)}  |  Training PDFs: {len(train_docs)}  |  "
            f"Validation PDFs: {len(validation_docs)}  |  Selected PDFs: {len(selected_docs)}  |  "
            f"Approved pages to push: {approved_pages}  |  Skipped non-approved pages: {skipped_pages}  |  "
            f"Filters: no reg flags + approved pages only"
        )
        self.schema_preview_view.setPlainText(
            build_custom_extraction_schema_preview(
                page_meta_keys=page_meta_keys,
                fact_keys=fact_keys,
            )
        )
        self.prompt_preview_view.setPlainText(
            build_custom_extraction_prompt_template(
                page_meta_keys=page_meta_keys,
                fact_keys=fact_keys,
            )
        )

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.config_browse_btn.setEnabled(not running)
        self.export_browse_btn.setEnabled(not running)
        if running:
            self.stage_label.setText("Running pipeline")
            self.stage_label.setProperty("tone", "accent")
        self.stage_label.style().unpolish(self.stage_label)
        self.stage_label.style().polish(self.stage_label)

    def run_pipeline(self) -> None:
        if self._worker_thread is not None:
            return
        resize_warning = self._resize_budget_warning()
        if resize_warning:
            QMessageBox.warning(
                self,
                "Pixel budget too small",
                resize_warning,
            )
            return
        included_doc_ids = self.selected_included_doc_ids()
        train_doc_ids = self.selected_train_doc_ids()
        validation_doc_ids = self.selected_validation_doc_ids()
        if not included_doc_ids:
            QMessageBox.warning(
                self,
                "No selected PDFs",
                "Select at least one eligible PDF to push. Only approved pages will be exported; flagged and other non-approved pages are skipped.",
            )
            return
        if not train_doc_ids:
            QMessageBox.warning(
                self,
                "No training PDFs",
                "Keep at least one eligible PDF in training before pushing.",
            )
            return
        self.log_view.clear()
        self.results_label.setText("Pipeline started.")
        argv = self._build_argv()
        worker = PushPipelineWorker(argv)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_emitted.connect(self._append_log)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        self._worker = worker
        self._worker_thread = thread
        self._set_running(True)
        thread.start()

    def _append_log(self, text: str) -> None:
        if not text:
            return
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _on_completed(self, summary: dict[str, Any]) -> None:
        _ = summary
        self.stage_label.setText("Pipeline finished")
        self.stage_label.setProperty("tone", "ok")
        self.stage_label.style().unpolish(self.stage_label)
        self.stage_label.style().polish(self.stage_label)
        lines = [line.strip() for line in self.log_view.toPlainText().splitlines() if line.strip()]
        pushed = [line.split(":", 1)[1].strip() for line in lines if line.startswith("PUSHED")]
        export_dir = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("EXPORT_DIR:")), None)
        parts = []
        if pushed:
            parts.append("Repos: " + ", ".join(pushed))
        if export_dir:
            parts.append(f"Export: {export_dir}")
        self.results_label.setText(" | ".join(parts) if parts else "Pipeline completed. Inspect the log for details.")

    def _on_failed(self, message: str) -> None:
        self.stage_label.setText("Pipeline failed")
        self.stage_label.setProperty("tone", "danger")
        self.stage_label.style().unpolish(self.stage_label)
        self.stage_label.style().polish(self.stage_label)
        self.results_label.setText("Pipeline failed. Inspect the log and error output.")
        self._append_log(f"\n{message}\n")

    def _on_finished(self) -> None:
        self._worker = None
        self._worker_thread = None
        self._set_running(False)


class DashboardWindow(QMainWindow):
    def __init__(self, startup_context: app_mod.StartupContext, *, dpi: int = 200) -> None:
        super().__init__()
        self.setObjectName("shellWindow")
        self.startup_context = startup_context
        self.import_dpi = int(dpi)
        self._import_thread: Optional[QThread] = None
        self._import_worker: Optional[WorkspaceImportWorker] = None
        self._documents_by_id: dict[str, WorkspaceDocumentSummary] = {}
        self._current_theme = load_theme_choice()
        self._nav_visible = bool(app_settings().value(NAV_VISIBLE_SETTING_KEY, True, type=bool))

        self.setWindowTitle("FineTree Modern Desktop Dashboard")
        self.resize(1540, 980)
        self._build_ui()
        self._build_menu()
        self.reload_workspace()
        self._apply_saved_theme()
        QTimer.singleShot(0, self._consume_startup_context)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("shellRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        nav = QFrame()
        self.nav_rail = nav
        nav.setObjectName("navRail")
        nav.setFixedWidth(220)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(18, 20, 18, 20)
        nav_layout.setSpacing(10)

        nav_top_row = QHBoxLayout()
        nav_top_row.setContentsMargins(0, 0, 0, 0)
        nav_top_row.setSpacing(8)
        brand = QLabel("FineTree")
        brand.setObjectName("navBrand")
        self.hide_nav_btn = QPushButton("Hide")
        self.hide_nav_btn.setObjectName("shellChromeBtn")
        self.hide_nav_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.hide_nav_btn.setToolTip("Hide navigation")
        self.hide_nav_btn.setStatusTip("Hide navigation")
        self.hide_nav_btn.setAccessibleName("Hide navigation")
        self.hide_nav_btn.setFixedHeight(32)
        nav_top_row.addWidget(brand, 1)
        nav_top_row.addWidget(self.hide_nav_btn, 0, Qt.AlignTop)
        caption = QLabel("Desktop dashboard")
        caption.setObjectName("navCaption")
        nav_layout.addLayout(nav_top_row)
        nav_layout.addWidget(caption)
        nav_layout.addSpacing(8)

        self.home_btn = self._nav_button("Home", self.style().standardIcon(QStyle.SP_DirHomeIcon))
        self.annotate_btn = self._nav_button("Annotate", _make_badge_icon("PDF", "#0f9fb6"))
        self.push_btn = self._nav_button("Push", _make_badge_icon("HF", "#d97706"))
        self.theme_btn = self._nav_button("Toggle theme", _make_badge_icon("T", "#64748b"), checkable=False)
        nav_layout.addWidget(self.home_btn)
        nav_layout.addWidget(self.annotate_btn)
        nav_layout.addWidget(self.push_btn)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.theme_btn)
        layout.addWidget(nav)

        content_shell = QWidget()
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.show_nav_row = QWidget()
        self.show_nav_row.setObjectName("shellTopRow")
        self.show_nav_row.setMaximumHeight(20)
        content_top_row = QHBoxLayout(self.show_nav_row)
        content_top_row.setContentsMargins(0, 0, 0, 1)
        content_top_row.setSpacing(2)
        self.show_nav_btn = QPushButton("Show menu")
        self.show_nav_btn.setObjectName("shellChromeBtn")
        self.show_nav_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.show_nav_btn.setToolTip("Show navigation")
        self.show_nav_btn.setStatusTip("Show navigation")
        self.show_nav_btn.setAccessibleName("Show navigation")
        self.show_nav_btn.setFixedHeight(18)
        content_top_row.addWidget(self.show_nav_btn, 0, Qt.AlignLeft)
        content_top_row.addStretch(1)
        content_layout.addWidget(self.show_nav_row, 0)

        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)
        layout.addWidget(content_shell, 1)

        self.home_view = HomeView()
        self.annotator_host = AnnotatorHost()
        self.push_view = PushView()
        self.stack.addWidget(self.home_view)
        self.stack.addWidget(self.annotator_host)
        self.stack.addWidget(self.push_view)

        self.home_btn.clicked.connect(self.show_home)
        self.annotate_btn.clicked.connect(self.show_annotate)
        self.push_btn.clicked.connect(self.show_push)
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.hide_nav_btn.clicked.connect(lambda: self.set_nav_visible(False))
        self.show_nav_btn.clicked.connect(lambda: self.set_nav_visible(True))
        self.home_view.import_pdf_requested.connect(self._choose_pdf_for_import)
        self.home_view.reset_approved_requested.connect(self.reset_all_approved_pages)
        self.home_view.open_document_requested.connect(self.open_workspace_document)
        self.home_view.prepare_document_requested.connect(self.prepare_workspace_document)
        self.home_view.checked_document_requested.connect(self.set_document_checked)
        self.home_view.review_document_requested.connect(self.set_document_reviewed)
        self.home_view.delete_document_requested.connect(self.delete_workspace_document)
        self.annotator_host.document_saved.connect(self._on_document_saved)
        self.annotator_host.current_document_changed.connect(self._on_current_document_changed)
        self.annotator_host.document_issues_changed.connect(self._on_document_issues_changed)

        self._apply_nav_visibility()
        self.statusBar().showMessage("Ready")
        self.show_home()

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        self.import_action = QAction("Import PDF", self)
        self.import_action.triggered.connect(self._choose_pdf_for_import)
        self.save_action = QAction("Save Current", self)
        self.save_action.triggered.connect(self.save_current_document)
        self.save_action.setEnabled(False)
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        navigate_menu = menu.addMenu("Navigate")
        self.home_action = QAction("Home", self)
        self.home_action.triggered.connect(self.show_home)
        self.annotate_action = QAction("Annotate", self)
        self.annotate_action.triggered.connect(self.show_annotate)
        self.push_action = QAction("Push", self)
        self.push_action.triggered.connect(self.show_push)
        navigate_menu.addAction(self.home_action)
        navigate_menu.addAction(self.annotate_action)
        navigate_menu.addAction(self.push_action)

        view_menu = menu.addMenu("View")
        self.theme_light_action = QAction("Light Theme", self, checkable=True)
        self.theme_dark_action = QAction("Dark Theme", self, checkable=True)
        self.show_nav_action = QAction("Show Navigation", self, checkable=True)
        self.show_nav_action.setShortcut("F9")
        self.theme_light_action.triggered.connect(lambda: self.set_theme("light"))
        self.theme_dark_action.triggered.connect(lambda: self.set_theme("dark"))
        self.show_nav_action.triggered.connect(self.set_nav_visible)
        self.show_nav_action.setChecked(self._nav_visible)
        view_menu.addAction(self.show_nav_action)
        view_menu.addSeparator()
        view_menu.addAction(self.theme_light_action)
        view_menu.addAction(self.theme_dark_action)

        help_menu = menu.addMenu("Help")
        shortcuts_action = QAction("Annotator Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def _nav_button(self, text: str, icon: QIcon, *, checkable: bool = True) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navButton")
        button.setIcon(icon)
        button.setIconSize(QSize(26, 26))
        button.setCheckable(checkable)
        return button

    def _apply_saved_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        self._current_theme = apply_theme(app, self._current_theme)
        self.theme_light_action.setChecked(self._current_theme == "light")
        self.theme_dark_action.setChecked(self._current_theme == "dark")

    def set_theme(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        resolved = save_theme_choice(theme_name)
        self._current_theme = apply_theme(app, resolved)
        self.theme_light_action.setChecked(self._current_theme == "light")
        self.theme_dark_action.setChecked(self._current_theme == "dark")
        self.statusBar().showMessage(f"Theme changed to {self._current_theme}.", 2500)

    def toggle_theme(self) -> None:
        self.set_theme("dark" if self._current_theme == "light" else "light")

    def _apply_nav_visibility(self) -> None:
        if hasattr(self, "nav_rail"):
            self.nav_rail.setVisible(self._nav_visible)
        if hasattr(self, "show_nav_row"):
            self.show_nav_row.setVisible(not self._nav_visible)
        if hasattr(self, "show_nav_btn"):
            self.show_nav_btn.setVisible(not self._nav_visible)
        if hasattr(self, "show_nav_action"):
            self.show_nav_action.setChecked(self._nav_visible)

    def set_nav_visible(self, visible: bool) -> None:
        self._nav_visible = bool(visible)
        app_settings().setValue(NAV_VISIBLE_SETTING_KEY, self._nav_visible)
        self._apply_nav_visibility()
        state_text = "shown" if self._nav_visible else "hidden"
        self.statusBar().showMessage(f"Navigation {state_text}. Use the button or F9 to toggle.", 2500)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.annotator_host.confirm_close_all_documents():
            event.accept()
            return
        event.ignore()

    def toggle_nav_visible(self) -> None:
        self.set_nav_visible(not self._nav_visible)

    def reload_workspace(self) -> None:
        documents = discover_workspace_documents()
        live_issue_summaries = self.annotator_host.live_document_issue_summaries()
        merged_documents: list[WorkspaceDocumentSummary] = []
        for document in documents:
            live_summary = live_issue_summaries.get(document.doc_id)
            if live_summary is None:
                merged_documents.append(document)
                continue
            reviewed = bool(document.reviewed) and int(getattr(live_summary, "reg_flag_count", 0)) == 0
            merged_documents.append(
                replace(
                    document,
                    reg_flag_count=int(getattr(live_summary, "reg_flag_count", 0)),
                    warning_count=int(getattr(live_summary, "warning_count", 0)),
                    pages_with_reg_flags=int(getattr(live_summary, "pages_with_reg_flags", 0)),
                    pages_with_warnings=int(getattr(live_summary, "pages_with_warnings", 0)),
                    checked=document.checked,
                    reviewed=reviewed,
                )
            )
        self._documents_by_id = {doc.doc_id: doc for doc in merged_documents}
        self.home_view.set_documents(merged_documents)
        self.push_view.set_documents(merged_documents)

    def show_home(self) -> None:
        self.reload_workspace()
        self.stack.setCurrentWidget(self.home_view)
        self.home_btn.setChecked(True)
        self.annotate_btn.setChecked(False)
        self.push_btn.setChecked(False)

    def show_annotate(self) -> None:
        if self.annotator_host.current_window() is None:
            self.annotator_host.show_placeholder()
        self.stack.setCurrentWidget(self.annotator_host)
        current_window = self.annotator_host.current_window()
        if current_window is not None:
            schedule_fit = getattr(current_window, "schedule_auto_fit_current_page", None)
            if callable(schedule_fit):
                schedule_fit()
        self.home_btn.setChecked(False)
        self.annotate_btn.setChecked(True)
        self.push_btn.setChecked(False)

    def show_push(self) -> None:
        self.reload_workspace()
        self.stack.setCurrentWidget(self.push_view)
        self.home_btn.setChecked(False)
        self.annotate_btn.setChecked(False)
        self.push_btn.setChecked(True)

    def open_workspace_document(self, doc_id: str) -> None:
        summary = self._documents_by_id.get(doc_id)
        if summary is None:
            summary = build_document_summary(doc_id)
        if summary.status == "Needs extraction" and summary.source_pdf is not None:
            QMessageBox.information(
                self,
                "Prepare document first",
                "This PDF still needs page images. Click Prepare from Home first.",
            )
            return
        context = DocumentContext.from_summary(summary)
        try:
            self.annotator_host.open_document(context)
        except Exception as exc:
            QMessageBox.warning(self, "Open document failed", str(exc))
            return
        self.show_annotate()
        self.statusBar().showMessage(f"Opened {context.title}.", 3000)

    def prepare_workspace_document(self, doc_id: str) -> None:
        summary = self._documents_by_id.get(doc_id)
        if summary is None:
            summary = build_document_summary(doc_id)
        if summary.source_pdf is None:
            QMessageBox.warning(self, "Prepare failed", "Managed source PDF not found for this document.")
            return
        self._start_import(summary.source_pdf, open_after=False)
        self.statusBar().showMessage(f"Preparing {doc_id} ...", 5000)

    def set_document_checked(self, doc_id: str, checked: bool) -> None:
        summary = self._documents_by_id.get(doc_id)
        if checked and summary is not None:
            is_complete = summary.page_count > 0 and summary.annotated_page_count >= summary.page_count
            if not is_complete:
                QMessageBox.warning(
                    self,
                    "Cannot mark checked",
                    "Finish annotating every page before marking this PDF as checked.",
                )
                self.reload_workspace()
                return
        set_workspace_document_checked(doc_id, checked)
        self.reload_workspace()
        state_text = "checked" if checked else "not checked"
        self.statusBar().showMessage(f"{doc_id} marked as {state_text}.", 3000)

    def set_document_reviewed(self, doc_id: str, reviewed: bool) -> None:
        summary = self._documents_by_id.get(doc_id)
        if reviewed and summary is not None and not summary.checked:
            QMessageBox.warning(
                self,
                "Cannot mark reviewed",
                "Mark this PDF as checked before review.",
            )
            self.reload_workspace()
            return
        if reviewed and summary is not None and summary.reg_flag_count > 0:
            QMessageBox.warning(
                self,
                "Cannot mark reviewed",
                (
                    f"{doc_id} has {summary.reg_flag_count} reg flag(s) across "
                    f"{summary.pages_with_reg_flags} page(s). Fix them before review."
                ),
            )
            self.reload_workspace()
            return
        set_workspace_document_reviewed(doc_id, reviewed)
        self.reload_workspace()
        state_text = "reviewed" if reviewed else "not reviewed"
        self.statusBar().showMessage(f"{doc_id} marked as {state_text}.", 3000)

    def delete_workspace_document(self, doc_id: str) -> None:
        summary = self._documents_by_id.get(doc_id)
        if summary is None:
            summary = build_document_summary(doc_id)

        display_name = summary.source_pdf.name if summary.source_pdf is not None else f"{doc_id}.pdf"
        answer = QMessageBox.question(
            self,
            "Remove PDF",
            (
                f"Delete {display_name} and all of its workspace data?\n\n"
                "This removes the PDF, extracted images, and annotations permanently."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        if not self.annotator_host.confirm_close_document(doc_id):
            return

        try:
            delete_workspace_document_files(doc_id)
        except Exception as exc:
            QMessageBox.warning(self, "Remove failed", str(exc))
            return

        self.annotator_host.close_document(doc_id)
        self.reload_workspace()
        self.show_home()
        self.statusBar().showMessage(f"Removed {display_name} and all associated data.", 4000)

    def reset_all_approved_pages(self) -> None:
        total_approved = sum(int(doc.approved_page_count or 0) for doc in self._documents_by_id.values())
        if total_approved <= 0:
            self.statusBar().showMessage("No approved pages to reset.", 2500)
            return
        for context, window in self.annotator_host.managed_windows():
            has_unsaved_changes = getattr(window, "has_unsaved_changes", None)
            if callable(has_unsaved_changes) and has_unsaved_changes():
                QMessageBox.warning(
                    self,
                    "Reset all approved",
                    (
                        "Save or discard changes in open annotator windows before resetting approved pages. "
                        f"Blocked by {context.doc_id}."
                    ),
                )
                return
        answer = QMessageBox.question(
            self,
            "Reset all approved",
            f"Are you sure you want to disapprove {total_approved} pages?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        changed_pages = 0
        open_windows_by_doc_id = {
            context.doc_id: window
            for context, window in self.annotator_host.managed_windows()
        }
        for summary in self._documents_by_id.values():
            window = open_windows_by_doc_id.get(summary.doc_id)
            live_changed = 0
            if window is not None:
                reset_live = getattr(window, "reset_all_approved_pages", None)
                if callable(reset_live):
                    live_changed = int(reset_live(mark_saved_state=True))
            disk_changed = int(reset_document_approved_pages(summary.annotations_path))
            changed_pages += max(live_changed, disk_changed)
        self.reload_workspace()
        self.statusBar().showMessage(f"Disapproved {changed_pages} page(s).", 5000)

    def _on_document_issues_changed(self, context: Optional[DocumentContext], summary: Any) -> None:
        if context is None or not context.managed:
            return
        current = self._documents_by_id.get(context.doc_id)
        if current is not None:
            self._documents_by_id[context.doc_id] = replace(
                current,
                reg_flag_count=int(getattr(summary, "reg_flag_count", 0)),
                warning_count=int(getattr(summary, "warning_count", 0)),
                pages_with_reg_flags=int(getattr(summary, "pages_with_reg_flags", 0)),
                pages_with_warnings=int(getattr(summary, "pages_with_warnings", 0)),
                checked=current.checked,
                reviewed=(current.reviewed and int(getattr(summary, "reg_flag_count", 0)) == 0),
            )
        if current is not None and current.reviewed and int(getattr(summary, "reg_flag_count", 0)) > 0:
            set_workspace_document_reviewed(context.doc_id, False)
            self.statusBar().showMessage(
                f"{context.doc_id} was auto-unreviewed because reg flags were introduced.",
                4000,
            )
            self.reload_workspace()
            return
        if self.stack.currentWidget() in {self.home_view, self.push_view}:
            self.reload_workspace()

    def open_ad_hoc_document(self, images_dir: Path, annotations_path: Path, pdf_path: Optional[Path] = None) -> None:
        context = DocumentContext(
            doc_id=images_dir.name,
            images_dir=images_dir,
            annotations_path=annotations_path,
            pdf_path=pdf_path,
            managed=False,
        )
        try:
            self.annotator_host.open_document(context)
        except Exception as exc:
            QMessageBox.warning(self, "Open document failed", str(exc))
            return
        self.show_annotate()

    def _choose_pdf_for_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose PDF", str(Path.cwd()), "PDF Files (*.pdf)")
        if not path:
            return
        self._start_import(Path(path))

    def _start_import(self, pdf_path: Path, *, open_after: bool = True) -> None:
        if self._import_thread is not None:
            QMessageBox.information(self, "Import in progress", "A PDF import is already running.")
            return
        worker = WorkspaceImportWorker(pdf_path, dpi=self.import_dpi)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(lambda result: self._on_import_completed(result, open_after=open_after))
        worker.failed.connect(self._on_import_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_import_finished)
        self._import_worker = worker
        self._import_thread = thread
        self.statusBar().showMessage(f"Importing {pdf_path.name} ...")
        thread.start()

    def _on_import_completed(self, result_obj: object, *, open_after: bool = True) -> None:
        result = result_obj if isinstance(result_obj, WorkspaceImportResult) else None
        if result is None:
            self.reload_workspace()
            return
        self.reload_workspace()
        if open_after:
            self.open_workspace_document(result.document.doc_id)
        if not open_after:
            self.statusBar().showMessage(
                f"Prepared {result.document.doc_id} ({result.document.page_count} page(s) ready).",
                5000,
            )
        elif result.opened_existing:
            self.statusBar().showMessage(
                f"Opened existing managed document {result.document.doc_id}.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                f"Imported {result.document.doc_id} ({result.document.page_count} page(s)).",
                5000,
            )

    def _on_import_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Import failed", message)
        self.statusBar().showMessage("Import failed.", 5000)

    def _on_import_finished(self) -> None:
        self._import_worker = None
        self._import_thread = None

    def _on_document_saved(self, payload: object) -> None:
        self.reload_workspace()
        message = "Workspace document saved."
        duration_ms = 3000
        if isinstance(payload, dict):
            warning_count = int(payload.get("warning_count") or 0)
            no_changes = bool(payload.get("no_changes"))
            backup_path = payload.get("backup_path")
            if no_changes:
                message = "Workspace document saved (no changes)."
            if warning_count > 0:
                message = f"{message.rstrip('.')} with {warning_count} warning(s)."
                duration_ms = 7000
            if backup_path:
                message = f"{message.rstrip('.')} Legacy backup created."
                duration_ms = max(duration_ms, 7000)
        self.statusBar().showMessage(message, duration_ms)

    def _on_current_document_changed(self, context: Optional[DocumentContext]) -> None:
        if context is None:
            self.save_action.setEnabled(False)
            self.setWindowTitle("FineTree Modern Desktop Dashboard")
            return
        self.save_action.setEnabled(True)
        self.setWindowTitle(f"FineTree Modern Desktop Dashboard - {context.title}")

    def save_current_document(self) -> None:
        window = self.annotator_host.current_window()
        if window is None:
            QMessageBox.information(self, "Save current", "No annotator document is open.")
            return
        window.save_annotations()

    def _consume_startup_context(self) -> None:
        context = self.startup_context
        if context.mode == "home":
            self.show_home()
            return
        if context.mode == "images-dir" and context.images_dir is not None and context.annotations_path is not None:
            self.open_ad_hoc_document(context.images_dir, context.annotations_path, pdf_path=context.pdf_path)
            return
        if context.mode == "pdf" and context.pdf_path is not None:
            self._start_import(context.pdf_path, open_after=True)
            return
        self.show_home()

    def _show_shortcuts(self) -> None:
        window = self.annotator_host.current_window()
        if window is None:
            QMessageBox.information(self, "Shortcuts", "Open a document to view annotator shortcuts.")
            return
        window.show_help_dialog()
