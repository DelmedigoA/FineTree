from __future__ import annotations

from typing import Iterable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .types import (
    AIActionCapabilities,
    AIActionKind,
    AIDialogDefaults,
    AIProvider,
    FEW_SHOT_SOURCE_CUSTOM_PAGES,
    FEW_SHOT_SOURCE_PRESET,
    FEW_SHOT_SOURCE_PREVIOUS_PAGES,
    action_label,
    provider_label,
)


class AIDialog(QDialog):
    state_changed = pyqtSignal()
    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(
        self,
        *,
        thinking_levels: Iterable[str],
        few_shot_presets: tuple[tuple[str, str], ...],
        fix_field_choices: tuple[tuple[str, bool], ...],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle("AI")
        self.resize(560, 680)
        self._fix_field_checks: dict[str, QCheckBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("AI")
        title.setObjectName("subtitleLabel")
        root.addWidget(title)

        summary = QLabel("Configure and run AI actions for the current page.")
        summary.setWordWrap(True)
        summary.setObjectName("hintText")
        root.addWidget(summary)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.provider_combo = QComboBox()
        self.provider_combo.addItem(provider_label(AIProvider.GEMINI), AIProvider.GEMINI.value)
        self.provider_combo.addItem(provider_label(AIProvider.QWEN), AIProvider.QWEN.value)
        self.action_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Action", self.action_combo)
        form.addRow("Model", self.model_combo)
        root.addLayout(form)

        self.context_label = QLabel("-")
        self.context_label.setWordWrap(True)
        self.context_label.setObjectName("hintText")
        root.addWidget(self.context_label)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setObjectName("hintText")
        root.addWidget(self.validation_label)

        self.options_box = QGroupBox("Options")
        options_layout = QFormLayout(self.options_box)
        options_layout.setContentsMargins(12, 12, 12, 12)
        options_layout.setHorizontalSpacing(8)
        options_layout.setVerticalSpacing(8)
        self.thinking_check = QCheckBox("Enable thinking")
        self.thinking_level_combo = QComboBox()
        self.thinking_level_combo.addItems(list(thinking_levels))
        self.few_shot_check = QCheckBox("Use few-shot examples")
        self.few_shot_source_combo = QComboBox()
        self.few_shot_source_combo.addItem("Preset", FEW_SHOT_SOURCE_PRESET)
        self.few_shot_source_combo.addItem("Previous pages", FEW_SHOT_SOURCE_PREVIOUS_PAGES)
        self.few_shot_source_combo.addItem("Custom pages", FEW_SHOT_SOURCE_CUSTOM_PAGES)
        self.few_shot_preset_combo = QComboBox()
        for preset_id, preset_label in few_shot_presets:
            self.few_shot_preset_combo.addItem(preset_label, preset_id)
        self.few_shot_previous_count_spin = QSpinBox()
        self.few_shot_previous_count_spin.setRange(1, 999)
        self.few_shot_previous_count_spin.setValue(2)
        self.few_shot_page_spec_edit = QLineEdit()
        self.few_shot_page_spec_edit.setPlaceholderText("1-3")
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setRange(-1.0, 2.0)
        self.temperature_spin.setSpecialValueText("Model default")
        self.temperature_spin.setValue(-1.0)
        self.max_facts_spin = QSpinBox()
        self.max_facts_spin.setRange(0, 999)
        options_layout.addRow("Thinking", self.thinking_check)
        options_layout.addRow("Thinking level", self.thinking_level_combo)
        options_layout.addRow("Few-shot", self.few_shot_check)
        options_layout.addRow("Few-shot source", self.few_shot_source_combo)
        options_layout.addRow("Few-shot preset", self.few_shot_preset_combo)
        options_layout.addRow("Previous count", self.few_shot_previous_count_spin)
        options_layout.addRow("Custom pages", self.few_shot_page_spec_edit)
        options_layout.addRow("Temperature", self.temperature_spin)
        options_layout.addRow("Max facts", self.max_facts_spin)
        root.addWidget(self.options_box)

        self.fix_box = QGroupBox("Fields to Fix")
        fix_layout = QVBoxLayout(self.fix_box)
        fix_layout.setContentsMargins(12, 12, 12, 12)
        fix_layout.setSpacing(8)
        fix_actions = QHBoxLayout()
        self.select_all_fields_btn = QPushButton("Select All")
        self.clear_all_fields_btn = QPushButton("Clear All")
        self.select_all_fields_btn.setObjectName("smallActionBtn")
        self.clear_all_fields_btn.setObjectName("smallActionBtn")
        fix_actions.addWidget(self.select_all_fields_btn)
        fix_actions.addWidget(self.clear_all_fields_btn)
        fix_actions.addStretch(1)
        fix_layout.addLayout(fix_actions)
        field_grid = QGridLayout()
        row = 0
        column = 0
        for field_name, checked_default in fix_field_choices:
            checkbox = QCheckBox(field_name)
            checkbox.setChecked(bool(checked_default))
            self._fix_field_checks[field_name] = checkbox
            field_grid.addWidget(checkbox, row, column)
            column += 1
            if column == 2:
                column = 0
                row += 1
        fix_layout.addLayout(field_grid)
        self.statement_type_check = QCheckBox("Allow statement_type update")
        fix_layout.addWidget(self.statement_type_check)
        root.addWidget(self.fix_box)

        self.prompt_toggle = QToolButton()
        self.prompt_toggle.setText("Customize Prompt")
        self.prompt_toggle.setCheckable(True)
        self.prompt_toggle.setChecked(False)
        self.prompt_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        root.addWidget(self.prompt_toggle, 0, Qt.AlignLeft)

        self.prompt_frame = QFrame()
        prompt_layout = QVBoxLayout(self.prompt_frame)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(6)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMinimumHeight(180)
        prompt_layout.addWidget(self.prompt_edit)
        self.prompt_frame.setVisible(False)
        root.addWidget(self.prompt_frame)

        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(4)
        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("subtitleLabel")
        self.fact_count_label = QLabel("Parsed facts: 0")
        self.fact_count_label.setObjectName("monoLabel")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fact_count_label)
        root.addWidget(status_frame)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.close_btn = QPushButton("Close")
        self.run_btn.setProperty("variant", "primary")
        self.stop_btn.setProperty("variant", "danger")
        self.stop_btn.setEnabled(False)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        actions.addStretch(1)
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.provider_combo.currentIndexChanged.connect(self.state_changed.emit)
        self.action_combo.currentIndexChanged.connect(self.state_changed.emit)
        self.model_combo.currentTextChanged.connect(self.state_changed.emit)
        self.thinking_check.toggled.connect(self._on_thinking_toggled)
        self.few_shot_check.toggled.connect(self._on_few_shot_toggled)
        self.temperature_spin.valueChanged.connect(lambda _value: self.state_changed.emit())
        self.max_facts_spin.valueChanged.connect(lambda _value: self.state_changed.emit())
        self.thinking_level_combo.currentTextChanged.connect(self.state_changed.emit)
        self.few_shot_source_combo.currentTextChanged.connect(self._on_few_shot_source_changed)
        self.few_shot_preset_combo.currentTextChanged.connect(self.state_changed.emit)
        self.few_shot_previous_count_spin.valueChanged.connect(lambda _value: self.state_changed.emit())
        self.few_shot_page_spec_edit.textChanged.connect(lambda _value: self.state_changed.emit())
        self.statement_type_check.toggled.connect(self.state_changed.emit)
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.close_btn.clicked.connect(self.close)
        self.prompt_toggle.toggled.connect(self._on_prompt_toggled)
        self.select_all_fields_btn.clicked.connect(self._select_all_fix_fields)
        self.clear_all_fields_btn.clicked.connect(self._clear_all_fix_fields)
        for checkbox in self._fix_field_checks.values():
            checkbox.toggled.connect(self.state_changed.emit)

        self._on_thinking_toggled(self.thinking_check.isChecked())
        self._on_few_shot_toggled(self.few_shot_check.isChecked())
        self._update_few_shot_fields_visibility()

    def _on_prompt_toggled(self, checked: bool) -> None:
        self.prompt_frame.setVisible(checked)

    def _on_thinking_toggled(self, checked: bool) -> None:
        self.thinking_level_combo.setEnabled(bool(checked))
        self.state_changed.emit()

    def _on_few_shot_toggled(self, checked: bool) -> None:
        enabled = bool(checked)
        self.few_shot_source_combo.setEnabled(enabled)
        self.few_shot_preset_combo.setEnabled(enabled)
        self.few_shot_previous_count_spin.setEnabled(enabled)
        self.few_shot_page_spec_edit.setEnabled(enabled)
        self._update_few_shot_fields_visibility()
        self.state_changed.emit()

    def _on_few_shot_source_changed(self, _text: str) -> None:
        self._update_few_shot_fields_visibility()
        self.state_changed.emit()

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        label = self.options_box.layout().labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def _update_few_shot_fields_visibility(self) -> None:
        use_few_shot = self.few_shot_check.isChecked()
        supports = getattr(self, "_supports_few_shot", False)
        dynamic_gt = getattr(self, "_dynamic_gt_few_shot", False)
        source = self.current_few_shot_source()

        show_source = supports and dynamic_gt and use_few_shot
        show_preset = supports and ((not dynamic_gt and use_few_shot) or (dynamic_gt and use_few_shot and source == FEW_SHOT_SOURCE_PRESET))
        show_previous = supports and dynamic_gt and use_few_shot and source == FEW_SHOT_SOURCE_PREVIOUS_PAGES
        show_custom = supports and dynamic_gt and use_few_shot and source == FEW_SHOT_SOURCE_CUSTOM_PAGES

        self._set_form_row_visible(self.few_shot_source_combo, show_source)
        self._set_form_row_visible(self.few_shot_preset_combo, show_preset)
        self._set_form_row_visible(self.few_shot_previous_count_spin, show_previous)
        self._set_form_row_visible(self.few_shot_page_spec_edit, show_custom)

    def _select_all_fix_fields(self) -> None:
        for checkbox in self._fix_field_checks.values():
            checkbox.setChecked(True)

    def _clear_all_fix_fields(self) -> None:
        for checkbox in self._fix_field_checks.values():
            checkbox.setChecked(False)

    def current_provider(self) -> AIProvider:
        value = self.provider_combo.currentData()
        return AIProvider(str(value))

    def current_action(self) -> AIActionKind:
        value = self.action_combo.currentData()
        return AIActionKind(str(value))

    def current_model(self) -> str:
        return self.model_combo.currentText().strip()

    def current_prompt(self) -> str:
        return self.prompt_edit.toPlainText()

    def current_temperature(self) -> float | None:
        if self.temperature_spin.value() <= self.temperature_spin.minimum():
            return None
        return float(self.temperature_spin.value())

    def current_few_shot_source(self) -> str:
        return str(self.few_shot_source_combo.currentData() or FEW_SHOT_SOURCE_PRESET)

    def current_few_shot_previous_count(self) -> int:
        return int(self.few_shot_previous_count_spin.value())

    def current_few_shot_page_spec(self) -> str:
        return self.few_shot_page_spec_edit.text().strip()

    def selected_fix_fields(self) -> set[str]:
        return {
            field_name
            for field_name, checkbox in self._fix_field_checks.items()
            if checkbox.isChecked()
        }

    def set_actions(self, actions: list[AIActionKind], *, current_action: Optional[AIActionKind] = None) -> None:
        previous = current_action or (self.current_action() if self.action_combo.count() else None)
        self.action_combo.blockSignals(True)
        self.action_combo.clear()
        for action in actions:
            self.action_combo.addItem(action_label(action), action.value)
        if previous is not None:
            index = self.action_combo.findData(previous.value)
            if index >= 0:
                self.action_combo.setCurrentIndex(index)
        self.action_combo.blockSignals(False)

    def set_model_choices(self, models: list[str], *, current_model: str) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for option in models:
            self.model_combo.addItem(option)
        self.model_combo.setCurrentText(current_model)
        self.model_combo.blockSignals(False)

    def set_context_summary(self, text: str) -> None:
        self.context_label.setText(text)

    def set_validation_message(self, text: str) -> None:
        self.validation_label.setText(text)
        self.validation_label.setVisible(bool(text))

    def set_prompt_text(self, text: str) -> None:
        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(text)
        self.prompt_edit.blockSignals(False)

    def set_temperature(self, value: float | None) -> None:
        self.temperature_spin.blockSignals(True)
        if value is None:
            self.temperature_spin.setValue(self.temperature_spin.minimum())
        else:
            self.temperature_spin.setValue(float(value))
        self.temperature_spin.blockSignals(False)

    def set_status(self, text: str, *, fact_count: int) -> None:
        self.status_label.setText(text)
        self.fact_count_label.setText(f"Parsed facts: {int(fact_count)}")

    def set_capabilities(self, capabilities: AIActionCapabilities) -> None:
        self._supports_few_shot = capabilities.supports_few_shot
        self.thinking_check.setVisible(capabilities.supports_thinking)
        self.thinking_level_combo.setVisible(capabilities.supports_thinking_level)
        thinking_level_label = self.options_box.layout().labelForField(self.thinking_level_combo)
        if thinking_level_label is not None:
            thinking_level_label.setVisible(capabilities.supports_thinking_level)

        self.few_shot_check.setVisible(capabilities.supports_few_shot)
        few_shot_label = self.options_box.layout().labelForField(self.few_shot_check)
        if few_shot_label is not None:
            few_shot_label.setVisible(capabilities.supports_few_shot)
        self._update_few_shot_fields_visibility()

        self.temperature_spin.setVisible(capabilities.supports_temperature)
        temperature_label = self.options_box.layout().labelForField(self.temperature_spin)
        if temperature_label is not None:
            temperature_label.setVisible(capabilities.supports_temperature)

        self.max_facts_spin.setVisible(capabilities.supports_max_facts)
        max_facts_label = self.options_box.layout().labelForField(self.max_facts_spin)
        if max_facts_label is not None:
            max_facts_label.setVisible(capabilities.supports_max_facts)

        self.fix_box.setVisible(capabilities.supports_fix_fields)
        self.statement_type_check.setVisible(capabilities.supports_statement_type_toggle)
        self.options_box.setVisible(
            capabilities.supports_thinking
            or capabilities.supports_few_shot
            or capabilities.supports_temperature
            or capabilities.supports_max_facts
        )
        self._on_thinking_toggled(self.thinking_check.isChecked())
        self._on_few_shot_toggled(self.few_shot_check.isChecked())

    def set_few_shot_context(self, *, provider: AIProvider, action: AIActionKind, supports_few_shot: bool) -> None:
        self._supports_few_shot = bool(supports_few_shot)
        self._dynamic_gt_few_shot = provider == AIProvider.GEMINI and action == AIActionKind.GROUND_TRUTH
        self._update_few_shot_fields_visibility()

    def apply_defaults(self, defaults: AIDialogDefaults) -> None:
        provider_index = self.provider_combo.findData(defaults.provider.value)
        if provider_index >= 0:
            self.provider_combo.setCurrentIndex(provider_index)
        self.set_temperature(defaults.temperature)
        self.thinking_check.setChecked(bool(defaults.enable_thinking))
        level_index = self.thinking_level_combo.findText(defaults.thinking_level)
        if level_index >= 0:
            self.thinking_level_combo.setCurrentIndex(level_index)
        self.few_shot_check.setChecked(bool(defaults.use_few_shot))
        source_index = self.few_shot_source_combo.findData(defaults.few_shot_source)
        if source_index >= 0:
            self.few_shot_source_combo.setCurrentIndex(source_index)
        preset_index = self.few_shot_preset_combo.findData(defaults.few_shot_preset)
        if preset_index >= 0:
            self.few_shot_preset_combo.setCurrentIndex(preset_index)
        self.few_shot_previous_count_spin.setValue(max(1, int(defaults.few_shot_previous_count)))
        self.few_shot_page_spec_edit.setText(str(defaults.few_shot_page_spec or ""))
        self.max_facts_spin.setValue(max(0, int(defaults.max_facts)))
        for field_name, checkbox in self._fix_field_checks.items():
            checkbox.setChecked(field_name in defaults.selected_fact_fields)
        self.statement_type_check.setChecked(bool(defaults.include_statement_type))

    def set_current_action(self, action: AIActionKind) -> None:
        index = self.action_combo.findData(action.value)
        if index >= 0:
            self.action_combo.setCurrentIndex(index)

    def set_running(self, running: bool) -> None:
        for widget in (
            self.provider_combo,
            self.action_combo,
            self.model_combo,
            self.thinking_check,
            self.thinking_level_combo,
            self.few_shot_check,
            self.few_shot_source_combo,
            self.few_shot_preset_combo,
            self.few_shot_previous_count_spin,
            self.few_shot_page_spec_edit,
            self.temperature_spin,
            self.max_facts_spin,
            self.statement_type_check,
            self.select_all_fields_btn,
            self.clear_all_fields_btn,
            self.prompt_toggle,
            self.prompt_edit,
        ):
            widget.setEnabled(not running)
        for checkbox in self._fix_field_checks.values():
            checkbox.setEnabled(not running)
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def set_run_enabled(self, enabled: bool) -> None:
        if not self.stop_btn.isEnabled():
            self.run_btn.setEnabled(enabled)


class AnnotateProgressDialog(QDialog):
    stop_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle("Annotate")
        self.setFixedSize(320, 144)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Annotate")
        title.setObjectName("subtitleLabel")
        root.addWidget(title)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.fact_count_label = QLabel("Parsed facts: 0")
        self.fact_count_label.setObjectName("monoLabel")
        root.addWidget(self.fact_count_label)

        actions = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.close_btn = QPushButton("Close")
        self.stop_btn.setProperty("variant", "danger")
        actions.addWidget(self.stop_btn)
        actions.addStretch(1)
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.close_btn.clicked.connect(self.hide)

    def set_status(self, text: str, *, fact_count: int) -> None:
        self.status_label.setText(text)
        self.fact_count_label.setText(f"Parsed facts: {int(fact_count)}")

    def set_running(self, running: bool) -> None:
        self.stop_btn.setEnabled(running)
        self.close_btn.setEnabled(not running)


__all__ = ["AIDialog", "AnnotateProgressDialog"]
