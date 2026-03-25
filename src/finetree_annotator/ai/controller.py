from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from ..annotation_core import PageState
from ..gemini_vlm import format_issue_summary_brief, load_issue_summary, resolve_supported_gemini_model_name
from ..provider_workers import GeminiFillWorker, GeminiStreamWorker, QwenStreamWorker
from ..qwen_vlm import current_qwen_gt_model_choices
from .bbox import (
    BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
    BBOX_MODE_PIXEL_AS_IS,
    normalize_ai_fact_payload,
)
from .dialog import AIDialog
from .payloads import (
    build_extraction_prompt,
    build_gemini_autocomplete_prompt,
    build_gemini_autocomplete_request_payload,
    build_gemini_fill_prompt,
    build_gemini_fill_request_payload,
    extraction_prompt_template,
    gemini_autocomplete_prompt_template,
    gemini_fill_prompt_template,
)
from .types import (
    AIActionCapabilities,
    AIActionKind,
    AIDialogDefaults,
    AIPageContext,
    AIProvider,
    AIWorkflowRequest,
    FEW_SHOT_PRESET_2015_TWO_SHOT,
    FEW_SHOT_PRESET_CHOICES,
    FEW_SHOT_PRESET_CLASSIC,
    provider_label,
)


class AIWorkflowController:
    def __init__(
        self,
        host: Any,
        *,
        thinking_levels: tuple[str, ...],
        fix_field_choices: tuple[tuple[str, bool], ...],
    ) -> None:
        self.host = host
        self.thinking_levels = tuple(thinking_levels)
        self.fix_field_choices = tuple(fix_field_choices)
        self.dialog: Optional[AIDialog] = None
        self._status_text = "Idle."
        self._status_fact_count = 0
        self._dialog_refresh_in_progress = False
        self._last_dialog_defaults_key: tuple[AIProvider, AIActionKind] | None = None
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts: tuple[int, int] | None = None

    def ensure_dialog(self) -> AIDialog:
        if self.dialog is None:
            dialog = AIDialog(
                parent=self.host,
                thinking_levels=self.thinking_levels,
                few_shot_presets=FEW_SHOT_PRESET_CHOICES,
                fix_field_choices=self.fix_field_choices,
            )
            dialog.state_changed.connect(self._on_dialog_state_changed)
            dialog.run_requested.connect(self.run_from_dialog)
            dialog.stop_requested.connect(self.stop_active_generation)
            self.dialog = dialog
        return self.dialog

    def open_dialog(
        self,
        *,
        provider: Optional[AIProvider] = None,
        action: Optional[AIActionKind] = None,
    ) -> None:
        dialog = self.ensure_dialog()
        target_provider = provider or AIProvider.GEMINI
        target_action = action or AIActionKind.GROUND_TRUTH
        self._dialog_refresh_in_progress = True
        try:
            self.refresh_dialog_state(
                provider=target_provider,
                action=target_action,
                reset_defaults=True,
            )
        finally:
            self._dialog_refresh_in_progress = False
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def refresh_dialog_state(
        self,
        *,
        provider: Optional[AIProvider] = None,
        action: Optional[AIActionKind] = None,
        reset_defaults: bool = False,
    ) -> None:
        dialog = self.dialog
        if dialog is None:
            return

        self._dialog_refresh_in_progress = True
        try:
            target_provider = provider or dialog.current_provider()
            actions = self._actions_for_provider(target_provider)
            dialog.set_actions(actions, current_action=action or (dialog.current_action() if dialog.action_combo.count() else None))
            if provider is not None:
                dialog.apply_defaults(self._defaults_for(target_provider, action or self._first_action_for_provider(target_provider)))
                provider_index = dialog.provider_combo.findData(target_provider.value)
                if provider_index >= 0:
                    dialog.provider_combo.setCurrentIndex(provider_index)
            current_action = action or dialog.current_action()
            if current_action not in actions:
                current_action = actions[0]
                dialog.set_current_action(current_action)
            if reset_defaults or self._last_dialog_defaults_key != (target_provider, current_action):
                defaults = self._defaults_for(target_provider, current_action)
                dialog.apply_defaults(defaults)
                self._last_dialog_defaults_key = (target_provider, current_action)

            model_choices = self._model_choices_for(target_provider)
            defaults = self._defaults_for(target_provider, current_action)
            current_model = dialog.current_model() or defaults.model
            if provider is not None or not current_model:
                current_model = defaults.model
            dialog.set_model_choices(model_choices, current_model=current_model)

            capabilities = self._capabilities_for(target_provider, current_action)
            dialog.set_capabilities(capabilities)
            context = self.host._ai_page_context()
            dialog.set_context_summary(self._context_summary(context))
            dialog.set_validation_message(self._validation_message(capabilities, context))
            dialog.set_prompt_text(self._build_prompt_for_dialog(target_provider, current_action, context))
            dialog.set_status(self._status_text, fact_count=self._status_fact_count)
            dialog.set_running(self.is_running())
            dialog.set_run_enabled(self._can_run(capabilities, context) and not self.is_running())
        finally:
            self._dialog_refresh_in_progress = False

    def _on_dialog_state_changed(self) -> None:
        if self._dialog_refresh_in_progress:
            return
        self.refresh_dialog_state()

    def is_running(self) -> bool:
        return any(
            thread is not None
            for thread in (
                self.host._gemini_stream_thread,
                self.host._gemini_fill_thread,
                self.host._qwen_stream_thread,
            )
        )

    def stop_active_generation(self) -> None:
        if self.host._gemini_stream_thread is not None and self.host._gemini_stream_worker is not None:
            self.host._gemini_stream_cancel_requested = True
            self.host._gemini_stream_worker.request_cancel()
            self._set_status("Stopping AI stream...", fact_count=self.host._gemini_stream_fact_count, running=True)
            return
        if self.host._qwen_stream_thread is not None and self.host._qwen_stream_worker is not None:
            self.host._qwen_stream_cancel_requested = True
            self.host._qwen_stream_worker.request_cancel()
            self._set_status("Stopping AI stream...", fact_count=self.host._qwen_stream_fact_count, running=True)
            return
        if self.host._gemini_fill_thread is not None and self.host._gemini_fill_worker is not None:
            self.host._gemini_fill_cancel_requested = True
            self.host._gemini_fill_worker.request_cancel()
            self._set_status("Stopping Fix...", fact_count=0, running=True)

    def run_from_dialog(self) -> None:
        dialog = self.dialog
        if dialog is None:
            return
        request = AIWorkflowRequest(
            provider=dialog.current_provider(),
            action=dialog.current_action(),
            model=dialog.current_model(),
            prompt_text=dialog.current_prompt().strip(),
            temperature=dialog.current_temperature(),
            enable_thinking=dialog.thinking_check.isChecked(),
            thinking_level=dialog.thinking_level_combo.currentText().strip().lower() or "minimal",
            use_few_shot=dialog.few_shot_check.isChecked(),
            few_shot_preset=str(dialog.few_shot_preset_combo.currentData() or FEW_SHOT_PRESET_CLASSIC),
            max_facts=int(dialog.max_facts_spin.value()),
            selected_fact_fields=dialog.selected_fix_fields(),
            include_statement_type=dialog.statement_type_check.isChecked(),
        )
        self.start_request(request)
        if self.is_running():
            dialog.close()

    def start_request(self, request: AIWorkflowRequest) -> None:
        if self.is_running():
            QMessageBox.information(self.host, "AI", "An AI action is already running.")
            return

        self.host._capture_current_state()
        context = self.host._ai_page_context()
        capabilities = self._capabilities_for(request.provider, request.action)
        validation_message = self._validation_message(capabilities, context)
        if validation_message:
            QMessageBox.information(self.host, "AI", validation_message)
            self.refresh_dialog_state()
            return
        if context is None:
            QMessageBox.warning(self.host, "AI", "No current page is loaded.")
            self.refresh_dialog_state()
            return

        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.GROUND_TRUTH:
            self._run_gemini_ground_truth(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.AUTO_COMPLETE:
            self._run_gemini_autocomplete(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.FIX_SELECTED:
            self._run_gemini_fix(request, context)
            return
        if request.provider == AIProvider.QWEN and request.action == AIActionKind.GROUND_TRUTH:
            self._run_qwen_ground_truth(request, context)
            return
        QMessageBox.warning(self.host, "AI", "That provider/action combination is not supported.")
        self.refresh_dialog_state()

    def _run_gemini_ground_truth(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        current_state = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self.host,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    "Generate AI ground truth and replace this page annotations?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        prompt_text = request.prompt_text.strip()
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return

        try:
            from ..gemini_vlm import ensure_gemini_backend_credentials
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Gemini backend is unavailable:\n{exc}")
            return

        model_name = resolve_supported_gemini_model_name(request.model.strip() or self.host._gemini_model_name)
        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return

        few_shot_examples = self._load_gemini_few_shot_examples(request)
        self.host._gemini_model_name = model_name
        self.host._gemini_temperature = request.temperature
        self.host._gemini_enable_thinking = request.enable_thinking
        self.host._gemini_thinking_level = request.thinking_level
        existing_meta = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index)).meta or {}
        self.host.page_states[context.page_name] = PageState(
            meta={**self.host._default_meta(context.page_index), **existing_meta},
            facts=[],
        )
        was_restoring = self.host._is_restoring_history
        self.host._is_restoring_history = True
        try:
            self.host.show_page(context.page_index)
        finally:
            self.host._is_restoring_history = was_restoring

        self._start_gemini_stream(
            page_path=context.page_path,
            page_name=context.page_name,
            prompt_text=prompt_text,
            model_name=model_name,
            gemini_api_key=gemini_api_key,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            thinking_level=request.thinking_level,
            few_shot_examples=few_shot_examples,
            mode="gt",
            max_facts=request.max_facts,
            apply_meta=True,
            initial_seen_facts=set(),
        )

    def _run_gemini_autocomplete(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        prompt_text = request.prompt_text.strip()
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return

        try:
            from ..gemini_vlm import ensure_gemini_backend_credentials
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Gemini backend is unavailable:\n{exc}")
            return

        model_name = resolve_supported_gemini_model_name(request.model.strip() or self.host._gemini_model_name)
        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return

        few_shot_examples = self._load_gemini_few_shot_examples(request)
        self.host._gemini_model_name = model_name
        self.host._gemini_temperature = request.temperature
        self.host._gemini_enable_thinking = request.enable_thinking
        self.host._gemini_thinking_level = request.thinking_level
        initial_seen_facts = {
            self.host._fact_uniqueness_key(payload)
            for payload in context.ordered_fact_payloads
        }
        self.host._gemini_autocomplete_snapshot = {
            "page_name": context.page_name,
            "ordered_fact_signature": context.ordered_fact_signature,
            "locked_fact_payloads": [deepcopy(payload) for payload in context.ordered_fact_payloads],
        }
        self._start_gemini_stream(
            page_path=context.page_path,
            page_name=context.page_name,
            prompt_text=prompt_text,
            model_name=model_name,
            gemini_api_key=gemini_api_key,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            thinking_level=request.thinking_level,
            few_shot_examples=few_shot_examples,
            mode="autocomplete",
            max_facts=request.max_facts,
            apply_meta=False,
            initial_seen_facts=initial_seen_facts,
        )

    def _run_gemini_fix(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        if not request.selected_fact_fields and not request.include_statement_type:
            QMessageBox.warning(self.host, "AI", "Choose at least one fact field or statement_type.")
            return
        prompt_text = request.prompt_text.strip()
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return

        try:
            from ..gemini_vlm import ensure_gemini_backend_credentials
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Gemini backend is unavailable:\n{exc}")
            return

        model_name = resolve_supported_gemini_model_name(request.model.strip() or self.host._gemini_model_name)
        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return

        self.host._gemini_model_name = model_name
        self.host._gemini_temperature = request.temperature
        self.host._gemini_enable_thinking = request.enable_thinking
        self.host._gemini_thinking_level = request.thinking_level
        self.host._gemini_fill_target_page = context.page_name
        self.host._gemini_fill_selected_fact_fields = set(request.selected_fact_fields)
        self.host._gemini_fill_include_statement_type = request.include_statement_type
        self.host._gemini_fill_snapshot = {
            "page_name": context.page_name,
            "selected_fact_nums": list(context.selected_fact_nums),
            "ordered_fact_signature": context.ordered_fact_signature,
        }
        self.host._gemini_fill_cancel_requested = False

        worker = GeminiFillWorker(
            image_path=context.page_path,
            prompt=prompt_text,
            model=model_name,
            api_key=gemini_api_key,
            allowed_fact_fields=set(request.selected_fact_fields),
            allow_statement_type=request.include_statement_type,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            thinking_level=request.thinking_level,
        )
        thread = QThread(self.host)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_gemini_fill_completed)
        worker.failed.connect(self._on_gemini_fill_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_gemini_fill_finished)

        self.host._gemini_fill_worker = worker
        self.host._gemini_fill_thread = thread
        self._set_status(
            f"Running Gemini Fix for {len(context.selected_fact_nums)} selected fact(s)...",
            fact_count=0,
            running=True,
        )
        self.host.statusBar().showMessage(f"Running Gemini Fix for {context.page_name}...", 3000)
        thread.start()

    def _run_qwen_ground_truth(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        current_state = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self.host,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    "Generate AI ground truth and replace this page annotations?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        prompt_text = request.prompt_text.strip()
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return

        try:
            from ..qwen_vlm import is_qwen_flash_model_requested, resolve_qwen_flash_api_key
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Qwen backend is unavailable:\n{exc}")
            return

        model_name = request.model.strip() or self.host._qwen_model_name
        qwen_config_path = self.host._resolve_qwen_config_path()
        is_qwen_flash = is_qwen_flash_model_requested(model_name)
        if is_qwen_flash:
            if not resolve_qwen_flash_api_key():
                QMessageBox.warning(
                    self.host,
                    "AI",
                    (
                        "Qwen hosted API key not found.\n\n"
                        "Set FINETREE_QWEN_FLASH_API_KEY, FINETREE_QWEN_API_KEY, "
                        "QWEN_API_KEY, or DASHSCOPE_API_KEY."
                    ),
                )
                return
        elif qwen_config_path is None:
            QMessageBox.warning(
                self.host,
                "AI",
                (
                    "Could not find Qwen fine-tune config.\n"
                    "Expected configs/qwen_ui_runpod_queue.yaml, "
                    "configs/finetune_qwen35a3_vl.yaml, or FINETREE_QWEN_CONFIG."
                ),
            )
            return

        self.host._qwen_model_name = model_name
        self.host._qwen_enable_thinking = request.enable_thinking
        few_shot_examples: Optional[list[dict[str, Any]]] = None
        if request.use_few_shot and is_qwen_flash:
            few_shot_examples = self._load_qwen_few_shot_examples(request)
        elif request.use_few_shot:
            self.host.statusBar().showMessage(
                "Qwen few-shot is currently enabled only for hosted Qwen 3.5 DashScope models; running standard mode.",
                7000,
            )

        existing_meta = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index)).meta or {}
        self.host.page_states[context.page_name] = PageState(
            meta={**self.host._default_meta(context.page_index), **existing_meta},
            facts=[],
        )
        was_restoring = self.host._is_restoring_history
        self.host._is_restoring_history = True
        try:
            self.host.show_page(context.page_index)
        finally:
            self.host._is_restoring_history = was_restoring

        self.host._qwen_stream_target_page = context.page_name
        self.host._qwen_stream_seen_facts = set()
        self.host._qwen_stream_fact_count = 0
        self.host._qwen_stream_cancel_requested = False

        worker = QwenStreamWorker(
            image_path=context.page_path,
            prompt=prompt_text,
            model=model_name,
            config_path=str(qwen_config_path) if qwen_config_path is not None else None,
            few_shot_examples=few_shot_examples,
            enable_thinking=request.enable_thinking,
        )
        thread = QThread(self.host)
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

        self.host._qwen_stream_worker = worker
        self.host._qwen_stream_thread = thread
        self._set_status(f"Streaming Qwen GT from {model_name}...", fact_count=0, running=True)
        self.host.statusBar().showMessage(f"Streaming Qwen GT for {context.page_name}...", 3000)
        thread.start()

    def _start_gemini_stream(
        self,
        *,
        page_path: Path,
        page_name: str,
        prompt_text: str,
        model_name: str,
        gemini_api_key: Optional[str],
        temperature: Optional[float],
        enable_thinking: bool,
        thinking_level: str,
        few_shot_examples: Optional[list[dict[str, Any]]],
        mode: str,
        max_facts: int,
        apply_meta: bool,
        initial_seen_facts: set[tuple[Any, ...]],
    ) -> None:
        self.host._gemini_stream_target_page = page_name
        self.host._gemini_stream_seen_facts = set(initial_seen_facts)
        self.host._gemini_stream_fact_count = 0
        self.host._gemini_autocomplete_buffered_facts = []
        self.host._gemini_autocomplete_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_autocomplete_last_bbox_scores = {}
        self.host._gemini_gt_buffered_facts = []
        self.host._gemini_gt_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_gt_last_bbox_scores = {}
        self.host._gemini_gt_live_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_gt_live_bbox_mode_locked = False
        self.host._gemini_gt_live_applied = False
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts = None
        self.host._gemini_stream_cancel_requested = False
        self.host._gemini_stream_mode = mode
        self.host._gemini_stream_apply_meta = bool(apply_meta)
        self.host._gemini_stream_max_facts = max(0, int(max_facts))
        self.host._gemini_stream_limit_reached = False

        worker = GeminiStreamWorker(
            image_path=page_path,
            prompt=prompt_text,
            model=model_name,
            mode=mode,
            api_key=gemini_api_key,
            few_shot_examples=few_shot_examples,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_level=thinking_level,
            max_facts=self.host._gemini_stream_max_facts,
            allow_partial_finalize_error=(mode == "autocomplete"),
        )
        thread = QThread(self.host)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk_received.connect(self._on_gemini_stream_chunk)
        worker.meta_received.connect(self._on_gemini_stream_meta)
        worker.fact_received.connect(self._on_gemini_stream_fact)
        worker.limit_reached.connect(self._on_gemini_stream_limit_reached)
        worker.completed.connect(self._on_gemini_stream_completed)
        worker.failed.connect(self._on_gemini_stream_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_gemini_stream_finished)

        self.host._gemini_stream_worker = worker
        self.host._gemini_stream_thread = thread
        status_text = (
            f"Running Gemini Auto Complete from {model_name}..."
            if mode == "autocomplete"
            else f"Streaming Gemini GT from {model_name}..."
        )
        self._set_status(status_text, fact_count=0, running=True)
        self.host.statusBar().showMessage(
            f"Running {'Gemini Auto Complete' if mode == 'autocomplete' else 'Gemini GT'} for {page_name}...",
            3000,
        )
        thread.start()

    def _load_gemini_few_shot_examples(self, request: AIWorkflowRequest) -> Optional[list[dict[str, Any]]]:
        if not request.use_few_shot:
            return None
        loaded_examples, warnings = self.host._load_gemini_few_shot_examples(preset=request.few_shot_preset)
        if loaded_examples:
            if warnings:
                self.host.statusBar().showMessage(
                    f"Few-shot loaded with warnings: {'; '.join(warnings[:2])}",
                    6500,
                )
            return loaded_examples
        warning_text = "; ".join(warnings[:2]) if warnings else "Few-shot preset unavailable."
        self.host.statusBar().showMessage(f"Few-shot fallback to standard mode: {warning_text}", 7000)
        return None

    def _load_qwen_few_shot_examples(self, request: AIWorkflowRequest) -> Optional[list[dict[str, Any]]]:
        loaded_examples, warnings = self.host._load_gemini_few_shot_examples(preset=request.few_shot_preset)
        if loaded_examples:
            if warnings:
                self.host.statusBar().showMessage(
                    f"Qwen few-shot loaded with warnings: {'; '.join(warnings[:2])}",
                    6500,
                )
            return loaded_examples
        warning_text = "; ".join(warnings[:2]) if warnings else "Few-shot preset unavailable."
        self.host.statusBar().showMessage(
            f"Qwen few-shot fallback to standard mode: {warning_text}",
            7000,
        )
        return None

    def _actions_for_provider(self, provider: AIProvider) -> list[AIActionKind]:
        if provider == AIProvider.QWEN:
            return [AIActionKind.GROUND_TRUTH]
        return [AIActionKind.GROUND_TRUTH, AIActionKind.AUTO_COMPLETE, AIActionKind.FIX_SELECTED]

    def _first_action_for_provider(self, provider: AIProvider) -> AIActionKind:
        return self._actions_for_provider(provider)[0]

    def _capabilities_for(self, provider: AIProvider, action: AIActionKind) -> AIActionCapabilities:
        if provider == AIProvider.QWEN:
            return AIActionCapabilities(
                supports_thinking=True,
                supports_thinking_level=False,
                supports_few_shot=True,
                supports_max_facts=False,
                replaces_existing_page_facts=True,
            )
        if action == AIActionKind.GROUND_TRUTH:
            return AIActionCapabilities(
                supports_thinking=True,
                supports_thinking_level=True,
                supports_few_shot=True,
                supports_temperature=True,
                supports_max_facts=True,
                replaces_existing_page_facts=True,
            )
        if action == AIActionKind.AUTO_COMPLETE:
            return AIActionCapabilities(
                supports_thinking=True,
                supports_thinking_level=True,
                supports_few_shot=True,
                supports_temperature=True,
                supports_max_facts=True,
                requires_existing_facts=True,
            )
        return AIActionCapabilities(
            supports_thinking=True,
            supports_thinking_level=True,
            supports_temperature=True,
            supports_fix_fields=True,
            supports_statement_type_toggle=True,
            requires_selected_facts=True,
        )

    def _defaults_for(self, provider: AIProvider, action: AIActionKind) -> AIDialogDefaults:
        default_fix_fields = {
            field_name
            for field_name, checked_default in self.fix_field_choices
            if checked_default
        }
        if provider == AIProvider.QWEN:
            return AIDialogDefaults(
                provider=provider,
                action=AIActionKind.GROUND_TRUTH,
                model=self.host._qwen_model_name,
                temperature=None,
                enable_thinking=bool(self.host._qwen_enable_thinking),
                thinking_level="high" if self.host._qwen_enable_thinking else "minimal",
                use_few_shot=True,
                few_shot_preset=FEW_SHOT_PRESET_CLASSIC,
                max_facts=0,
                selected_fact_fields=default_fix_fields,
                include_statement_type=False,
            )
        if action == AIActionKind.GROUND_TRUTH:
            few_shot_default = FEW_SHOT_PRESET_2015_TWO_SHOT
            use_few_shot = True
        elif action == AIActionKind.AUTO_COMPLETE:
            few_shot_default = FEW_SHOT_PRESET_CLASSIC
            use_few_shot = False
        else:
            few_shot_default = FEW_SHOT_PRESET_CLASSIC
            use_few_shot = False
        return AIDialogDefaults(
            provider=provider,
            action=action,
            model=self.host._gemini_model_name,
            temperature=getattr(self.host, "_gemini_temperature", None),
            enable_thinking=bool(self.host._gemini_enable_thinking),
            thinking_level=self.host._gemini_thinking_level,
            use_few_shot=use_few_shot,
            few_shot_preset=few_shot_default,
            max_facts=0,
            selected_fact_fields=default_fix_fields,
            include_statement_type=False,
        )

    def _model_choices_for(self, provider: AIProvider) -> list[str]:
        if provider == AIProvider.QWEN:
            config_path = self.host._resolve_qwen_config_path()
            return list(current_qwen_gt_model_choices(str(config_path) if config_path is not None else None))
        from ..gemini_vlm import SUPPORTED_GEMINI_MODELS

        return list(SUPPORTED_GEMINI_MODELS)

    def _context_summary(self, context: Optional[AIPageContext]) -> str:
        if context is None:
            return "No page loaded."
        return (
            f"Page: {context.page_name} | "
            f"Existing facts: {context.existing_fact_count} | "
            f"Selected facts: {context.selected_fact_count}"
        )

    def _validation_message(self, capabilities: AIActionCapabilities, context: Optional[AIPageContext]) -> str:
        if context is None:
            return "Load a page to run AI."
        if capabilities.requires_existing_facts and context.existing_fact_count <= 0:
            return "Auto Complete requires at least one existing fact on the current page."
        if capabilities.requires_selected_facts and context.selected_fact_count <= 0:
            return "Select one or more facts before running Fix."
        return ""

    def _can_run(self, capabilities: AIActionCapabilities, context: Optional[AIPageContext]) -> bool:
        return context is not None and not self._validation_message(capabilities, context)

    def _build_prompt_for_dialog(
        self,
        provider: AIProvider,
        action: AIActionKind,
        context: Optional[AIPageContext],
    ) -> str:
        if context is None:
            return ""
        if provider == AIProvider.QWEN or action == AIActionKind.GROUND_TRUTH:
            return build_extraction_prompt(
                extraction_prompt_template(),
                context.page_path,
                image_dimensions=context.image_dimensions,
            )
        if action == AIActionKind.AUTO_COMPLETE:
            request_payload = build_gemini_autocomplete_request_payload(
                page_name=context.page_name,
                page_meta=context.page_meta,
                ordered_fact_payloads=context.ordered_fact_payloads,
            )
            return build_gemini_autocomplete_prompt(
                gemini_autocomplete_prompt_template(),
                request_payload=request_payload,
                image_dimensions=context.image_dimensions,
            )
        dialog = self.dialog
        selected_fact_fields = dialog.selected_fix_fields() if dialog is not None else set()
        include_statement_type = bool(dialog.statement_type_check.isChecked()) if dialog is not None else False
        request_payload = build_gemini_fill_request_payload(
            page_name=context.page_name,
            page_meta=context.page_meta,
            ordered_fact_payloads=context.ordered_fact_payloads,
            selected_fact_nums=context.selected_fact_nums,
            selected_fact_fields=selected_fact_fields,
            include_statement_type=include_statement_type,
        )
        return build_gemini_fill_prompt(
            gemini_fill_prompt_template(),
            request_payload=request_payload,
            selected_fact_fields=selected_fact_fields,
            include_statement_type=include_statement_type,
        )

    def _set_status(self, text: str, *, fact_count: int, running: bool) -> None:
        self._status_text = text
        self._status_fact_count = int(fact_count)
        if self.dialog is not None:
            self.dialog.set_status(text, fact_count=self._status_fact_count)
            self.dialog.set_running(running)
            self.dialog.set_run_enabled(not running)

    def _current_gemini_stream_issue_summary(self) -> tuple[dict[str, Any] | None, Path | None]:
        worker = self.host._gemini_stream_worker
        if worker is None:
            return None, None
        summary = getattr(worker, "issue_summary", None)
        session_dir = getattr(worker, "session_dir", None)
        if summary is None and isinstance(session_dir, Path):
            summary = load_issue_summary(session_dir)
        return summary, session_dir if isinstance(session_dir, Path) else None

    def _apply_finalized_gemini_stream_payloads(self, extraction_facts: list[Any]) -> None:
        mode = self.host._gemini_stream_mode
        finalized_payloads: list[dict[str, Any]] = []
        for fact in extraction_facts:
            if hasattr(fact, "model_dump"):
                fact_payload = fact.model_dump(mode="json")
            elif isinstance(fact, dict):
                fact_payload = fact
            else:
                continue
            if mode == "autocomplete":
                normalized_payload = self.host._normalized_autocomplete_generated_fact_payload(fact_payload)
            else:
                normalized_payload = self.host._normalized_stream_fact_payload(fact_payload)
            if normalized_payload is None:
                continue
            finalized_payloads.append(normalized_payload)
        self.host._gemini_stream_seen_facts = {
            self.host._fact_uniqueness_key(payload)
            for payload in finalized_payloads
        }
        self.host._gemini_stream_fact_count = len(finalized_payloads)
        if mode == "autocomplete":
            self.host._gemini_autocomplete_buffered_facts = finalized_payloads
        elif mode == "gt":
            self.host._gemini_gt_buffered_facts = finalized_payloads

    def _gt_finalize_counts(
        self,
        *,
        live_fact_count: int,
        finalized_fact_count: int,
        summary: dict[str, Any] | None,
    ) -> tuple[int, int]:
        if isinstance(summary, dict):
            streamed_count = int(summary.get("streamed_fact_count") or 0)
            kept_count = int(summary.get("kept_fact_count") or finalized_fact_count)
            if streamed_count > 0:
                return streamed_count, kept_count
        return live_fact_count, finalized_fact_count

    def _gt_should_retain_live_result(
        self,
        *,
        live_fact_count: int,
        finalized_fact_count: int,
        summary: dict[str, Any] | None,
    ) -> tuple[bool, int, int]:
        streamed_count, kept_count = self._gt_finalize_counts(
            live_fact_count=live_fact_count,
            finalized_fact_count=finalized_fact_count,
            summary=summary,
        )
        return kept_count < streamed_count, streamed_count, kept_count

    def _gt_finalize_retained_live_warning(
        self,
        *,
        streamed_count: int,
        kept_count: int,
        summary: dict[str, Any] | None,
        session_dir: Path | None,
    ) -> str:
        base_detail = format_issue_summary_brief(
            summary,
            session_dir=session_dir,
            streamed_live_facts_remain=True,
            no_changes_applied=False,
        )
        retain_line = (
            f"Preserved the {streamed_count} live-streamed fact(s) because final validation retained only "
            f"{kept_count}."
        )
        if base_detail:
            return f"{base_detail}\n{retain_line}"
        return retain_line

    def _on_gemini_stream_chunk(self, text: str) -> None:
        _ = text

    def _on_gemini_stream_limit_reached(self) -> None:
        self.host._gemini_stream_limit_reached = True

    def _on_gemini_stream_meta(self, meta_payload: dict[str, Any]) -> None:
        page_name = self.host._gemini_stream_target_page
        if page_name is None:
            return
        if self.host._gemini_stream_apply_meta:
            self.host._apply_stream_meta(page_name, meta_payload)
        status = "Meta received. Streaming facts..." if self.host._gemini_stream_apply_meta else "Locked context loaded. Streaming missing facts..."
        self._set_status(status, fact_count=self.host._gemini_stream_fact_count, running=True)

    def _on_gemini_stream_fact(self, fact_payload: dict[str, Any]) -> None:
        page_name = self.host._gemini_stream_target_page
        if page_name is None:
            return
        added = self.host._apply_stream_fact(
            page_name,
            fact_payload,
            seen_facts=self.host._gemini_stream_seen_facts,
            stream_source="gemini",
        )
        if added:
            self.host._gemini_stream_fact_count += 1
            if self.host._gemini_stream_mode == "gt":
                self.host._update_gemini_gt_live_stream(page_name)
            progress_text = (
                f"Streaming missing facts... {self.host._gemini_stream_fact_count} parsed"
                if self.host._gemini_stream_mode == "autocomplete"
                else f"Streaming facts... {self.host._gemini_stream_fact_count} parsed"
            )
            self._set_status(progress_text, fact_count=self.host._gemini_stream_fact_count, running=True)

    def _on_gemini_stream_completed(self, extraction_obj: Any) -> None:
        page_name = self.host._gemini_stream_target_page
        if page_name is None:
            return
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts = None
        pre_finalize_gt_buffer: list[dict[str, Any]] = []
        pre_finalize_gt_seen_facts: set[tuple[Any, ...]] = set()
        pre_finalize_gt_count = 0
        if self.host._gemini_stream_mode == "gt":
            pre_finalize_gt_buffer = [deepcopy(payload) for payload in self.host._gemini_gt_buffered_facts]
            pre_finalize_gt_seen_facts = set(self.host._gemini_stream_seen_facts)
            pre_finalize_gt_count = len(pre_finalize_gt_buffer)
        try:
            extraction_meta = getattr(extraction_obj, "meta", None)
            if hasattr(extraction_meta, "model_dump"):
                meta_payload = extraction_meta.model_dump(mode="json")
            elif isinstance(extraction_meta, dict):
                meta_payload = extraction_meta
            else:
                meta_payload = {}
            if self.host._gemini_stream_apply_meta:
                self.host._apply_stream_meta(page_name, meta_payload)
            extraction_facts = getattr(extraction_obj, "facts", [])
            if not isinstance(extraction_facts, list):
                extraction_facts = []
            if self.host._gemini_stream_mode in {"autocomplete", "gt"}:
                self._apply_finalized_gemini_stream_payloads(extraction_facts)
            else:
                for fact in extraction_facts:
                    if hasattr(fact, "model_dump"):
                        fact_payload = fact.model_dump(mode="json")
                    elif isinstance(fact, dict):
                        fact_payload = fact
                    else:
                        continue
                    added = self.host._apply_stream_fact(
                        page_name,
                        fact_payload,
                        seen_facts=self.host._gemini_stream_seen_facts,
                        stream_source="gemini",
                    )
                    if added:
                        self.host._gemini_stream_fact_count += 1
            if self.host._gemini_stream_mode == "autocomplete":
                merged, error_message = self.host._merge_autocomplete_buffered_facts(page_name)
                if not merged:
                    self._on_gemini_stream_failed(error_message or "Auto Complete could not merge results.")
                    return
            elif self.host._gemini_stream_mode == "gt":
                summary, _session_dir = self._current_gemini_stream_issue_summary()
                finalized_gt_count = len(self.host._gemini_gt_buffered_facts)
                retain_live, streamed_count, kept_count = self._gt_should_retain_live_result(
                    live_fact_count=pre_finalize_gt_count,
                    finalized_fact_count=finalized_gt_count,
                    summary=summary,
                )
                if retain_live and pre_finalize_gt_count > 0:
                    self.host._gemini_gt_buffered_facts = pre_finalize_gt_buffer
                    self.host._gemini_stream_seen_facts = set(pre_finalize_gt_seen_facts)
                    self.host._gemini_stream_fact_count = pre_finalize_gt_count
                    self.host._gemini_gt_live_applied = True
                    self._gemini_gt_finalize_retained_live = True
                    self._gemini_gt_finalize_retained_counts = (streamed_count, kept_count)
                else:
                    merged, error_message = self.host._merge_gemini_gt_buffered_facts(page_name)
                    if not merged:
                        self._on_gemini_stream_failed(error_message or "Ground Truth could not merge results.")
                        return
        except Exception as exc:
            self._on_gemini_stream_failed(f"Final parse failed: {exc}")
            return

        summary, session_dir = self._current_gemini_stream_issue_summary()
        if self.host._gemini_stream_mode == "gt" and self._gemini_gt_finalize_retained_live:
            streamed_count, kept_count = self._gemini_gt_finalize_retained_counts or (
                self.host._gemini_stream_fact_count,
                len(self.host._gemini_gt_buffered_facts),
            )
            QMessageBox.warning(
                self.host,
                "AI warning",
                self._gt_finalize_retained_live_warning(
                    streamed_count=streamed_count,
                    kept_count=kept_count,
                    summary=summary,
                    session_dir=session_dir,
                ),
            )
        elif summary is not None and int(summary.get("issue_count") or 0) > 0:
            QMessageBox.warning(
                self.host,
                "AI warning",
                format_issue_summary_brief(
                    summary,
                    session_dir=session_dir,
                    streamed_live_facts_remain=False,
                    no_changes_applied=False,
                ),
            )

        if self.host._gemini_stream_mode != "autocomplete" or self.host._gemini_stream_fact_count > 0:
            self.host._record_history_snapshot()
        completion_status, completion_message = self._gemini_stream_completion_text()
        self._set_status(completion_status, fact_count=self.host._gemini_stream_fact_count, running=False)
        self.host.statusBar().showMessage(completion_message, 6000)

    def _on_gemini_stream_failed(self, message: str) -> None:
        title = "AI failed"
        self._set_status(f"Error: {message}", fact_count=self.host._gemini_stream_fact_count, running=False)
        summary, session_dir = self._current_gemini_stream_issue_summary()
        if summary is not None:
            detail = format_issue_summary_brief(
                summary,
                session_dir=session_dir,
                streamed_live_facts_remain=(self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied),
                no_changes_applied=self.host._gemini_stream_mode in {"autocomplete", "gt"} and not self.host._gemini_gt_live_applied,
            )
        elif self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied:
            detail = f"{message}\n\nSome streamed facts were rendered live and remain on the page."
        elif self.host._gemini_stream_mode in {"autocomplete", "gt"}:
            detail = f"{message}\n\nNo changes were applied."
        else:
            detail = f"{message}\n\nAny facts already streamed remain on the page."
        QMessageBox.warning(self.host, title, detail)

    def _on_gemini_stream_finished(self) -> None:
        if self.host._gemini_stream_cancel_requested and not self.host._gemini_stream_limit_reached:
            stopped_message = (
                (
                    f"Ground Truth stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, live-streamed facts kept)."
                )
                if self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied
                else (
                    f"Ground Truth stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, no changes applied)."
                )
                if self.host._gemini_stream_mode == "gt"
                else f"Auto Complete stopped ({self.host._gemini_stream_fact_count} fact(s) parsed)."
            )
            self._set_status(stopped_message, fact_count=self.host._gemini_stream_fact_count, running=False)
            self.host.statusBar().showMessage(stopped_message, 5000)
        self.host._gemini_stream_thread = None
        self.host._gemini_stream_worker = None
        self.host._gemini_stream_target_page = None
        self.host._gemini_stream_cancel_requested = False
        self.host._gemini_stream_limit_reached = False
        self.host._gemini_stream_max_facts = 0
        self.host._gemini_stream_apply_meta = True
        self.host._gemini_stream_mode = "gt"
        self.host._gemini_autocomplete_snapshot = None
        self.host._gemini_autocomplete_buffered_facts = []
        self.host._gemini_autocomplete_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_autocomplete_last_bbox_scores = {}
        self.host._gemini_gt_buffered_facts = []
        self.host._gemini_gt_last_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_gt_last_bbox_scores = {}
        self.host._gemini_gt_live_bbox_mode = BBOX_MODE_PIXEL_AS_IS
        self.host._gemini_gt_live_bbox_mode_locked = False
        self.host._gemini_gt_live_applied = False
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts = None
        self.refresh_dialog_state()

    def _gemini_stream_completion_text(self) -> tuple[str, str]:
        if self.host._gemini_stream_mode == "autocomplete":
            bbox_mode_label = (
                "normalized_1000->pixel"
                if self.host._gemini_autocomplete_last_bbox_mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL
                else "pixel"
            )
            text = (
                f"Auto Complete complete. Merged {self.host._gemini_stream_fact_count} new fact(s) "
                f"(bbox mode: {bbox_mode_label})."
            )
            return text, text
        bbox_mode_label = (
            "normalized_1000->pixel"
            if self.host._gemini_gt_last_bbox_mode == BBOX_MODE_NORMALIZED_1000_TO_PIXEL
            else "pixel"
        )
        pixel_score = float(self.host._gemini_gt_last_bbox_scores.get(BBOX_MODE_PIXEL_AS_IS, 0.0))
        normalized_score = float(self.host._gemini_gt_last_bbox_scores.get(BBOX_MODE_NORMALIZED_1000_TO_PIXEL, 0.0))
        if self._gemini_gt_finalize_retained_live:
            streamed_count, kept_count = self._gemini_gt_finalize_retained_counts or (
                self.host._gemini_stream_fact_count,
                self.host._gemini_stream_fact_count,
            )
            text = (
                f"Ground Truth complete with warnings. Preserved {self.host._gemini_stream_fact_count} "
                f"live-streamed fact(s); final validation retained only {kept_count}/{streamed_count} "
                f"(bbox mode: {bbox_mode_label}, pixel={pixel_score:.3f}, normalized={normalized_score:.3f})."
            )
            return text, text
        text = (
            f"Ground Truth complete. Parsed {self.host._gemini_stream_fact_count} fact(s) "
            f"(bbox mode: {bbox_mode_label}, pixel={pixel_score:.3f}, normalized={normalized_score:.3f})."
        )
        return text, text

    def _on_gemini_fill_completed(self, patch_payload: dict[str, Any]) -> None:
        self.host._on_gemini_fill_completed(patch_payload)
        if self.host._gemini_fill_snapshot is not None:
            updated_count = len(patch_payload.get("fact_updates") or [])
            self._set_status(f"Fix complete. Updated {updated_count} fact(s).", fact_count=updated_count, running=False)

    def _on_gemini_fill_failed(self, message: str) -> None:
        self._set_status(f"Error: {message}", fact_count=0, running=False)
        self.host._on_gemini_fill_failed(message)

    def _on_gemini_fill_finished(self) -> None:
        self.host._on_gemini_fill_finished()
        self.refresh_dialog_state()

    def _on_qwen_stream_chunk(self, text: str) -> None:
        _ = text

    def _on_qwen_stream_meta(self, meta_payload: dict[str, Any]) -> None:
        page_name = self.host._qwen_stream_target_page
        if page_name is None:
            return
        self.host._apply_stream_meta(page_name, meta_payload)
        self._set_status("Meta received. Streaming facts...", fact_count=self.host._qwen_stream_fact_count, running=True)

    def _on_qwen_stream_fact(self, fact_payload: dict[str, Any]) -> None:
        page_name = self.host._qwen_stream_target_page
        if page_name is None:
            return
        added = self.host._apply_stream_fact(
            page_name,
            fact_payload,
            seen_facts=self.host._qwen_stream_seen_facts,
            stream_source="qwen",
        )
        if added:
            self.host._qwen_stream_fact_count += 1
            self._set_status(
                f"Streaming facts... {self.host._qwen_stream_fact_count} parsed",
                fact_count=self.host._qwen_stream_fact_count,
                running=True,
            )

    def _on_qwen_stream_completed(self, extraction_obj: Any) -> None:
        page_name = self.host._qwen_stream_target_page
        if page_name is None:
            return
        try:
            meta_payload = extraction_obj.meta.model_dump(mode="json")
            self.host._apply_stream_meta(page_name, meta_payload)
            for fact in extraction_obj.facts:
                payload = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact
                if not isinstance(payload, dict):
                    continue
                added = self.host._apply_stream_fact(
                    page_name,
                    payload,
                    seen_facts=self.host._qwen_stream_seen_facts,
                    stream_source="qwen",
                )
                if added:
                    self.host._qwen_stream_fact_count += 1
        except Exception as exc:
            self._on_qwen_stream_failed(f"Final parse failed: {exc}")
            return

        self.host._record_history_snapshot()
        message = f"Qwen GT complete. Parsed {self.host._qwen_stream_fact_count} fact(s)."
        self._set_status(message, fact_count=self.host._qwen_stream_fact_count, running=False)
        self.host.statusBar().showMessage(message, 6000)

    def _on_qwen_stream_failed(self, message: str) -> None:
        self._set_status(f"Error: {message}", fact_count=self.host._qwen_stream_fact_count, running=False)
        QMessageBox.warning(self.host, "AI failed", f"{message}\n\nAny facts already streamed remain on the page.")

    def _on_qwen_stream_finished(self) -> None:
        if self.host._qwen_stream_cancel_requested:
            stopped = f"Qwen GT stopped ({self.host._qwen_stream_fact_count} fact(s) parsed)."
            self._set_status(stopped, fact_count=self.host._qwen_stream_fact_count, running=False)
            self.host.statusBar().showMessage(stopped, 5000)
        self.host._qwen_stream_thread = None
        self.host._qwen_stream_worker = None
        self.host._qwen_stream_target_page = None
        self.refresh_dialog_state()


__all__ = ["AIWorkflowController"]
