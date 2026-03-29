from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from ..annotation_core import PageState
from ..gemini_vlm import (
    DEFAULT_GEMINI_MODEL,
    format_issue_summary_brief,
    is_vertex_gemini_model_requested,
    load_issue_summary,
    resolve_supported_gemini_model_name,
)
from ..local_doctr import LOCAL_DOCTR_MODEL_NAME
from ..provider_workers import GeminiFillWorker, GeminiStreamWorker, LocalDocTRBBoxWorker, QwenStreamWorker
from ..qwen_vlm import DEFAULT_QWEN_BBOX_MAX_PIXELS, current_qwen_gt_model_choices
from ..vision_resize import prepared_dimensions_for_max_pixels
from .bbox import (
    BBOX_MODE_NORMALIZED_1000_TO_PIXEL,
    BBOX_MODE_PIXEL_AS_IS,
    normalize_ai_fact_payload,
)
from .dialog import AIDialog, AnnotateProgressDialog
from .payloads import (
    build_extraction_prompt,
    build_gemini_bbox_only_prompt,
    build_gemini_autocomplete_prompt,
    build_gemini_autocomplete_request_payload,
    build_gemini_fill_prompt,
    build_gemini_fill_request_payload,
    build_gemini_fix_drawn_prompt,
    build_gemini_fix_drawn_request_payload,
    build_qwen_bbox_only_prompt,
    extraction_prompt_template,
    gemini_autocomplete_prompt_template,
    gemini_fill_prompt_template,
    system_prompt_template,
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
    LocalDetectorBackend,
    local_detector_backend_label,
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
        self.annotate_progress_dialog: Optional[AnnotateProgressDialog] = None
        self._status_text = "Idle."
        self._status_fact_count = 0
        self._dialog_refresh_in_progress = False
        self._last_dialog_defaults_key: tuple[AIProvider, AIActionKind] | None = None
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts: tuple[int, int] | None = None
        self._document_auto_annotate_pending_completion: tuple[str, int] | None = None

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

    def ensure_annotate_progress_dialog(self) -> AnnotateProgressDialog:
        if self.annotate_progress_dialog is None:
            dialog = AnnotateProgressDialog(parent=self.host)
            dialog.stop_requested.connect(self.stop_active_generation)
            self.annotate_progress_dialog = dialog
        return self.annotate_progress_dialog

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
                self.host._local_doctr_thread,
                self.host._gemini_stream_thread,
                self.host._gemini_fill_thread,
                self.host._qwen_stream_thread,
            )
        )

    def stop_active_generation(self) -> None:
        if self.host._local_doctr_thread is not None and self.host._local_doctr_worker is not None:
            self.host._local_doctr_cancel_requested = True
            self.host._local_doctr_worker.request_cancel()
            backend_label = local_detector_backend_label(self.host._local_doctr_backend)
            self._set_status(
                f"Stopping local detector ({backend_label})...",
                fact_count=self.host._local_doctr_fact_count,
                running=True,
            )
            return
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

        if request.provider == AIProvider.DOCTR and request.action == AIActionKind.BBOX_ONLY:
            self._run_local_doctr_bbox_only(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.GROUND_TRUTH:
            self._run_gemini_ground_truth(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.BBOX_ONLY:
            self._run_gemini_bbox_only(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.AUTO_COMPLETE:
            self._run_gemini_autocomplete(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.FIX_SELECTED:
            self._run_gemini_fix(request, context)
            return
        if request.provider == AIProvider.GEMINI and request.action == AIActionKind.FIX_DRAWN:
            self._run_gemini_fix_drawn(request, context)
            return
        if request.provider == AIProvider.QWEN and request.action == AIActionKind.GROUND_TRUTH:
            self._run_qwen_ground_truth(request, context)
            return
        if request.provider == AIProvider.QWEN and request.action == AIActionKind.BBOX_ONLY:
            self._run_qwen_bbox_only(request, context)
            return
        QMessageBox.warning(self.host, "AI", "That provider/action combination is not supported.")
        self.refresh_dialog_state()

    def _page_context_for_name(self, page_name: str) -> AIPageContext | None:
        page_idx = self.host._page_index_by_name(page_name)
        if page_idx < 0:
            return None
        if self.host.current_index != page_idx:
            self.host.show_page(page_idx)
        context = self.host._ai_page_context()
        if context is None or context.page_name != page_name:
            return None
        return context

    def _start_annotate_for_context(
        self,
        context: AIPageContext,
        *,
        mode: str,
        show_progress_dialog: bool,
        show_busy_message: bool = True,
    ) -> bool:
        if self.is_running():
            if show_busy_message:
                QMessageBox.information(self.host, "AI", "An AI action is already running.")
            return False

        try:
            from ..gemini_vlm import ensure_gemini_backend_credentials
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Gemini backend is unavailable:\n{exc}")
            return False

        model_name = DEFAULT_GEMINI_MODEL
        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return False

        request = AIWorkflowRequest(
            provider=AIProvider.GEMINI,
            action=AIActionKind.GROUND_TRUTH,
            model=model_name,
            prompt_text=self._build_prompt_for_dialog(AIProvider.GEMINI, AIActionKind.GROUND_TRUTH, context),
            use_few_shot=True,
            few_shot_preset=FEW_SHOT_PRESET_2015_TWO_SHOT,
        )
        prompt_text = request.prompt_text.strip()
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return False
        few_shot_examples = self._load_gemini_few_shot_examples(request)
        self.host._gemini_model_name = model_name
        self.host._gemini_temperature = None
        self.host._gemini_enable_thinking = False
        self.host._gemini_thinking_level = "minimal"

        current_state = deepcopy(
            self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        )
        self.host._auto_annotate_active = True
        self.host._auto_annotate_phase = "gemini"
        self.host._auto_annotate_mode = mode
        self.host._auto_annotate_target_page = context.page_name
        self.host._auto_annotate_original_state = current_state
        self.host._auto_annotate_restore_on_finish = True
        self.host._auto_annotate_gemini_payloads = []
        self.host._auto_annotate_last_match_debug = {}
        self.host._gemini_gt_live_updates_enabled = False

        if show_progress_dialog:
            progress_dialog = self.ensure_annotate_progress_dialog()
            progress_dialog.set_status("Starting Gemini annotations...", fact_count=0)
            progress_dialog.set_running(True)
            progress_dialog.show()
            progress_dialog.raise_()
            progress_dialog.activateWindow()

        self._start_gemini_stream(
            page_path=context.page_path,
            page_name=context.page_name,
            prompt_text=prompt_text,
            model_name=model_name,
            gemini_api_key=gemini_api_key,
            system_prompt=None,
            temperature=None,
            enable_thinking=False,
            thinking_level="minimal",
            few_shot_examples=few_shot_examples,
            mode="gt",
            max_facts=0,
            apply_meta=False,
            initial_seen_facts=set(),
        )
        return True

    def start_annotate(self) -> None:
        context = self.host._ai_page_context()
        if context is None:
            QMessageBox.warning(self.host, "AI", "No current page is loaded.")
            self.refresh_dialog_state()
            return
        self._start_annotate_for_context(context, mode="annotate", show_progress_dialog=True)

    def start_auto_annotate(self) -> None:
        self.start_annotate()

    def start_document_auto_annotate_page(self, page_name: str) -> bool:
        context = self._page_context_for_name(page_name)
        if context is None:
            return False
        return self._start_annotate_for_context(
            context,
            mode="document",
            show_progress_dialog=False,
            show_busy_message=False,
        )

    def _start_align_bboxes_for_context(self, context: AIPageContext, *, mode: str) -> bool:
        if self.is_running():
            QMessageBox.information(self.host, "AI", "An AI action is already running.")
            return False

        ordered_fact_payloads = self.host._ordered_fact_payloads_by_geometry(context.ordered_fact_payloads)
        if not ordered_fact_payloads:
            QMessageBox.warning(self.host, "AI", "No current annotations are available to align.")
            return False

        self.host._auto_annotate_active = True
        self.host._auto_annotate_phase = "detector"
        self.host._auto_annotate_mode = mode
        self.host._auto_annotate_target_page = context.page_name
        self.host._auto_annotate_original_state = deepcopy(
            self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        )
        self.host._auto_annotate_restore_on_finish = True
        self.host._auto_annotate_gemini_payloads = [deepcopy(payload) for payload in ordered_fact_payloads]
        self.host._auto_annotate_last_match_debug = {}
        self.host._gemini_gt_live_updates_enabled = True

        self._start_local_doctr_worker(
            image_path=context.page_path,
            page_name=context.page_name,
            backend=LocalDetectorBackend.MERGED,
            max_facts=0,
        )
        return True

    def start_align_bboxes_for_page(self, page_name: str, *, mode: str = "align") -> bool:
        context = self._page_context_for_name(page_name)
        if context is None:
            return False
        return self._start_align_bboxes_for_context(context, mode=mode)

    def start_align_bboxes(self) -> None:
        context = self.host._ai_page_context()
        if context is None:
            QMessageBox.warning(self.host, "AI", "No current page is loaded.")
            self.refresh_dialog_state()
            return
        self._start_align_bboxes_for_context(context, mode="align")

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

        model_name = resolve_supported_gemini_model_name(request.model.strip() or self.host._gemini_model_name)
        use_tuned_model = is_vertex_gemini_model_requested(model_name)
        prompt_text = (
            self._build_prompt_for_dialog(
                AIProvider.GEMINI,
                AIActionKind.GROUND_TRUTH,
                context,
            ).strip()
            if use_tuned_model
            else request.prompt_text.strip()
        )
        if not prompt_text:
            QMessageBox.warning(self.host, "AI", "Prompt cannot be empty.")
            return
        system_prompt = system_prompt_template() if use_tuned_model else None
        if use_tuned_model and not system_prompt:
            QMessageBox.warning(self.host, "AI", "System prompt cannot be empty.")
            return

        try:
            from ..gemini_vlm import ensure_gemini_backend_credentials
        except Exception as exc:
            QMessageBox.warning(self.host, "AI", f"Gemini backend is unavailable:\n{exc}")
            return

        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return

        few_shot_examples = None
        effective_temperature = 0.0 if use_tuned_model else request.temperature
        if not use_tuned_model:
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
            system_prompt=system_prompt,
            temperature=effective_temperature,
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
            system_prompt=None,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            thinking_level=request.thinking_level,
            few_shot_examples=few_shot_examples,
            mode="autocomplete",
            max_facts=request.max_facts,
            apply_meta=False,
            initial_seen_facts=initial_seen_facts,
        )

    def _run_gemini_bbox_only(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        current_state = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self.host,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    "Extract Gemini bounding boxes only and replace this page annotations?"
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
        if is_vertex_gemini_model_requested(model_name):
            QMessageBox.warning(
                self.host,
                "AI",
                "BBoxes + Values mode currently supports standard Gemini models only. Select gemini-3-flash-preview or another non-tuned Gemini model.",
            )
            return
        gemini_api_key, auth_error = ensure_gemini_backend_credentials(model_name)
        if auth_error:
            QMessageBox.warning(self.host, "AI", auth_error)
            return

        few_shot_examples = self._load_gemini_bbox_only_few_shot_examples(request)
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
            system_prompt=None,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            thinking_level=request.thinking_level,
            few_shot_examples=few_shot_examples,
            mode="bbox_only",
            max_facts=request.max_facts,
            apply_meta=False,
            initial_seen_facts=set(),
        )

    def _run_local_doctr_bbox_only(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        backend_label = local_detector_backend_label(request.local_detector_backend)
        current_state = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        if current_state.facts:
            answer = QMessageBox.question(
                self.host,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    f"Run the local detector ({backend_label}) and replace this page annotations?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        existing_meta = current_state.meta or {}
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

        self._start_local_doctr_worker(
            image_path=context.page_path,
            page_name=context.page_name,
            backend=request.local_detector_backend,
            max_facts=request.max_facts,
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

    def _run_gemini_fix_drawn(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        if not context.hand_drawn_fact_nums:
            QMessageBox.warning(self.host, "AI", "No hand-drawn bboxes to fix.")
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

        from ..schema_contract import SchemaRegistry

        all_patch_fields = set(SchemaRegistry.get_prompt_contract("gemini_fill")["fact_patch_fields"])
        all_patch_fields.add("value")

        self.host._gemini_model_name = model_name
        self.host._gemini_temperature = request.temperature
        self.host._gemini_enable_thinking = request.enable_thinking
        self.host._gemini_thinking_level = request.thinking_level
        self.host._gemini_fill_target_page = context.page_name
        self.host._gemini_fill_selected_fact_fields = all_patch_fields
        self.host._gemini_fill_include_statement_type = False
        self.host._gemini_fill_snapshot = {
            "page_name": context.page_name,
            "selected_fact_nums": list(context.hand_drawn_fact_nums),
            "ordered_fact_signature": context.ordered_fact_signature,
        }
        self.host._gemini_fill_cancel_requested = False

        worker = GeminiFillWorker(
            image_path=context.page_path,
            prompt=prompt_text,
            model=model_name,
            api_key=gemini_api_key,
            allowed_fact_fields=all_patch_fields,
            allow_statement_type=False,
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
        drawn_count = len(context.hand_drawn_fact_nums)
        self._set_status(
            f"Running Fix Drawn for {drawn_count} hand-drawn bbox(es)...",
            fact_count=0,
            running=True,
        )
        self.host.statusBar().showMessage(f"Running Fix Drawn for {context.page_name}...", 3000)
        thread.start()

    def _start_local_doctr_worker(
        self,
        *,
        image_path: Path,
        page_name: str,
        backend: LocalDetectorBackend,
        max_facts: int,
    ) -> None:
        backend_label = local_detector_backend_label(backend)
        self.host._local_doctr_target_page = page_name
        self.host._local_doctr_seen_facts = set()
        self.host._local_doctr_fact_count = 0
        self.host._local_doctr_cancel_requested = False
        self.host._local_doctr_buffered_facts = []
        self.host._local_doctr_backend = backend

        worker = LocalDocTRBBoxWorker(
            image_path=image_path,
            max_facts=max_facts,
            backend=backend,
        )
        thread = QThread(self.host)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.fact_received.connect(self._on_local_doctr_fact)
        worker.completed.connect(self._on_local_doctr_completed)
        worker.failed.connect(self._on_local_doctr_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_local_doctr_finished)

        self.host._local_doctr_worker = worker
        self.host._local_doctr_thread = thread
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
            if getattr(self.host, "_auto_annotate_mode", "") == "align":
                self._set_status("Align bboxes: running local detectors...", fact_count=0, running=True)
                self.host.statusBar().showMessage(f"Align bboxes: running local detectors for {page_name}...", 3000)
            else:
                self._set_status("Auto-Annotate: running local detectors...", fact_count=0, running=True)
                self.host.statusBar().showMessage(f"Auto-Annotate: running local detectors for {page_name}...", 3000)
        elif backend == LocalDetectorBackend.MERGED:
            self._set_status("Running local detectors (stock + fine-tuned)...", fact_count=0, running=True)
            self.host.statusBar().showMessage(
                f"Running local detectors (stock + fine-tuned) for {page_name}...",
                3000,
            )
        else:
            self._set_status(
                f"Running local detector ({backend_label}) BBoxes + Values...",
                fact_count=0,
                running=True,
            )
            self.host.statusBar().showMessage(
                f"Running local detector ({backend_label}) for {page_name}...",
                3000,
            )
        thread.start()

    def _run_qwen_ground_truth(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        self._run_qwen_stream(request, context, mode="gt")

    def _run_qwen_bbox_only(self, request: AIWorkflowRequest, context: AIPageContext) -> None:
        self._run_qwen_stream(request, context, mode="bbox_only")

    def _run_qwen_stream(self, request: AIWorkflowRequest, context: AIPageContext, *, mode: str) -> None:
        current_state = self.host.page_states.get(context.page_name, self.host._default_state(context.page_index))
        replace_message = (
            "Generate AI ground truth and replace this page annotations?"
            if mode == "gt"
            else "Extract Qwen bounding boxes + values and replace this page annotations?"
        )
        if current_state.facts:
            answer = QMessageBox.question(
                self.host,
                "Replace current annotations?",
                (
                    f"Current page already has {len(current_state.facts)} bbox(es).\n"
                    f"{replace_message}"
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
            few_shot_examples = (
                self._load_qwen_bbox_only_few_shot_examples(request)
                if mode == "bbox_only"
                else self._load_qwen_few_shot_examples(request)
            )
        elif request.use_few_shot:
            self.host.statusBar().showMessage(
                "Qwen few-shot is currently enabled only for hosted Qwen 3.5 DashScope models; running standard mode.",
                7000,
            )
        prepared_size: tuple[int, int] | None = None
        bbox_original_size: tuple[float, float] | None = None
        if mode == "bbox_only":
            image_dimensions = context.image_dimensions
            if image_dimensions is None:
                QMessageBox.warning(self.host, "AI", "Current image dimensions are unavailable for Qwen bbox-only mode.")
                return
            prepared_h, prepared_w = prepared_dimensions_for_max_pixels(
                original_width=float(image_dimensions[0]),
                original_height=float(image_dimensions[1]),
                max_pixels=DEFAULT_QWEN_BBOX_MAX_PIXELS,
            )
            prepared_size = (int(prepared_w), int(prepared_h))
            bbox_original_size = (float(image_dimensions[0]), float(image_dimensions[1]))
            QMessageBox.information(
                self.host,
                "Qwen BBoxes + Values",
                (
                    "Current bundled FineTree Qwen checkpoints were trained on no-bbox targets.\n\n"
                    "Qwen BBoxes + Values is experimental unless the selected model was separately trained "
                    "for bbox output."
                ),
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
        self.host._qwen_stream_mode = mode
        self.host._qwen_bbox_buffered_facts = []
        self.host._qwen_bbox_original_size = None
        self.host._qwen_bbox_prepared_size = None
        self.host._qwen_bbox_max_pixels = None
        if mode == "bbox_only":
            self.host._qwen_bbox_original_size = bbox_original_size
            self.host._qwen_bbox_prepared_size = prepared_size
            self.host._qwen_bbox_max_pixels = int(DEFAULT_QWEN_BBOX_MAX_PIXELS)

        worker = QwenStreamWorker(
            image_path=context.page_path,
            prompt=prompt_text,
            model=model_name,
            mode=mode,
            config_path=str(qwen_config_path) if qwen_config_path is not None else None,
            few_shot_examples=few_shot_examples,
            enable_thinking=request.enable_thinking,
            prepared_size=prepared_size,
            original_size=bbox_original_size,
            bbox_max_pixels=self.host._qwen_bbox_max_pixels,
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
        self._set_status(
            (
                f"Running Qwen BBoxes + Values from {model_name}..."
                if mode == "bbox_only"
                else f"Streaming Qwen GT from {model_name}..."
            ),
            fact_count=0,
            running=True,
        )
        self.host.statusBar().showMessage(
            (
                f"Running Qwen BBoxes + Values for {context.page_name}..."
                if mode == "bbox_only"
                else f"Streaming Qwen GT for {context.page_name}..."
            ),
            3000,
        )
        thread.start()

    def _start_gemini_stream(
        self,
        *,
        page_path: Path,
        page_name: str,
        prompt_text: str,
        model_name: str,
        gemini_api_key: Optional[str],
        system_prompt: Optional[str],
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
            system_prompt=system_prompt,
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
            else f"Running Gemini BBoxes + Values from {model_name}..."
            if mode == "bbox_only"
            else f"Streaming Gemini GT from {model_name}..."
        )
        self._set_status(status_text, fact_count=0, running=True)
        self.host.statusBar().showMessage(
            (
                f"Running Gemini Auto Complete for {page_name}..."
                if mode == "autocomplete"
                else f"Running Gemini BBoxes + Values for {page_name}..."
                if mode == "bbox_only"
                else f"Running Gemini GT for {page_name}..."
            ),
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

    def _load_gemini_bbox_only_few_shot_examples(self, request: AIWorkflowRequest) -> Optional[list[dict[str, Any]]]:
        loaded_examples = self._load_gemini_few_shot_examples(request)
        if not loaded_examples:
            return None
        transformed: list[dict[str, Any]] = []
        invalid_count = 0
        for raw_example in loaded_examples:
            if not isinstance(raw_example, dict):
                invalid_count += 1
                continue
            expected_json = str(raw_example.get("expected_json") or "").strip()
            if not expected_json:
                invalid_count += 1
                continue
            try:
                import json

                payload = json.loads(expected_json)
            except Exception:
                invalid_count += 1
                continue
            pages = payload.get("pages") if isinstance(payload, dict) else None
            if not isinstance(pages, list):
                invalid_count += 1
                continue
            projected_pages: list[dict[str, Any]] = []
            for raw_page in pages:
                if not isinstance(raw_page, dict):
                    continue
                raw_facts = raw_page.get("facts") if isinstance(raw_page.get("facts"), list) else []
                projected_pages.append(
                    {
                        "image": raw_page.get("image"),
                        "meta": raw_page.get("meta") if isinstance(raw_page.get("meta"), dict) else {},
                        "facts": [
                            {
                                "bbox": fact.get("bbox"),
                                "value": fact.get("value"),
                            }
                            for fact in raw_facts
                            if isinstance(fact, dict) and fact.get("bbox") is not None
                        ],
                    }
                )
            transformed.append(
                {
                    **raw_example,
                    "expected_json": json.dumps({"pages": projected_pages}, ensure_ascii=False, separators=(",", ":")),
                }
            )
        if invalid_count:
            self.host.statusBar().showMessage(
                f"BBoxes + Values few-shot skipped {invalid_count} invalid example(s).",
                6000,
            )
        return transformed or None

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

    def _load_qwen_bbox_only_few_shot_examples(self, request: AIWorkflowRequest) -> Optional[list[dict[str, Any]]]:
        loaded_examples = self._load_qwen_few_shot_examples(request)
        if not loaded_examples:
            return None
        transformed: list[dict[str, Any]] = []
        invalid_count = 0
        for raw_example in loaded_examples:
            if not isinstance(raw_example, dict):
                invalid_count += 1
                continue
            expected_json = str(raw_example.get("expected_json") or "").strip()
            if not expected_json:
                invalid_count += 1
                continue
            try:
                import json

                payload = json.loads(expected_json)
            except Exception:
                invalid_count += 1
                continue
            pages = payload.get("pages") if isinstance(payload, dict) else None
            if not isinstance(pages, list):
                invalid_count += 1
                continue
            projected_pages: list[dict[str, Any]] = []
            for raw_page in pages:
                if not isinstance(raw_page, dict):
                    continue
                raw_facts = raw_page.get("facts") if isinstance(raw_page.get("facts"), list) else []
                projected_pages.append(
                    {
                        "image": raw_page.get("image"),
                        "meta": raw_page.get("meta") if isinstance(raw_page.get("meta"), dict) else {},
                        "facts": [
                            {
                                "bbox": fact.get("bbox"),
                                "value": fact.get("value"),
                            }
                            for fact in raw_facts
                            if isinstance(fact, dict) and fact.get("bbox") is not None
                        ],
                    }
                )
            transformed.append(
                {
                    **raw_example,
                    "expected_json": json.dumps({"pages": projected_pages}, ensure_ascii=False, separators=(",", ":")),
                }
            )
        if invalid_count:
            self.host.statusBar().showMessage(
                f"Qwen BBoxes + Values few-shot skipped {invalid_count} invalid example(s).",
                6000,
            )
        return transformed or None

    def _actions_for_provider(self, provider: AIProvider) -> list[AIActionKind]:
        if provider == AIProvider.DOCTR:
            return [AIActionKind.BBOX_ONLY]
        if provider == AIProvider.QWEN:
            return [AIActionKind.GROUND_TRUTH, AIActionKind.BBOX_ONLY]
        return [AIActionKind.GROUND_TRUTH, AIActionKind.BBOX_ONLY, AIActionKind.AUTO_COMPLETE, AIActionKind.FIX_SELECTED, AIActionKind.FIX_DRAWN]

    def _first_action_for_provider(self, provider: AIProvider) -> AIActionKind:
        return self._actions_for_provider(provider)[0]

    def _capabilities_for(self, provider: AIProvider, action: AIActionKind) -> AIActionCapabilities:
        if provider == AIProvider.DOCTR:
            return AIActionCapabilities(
                supports_max_facts=True,
                replaces_existing_page_facts=True,
            )
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
        if action == AIActionKind.BBOX_ONLY:
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
        if action == AIActionKind.FIX_DRAWN:
            return AIActionCapabilities(
                supports_thinking=True,
                supports_thinking_level=True,
                supports_temperature=True,
                requires_existing_facts=True,
                requires_hand_drawn_facts=True,
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
        if provider == AIProvider.DOCTR:
            return AIDialogDefaults(
                provider=provider,
                action=AIActionKind.BBOX_ONLY,
                model=LOCAL_DOCTR_MODEL_NAME,
                temperature=None,
                enable_thinking=False,
                thinking_level="minimal",
                use_few_shot=False,
                few_shot_preset=FEW_SHOT_PRESET_CLASSIC,
                max_facts=0,
                selected_fact_fields=default_fix_fields,
                include_statement_type=False,
            )
        if provider == AIProvider.QWEN:
            return AIDialogDefaults(
                provider=provider,
                action=action,
                model=self.host._qwen_model_name,
                temperature=None,
                enable_thinking=bool(self.host._qwen_enable_thinking),
                thinking_level="high" if self.host._qwen_enable_thinking else "minimal",
                use_few_shot=(action == AIActionKind.BBOX_ONLY),
                few_shot_preset=FEW_SHOT_PRESET_CLASSIC,
                max_facts=0,
                selected_fact_fields=default_fix_fields,
                include_statement_type=False,
            )
        if action == AIActionKind.GROUND_TRUTH:
            few_shot_default = FEW_SHOT_PRESET_2015_TWO_SHOT
            use_few_shot = False
        elif action == AIActionKind.BBOX_ONLY:
            few_shot_default = FEW_SHOT_PRESET_CLASSIC
            use_few_shot = True
        elif action == AIActionKind.AUTO_COMPLETE:
            few_shot_default = FEW_SHOT_PRESET_CLASSIC
            use_few_shot = False
        else:
            few_shot_default = FEW_SHOT_PRESET_CLASSIC
            use_few_shot = False
        model_name = self.host._gemini_model_name
        temperature = getattr(self.host, "_gemini_temperature", None)
        return AIDialogDefaults(
            provider=provider,
            action=action,
            model=model_name,
            temperature=temperature,
            enable_thinking=bool(self.host._gemini_enable_thinking),
            thinking_level=self.host._gemini_thinking_level,
            use_few_shot=use_few_shot,
            few_shot_preset=few_shot_default,
            max_facts=0,
            selected_fact_fields=default_fix_fields,
            include_statement_type=False,
        )

    def _model_choices_for(self, provider: AIProvider) -> list[str]:
        if provider == AIProvider.DOCTR:
            return [LOCAL_DOCTR_MODEL_NAME]
        if provider == AIProvider.QWEN:
            config_path = self.host._resolve_qwen_config_path()
            return list(current_qwen_gt_model_choices(str(config_path) if config_path is not None else None))
        from ..gemini_vlm import SUPPORTED_GEMINI_MODELS

        return list(SUPPORTED_GEMINI_MODELS)

    def _context_summary(self, context: Optional[AIPageContext]) -> str:
        if context is None:
            return "No page loaded."
        summary = (
            f"Page: {context.page_name} | "
            f"Existing facts: {context.existing_fact_count} | "
            f"Selected facts: {context.selected_fact_count}"
        )
        if context.hand_drawn_fact_nums:
            summary += f" | Hand-drawn: {len(context.hand_drawn_fact_nums)}"
        return summary

    def _validation_message(self, capabilities: AIActionCapabilities, context: Optional[AIPageContext]) -> str:
        if context is None:
            return "Load a page to run AI."
        if capabilities.requires_existing_facts and context.existing_fact_count <= 0:
            return "Auto Complete requires at least one existing fact on the current page."
        if capabilities.requires_selected_facts and context.selected_fact_count <= 0:
            return "Select one or more facts before running Fix."
        if capabilities.requires_hand_drawn_facts and not context.hand_drawn_fact_nums:
            return "Draw one or more bboxes on the page before running Fix Drawn."
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
        if provider == AIProvider.DOCTR:
            image_size = (
                f"{int(context.image_dimensions[0])} x {int(context.image_dimensions[1])} pixels"
                if context.image_dimensions is not None
                else "unknown"
            )
            return (
                "Local detector bbox-only mode:\n"
                f"- Current image size: {image_size}.\n"
                "- Runs the fine-tuned detector locally on the current page image.\n"
                "- Crop-recognizes detected boxes locally to fill `value` when possible.\n"
                "- Keeps empty values when crop recognition fails.\n"
                "- Filters out exact `31` and exact years from 2000 through 2026.\n"
                "- Emits only `bbox` and `value` per fact.\n"
                "- Uses original-image pixel coordinates in `[x, y, w, h]` format.\n"
                "- Prompt customization is informational only in this mode."
            )
        if provider == AIProvider.QWEN and action == AIActionKind.BBOX_ONLY:
            image_dimensions = context.image_dimensions
            prepared_dimensions: tuple[int, int] | None = None
            if image_dimensions is not None:
                prepared_h, prepared_w = prepared_dimensions_for_max_pixels(
                    original_width=float(image_dimensions[0]),
                    original_height=float(image_dimensions[1]),
                    max_pixels=DEFAULT_QWEN_BBOX_MAX_PIXELS,
                )
                prepared_dimensions = (int(prepared_w), int(prepared_h))
            return build_qwen_bbox_only_prompt(
                context.page_path,
                image_dimensions=image_dimensions,
                prepared_dimensions=prepared_dimensions,
                max_pixels=DEFAULT_QWEN_BBOX_MAX_PIXELS,
            )
        if provider == AIProvider.QWEN or action == AIActionKind.GROUND_TRUTH:
            return build_extraction_prompt(
                extraction_prompt_template(),
                context.page_path,
                image_dimensions=context.image_dimensions,
            )
        if action == AIActionKind.BBOX_ONLY:
            return build_gemini_bbox_only_prompt(
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
        if action == AIActionKind.FIX_DRAWN:
            request_payload = build_gemini_fix_drawn_request_payload(
                page_name=context.page_name,
                page_meta=context.page_meta,
                ordered_fact_payloads=context.ordered_fact_payloads,
                hand_drawn_fact_nums=context.hand_drawn_fact_nums,
            )
            return build_gemini_fix_drawn_prompt(
                request_payload=request_payload,
                hand_drawn_fact_nums=context.hand_drawn_fact_nums,
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
        if self.annotate_progress_dialog is not None:
            self.annotate_progress_dialog.set_status(text, fact_count=self._status_fact_count)
            self.annotate_progress_dialog.set_running(running)

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
        elif mode in {"gt", "bbox_only"}:
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

    def _start_auto_annotate_detector(self, page_name: str) -> None:
        page_idx = self.host._page_index_by_name(page_name)
        if page_idx < 0:
            self._on_local_doctr_failed("Target page is unavailable.")
            return
        self.host._auto_annotate_phase = "detector"
        self._start_local_doctr_worker(
            image_path=self.host.images_dir / page_name,
            page_name=page_name,
            backend=LocalDetectorBackend.MERGED,
            max_facts=0,
        )

    def _on_local_doctr_fact(self, fact_payload: dict[str, Any]) -> None:
        page_name = self.host._local_doctr_target_page
        if page_name is None:
            return
        added = self.host._apply_stream_fact(
            page_name,
            fact_payload,
            seen_facts=self.host._local_doctr_seen_facts,
            stream_source="local_doctr",
        )
        if added:
            self.host._local_doctr_fact_count += 1
            if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
                if getattr(self.host, "_auto_annotate_mode", "") == "align":
                    status = f"Align bboxes: detector parsed {self.host._local_doctr_fact_count} fact(s)..."
                else:
                    status = f"Auto-Annotate: detector parsed {self.host._local_doctr_fact_count} fact(s)..."
            else:
                status = f"Extracting local bbox+value facts... {self.host._local_doctr_fact_count} parsed"
            self._set_status(status, fact_count=self.host._local_doctr_fact_count, running=True)

    def _on_local_doctr_completed(self, extraction_obj: Any) -> None:
        page_name = self.host._local_doctr_target_page
        if page_name is None:
            return
        try:
            finalized_payloads: list[dict[str, Any]] = []
            for fact in getattr(extraction_obj, "facts", []) or []:
                payload = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact
                if not isinstance(payload, dict):
                    continue
                normalized_payload = self.host._normalized_stream_fact_payload(payload)
                if normalized_payload is None:
                    continue
                finalized_payloads.append(normalized_payload)
            self.host._local_doctr_buffered_facts = finalized_payloads
            self.host._local_doctr_seen_facts = {
                self.host._fact_uniqueness_key(payload)
                for payload in finalized_payloads
            }
            self.host._local_doctr_fact_count = len(finalized_payloads)

            page_idx = self.host._page_index_by_name(page_name)
            if page_idx < 0:
                self._on_local_doctr_failed("Target page is unavailable.")
                return
            state = self.host.page_states.get(page_name, self.host._default_state(page_idx))
            if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
                merged_payloads, match_debug = self.host._merge_auto_annotate_bbox_payloads(
                    page_name=page_name,
                    gemini_payloads=self.host._auto_annotate_gemini_payloads,
                    detector_payloads=finalized_payloads,
                )
                self.host._auto_annotate_last_match_debug = match_debug
                self.host.page_states[page_name] = PageState(
                    meta=dict(state.meta or {}),
                    facts=self.host._build_box_records_from_fact_payloads(merged_payloads),
                )
            else:
                self.host.page_states[page_name] = PageState(
                    meta=dict(state.meta or {}),
                    facts=self.host._build_box_records_from_fact_payloads(finalized_payloads),
                )
            self.host._apply_page_state_to_scene(page_name)
        except Exception as exc:
            self._on_local_doctr_failed(f"Final parse failed: {exc}")
            return

        self.host._record_history_snapshot()
        backend_label = local_detector_backend_label(self.host._local_doctr_backend)
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
            match_debug = getattr(self.host, "_auto_annotate_last_match_debug", {}) or {}
            matched_count = int(match_debug.get("matched_count") or 0)
            gemini_count = int(match_debug.get("gemini_count") or 0)
            if getattr(self.host, "_auto_annotate_mode", "") == "align":
                message = (
                    f"Align bboxes complete. Matched {matched_count}/{gemini_count} fact(s); "
                    f"detector parsed {self.host._local_doctr_fact_count} fact(s)."
                )
            else:
                message = f"Auto-Annotate complete. Detector parsed {self.host._local_doctr_fact_count} fact(s)."
        elif self.host._local_doctr_backend == LocalDetectorBackend.MERGED:
            message = f"Local detectors merged. Parsed {self.host._local_doctr_fact_count} fact(s)."
        else:
            message = (
                f"Local detector ({backend_label}) BBoxes + Values complete. "
                f"Parsed {self.host._local_doctr_fact_count} fact(s)."
            )
        self._set_status(message, fact_count=self.host._local_doctr_fact_count, running=False)
        self.host.statusBar().showMessage(message, 6000)
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
            if getattr(self.host, "_auto_annotate_mode", "") == "document":
                final_state = self.host.page_states.get(page_name, self.host._default_state(page_idx))
                self._document_auto_annotate_pending_completion = (page_name, len(final_state.facts))

    def _on_local_doctr_failed(self, message: str) -> None:
        self._document_auto_annotate_pending_completion = None
        backend_label = local_detector_backend_label(self.host._local_doctr_backend)
        self._set_status(
            f"Error ({backend_label}): {message}",
            fact_count=self.host._local_doctr_fact_count,
            running=False,
        )
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
            if getattr(self.host, "_auto_annotate_mode", "") == "document":
                page_name = self.host._auto_annotate_target_page or self.host._local_doctr_target_page or ""
                self.host._record_history_snapshot()
                self.host.document_auto_annotate_page_failed.emit(page_name, message)
                return
            title = "Align bboxes warning" if getattr(self.host, "_auto_annotate_mode", "") == "align" else "Auto-Annotate warning"
            kept_text = (
                "Kept the current annotations on the page."
                if getattr(self.host, "_auto_annotate_mode", "") == "align"
                else "Kept Gemini ground truth results on the page."
            )
            self.host._record_history_snapshot()
            QMessageBox.warning(
                self.host,
                title,
                f"Local detector ({backend_label}) failed:\n{message}\n\n{kept_text}",
            )
            return
        QMessageBox.warning(
            self.host,
            "AI failed",
            f"Local detector ({backend_label}) failed:\n{message}\n\nNo buffered facts were applied to the page.",
        )

    def _on_local_doctr_finished(self) -> None:
        pending_completion = self._document_auto_annotate_pending_completion
        self._document_auto_annotate_pending_completion = None
        if self.host._local_doctr_cancel_requested:
            backend_label = local_detector_backend_label(self.host._local_doctr_backend)
            if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
                self.host._record_history_snapshot()
                auto_mode = getattr(self.host, "_auto_annotate_mode", "")
                if auto_mode == "align":
                    stopped = (
                        f"Align bboxes stopped during detector step "
                        f"({self.host._local_doctr_fact_count} fact(s) parsed, no changes applied)."
                    )
                else:
                    stopped = (
                        f"Auto-Annotate stopped during detector step "
                        f"({self.host._local_doctr_fact_count} fact(s) parsed, Gemini GT kept)."
                    )
            elif self.host._local_doctr_backend == LocalDetectorBackend.MERGED:
                stopped = (
                    f"Local detectors stopped ({self.host._local_doctr_fact_count} fact(s) parsed, "
                    "no changes applied)."
                )
            else:
                stopped = (
                    f"Local detector ({backend_label}) BBoxes + Values stopped "
                    f"({self.host._local_doctr_fact_count} fact(s) parsed, "
                    "no changes applied)."
                    )
            self._set_status(stopped, fact_count=self.host._local_doctr_fact_count, running=False)
            self.host.statusBar().showMessage(stopped, 5000)
            if (
                self.host._auto_annotate_active
                and self.host._auto_annotate_phase == "detector"
                and getattr(self.host, "_auto_annotate_mode", "") == "document"
            ):
                page_name = self.host._auto_annotate_target_page or ""
                self.host.document_auto_annotate_page_stopped.emit(page_name, stopped)
        self.host._local_doctr_thread = None
        self.host._local_doctr_worker = None
        self.host._local_doctr_target_page = None
        self.host._local_doctr_backend = LocalDetectorBackend.MERGED
        self.host._local_doctr_seen_facts = set()
        self.host._local_doctr_buffered_facts = []
        self.host._local_doctr_cancel_requested = False
        self.host._auto_annotate_last_match_debug = {}
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "detector":
            self.host._reset_auto_annotate_state()
        if pending_completion is not None and not self.host._local_doctr_cancel_requested:
            page_name, fact_count = pending_completion
            self.host.document_auto_annotate_page_completed.emit(page_name, fact_count)
        self.refresh_dialog_state()

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
        if self.host._gemini_stream_apply_meta:
            status = "Meta received. Streaming facts..."
        elif self.host._auto_annotate_active and self.host._auto_annotate_phase == "gemini":
            if getattr(self.host, "_auto_annotate_mode", "") == "annotate":
                status = "Annotate: streaming Gemini ground truth..."
            else:
                status = "Auto-Annotate: streaming Gemini ground truth..."
        elif self.host._gemini_stream_mode == "gt":
            status = "Streaming facts..."
        else:
            status = "Locked context loaded. Streaming missing facts..."
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
            if self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_updates_enabled:
                self.host._update_gemini_gt_live_stream(page_name)
            progress_text = (
                f"Streaming missing facts... {self.host._gemini_stream_fact_count} parsed"
                if self.host._gemini_stream_mode == "autocomplete"
                else f"Extracting bbox+value facts... {self.host._gemini_stream_fact_count} parsed"
                if self.host._gemini_stream_mode == "bbox_only"
                else f"Annotate: {self.host._gemini_stream_fact_count} parsed"
                if self.host._auto_annotate_active
                and self.host._auto_annotate_phase == "gemini"
                and getattr(self.host, "_auto_annotate_mode", "") == "annotate"
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
            if self.host._gemini_stream_mode in {"autocomplete", "gt", "bbox_only"}:
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
                if self.host._auto_annotate_active and self.host._auto_annotate_phase == "gemini":
                    if getattr(self.host, "_auto_annotate_mode", "") == "annotate":
                        merged, error_message = self.host._merge_gemini_gt_buffered_facts(page_name)
                        if not merged:
                            self._on_gemini_stream_failed(error_message or "Ground Truth could not merge results.")
                            return
                        self.host._apply_stream_meta(page_name, meta_payload)
                        count = self.host._gemini_stream_fact_count
                        self.host._auto_annotate_restore_on_finish = False
                        self.host._reset_auto_annotate_state()
                        self.host._record_history_snapshot()
                        msg = f"Annotate complete. Parsed {count} fact(s)."
                        self._set_status(msg, fact_count=count, running=False)
                        self.host.statusBar().showMessage(msg, 6000)
                        return
                    resolved_payloads, bbox_mode = self.host._resolve_autocomplete_bbox_mode(
                        page_name,
                        self.host._gemini_gt_buffered_facts,
                    )
                    self.host._gemini_gt_last_bbox_mode = bbox_mode
                    self.host._gemini_gt_last_bbox_scores = dict(
                        self.host._gemini_autocomplete_last_bbox_scores
                        or {
                            BBOX_MODE_PIXEL_AS_IS: 0.0,
                            BBOX_MODE_NORMALIZED_1000_TO_PIXEL: 0.0,
                        }
                    )
                    self.host._auto_annotate_gemini_payloads = [deepcopy(payload) for payload in resolved_payloads]
                    self.host._apply_stream_meta(page_name, meta_payload)
                    merged_records = self.host._build_box_records_from_fact_payloads(resolved_payloads)
                    page_idx = self.host._page_index_by_name(page_name)
                    state = self.host.page_states.get(page_name, self.host._default_state(page_idx))
                    self.host.page_states[page_name] = PageState(meta=dict(state.meta or {}), facts=merged_records)
                    self.host._gemini_stream_fact_count = len(merged_records)
                    self.host._gemini_gt_live_applied = False
                    self.host._apply_page_state_to_scene(page_name)
                    self._start_auto_annotate_detector(page_name)
                    return
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
            elif self.host._gemini_stream_mode == "bbox_only":
                merged, error_message = self.host._merge_gemini_gt_buffered_facts(page_name)
                if not merged:
                    self._on_gemini_stream_failed(error_message or "BBoxes + Values could not merge results.")
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
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "gemini":
            page_name = self.host._auto_annotate_target_page
            if page_name is not None and getattr(self.host, "_auto_annotate_restore_on_finish", False):
                self.host._restore_page_state_snapshot(page_name, self.host._auto_annotate_original_state)
            auto_mode = getattr(self.host, "_auto_annotate_mode", "")
            if auto_mode == "document":
                self.host.document_auto_annotate_page_failed.emit(page_name or "", message)
                return
            title = "Annotate failed" if auto_mode == "annotate" else "Auto-Annotate failed"
        summary, session_dir = self._current_gemini_stream_issue_summary()
        if summary is not None:
            detail = format_issue_summary_brief(
                summary,
                session_dir=session_dir,
                streamed_live_facts_remain=(self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied),
                no_changes_applied=self.host._gemini_stream_mode in {"autocomplete", "gt", "bbox_only"} and not self.host._gemini_gt_live_applied,
            )
        elif self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied:
            detail = f"{message}\n\nSome streamed facts were rendered live and remain on the page."
        elif self.host._gemini_stream_mode in {"autocomplete", "gt", "bbox_only"}:
            detail = f"{message}\n\nNo changes were applied."
        else:
            detail = f"{message}\n\nAny facts already streamed remain on the page."
        QMessageBox.warning(self.host, title, detail)

    def _on_gemini_stream_finished(self) -> None:
        if self.host._gemini_stream_cancel_requested and not self.host._gemini_stream_limit_reached:
            if (
                self.host._auto_annotate_active
                and self.host._auto_annotate_phase == "gemini"
                and getattr(self.host, "_auto_annotate_mode", "") == "annotate"
            ):
                stopped_message = (
                    f"Annotate stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, no changes applied)."
                )
            elif self.host._gemini_stream_mode == "gt" and self.host._gemini_gt_live_applied:
                stopped_message = (
                    f"Ground Truth stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, live-streamed facts kept)."
                )
            elif self.host._gemini_stream_mode == "gt":
                stopped_message = (
                    f"Ground Truth stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, no changes applied)."
                )
            elif self.host._gemini_stream_mode == "bbox_only":
                stopped_message = (
                    f"BBoxes + Values stopped ({self.host._gemini_stream_fact_count} fact(s) parsed, no changes applied)."
                )
            else:
                stopped_message = f"Auto Complete stopped ({self.host._gemini_stream_fact_count} fact(s) parsed)."
            self._set_status(stopped_message, fact_count=self.host._gemini_stream_fact_count, running=False)
            self.host.statusBar().showMessage(stopped_message, 5000)
            if (
                self.host._auto_annotate_active
                and self.host._auto_annotate_phase == "gemini"
                and getattr(self.host, "_auto_annotate_mode", "") == "document"
            ):
                page_name = self.host._auto_annotate_target_page or ""
                self.host.document_auto_annotate_page_stopped.emit(page_name, stopped_message)
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
        self.host._gemini_gt_live_updates_enabled = True
        self._gemini_gt_finalize_retained_live = False
        self._gemini_gt_finalize_retained_counts = None
        if self.host._auto_annotate_active and self.host._auto_annotate_phase == "gemini":
            page_name = self.host._auto_annotate_target_page
            if page_name is not None and getattr(self.host, "_auto_annotate_restore_on_finish", False):
                self.host._restore_page_state_snapshot(page_name, self.host._auto_annotate_original_state)
            self.host._reset_auto_annotate_state()
        if self.annotate_progress_dialog is not None:
            self.annotate_progress_dialog.hide()
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
        if self.host._gemini_stream_mode == "bbox_only":
            text = (
                f"BBoxes + Values complete. Parsed {self.host._gemini_stream_fact_count} fact(s) "
                f"(bbox mode: {bbox_mode_label}, pixel={pixel_score:.3f}, normalized={normalized_score:.3f})."
            )
            return text, text
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
        if self.host._qwen_stream_mode == "bbox_only":
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
                (
                    f"Extracting bbox+value facts... {self.host._qwen_stream_fact_count} parsed"
                    if self.host._qwen_stream_mode == "bbox_only"
                    else f"Streaming facts... {self.host._qwen_stream_fact_count} parsed"
                ),
                fact_count=self.host._qwen_stream_fact_count,
                running=True,
            )

    def _on_qwen_stream_completed(self, extraction_obj: Any) -> None:
        page_name = self.host._qwen_stream_target_page
        if page_name is None:
            return
        try:
            if self.host._qwen_stream_mode != "bbox_only":
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
            else:
                finalized_payloads: list[dict[str, Any]] = []
                for fact in getattr(extraction_obj, "facts", []) or []:
                    payload = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact
                    if not isinstance(payload, dict):
                        continue
                    normalized_payload = self.host._normalized_stream_fact_payload(payload)
                    if normalized_payload is None:
                        continue
                    finalized_payloads.append(normalized_payload)
                self.host._qwen_bbox_buffered_facts = finalized_payloads
                self.host._qwen_stream_seen_facts = {
                    self.host._fact_uniqueness_key(payload)
                    for payload in finalized_payloads
                }
                self.host._qwen_stream_fact_count = len(finalized_payloads)
                merged, error_message = self.host._merge_qwen_bbox_buffered_facts(page_name)
                if not merged:
                    self._on_qwen_stream_failed(error_message or "Qwen BBoxes + Values could not merge results.")
                    return
        except Exception as exc:
            self._on_qwen_stream_failed(f"Final parse failed: {exc}")
            return

        self.host._record_history_snapshot()
        message = (
            f"Qwen BBoxes + Values complete. Parsed {self.host._qwen_stream_fact_count} fact(s)."
            if self.host._qwen_stream_mode == "bbox_only"
            else f"Qwen GT complete. Parsed {self.host._qwen_stream_fact_count} fact(s)."
        )
        self._set_status(message, fact_count=self.host._qwen_stream_fact_count, running=False)
        self.host.statusBar().showMessage(message, 6000)

    def _on_qwen_stream_failed(self, message: str) -> None:
        self._set_status(f"Error: {message}", fact_count=self.host._qwen_stream_fact_count, running=False)
        details = (
            "No buffered facts were applied to the page."
            if self.host._qwen_stream_mode == "bbox_only"
            else "Any facts already streamed remain on the page."
        )
        QMessageBox.warning(self.host, "AI failed", f"{message}\n\n{details}")

    def _on_qwen_stream_finished(self) -> None:
        if self.host._qwen_stream_cancel_requested:
            stopped = (
                f"Qwen BBoxes + Values stopped ({self.host._qwen_stream_fact_count} fact(s) parsed, no changes applied)."
                if self.host._qwen_stream_mode == "bbox_only"
                else f"Qwen GT stopped ({self.host._qwen_stream_fact_count} fact(s) parsed)."
            )
            self._set_status(stopped, fact_count=self.host._qwen_stream_fact_count, running=False)
            self.host.statusBar().showMessage(stopped, 5000)
        self.host._qwen_stream_thread = None
        self.host._qwen_stream_worker = None
        self.host._qwen_stream_target_page = None
        self.host._qwen_stream_mode = "gt"
        self.host._qwen_bbox_buffered_facts = []
        self.host._qwen_bbox_original_size = None
        self.host._qwen_bbox_prepared_size = None
        self.host._qwen_bbox_max_pixels = None
        self.refresh_dialog_state()


__all__ = ["AIWorkflowController"]
