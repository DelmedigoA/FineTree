from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from .fact_ordering import normalize_document_meta
from .schemas import PageExtraction

DEFAULT_QWEN_STREAM_MAX_NEW_TOKENS = 10_000


class GeminiStreamWorker(QObject):
    chunk_received = pyqtSignal(str)
    meta_received = pyqtSignal(dict)
    fact_received = pyqtSignal(dict)
    limit_reached = pyqtSignal()
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        image_path: Path,
        prompt: str,
        model: str,
        mode: str = "gt",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        thinking_level: Optional[str] = None,
        max_facts: int = 0,
        allow_partial_finalize_error: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.mode = str(mode or "gt").strip().lower()
        self.api_key = api_key
        self.system_prompt = str(system_prompt or "").strip() or None
        self.few_shot_examples = few_shot_examples
        self.temperature = float(temperature) if temperature is not None else None
        self.enable_thinking = bool(enable_thinking)
        self.thinking_level = str(thinking_level).strip().lower() if thinking_level is not None else None
        self.max_facts = max(0, int(max_facts))
        self.allow_partial_finalize_error = bool(allow_partial_finalize_error)
        self._cancel_requested = False
        self._limit_reached = False
        self._latest_meta_payload: Optional[dict[str, Any]] = None
        self._emitted_facts: list[dict[str, Any]] = []
        self.session_dir: Optional[Path] = None
        self.issue_summary: Optional[dict[str, Any]] = None

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _build_lenient_stream_result(self) -> Any:
        meta_payload = self._latest_meta_payload or {
            "entity_name": None,
            "page_num": None,
            "page_type": "other",
            "statement_type": None,
            "title": None,
        }
        meta_obj = SimpleNamespace(model_dump=lambda mode="json", _meta=deepcopy(meta_payload): deepcopy(_meta))
        fact_objs = [
            SimpleNamespace(model_dump=lambda mode="json", _fact=deepcopy(fact): deepcopy(_fact))
            for fact in self._emitted_facts
        ]
        return SimpleNamespace(meta=meta_obj, facts=fact_objs)

    def _build_partial_extraction(self) -> Any:
        meta_payload = self._latest_meta_payload or {
            "entity_name": None,
            "page_num": None,
            "page_type": "other",
            "statement_type": None,
            "title": None,
        }
        try:
            return PageExtraction.model_validate(
                {
                    "images_dir": str(self.image_path.parent),
                    "metadata": normalize_document_meta({}),
                    "pages": [
                        {
                            "image": self.image_path.name,
                            "meta": meta_payload,
                            "facts": self._emitted_facts,
                        }
                    ],
                }
            )
        except Exception:
            return self._build_lenient_stream_result()

    def run(self) -> None:
        try:
            from .gemini_vlm import (
                StreamingBBoxOnlyParser,
                StreamingPageExtractionParser,
                _create_gemini_log_session,
                load_issue_summary,
                resolve_supported_gemini_model_name,
                stream_content_from_image,
            )

            resolved_model = resolve_supported_gemini_model_name(self.model)
            self.session_dir = _create_gemini_log_session(
                operation="stream_content",
                model=resolved_model,
                image_path=self.image_path,
                prompt=self.prompt,
                mime_type=None,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                thinking_level=self.thinking_level,
                few_shot_examples=self.few_shot_examples,
            )
            parser = (
                StreamingBBoxOnlyParser(image_path=self.image_path, session_dir=self.session_dir)
                if self.mode == "bbox_only"
                else StreamingPageExtractionParser(image_path=self.image_path, session_dir=self.session_dir)
            )
            for chunk in stream_content_from_image(
                image_path=self.image_path,
                prompt=self.prompt,
                model=resolved_model,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                few_shot_examples=self.few_shot_examples,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                thinking_level=self.thinking_level,
                response_mime_type="application/json" if self.mode in {"gt", "bbox_only"} else None,
                media_resolution="high" if self.mode in {"gt", "bbox_only"} else None,
                session_dir=self.session_dir,
            ):
                if self._cancel_requested:
                    break
                if not chunk:
                    continue
                self.chunk_received.emit(chunk)
                meta, facts = parser.feed(chunk)
                if meta is not None:
                    self._latest_meta_payload = meta
                    self.meta_received.emit(meta)
                for fact in facts:
                    if self._cancel_requested:
                        break
                    self.fact_received.emit(fact)
                    self._emitted_facts.append(fact)
                    if self.max_facts > 0 and len(self._emitted_facts) >= self.max_facts:
                        self._limit_reached = True
                        self.limit_reached.emit()
                        self._cancel_requested = True
                        break

            if self._limit_reached:
                self.completed.emit(self._build_partial_extraction())
            elif not self._cancel_requested:
                try:
                    extraction = parser.finalize()
                except Exception:
                    if self.session_dir is not None:
                        self.issue_summary = load_issue_summary(self.session_dir)
                    if (
                        self.allow_partial_finalize_error
                        and self.issue_summary is None
                        and (self._latest_meta_payload is not None or self._emitted_facts)
                    ):
                        self.completed.emit(self._build_lenient_stream_result())
                    else:
                        raise
                else:
                    if self.session_dir is not None:
                        self.issue_summary = load_issue_summary(self.session_dir)
                    self.completed.emit(extraction)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class GeminiFillWorker(QObject):
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        image_path: Path,
        prompt: str,
        model: str,
        api_key: Optional[str],
        allowed_fact_fields: set[str],
        allow_statement_type: bool,
        enable_thinking: bool,
        thinking_level: Optional[str] = None,
        temperature: Optional[float] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.allowed_fact_fields = set(allowed_fact_fields)
        self.allow_statement_type = bool(allow_statement_type)
        self.temperature = float(temperature) if temperature is not None else None
        self.enable_thinking = bool(enable_thinking)
        self.thinking_level = str(thinking_level).strip().lower() if thinking_level is not None else None
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            from .gemini_vlm import generate_content_from_image, parse_selected_field_patch_text

            raw_text = generate_content_from_image(
                image_path=self.image_path,
                prompt=self.prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                thinking_level=self.thinking_level,
            )
            if self._cancel_requested:
                return
            parsed = parse_selected_field_patch_text(
                raw_text,
                allowed_fact_fields=self.allowed_fact_fields,
                allow_statement_type=self.allow_statement_type,
            )
            if self._cancel_requested:
                return
            self.completed.emit(parsed)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


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
        max_new_tokens: int = DEFAULT_QWEN_STREAM_MAX_NEW_TOKENS,
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

            parser = StreamingPageExtractionParser(image_path=self.image_path)
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


__all__ = [
    "DEFAULT_QWEN_STREAM_MAX_NEW_TOKENS",
    "GeminiFillWorker",
    "GeminiStreamWorker",
    "QwenStreamWorker",
]
