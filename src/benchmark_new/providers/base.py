from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..models import DocumentInput, ProgressSnapshot, ProviderRunOutput


@dataclass(frozen=True)
class ProviderOptions:
    provider: str
    model: str
    max_pixels: int = 1_400_000
    max_tokens: int = 24_000
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    instruction: str | None = None
    enable_thinking: bool = False
    thinking_level: str | None = None


ProgressCallback = Callable[[ProgressSnapshot], None]


class ProviderRunner(Protocol):
    def __call__(
        self,
        documents: tuple[DocumentInput, ...],
        *,
        options: ProviderOptions,
        run_dir: Path,
        dataset_version_id: str | None,
        dataset_name: str | None,
        split: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ProviderRunOutput: ...


def get_provider_runner(provider_name: str) -> ProviderRunner:
    normalized = str(provider_name or "").strip().lower()
    if normalized == "finetree_vllm":
        from .finetree_vllm import run_finetree_vllm_inference

        return run_finetree_vllm_inference
    if normalized == "gemini":
        from .gemini import run_gemini_inference

        return run_gemini_inference
    raise ValueError(f"Unsupported provider: {provider_name}")
