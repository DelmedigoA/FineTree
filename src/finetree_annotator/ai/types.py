from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AIProvider(str, Enum):
    DOCTR = "doctr"
    GEMINI = "gemini"
    QWEN = "qwen"


class AIActionKind(str, Enum):
    GROUND_TRUTH = "ground_truth"
    BBOX_ONLY = "bbox_only"
    AUTO_COMPLETE = "auto_complete"
    FIX_SELECTED = "fix_selected"
    FIX_DRAWN = "fix_drawn"


class LocalDetectorBackend(str, Enum):
    FINE_TUNED = "fine_tuned"
    DOCTR_PRETRAINED = "doctr_pretrained"
    MERGED = "merged"


FEW_SHOT_PRESET_ONE_SHOT = "test_1"
FEW_SHOT_PRESET_2015_TWO_SHOT = "2015_2"
FEW_SHOT_PRESET_CLASSIC = "classic_4"
FEW_SHOT_PRESET_EXTENDED = "extended_7"

FEW_SHOT_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    (FEW_SHOT_PRESET_ONE_SHOT, "Test 1-shot"),
    (FEW_SHOT_PRESET_2015_TWO_SHOT, "2015 2-shot"),
    (FEW_SHOT_PRESET_CLASSIC, "Classic 4-shot"),
    (FEW_SHOT_PRESET_EXTENDED, "Extended 7-shot"),
)

FEW_SHOT_PRESET_SUMMARY: dict[str, str] = {
    FEW_SHOT_PRESET_ONE_SHOT: "test(1): page_num 4",
    FEW_SHOT_PRESET_2015_TWO_SHOT: "2015(2): pages 4,11",
    FEW_SHOT_PRESET_CLASSIC: "classic(4): test 1,4,9,2",
    FEW_SHOT_PRESET_EXTENDED: "extended(7): test 9,4,5,10 | pdf_3 18,23 | pdf_2 8",
}

FEW_SHOT_PRESET_HELP_TEXT = " | ".join(
    FEW_SHOT_PRESET_SUMMARY.get(preset_id, preset_id)
    for preset_id, _label in FEW_SHOT_PRESET_CHOICES
)


@dataclass(frozen=True)
class AIActionCapabilities:
    supports_thinking: bool = False
    supports_thinking_level: bool = False
    supports_few_shot: bool = False
    supports_temperature: bool = False
    supports_max_facts: bool = False
    supports_fix_fields: bool = False
    supports_statement_type_toggle: bool = False
    requires_existing_facts: bool = False
    requires_selected_facts: bool = False
    requires_hand_drawn_facts: bool = False
    replaces_existing_page_facts: bool = False


@dataclass(frozen=True)
class AIPageContext:
    page_path: Path
    page_name: str
    page_index: int
    page_meta: dict[str, Any]
    ordered_fact_payloads: list[dict[str, Any]]
    ordered_fact_signature: list[tuple[Any, ...]]
    selected_fact_nums: list[int]
    existing_fact_count: int
    selected_fact_count: int
    image_dimensions: Optional[tuple[float, float]]
    hand_drawn_fact_nums: tuple[int, ...] = ()


@dataclass
class AIWorkflowRequest:
    provider: AIProvider
    action: AIActionKind
    model: str
    prompt_text: str
    temperature: Optional[float] = None
    enable_thinking: bool = False
    thinking_level: str = "minimal"
    use_few_shot: bool = False
    few_shot_preset: str = ""
    max_facts: int = 0
    selected_fact_fields: set[str] = field(default_factory=set)
    include_statement_type: bool = False
    local_detector_backend: LocalDetectorBackend = LocalDetectorBackend.MERGED


@dataclass(frozen=True)
class AIDialogDefaults:
    provider: AIProvider
    action: AIActionKind
    model: str
    temperature: Optional[float]
    enable_thinking: bool
    thinking_level: str
    use_few_shot: bool
    few_shot_preset: str
    max_facts: int
    selected_fact_fields: set[str]
    include_statement_type: bool


def action_label(action: AIActionKind) -> str:
    if action == AIActionKind.GROUND_TRUTH:
        return "Ground Truth"
    if action == AIActionKind.BBOX_ONLY:
        return "BBoxes + Values"
    if action == AIActionKind.AUTO_COMPLETE:
        return "Auto Complete"
    if action == AIActionKind.FIX_DRAWN:
        return "Fix Drawn"
    return "Fix"


def provider_label(provider: AIProvider) -> str:
    if provider == AIProvider.DOCTR:
        return "Local detector"
    if provider == AIProvider.GEMINI:
        return "Gemini"
    return "Qwen"


def local_detector_backend_label(backend: LocalDetectorBackend) -> str:
    if backend == LocalDetectorBackend.MERGED:
        return "stock + fine-tuned"
    if backend == LocalDetectorBackend.DOCTR_PRETRAINED:
        return "stock docTR"
    return "fine-tuned"


__all__ = [
    "AIActionCapabilities",
    "AIActionKind",
    "AIDialogDefaults",
    "AIPageContext",
    "AIProvider",
    "AIWorkflowRequest",
    "FEW_SHOT_PRESET_2015_TWO_SHOT",
    "FEW_SHOT_PRESET_CHOICES",
    "FEW_SHOT_PRESET_CLASSIC",
    "FEW_SHOT_PRESET_EXTENDED",
    "FEW_SHOT_PRESET_HELP_TEXT",
    "FEW_SHOT_PRESET_ONE_SHOT",
    "FEW_SHOT_PRESET_SUMMARY",
    "LocalDetectorBackend",
    "action_label",
    "local_detector_backend_label",
    "provider_label",
]
