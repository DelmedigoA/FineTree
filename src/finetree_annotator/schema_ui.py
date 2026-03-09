from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .schema_registry import SchemaRegistry


@dataclass(frozen=True)
class UiFieldDescriptor:
    model: str
    key: str
    label: str
    help_text: str
    options: tuple[str, ...]


def _field_enum_options(model_name: str, field_name: str) -> tuple[str, ...]:
    model_spec = SchemaRegistry.get_model_spec(model_name)
    for field in model_spec.fields:
        if field.key == field_name:
            return tuple(str(value) for value in field.enum_values)
    return ()


def enum_options(model_name: str, field_name: str, *, include_blank: bool = True) -> tuple[str, ...]:
    values = _field_enum_options(model_name, field_name)
    if include_blank:
        return ("", *values)
    return values


@lru_cache(maxsize=1)
def ui_descriptors() -> dict[str, UiFieldDescriptor]:
    keys: list[tuple[str, str]] = [
        ("page_meta", "statement_type"),
        ("metadata", "entity_type"),
        ("metadata", "report_scope"),
        ("fact", "period_type"),
        ("fact", "duration_type"),
        ("fact", "recurring_period"),
        ("fact", "value_type"),
        ("fact", "value_context"),
        ("fact", "balance_type"),
        ("fact", "row_role"),
        ("fact", "aggregation_role"),
        ("fact", "path_source"),
    ]
    out: dict[str, UiFieldDescriptor] = {}
    for model_name, field_name in keys:
        model_spec = SchemaRegistry.get_model_spec(model_name)
        field_spec = next((field for field in model_spec.fields if field.key == field_name), None)
        if field_spec is None:
            continue
        descriptor = UiFieldDescriptor(
            model=model_name,
            key=field_name,
            label=field_spec.label,
            help_text=field_spec.description,
            options=enum_options(model_name, field_name, include_blank=True),
        )
        out[f"{model_name}.{field_name}"] = descriptor
    return out


def schema_ui_snapshot() -> dict[str, Any]:
    descriptors = ui_descriptors()
    return {
        key: {
            "model": descriptor.model,
            "field": descriptor.key,
            "label": descriptor.label,
            "help_text": descriptor.help_text,
            "options": list(descriptor.options),
        }
        for key, descriptor in descriptors.items()
    }


__all__ = [
    "UiFieldDescriptor",
    "enum_options",
    "schema_ui_snapshot",
    "ui_descriptors",
]
