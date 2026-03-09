from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, get_args, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic.fields import FieldInfo

from .schemas import CURRENT_SCHEMA_VERSION, Document, Fact, Metadata, PageMeta
LEGACY_COMPAT_MIN_VERSION = 1
LEGACY_COMPAT_MAX_VERSION = 2
SCHEMA_MODEL_MAP: dict[str, type[BaseModel]] = {
    "metadata": Metadata,
    "page_meta": PageMeta,
    "fact": Fact,
    "document": Document,
}
PROMPT_RUNTIME_OWNED_KEYS: tuple[str, ...] = ("schema_version",)
PROMPT_RUNTIME_OWNED_PAGE_META_KEYS: tuple[str, ...] = ("annotation_note", "annotation_status")
PROMPT_RUNTIME_OWNED_FACT_KEYS: tuple[str, ...] = ("fact_num", "fact_equation")
PROMPT_REQUIRED_FACT_KEYS: tuple[str, ...] = (
    "equation",
    "balance_type",
    "natural_sign",
    "row_role",
    "aggregation_role",
    "comment_ref",
    "note_flag",
    "note_name",
    "note_num",
    "note_ref",
    "period_type",
    "period_start",
    "period_end",
    "path_source",
)

_FIELD_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("metadata", "language"): "Document language",
    ("metadata", "reading_direction"): "Document reading direction",
    ("metadata", "company_name"): "Legal entity name",
    ("metadata", "company_id"): "Entity registration ID",
    ("metadata", "report_year"): "Report year",
    ("metadata", "entity_type"): "Entity category",
    ("metadata", "report_scope"): "Scope of reported values",
    ("page_meta", "entity_name"): "Entity name on page",
    ("page_meta", "page_num"): "Printed page number",
    ("page_meta", "page_type"): "Structural page classification",
    ("page_meta", "statement_type"): "Statement/report classification",
    ("page_meta", "title"): "Page title",
    ("page_meta", "annotation_note"): "Annotation-only note to revisit the page later",
    ("page_meta", "annotation_status"): "Annotation-only page decision status",
    ("fact", "value"): "Extracted value text",
    ("fact", "fact_num"): "Stable persisted fact order number",
    ("fact", "equation"): "Optional arithmetic expression tied to the fact",
    ("fact", "fact_equation"): "Optional arithmetic expression referencing fact numbers",
    ("fact", "balance_type"): "Accounting balance side (debit/credit)",
    ("fact", "natural_sign"): "Sign implied by the value text",
    ("fact", "row_role"): "Whether row is a detail line or a subtotal/total line",
    ("fact", "aggregation_role"): "Aggregation role for equation polarity",
    ("fact", "value_type"): "Value semantic type",
    ("fact", "currency"): "Currency code",
    ("fact", "scale"): "Scale multiplier",
    ("fact", "date"): "Observed date token",
    ("fact", "value_context"): "Whether value is textual, tabular, or mixed",
    ("fact", "period_type"): "Period semantic type",
    ("fact", "period_start"): "Period start (YYYY-MM-DD)",
    ("fact", "period_end"): "Period end (YYYY-MM-DD)",
    ("fact", "duration_type"): "Duration recurrence type",
    ("fact", "recurring_period"): "Cadence for recurring durations",
    ("fact", "note_flag"): "Whether fact is a note item",
    ("fact", "note_num"): "Note number",
    ("fact", "note_name"): "Note title",
    ("fact", "path"): "Hierarchy path labels",
    ("fact", "path_source"): "Path derivation source",
    ("fact", "note_ref"): "Fact note reference",
    ("fact", "comment_ref"): "Fact comment reference",
}


@dataclass(frozen=True)
class FieldSpec:
    key: str
    type_name: str
    nullable: bool
    required: bool
    default: Any
    enum_values: tuple[Any, ...]
    read_aliases: tuple[str, ...]
    label: str
    description: str


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_qualname: str
    fields: tuple[FieldSpec, ...]
    canonical_write_keys: tuple[str, ...]
    read_alias_keys: tuple[str, ...]


def _infer_type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation)
    return str(origin).replace("typing.", "")


def _extract_nullable(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = get_args(annotation)
    return type(None) in args


def _extract_enum_values(annotation: Any) -> tuple[Any, ...]:
    enum_cls = _extract_enum_cls(annotation)
    if enum_cls is None:
        return ()
    return tuple(member.value for member in enum_cls)


def _extract_enum_cls(annotation: Any) -> Optional[type[Enum]]:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, Enum):
            return arg
    return None


def _aliases_from_validation_alias(validation_alias: Any) -> tuple[str, ...]:
    if validation_alias is None:
        return ()
    if isinstance(validation_alias, str):
        return (validation_alias,)
    if isinstance(validation_alias, AliasChoices):
        aliases: list[str] = []
        for choice in validation_alias.choices:
            aliases.extend(list(_aliases_from_validation_alias(choice)))
        return tuple(dict.fromkeys(aliases))
    if isinstance(validation_alias, AliasPath):
        if len(validation_alias.path) == 1 and isinstance(validation_alias.path[0], str):
            return (validation_alias.path[0],)
        return ()
    return ()


def _default_value(field: FieldInfo) -> Any:
    if field.is_required():
        return None
    if field.default_factory is not None:
        try:
            return field.default_factory()
        except Exception:
            return None
    return field.default


def _field_label(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def _field_spec(model_name: str, field_name: str, field: FieldInfo) -> FieldSpec:
    aliases = tuple(alias for alias in _aliases_from_validation_alias(field.validation_alias) if alias != field_name)
    return FieldSpec(
        key=field_name,
        type_name=_infer_type_name(field.annotation),
        nullable=_extract_nullable(field.annotation),
        required=field.is_required(),
        default=_default_value(field),
        enum_values=_extract_enum_values(field.annotation),
        read_aliases=aliases,
        label=_field_label(field_name),
        description=_FIELD_DESCRIPTIONS.get((model_name, field_name), ""),
    )


@lru_cache(maxsize=1)
def _model_specs() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for model_name, model_cls in SCHEMA_MODEL_MAP.items():
        fields = tuple(_field_spec(model_name, field_name, field) for field_name, field in model_cls.model_fields.items())
        alias_keys: list[str] = []
        for field in fields:
            alias_keys.extend(field.read_aliases)
        specs[model_name] = ModelSpec(
            model_name=model_name,
            model_qualname=f"{model_cls.__module__}.{model_cls.__qualname__}",
            fields=fields,
            canonical_write_keys=tuple(field.key for field in fields),
            read_alias_keys=tuple(dict.fromkeys(alias_keys)),
        )
    return specs


class SchemaRegistry:
    @staticmethod
    def current_version() -> int:
        return CURRENT_SCHEMA_VERSION

    @staticmethod
    def compatibility_window() -> tuple[int, int]:
        return LEGACY_COMPAT_MIN_VERSION, LEGACY_COMPAT_MAX_VERSION

    @staticmethod
    def model_names() -> tuple[str, ...]:
        return tuple(_model_specs().keys())

    @staticmethod
    def get_model_spec(model_name: str) -> ModelSpec:
        key = str(model_name or "").strip().lower()
        specs = _model_specs()
        if key not in specs:
            raise KeyError(f"Unknown model spec: {model_name}")
        return specs[key]

    @staticmethod
    def get_prompt_contract(mode: str) -> dict[str, Any]:
        normalized_mode = str(mode or "").strip().lower()
        metadata_spec = SchemaRegistry.get_model_spec("metadata")
        page_meta_spec = SchemaRegistry.get_model_spec("page_meta")
        fact_spec = SchemaRegistry.get_model_spec("fact")
        document_spec = SchemaRegistry.get_model_spec("document")
        enum_lookup: dict[str, tuple[Any, ...]] = {
            field.key: field.enum_values
            for field in (
                *metadata_spec.fields,
                *page_meta_spec.fields,
                *fact_spec.fields,
            )
            if field.enum_values
        }
        extraction_contract = {
            "schema_version": SchemaRegistry.current_version(),
            "top_level_keys": [key for key in document_spec.canonical_write_keys if key not in PROMPT_RUNTIME_OWNED_KEYS],
            "metadata_keys": list(metadata_spec.canonical_write_keys),
            "page_meta_keys": [
                key for key in page_meta_spec.canonical_write_keys
                if key not in PROMPT_RUNTIME_OWNED_PAGE_META_KEYS
            ],
            "fact_keys": [
                key for key in fact_spec.canonical_write_keys
                if key not in PROMPT_RUNTIME_OWNED_FACT_KEYS
            ],
            "required_prompt_fact_keys": list(PROMPT_REQUIRED_FACT_KEYS),
            "legacy_fact_aliases": list(fact_spec.read_alias_keys),
            "enums": {
                "page_types": list(enum_lookup.get("page_type", ())),
                "statement_types": list(enum_lookup.get("statement_type", ())),
                "value_types": list(enum_lookup.get("value_type", ())),
                "value_contexts": list(enum_lookup.get("value_context", ())),
                "balance_types": list(enum_lookup.get("balance_type", ())),
                "natural_signs": list(enum_lookup.get("natural_sign", ())),
                "row_roles": list(enum_lookup.get("row_role", ())),
                "aggregation_roles": list(enum_lookup.get("aggregation_role", ())),
                "currencies": list(enum_lookup.get("currency", ())),
                "scales": list(enum_lookup.get("scale", ())),
                "entity_types": list(enum_lookup.get("entity_type", ())),
                "report_scope": list(enum_lookup.get("report_scope", ())),
                "period_types": list(enum_lookup.get("period_type", ())),
                "path_sources": list(enum_lookup.get("path_source", ())),
            },
        }
        if normalized_mode in {"extract", "extraction"}:
            return extraction_contract
        if normalized_mode in {"gemini_fill", "patch"}:
            return {
                "schema_version": SchemaRegistry.current_version(),
                "statement_types": list(enum_lookup.get("statement_type", ())),
                "fact_patch_fields": [
                    "equation",
                    "period_type",
                    "period_start",
                    "period_end",
                    "duration_type",
                    "recurring_period",
                    "value_context",
                    "balance_type",
                    "natural_sign",
                    "row_role",
                    "aggregation_role",
                    "path_source",
                    "value_type",
                    "currency",
                    "scale",
                    "date",
                    "comment_ref",
                    "note_ref",
                    "note_name",
                ],
                "period_types": list(enum_lookup.get("period_type", ())),
                "path_sources": list(enum_lookup.get("path_source", ())),
                "value_types": list(enum_lookup.get("value_type", ())),
                "currencies": list(enum_lookup.get("currency", ())),
                "scales": list(enum_lookup.get("scale", ())),
            }
        raise KeyError(f"Unknown prompt contract mode: {mode}")


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_COMPAT_MAX_VERSION",
    "LEGACY_COMPAT_MIN_VERSION",
    "ModelSpec",
    "FieldSpec",
    "PROMPT_RUNTIME_OWNED_KEYS",
    "PROMPT_RUNTIME_OWNED_PAGE_META_KEYS",
    "PROMPT_RUNTIME_OWNED_FACT_KEYS",
    "PROMPT_REQUIRED_FACT_KEYS",
    "SchemaRegistry",
]
