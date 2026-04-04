"""Schema metadata and enum options endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..schema_contract import CANONICAL_FACT_KEYS, CURRENCY_VALUES, SCALE_VALUES
from ..schema_ui import enum_options

router = APIRouter(prefix="/api/schema", tags=["schema"])


@router.get("/options")
def get_schema_options() -> dict[str, Any]:
    return {
        "canonical_fact_keys": list(CANONICAL_FACT_KEYS),
        "currency_values": list(CURRENCY_VALUES),
        "scale_values": list(SCALE_VALUES),
        "enums": {
            "page_type": list(enum_options("page_meta", "page_type")),
            "statement_type": list(enum_options("page_meta", "statement_type")),
            "value_type": list(enum_options("fact", "value_type")),
            "value_context": list(enum_options("fact", "value_context")),
            "natural_sign": list(enum_options("fact", "natural_sign")),
            "row_role": list(enum_options("fact", "row_role")),
            "period_type": list(enum_options("fact", "period_type")),
            "duration_type": list(enum_options("fact", "duration_type")),
            "recurring_period": list(enum_options("fact", "recurring_period")),
            "path_source": list(enum_options("fact", "path_source")),
            "entity_type": list(enum_options("metadata", "entity_type")),
            "report_scope": list(enum_options("metadata", "report_scope")),
        },
    }
