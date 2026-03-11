from __future__ import annotations

from copy import deepcopy
from typing import Any

from .equation_integrity import equation_integrity_reg_flags, resequence_fact_numbers_and_remap_fact_equations
from .fact_normalization import LEGACY_FACT_KEYS, normalize_annotation_payload, normalize_fact_payload
from .fact_ordering import compact_document_meta
from .schema_registry import CURRENT_SCHEMA_VERSION
from .schemas import PageMeta


class EquationIntegrityError(ValueError):
    def __init__(self, message: str, *, findings: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.findings: list[dict[str, Any]] = list(findings or [])


def _assign_missing_fact_numbers(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_facts: list[dict[str, Any]] = []
    for raw_fact in facts:
        normalized_fact, _fact_warnings = normalize_fact_payload(raw_fact, include_bbox=False)
        normalized_facts.append(normalized_fact)
    return resequence_fact_numbers_and_remap_fact_equations(normalized_facts)


def _metadata_dict(payload: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, dict):
        return compact_document_meta(raw_metadata)
    return compact_document_meta(payload.get("document_meta"))


def detect_payload_schema_version(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("schema_version")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def payload_uses_legacy_aliases(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if "document_meta" in payload:
        return True
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return False
    for page in pages:
        if not isinstance(page, dict):
            continue
        meta = page.get("meta")
        if isinstance(meta, dict) and "type" in meta:
            return True
        facts = page.get("facts")
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            if any(alias in fact for alias in LEGACY_FACT_KEYS):
                return True
            value_type = str(fact.get("value_type") or "").strip().lower()
            if value_type in {"%", "regular", "percentage"}:
                return True
    return False


def payload_requires_migration(payload: Any, *, target_version: int = CURRENT_SCHEMA_VERSION) -> bool:
    if not isinstance(payload, dict):
        return False
    version = detect_payload_schema_version(payload)
    if version is None:
        return True
    if version != int(target_version):
        return True
    return payload_uses_legacy_aliases(payload)


def normalize_payload(
    payload: Any,
    *,
    from_version: str | int = "auto",
    to_version: int = CURRENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    _ = from_version
    if not isinstance(payload, dict):
        return {
            "schema_version": int(to_version),
            "images_dir": None,
            "metadata": {},
            "pages": [],
        }
    normalized_payload, _findings = normalize_annotation_payload(deepcopy(payload))
    canonical_pages: list[dict[str, Any]] = []
    for page in normalized_payload.get("pages", []):
        if not isinstance(page, dict):
            continue
        image = page.get("image")
        image_name = str(image).strip() if isinstance(image, str) else None
        raw_meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
        try:
            meta = PageMeta.model_validate(raw_meta).model_dump(mode="json")
        except Exception:
            meta = PageMeta.model_validate({}).model_dump(mode="json")
        facts_out: list[dict[str, Any]] = []
        raw_facts = [fact for fact in page.get("facts", []) if isinstance(fact, dict)]
        normalized_facts = _assign_missing_fact_numbers(raw_facts)
        for raw_fact, normalized_fact in zip(raw_facts, normalized_facts):
            include_bbox = "bbox" in raw_fact
            if include_bbox:
                normalized_fact["bbox"] = _normalize_bbox_to_list(raw_fact.get("bbox"))
            facts_out.append(normalized_fact)
        canonical_pages.append(
            {
                "image": image_name,
                "meta": meta,
                "facts": facts_out,
            }
        )

    images_dir = normalized_payload.get("images_dir")
    normalized_images_dir = str(images_dir).strip() if isinstance(images_dir, str) else None
    return {
        "schema_version": int(to_version),
        "images_dir": normalized_images_dir or None,
        "metadata": _metadata_dict(normalized_payload),
        "pages": canonical_pages,
    }


def _normalize_bbox_to_list(raw_bbox: Any) -> list[float]:
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        x = _to_float(raw_bbox[0], 0.0)
        y = _to_float(raw_bbox[1], 0.0)
        w = max(_to_float(raw_bbox[2], 1.0), 1.0)
        h = max(_to_float(raw_bbox[3], 1.0), 1.0)
        return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
    if isinstance(raw_bbox, dict):
        x = _to_float(raw_bbox.get("x"), 0.0)
        y = _to_float(raw_bbox.get("y"), 0.0)
        w = max(_to_float(raw_bbox.get("w"), 1.0), 1.0)
        h = max(_to_float(raw_bbox.get("h"), 1.0), 1.0)
        return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
    return [0.0, 0.0, 1.0, 1.0]


def load_any_schema(
    payload: Any,
    *,
    from_version: str | int = "auto",
    to_version: int = CURRENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    return normalize_payload(payload, from_version=from_version, to_version=to_version)


def _equation_guard_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return findings
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_image = str(page.get("image") or "").strip() or "<unknown_page>"
        meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
        statement_type = str(meta.get("statement_type") or "").strip() or None
        facts = [fact for fact in page.get("facts", []) if isinstance(fact, dict)]
        for finding in equation_integrity_reg_flags(facts, statement_type=statement_type):
            findings.append({**finding, "page_image": page_image})
    return findings


def _equation_guard_message(findings: list[dict[str, Any]]) -> str:
    ordered = sorted(
        findings,
        key=lambda finding: (
            str(finding.get("page_image") or ""),
            int(finding.get("fact_num")) if isinstance(finding.get("fact_num"), int) else 0,
            str(finding.get("code") or ""),
        ),
    )
    lines: list[str] = []
    for finding in ordered[:8]:
        page_image = str(finding.get("page_image") or "<unknown_page>")
        fact_num = finding.get("fact_num")
        fact_label = f"f{fact_num}" if isinstance(fact_num, int) else "f?"
        code = str(finding.get("code") or "equation_integrity_error")
        message = str(finding.get("message") or "").strip()
        if message:
            lines.append(f"{page_image} {fact_label} [{code}] {message}")
        else:
            lines.append(f"{page_image} {fact_label} [{code}]")
    remaining = len(ordered) - len(lines)
    if remaining > 0:
        lines.append(f"...and {remaining} more integrity issue(s).")
    return "Equation integrity check failed; save aborted.\n" + "\n".join(lines)


def canonicalize_with_findings(
    payload: Any,
    *,
    to_version: int = CURRENT_SCHEMA_VERSION,
    strict_equation_guards: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = normalize_payload(payload, from_version="auto", to_version=to_version)
    equation_findings = _equation_guard_findings(canonical)
    if strict_equation_guards and equation_findings:
        raise EquationIntegrityError(_equation_guard_message(equation_findings), findings=equation_findings)
    return canonical, equation_findings


def save_canonical(payload: Any, *, to_version: int = CURRENT_SCHEMA_VERSION) -> dict[str, Any]:
    canonical, _equation_findings = canonicalize_with_findings(
        payload,
        to_version=to_version,
        strict_equation_guards=True,
    )
    return canonical


__all__ = [
    "canonicalize_with_findings",
    "EquationIntegrityError",
    "detect_payload_schema_version",
    "load_any_schema",
    "normalize_payload",
    "payload_requires_migration",
    "payload_uses_legacy_aliases",
    "save_canonical",
]
