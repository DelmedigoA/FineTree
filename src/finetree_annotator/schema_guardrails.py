from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from .schema_registry import SchemaRegistry

_STRING_LITERAL_RE = re.compile(r"""(?P<q>['"])(?P<value>[A-Za-z_][A-Za-z0-9_]*)\1""")
_HIGH_RISK_LEGACY_KEYS = {
    "document_meta",
    "ref_comment",
    "ref_note",
    "note_reference",
    "refference",
    "is_beur",
    "beur_num",
    "beur_number",
}


def schema_keys_for_literal_scan() -> set[str]:
    keys: set[str] = set(_HIGH_RISK_LEGACY_KEYS)
    for model_name in SchemaRegistry.model_names():
        spec = SchemaRegistry.get_model_spec(model_name)
        for alias in spec.read_alias_keys:
            if alias in _HIGH_RISK_LEGACY_KEYS:
                keys.add(alias)
    return keys


def scan_raw_schema_key_literals(
    root: Path,
    *,
    include_globs: Iterable[str],
    allow_relative_paths: Iterable[str],
    scanned_keys: set[str] | None = None,
) -> list[tuple[Path, int, str]]:
    target_keys = scanned_keys if scanned_keys is not None else schema_keys_for_literal_scan()
    allowed = {str(Path(path)) for path in allow_relative_paths}
    findings: list[tuple[Path, int, str]] = []
    for pattern in include_globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            if rel in allowed:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                for match in _STRING_LITERAL_RE.finditer(line):
                    value = match.group("value")
                    if value in target_keys:
                        findings.append((path, line_no, value))
    return findings


__all__ = [
    "scan_raw_schema_key_literals",
    "schema_keys_for_literal_scan",
]
