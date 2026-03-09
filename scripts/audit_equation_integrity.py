#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.equation_integrity import audit_and_rebuild_financial_facts  # noqa: E402
from finetree_annotator.schema_io import save_canonical  # noqa: E402


def _iter_files(root: Path, annotations_glob: str) -> list[Path]:
    pattern = Path(annotations_glob).expanduser()
    search_pattern = str(pattern) if pattern.is_absolute() else str(root / annotations_glob)
    return sorted(Path(path).resolve() for path in glob.glob(search_pattern, recursive=True) if Path(path).is_file())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _audit_payload(payload: dict[str, Any], *, apply_repairs: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated_payload = dict(payload)
    pages = updated_payload.get("pages")
    findings: list[dict[str, Any]] = []
    if not isinstance(pages, list):
        return updated_payload, findings
    next_pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_out = dict(page)
        meta = page_out.get("meta") if isinstance(page_out.get("meta"), dict) else {}
        statement_type = str(meta.get("statement_type") or "").strip() or None
        facts = page_out.get("facts")
        if not isinstance(facts, list):
            next_pages.append(page_out)
            continue
        raw_facts = [fact for fact in facts if isinstance(fact, dict)]
        rebuilt_facts, page_findings = audit_and_rebuild_financial_facts(
            raw_facts,
            statement_type=statement_type,
            apply_repairs=apply_repairs,
        )
        if apply_repairs:
            next_facts: list[dict[str, Any]] = []
            rebuilt_iter = iter(rebuilt_facts)
            for fact in facts:
                if isinstance(fact, dict):
                    repaired = next(rebuilt_iter)
                    if "bbox" in fact:
                        repaired = {**repaired, "bbox": fact.get("bbox")}
                    next_facts.append(repaired)
                else:
                    next_facts.append(fact)
            page_out["facts"] = next_facts
        for finding in page_findings:
            findings.append(
                {
                    "page_index": page_index,
                    "page_image": str(page_out.get("image") or f"page_{page_index + 1}"),
                    **finding,
                }
            )
        next_pages.append(page_out)
    updated_payload["pages"] = next_pages
    return updated_payload, findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and optionally repair fact equation integrity.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--annotations-glob", default="data/annotations/*.json", help="Glob pattern for annotation JSON files.")
    parser.add_argument("--write", action="store_true", help="Apply repairs in-place.")
    parser.add_argument("--json", action="store_true", help="Print full report as JSON.")
    parser.add_argument("--limit", type=int, default=200, help="Max findings to print in text mode.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    files = _iter_files(root, args.annotations_glob)

    all_findings: list[dict[str, Any]] = []
    changed_files = 0
    for path in files:
        payload = _load_json(path)
        rebuilt_payload, findings = _audit_payload(payload, apply_repairs=args.write)
        all_findings.extend([{**finding, "file": str(path.relative_to(root))} for finding in findings])
        if args.write and rebuilt_payload != payload:
            canonical_payload = save_canonical(rebuilt_payload)
            _write_json(path, canonical_payload)
            changed_files += 1

    report = {
        "annotations_glob": args.annotations_glob,
        "files_scanned": len(files),
        "files_changed": changed_files,
        "findings_count": len(all_findings),
        "findings": all_findings,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(
        f"EQUATION_INTEGRITY_REPORT files={report['files_scanned']} changed={report['files_changed']} "
        f"findings={report['findings_count']}"
    )
    for finding in all_findings[: max(0, int(args.limit))]:
        print(
            f"- {finding['file']} | page={finding['page_image']} | fact_num={finding.get('fact_num')} "
            f"| code={finding.get('code')} | severity={finding.get('severity')} | {finding.get('message')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
