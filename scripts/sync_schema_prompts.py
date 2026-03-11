#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finetree_annotator.schema_contract import (  # noqa: E402
    default_extraction_prompt_template,
    default_gemini_autocomplete_prompt_template,
    default_gemini_fill_prompt_template,
)


def _targets(root: Path) -> list[tuple[Path, str]]:
    prompts_root = root / "prompts"
    return [
        (prompts_root / "extraction_prompt.txt", default_extraction_prompt_template().strip() + "\n"),
        (prompts_root / "gemini_fill_prompt.txt", default_gemini_fill_prompt_template().strip() + "\n"),
        (prompts_root / "gemini_autocomplete_prompt.txt", default_gemini_autocomplete_prompt_template().strip() + "\n"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync prompt text files from schema-generated templates.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--check", action="store_true", help="Fail if prompt files are out of sync.")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    changed: list[Path] = []
    for path, expected in _targets(root):
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == expected:
            continue
        changed.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")

    if args.check and changed:
        print("PROMPT_SYNC_OUT_OF_DATE:")
        for path in changed:
            print(f"- {path}")
        return 1
    if changed:
        print(f"PROMPT_SYNC_UPDATED: {len(changed)} file(s)")
    else:
        print("PROMPT_SYNC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
