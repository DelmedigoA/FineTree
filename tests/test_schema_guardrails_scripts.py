from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from finetree_annotator.schema_contract import default_extraction_prompt_template, default_gemini_fill_prompt_template


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / name


def test_check_schema_key_literals_script_detects_violations(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "allowed.py"
    allowed.parent.mkdir(parents=True, exist_ok=True)
    allowed.write_text('X = "document_meta"\n', encoding="utf-8")
    bad = tmp_path / "src" / "bad.py"
    bad.write_text('Y = "document_meta"\n', encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path("check_schema_key_literals.py")),
            "--root",
            str(tmp_path),
            "--include-glob",
            "src/**/*.py",
            "--allow-path",
            "src/allowed.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "SCHEMA_KEY_LITERAL_VIOLATIONS" in (proc.stdout + proc.stderr)


def test_sync_schema_prompts_script_check_mode(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = prompts_dir / "extraction_prompt.txt"
    fill_path = prompts_dir / "gemini_fill_prompt.txt"
    extraction_path.write_text(default_extraction_prompt_template().strip() + "\n", encoding="utf-8")
    fill_path.write_text(default_gemini_fill_prompt_template().strip() + "\n", encoding="utf-8")

    ok_proc = subprocess.run(
        [sys.executable, str(_script_path("sync_schema_prompts.py")), "--root", str(tmp_path), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_proc.returncode == 0

    extraction_path.write_text("out of sync\n", encoding="utf-8")
    fail_proc = subprocess.run(
        [sys.executable, str(_script_path("sync_schema_prompts.py")), "--root", str(tmp_path), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail_proc.returncode == 1
    assert "PROMPT_SYNC_OUT_OF_DATE" in (fail_proc.stdout + fail_proc.stderr)
