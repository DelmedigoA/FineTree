from __future__ import annotations

import json
import tarfile
from pathlib import Path

from finetree_annotator import workspace_backup


def _write(root: Path, relative_path: str, content: str | bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
        return
    target.write_text(content, encoding="utf-8")


def test_plan_workspace_backup_excludes_generated_noise(tmp_path: Path) -> None:
    _write(tmp_path, "data/raw_pdfs/doc.pdf", b"%PDF-1.4")
    _write(tmp_path, "data/pdf_images/doc/page_0001.png", b"png")
    _write(tmp_path, "data/annotations/doc.json", '{"pages":[]}')
    _write(tmp_path, "data/annotations/_backup/doc/old.json", '{"pages":[]}')
    _write(tmp_path, "data/annotations_backup_20260307T140920/legacy.json", '{"pages":[]}')
    _write(tmp_path, "data/doctr_logs/page.json", '{"text":"debug"}')
    _write(tmp_path, "data/workspace_review_state.json", '{"checked_doc_ids":[]}')
    _write(tmp_path, "db/finetree.db", b"sqlite")
    _write(tmp_path, "data/.DS_Store", b"junk")

    plan = workspace_backup.plan_workspace_backup(tmp_path)
    planned_files = [item.relative_path for item in plan.files]

    assert "data/raw_pdfs/doc.pdf" in planned_files
    assert "data/pdf_images/doc/page_0001.png" in planned_files
    assert "data/annotations/doc.json" in planned_files
    assert "data/workspace_review_state.json" in planned_files
    assert "db/finetree.db" in planned_files
    assert "data/annotations/_backup/doc/old.json" not in planned_files
    assert "data/annotations_backup_20260307T140920/legacy.json" not in planned_files
    assert "data/doctr_logs/page.json" not in planned_files
    assert "data/.DS_Store" not in planned_files


def test_create_workspace_backup_writes_archive_and_manifest(tmp_path: Path) -> None:
    _write(tmp_path, "data/raw_pdfs/doc.pdf", b"%PDF-1.4")
    _write(tmp_path, "data/pdf_images/doc/page_0001.png", b"png")
    _write(tmp_path, "data/annotations/doc.json", '{"pages":[]}')
    _write(tmp_path, "data/workspace_review_state.json", '{"checked_doc_ids":[]}')
    _write(tmp_path, "db/finetree.db", b"sqlite")

    result = workspace_backup.create_workspace_backup(
        tmp_path,
        output_path="backups/test-backup.tar.gz",
    )

    archive_path = Path(result["archive_path"])
    manifest_path = Path(result["manifest_path"])
    sha256_path = Path(result["sha256_path"])

    assert archive_path.is_file()
    assert manifest_path.is_file()
    assert sha256_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["archive_path"] == "backups/test-backup.tar.gz"
    assert manifest["archive_sha256"] == result["archive_sha256"]
    assert manifest["file_count"] == 5

    with tarfile.open(archive_path, "r:gz") as archive:
        members = set(archive.getnames())
        assert "data/raw_pdfs/doc.pdf" in members
        assert "data/pdf_images/doc/page_0001.png" in members
        assert "data/annotations/doc.json" in members
        assert "data/workspace_review_state.json" in members
        assert "db/finetree.db" in members
        assert workspace_backup.BACKUP_MANIFEST_NAME in members

    assert sha256_path.read_text(encoding="utf-8").strip().endswith("test-backup.tar.gz")
