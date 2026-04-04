from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .annotation_backups import atomic_write_text, utc_timestamp_compact

DEFAULT_INCLUDE_PATHS: tuple[str, ...] = (
    "data/raw_pdfs",
    "data/pdf_images",
    "data/annotations",
    "data/finetune",
    "data/workspace_review_state.json",
    "db/finetree.db",
)

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "**/.DS_Store",
    "data/annotations/_backup/**",
    "data/annotations_backup_*/**",
    "data/doctr_logs/**",
)

BACKUP_MANIFEST_NAME = "backup_manifest.json"


@dataclass(frozen=True)
class BackupFile:
    relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class BackupPlan:
    include_paths: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    files: tuple[BackupFile, ...]
    missing_paths: tuple[str, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


def _resolve_rooted_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Backup path must stay inside repository root: {raw_path}") from exc
    return resolved


def _normalize_relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _pattern_matches(path_text: str, pattern: str) -> bool:
    normalized = pattern.strip().rstrip("/")
    if not normalized:
        return False
    if normalized.endswith("/**"):
        prefix_pattern = normalized[:-3].rstrip("/")
        return fnmatch.fnmatch(path_text, prefix_pattern) or fnmatch.fnmatch(path_text, normalized)
    return fnmatch.fnmatch(path_text, normalized)


def _is_excluded(relative_path: PurePosixPath, exclude_patterns: Sequence[str]) -> bool:
    candidates = [relative_path.as_posix()]
    candidates.extend(
        parent.as_posix() for parent in relative_path.parents if parent.as_posix() != "."
    )
    for candidate in candidates:
        if any(_pattern_matches(candidate, pattern) for pattern in exclude_patterns):
            return True
    return False


def _iter_selected_files(
    root: Path,
    include_paths: Sequence[str],
    exclude_patterns: Sequence[str],
) -> tuple[list[BackupFile], list[str]]:
    selected: dict[str, BackupFile] = {}
    missing_paths: list[str] = []

    for raw_include in include_paths:
        include_path = _resolve_rooted_path(root, raw_include)
        if not include_path.exists():
            missing_paths.append(str(raw_include))
            continue

        relative_include = PurePosixPath(_normalize_relative_path(root, include_path))
        if _is_excluded(relative_include, exclude_patterns):
            continue

        if include_path.is_file():
            stat = include_path.stat()
            selected[relative_include.as_posix()] = BackupFile(
                relative_path=relative_include.as_posix(),
                size_bytes=stat.st_size,
            )
            continue

        for current_root, dir_names, file_names in os.walk(include_path):
            current_path = Path(current_root)
            relative_dir = PurePosixPath(_normalize_relative_path(root, current_path))
            dir_names[:] = [
                name
                for name in sorted(dir_names)
                if not _is_excluded(relative_dir / name, exclude_patterns)
            ]
            for file_name in sorted(file_names):
                relative_file = relative_dir / file_name
                if _is_excluded(relative_file, exclude_patterns):
                    continue
                file_path = current_path / file_name
                stat = file_path.stat()
                selected[relative_file.as_posix()] = BackupFile(
                    relative_path=relative_file.as_posix(),
                    size_bytes=stat.st_size,
                )

    files = sorted(selected.values(), key=lambda item: item.relative_path)
    return files, sorted(set(missing_paths))


def plan_workspace_backup(
    root: Path,
    *,
    include_paths: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> BackupPlan:
    resolved_root = Path(root).expanduser().resolve()
    normalized_includes = tuple(str(path) for path in (include_paths or DEFAULT_INCLUDE_PATHS))
    normalized_excludes = tuple(str(pattern) for pattern in (exclude_patterns or DEFAULT_EXCLUDE_PATTERNS))
    files, missing_paths = _iter_selected_files(
        resolved_root,
        include_paths=normalized_includes,
        exclude_patterns=normalized_excludes,
    )
    return BackupPlan(
        include_paths=normalized_includes,
        exclude_patterns=normalized_excludes,
        files=tuple(files),
        missing_paths=tuple(missing_paths),
    )


def _default_archive_path(root: Path, *, stamp: str) -> Path:
    return root / "backups" / f"finetree-data-backup-{stamp}.tar.gz"


def _resolve_output_path(root: Path, output_path: str | Path | None, *, stamp: str) -> Path:
    if output_path is None:
        return _default_archive_path(root, stamp=stamp)
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_manifest(
    *,
    root: Path,
    archive_path: Path,
    created_at_utc: str,
    plan: BackupPlan,
) -> dict[str, Any]:
    try:
        archive_ref = archive_path.relative_to(root).as_posix()
    except ValueError:
        archive_ref = str(archive_path)
    return {
        "created_at_utc": created_at_utc,
        "archive_path": archive_ref,
        "include_paths": list(plan.include_paths),
        "exclude_patterns": list(plan.exclude_patterns),
        "file_count": plan.file_count,
        "total_size_bytes": plan.total_size_bytes,
        "missing_paths": list(plan.missing_paths),
        "files": [
            {"path": item.relative_path, "size_bytes": item.size_bytes}
            for item in plan.files
        ],
    }


def create_workspace_backup(
    root: Path,
    *,
    output_path: str | Path | None = None,
    include_paths: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    stamp = utc_timestamp_compact()
    archive_path = _resolve_output_path(resolved_root, output_path, stamp=stamp)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    plan = plan_workspace_backup(
        resolved_root,
        include_paths=include_paths,
        exclude_patterns=exclude_patterns,
    )
    if not plan.files:
        raise RuntimeError("No files matched the requested backup scope.")

    manifest_payload = _archive_manifest(
        root=resolved_root,
        archive_path=archive_path,
        created_at_utc=stamp,
        plan=plan,
    )
    manifest_bytes = (json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    tmp_archive_path = archive_path.with_name(archive_path.name + ".tmp")
    with tarfile.open(tmp_archive_path, mode="w:gz") as archive:
        for item in plan.files:
            archive.add(
                resolved_root / item.relative_path,
                arcname=item.relative_path,
                recursive=False,
            )
        manifest_info = tarfile.TarInfo(name=BACKUP_MANIFEST_NAME)
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
    tmp_archive_path.replace(archive_path)

    archive_sha256 = _file_sha256(archive_path)
    manifest_payload["archive_sha256"] = archive_sha256

    manifest_path = archive_path.with_name(archive_path.name + ".manifest.json")
    sha256_path = archive_path.with_name(archive_path.name + ".sha256")
    atomic_write_text(manifest_path, json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(sha256_path, f"{archive_sha256}  {archive_path.name}\n")

    return {
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "sha256_path": str(sha256_path),
        "archive_sha256": archive_sha256,
        "file_count": plan.file_count,
        "total_size_bytes": plan.total_size_bytes,
        "missing_paths": list(plan.missing_paths),
        "include_paths": list(plan.include_paths),
        "exclude_patterns": list(plan.exclude_patterns),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tar.gz backup of the managed FineTree workspace data."
    )
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--output", help="Output archive path. Defaults to backups/finetree-data-backup-<timestamp>.tar.gz")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional repo-relative path to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional glob pattern to exclude. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-doctr-logs",
        action="store_true",
        help="Include data/doctr_logs in the backup.",
    )
    parser.add_argument(
        "--include-annotation-history",
        action="store_true",
        help="Include annotation backup history under data/annotations/_backup and data/annotations_backup_*.",
    )
    return parser.parse_args(argv)


def _merge_include_paths(args: argparse.Namespace) -> tuple[str, ...]:
    include_paths = list(DEFAULT_INCLUDE_PATHS)
    for raw_path in args.include:
        normalized = str(raw_path).strip()
        if normalized and normalized not in include_paths:
            include_paths.append(normalized)
    if args.include_doctr_logs and "data/doctr_logs" not in include_paths:
        include_paths.append("data/doctr_logs")
    return tuple(include_paths)


def _merge_exclude_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
    if args.include_annotation_history:
        exclude_patterns = [
            pattern
            for pattern in exclude_patterns
            if not pattern.startswith("data/annotations/_backup")
            and not pattern.startswith("data/annotations_backup_")
        ]
    if args.include_doctr_logs:
        exclude_patterns = [
            pattern for pattern in exclude_patterns if not pattern.startswith("data/doctr_logs")
        ]
    for pattern in args.exclude:
        normalized = str(pattern).strip()
        if normalized:
            exclude_patterns.append(normalized)
    return tuple(exclude_patterns)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    result = create_workspace_backup(
        root,
        output_path=args.output,
        include_paths=_merge_include_paths(args),
        exclude_patterns=_merge_exclude_patterns(args),
    )
    print(
        "BACKUP: "
        f"archive={result['archive_path']} "
        f"manifest={result['manifest_path']} "
        f"sha256={result['sha256_path']} "
        f"files={result['file_count']} "
        f"bytes={result['total_size_bytes']}"
    )
    if result["missing_paths"]:
        print("SKIPPED_MISSING: " + ", ".join(result["missing_paths"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
