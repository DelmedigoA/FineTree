from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tomllib


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scripts_from_pyproject(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return []
    return sorted(str(name) for name in scripts)


def _scripts_from_distribution(dist_name: str = "finetree-annotator") -> list[str]:
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return []
    scripts = [entry.name for entry in dist.entry_points if entry.group == "console_scripts"]
    return sorted(set(scripts))


def _resolve_scripts() -> list[str]:
    scripts = _scripts_from_distribution()
    if scripts:
        return scripts
    pyproject_path = _project_root() / "pyproject.toml"
    if not pyproject_path.is_file():
        return []
    return _scripts_from_pyproject(pyproject_path)


def main() -> int:
    scripts = _resolve_scripts()
    if not scripts:
        print("No CLI commands found.")
        return 1
    for name in scripts:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
