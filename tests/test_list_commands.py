from __future__ import annotations

from types import SimpleNamespace

from finetree_annotator import list_commands


def test_scripts_from_pyproject_sorted(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        "[project.scripts]\n"
        'finetree-ft-train = "a:b"\n'
        'finetree-qwen-gt = "c:d"\n',
        encoding="utf-8",
    )
    assert list_commands._scripts_from_pyproject(pyproject) == [
        "finetree-ft-train",
        "finetree-qwen-gt",
    ]


def test_resolve_scripts_uses_distribution(monkeypatch) -> None:
    fake_dist = SimpleNamespace(
        entry_points=[
            SimpleNamespace(group="console_scripts", name="finetree-b"),
            SimpleNamespace(group="other", name="skip-me"),
            SimpleNamespace(group="console_scripts", name="finetree-a"),
        ]
    )
    monkeypatch.setattr(list_commands.metadata, "distribution", lambda _name: fake_dist)
    assert list_commands._resolve_scripts() == ["finetree-a", "finetree-b"]
