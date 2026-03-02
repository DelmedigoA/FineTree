from __future__ import annotations

from finetree_annotator import gemini_vlm


def test_resolve_api_key_prefers_google_api_key_env(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
    monkeypatch.setenv("FINETREE_GEMINI_API_KEY", "finetree-env-key")
    assert gemini_vlm.resolve_api_key() == "google-env-key"


def test_resolve_api_key_supports_finetree_env_name(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("FINETREE_GEMINI_API_KEY", "finetree-env-key")
    assert gemini_vlm.resolve_api_key() == "finetree-env-key"


def test_resolve_api_key_uses_doppler_when_env_missing(monkeypatch) -> None:
    gemini_vlm._api_key_from_doppler.cache_clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("FINETREE_GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_vlm, "_api_key_from_config", lambda: None)
    monkeypatch.setattr(gemini_vlm.shutil, "which", lambda _cmd: "/usr/local/bin/doppler")

    class _Proc:
        stdout = "doppler-key\n"

    def _fake_run(cmd, check, capture_output, text, timeout):
        assert cmd[:3] == ["doppler", "secrets", "get"]
        assert check is True
        assert capture_output is True
        assert text is True
        assert timeout == 5
        return _Proc()

    monkeypatch.setattr(gemini_vlm.subprocess, "run", _fake_run)
    assert gemini_vlm.resolve_api_key() == "doppler-key"
