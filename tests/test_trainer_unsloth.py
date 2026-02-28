from __future__ import annotations

from finetree_annotator.finetune import trainer_unsloth


def test_effective_vision_max_seq_length_clamps_small_values(capsys) -> None:
    out = trainer_unsloth._effective_vision_max_seq_length(256)
    captured = capsys.readouterr()

    assert out == 512
    assert "too small for Qwen3.5-VL image tokenization" in captured.out


def test_effective_vision_max_seq_length_keeps_safe_values() -> None:
    assert trainer_unsloth._effective_vision_max_seq_length(768) == 768


def test_extract_mm_token_count_parses_expected_field() -> None:
    msg = "Mismatch in `image` token count between text and `input_ids`. Got ids=[252] and text=[352]."
    assert trainer_unsloth._extract_mm_token_count(msg, "ids") == 252
    assert trainer_unsloth._extract_mm_token_count(msg, "text") == 352
    assert trainer_unsloth._extract_mm_token_count(msg, "missing") is None
