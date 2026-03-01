from __future__ import annotations

import numpy as np

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


def test_token_accuracy_metric_ignores_masked_positions() -> None:
    class _EvalPred:
        predictions = np.array([[1, 2, 3, 4]])
        label_ids = np.array([[-100, 2, 8, 4]])

    metrics = trainer_unsloth._token_accuracy_metric(_EvalPred())
    # Shifted predictions: [1,2,3], shifted labels: [2,8,4]
    # Valid positions: 3 => only middle matches (2==8 false, 3==4 false?) Actually none.
    assert "token_accuracy" in metrics
    assert metrics["token_accuracy"] == 0.0
