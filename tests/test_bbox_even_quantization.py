from __future__ import annotations

from finetree_annotator.finetune.bbox_even_quantization import (
    quantize_yxyx_bbox_to_even,
    snap_even_edge_pair,
)


def test_snap_even_edge_pair_biases_outward_on_tie() -> None:
    snapped_min, snapped_max, adjusted = snap_even_edge_pair(
        5.0,
        7.0,
        lower_bound=0.0,
        upper_bound=10.0,
    )

    assert (snapped_min, snapped_max) == (4, 8)
    assert adjusted is False


def test_quantize_yxyx_bbox_to_even_stays_valid_near_upper_bound() -> None:
    bbox, adjusted = quantize_yxyx_bbox_to_even(
        999.1,
        999.1,
        999.7,
        999.7,
        upper_bound=1000,
    )

    assert bbox == [998, 998, 1000, 1000]
    assert adjusted is False


def test_snap_even_edge_pair_expands_when_clamping_would_collapse_box() -> None:
    snapped_min, snapped_max, adjusted = snap_even_edge_pair(
        2.1,
        2.2,
        lower_bound=0.0,
        upper_bound=2.0,
    )

    assert (snapped_min, snapped_max) == (0, 2)
    assert adjusted is True
