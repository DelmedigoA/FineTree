from __future__ import annotations

from finetree_annotator.benchmark.config import DEFAULT_AGGREGATE_WEIGHTS, DEFAULT_META_FIELD_WEIGHTS
from finetree_annotator.benchmark.scoring import aggregate_mapping_metrics, evaluate_document_detailed


def _fact(
    *,
    fact_num: int | None = None,
    period_type: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    duration_type: str | None = None,
    row_role: str = "detail",
    comment_ref: str | None = None,
    note_flag: bool = False,
    note_name: str | None = None,
    note_num: int | None = None,
    note_ref: str | None = None,
    path_source: str | None = None,
    currency: str | None = None,
    scale: int | None = None,
    value_type: str | None = None,
    value_context: str | None = None,
) -> dict:
    payload = {
        "value": "1",
        "path": ["fact"],
        "row_role": row_role,
        "comment_ref": comment_ref,
        "note_flag": note_flag,
        "note_name": note_name,
        "note_num": note_num,
        "note_ref": note_ref,
        "path_source": path_source,
        "currency": currency,
        "scale": scale,
        "value_type": value_type,
        "value_context": value_context,
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "duration_type": duration_type,
    }
    if fact_num is not None:
        payload["fact_num"] = fact_num
    return payload


def _page(
    *,
    entity: str,
    page_num: str,
    page_type: str,
    statement_type: str | None,
    title: str,
    facts: list[dict],
) -> dict:
    return {
        "image": "page_0001.png",
        "meta": {
            "entity_name": entity,
            "page_num": page_num,
            "page_type": page_type,
            "statement_type": statement_type,
            "title": title,
        },
        "facts": facts,
    }


def _doc(*pages: dict) -> dict:
    return {
        "schema_version": 3,
        "images_dir": None,
        "metadata": {},
        "pages": list(pages),
    }


def test_evaluate_document_detailed_returns_expected_metrics_for_count_only() -> None:
    gt_doc = _doc(_page(entity="Acme", page_num="1", page_type="title", statement_type=None, title="Annual", facts=[_fact()]))
    pred_doc = _doc(_page(entity="Acme", page_num="1", page_type="title", statement_type=None, title="Annual", facts=[_fact()]))
    report = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
    )
    assert report["overall_score"] == 1.0
    assert report["meta_score"] == 1.0
    assert report["facts_score"] == 1.0
    assert report["facts_count_score"] == 1.0
    assert report["field_scores"]["entity_name"]["used_score"] == 1.0


def test_fact_quality_assigns_missing_prediction_fact_num_and_matches_by_gt_fact_num() -> None:
    gt_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[
                _fact(
                    fact_num=1,
                    period_type="instant",
                    period_start="2024-01-01",
                    period_end="2024-01-01",
                    row_role="detail",
                    comment_ref="Note A",
                    note_flag=True,
                    note_name="Debt",
                    note_num=4,
                    note_ref="4",
                    path_source="observed",
                    currency="ILS",
                    scale=1,
                    value_type="amount",
                    value_context="tabular",
                ),
                _fact(
                    fact_num=2,
                    period_type="duration",
                    period_start="2024-01-01",
                    period_end="2024-12-31",
                    duration_type="recurrent",
                    row_role="total",
                    comment_ref="Note B",
                    note_flag=False,
                    note_name="Revenue",
                    note_num=None,
                    note_ref=None,
                    path_source="inferred",
                    currency="USD",
                    scale=1000,
                    value_type="percent",
                    value_context="mixed",
                ),
            ],
        )
    )
    pred_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[
                _fact(
                    period_type="instant",
                    period_start="2024-01-01",
                    period_end="2024-01-01",
                    row_role="detail",
                    comment_ref="Note A",
                    note_flag=True,
                    note_name="Debt",
                    note_num=4,
                    note_ref="4",
                    path_source="observed",
                    currency="ILS",
                    scale=1,
                    value_type="amount",
                    value_context="tabular",
                ),
                _fact(
                    period_type="duration",
                    period_start="2024-01-01",
                    period_end="2024-12-31",
                    duration_type="recurrent",
                    row_role="total",
                    comment_ref="Note Bee",
                    note_flag=False,
                    note_name="Revenue",
                    note_num=None,
                    note_ref=None,
                    path_source="inferred",
                    currency="USD",
                    scale=1000,
                    value_type="percent",
                    value_context="mixed",
                ),
            ],
        )
    )
    report = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
        facts_method="fact_quality_v1",
    )
    first_pair, second_pair = report["page_reports"][0]["report"]["facts_score"]["fact_reports"]
    assert first_pair["match_strategy"] == "fact_num"
    assert first_pair["pred_fact"]["fact_num"] == 1
    assert second_pair["match_strategy"] == "fact_num"
    assert second_pair["pred_fact"]["fact_num"] == 2
    assert report["date_score"] == 1.0
    assert report["comment_ref_score"] < 1.0
    assert report["facts_count_score"] == 1.0
    assert report["facts_score"] < 1.0


def test_fact_quality_uses_page_order_fallback_when_gt_fact_num_is_missing() -> None:
    gt_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS")],
        )
    )
    pred_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(fact_num=9, period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS")],
        )
    )
    report = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
        facts_method="fact_quality_v1",
    )
    pair = report["page_reports"][0]["report"]["facts_score"]["fact_reports"][0]
    assert pair["match_strategy"] == "page_order"
    assert report["date_score"] == 1.0
    assert report["currency_score"] == 1.0


def test_fact_quality_unmatched_facts_score_zero() -> None:
    gt_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(fact_num=1, period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS")],
        )
    )
    pred_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[
                _fact(fact_num=99, period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS"),
                _fact(fact_num=100, period_type="duration", period_start="2024-01-01", period_end="2024-12-31", currency="USD"),
            ],
        )
    )
    report = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
        facts_method="fact_quality_v1",
    )
    facts = report["page_reports"][0]["report"]["facts_score"]
    assert facts["facts_count_score"] == 0.0
    assert facts["date_score"] == 0.0
    assert facts["currency_score"] == 0.0
    assert facts["fact_pair_count"] == 3


def test_fact_quality_date_score_requires_all_date_fields_to_match() -> None:
    gt_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(fact_num=1, period_type="duration", period_start="2024-01-01", period_end="2024-12-31", duration_type="recurrent")],
        )
    )
    pred_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(fact_num=1, period_type="duration", period_start="2024-01-01", period_end="2024-12-31")],
        )
    )
    report = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
        facts_method="fact_quality_v1",
    )
    assert report["date_score"] == 0.0


def test_aggregate_mapping_metrics_exposes_leaderboard_submetrics() -> None:
    gt_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(fact_num=1, period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS")],
        )
    )
    pred_doc = _doc(
        _page(
            entity="Acme",
            page_num="1",
            page_type="title",
            statement_type=None,
            title="Annual",
            facts=[_fact(period_type="instant", period_start="2024-01-01", period_end="2024-01-01", currency="ILS")],
        )
    )
    metrics = evaluate_document_detailed(
        pred_doc,
        gt_doc,
        meta_rules=DEFAULT_META_FIELD_WEIGHTS,
        aggregate_weights=DEFAULT_AGGREGATE_WEIGHTS,
        facts_method="fact_quality_v1",
    )
    aggregate = aggregate_mapping_metrics([{"metrics": metrics}])
    assert aggregate["mapping_count"] == 1
    assert aggregate["entity_score"] == 1.0
    assert aggregate["title_score"] == 1.0
    assert aggregate["page_type_score"] == 1.0
    assert aggregate["statement_type_score"] == 1.0
    assert aggregate["facts_count_score"] == 1.0
    assert aggregate["date_score"] == 1.0
    assert aggregate["currency_score"] == 1.0
