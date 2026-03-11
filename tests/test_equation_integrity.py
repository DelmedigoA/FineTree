from __future__ import annotations

from finetree_annotator.equation_integrity import (
    audit_and_rebuild_financial_facts,
    remap_fact_equation_references,
    resequence_fact_numbers_and_remap_fact_equations,
)


def test_audit_and_rebuild_financial_facts_rebuilds_from_fact_equation() -> None:
    facts = [
        {"fact_num": 10, "value": "269968", "natural_sign": "positive", "path": ["רכוש קבוע", "עלות"]},
        {
            "fact_num": 12,
            "value": "209255",
            "natural_sign": "positive",
            "path": ["רכוש קבוע", "בניכוי - פחת שנצבר"],
        },
        {
            "fact_num": 14,
            "value": "60713",
            "natural_sign": "positive",
            "path": ["רכוש קבוע", 'סה"כ רכוש קבוע'],
            "equation": "- 269968 + 209255",
            "fact_equation": "- f10 + f12",
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(
        facts,
        statement_type="balance_sheet",
        apply_repairs=True,
    )

    by_num = {fact["fact_num"]: fact for fact in rebuilt}
    assert by_num[10]["row_role"] == "detail"
    assert by_num[12]["row_role"] == "detail"
    assert by_num[14]["row_role"] == "total"
    assert by_num[14]["equations"][0]["equation"] == "- 269968 + 209255"
    assert by_num[14]["equations"][0]["fact_equation"] == "- f10 + f12"
    assert any(finding.get("code") == "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_treats_dash_as_zero_in_equation() -> None:
    facts = [
        {"fact_num": 1, "value": "-", "path": ["A"]},
        {"fact_num": 2, "value": "5", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "5",
            "row_role": "total",
            "path": ["A", "total"],
            "equation": "5",
            "fact_equation": "f1 + f2",
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)
    total_fact = next(fact for fact in rebuilt if fact["fact_num"] == 3)
    assert total_fact["equations"][0]["equation"] == "0 + 5"
    assert total_fact["equations"][0]["fact_equation"] == "f1 + f2"
    assert all(finding.get("code") != "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_preserves_subtractive_dash_sign() -> None:
    facts = [
        {"fact_num": 1, "value": "100", "path": ["A"]},
        {"fact_num": 2, "value": "-", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "100",
            "row_role": "total",
            "path": ["A", "total"],
            "equation": "100 - 0",
            "fact_equation": "f1 - f2",
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)
    total_fact = next(fact for fact in rebuilt if fact["fact_num"] == 3)
    assert total_fact["equations"] == [{"equation": "100 - 0", "fact_equation": "f1 - f2"}]
    assert all(finding.get("code") != "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_treats_negative_children_as_signed_references() -> None:
    facts = [
        {"fact_num": 1, "value": "(5)", "path": ["A"]},
        {"fact_num": 2, "value": "100", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "95",
            "row_role": "total",
            "path": ["A", "total"],
            "equation": "100 - 5",
            "fact_equation": "f2 + f1",
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)
    total_fact = next(fact for fact in rebuilt if fact["fact_num"] == 3)
    assert total_fact["equations"] == [{"equation": "100 - 5", "fact_equation": "f2 + f1"}]
    assert all(finding.get("code") != "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_supports_same_child_in_multiple_parents() -> None:
    facts = [
        {"fact_num": 1, "value": "100", "path": ["A"]},
        {
            "fact_num": 2,
            "value": "95",
            "row_role": "total",
            "fact_equation": "- f1",
            "path": ["A", "total"],
        },
        {
            "fact_num": 3,
            "value": "100",
            "row_role": "total",
            "fact_equation": "f1",
            "path": ["A", "reconciliation"],
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)

    by_num = {fact["fact_num"]: fact for fact in rebuilt}
    assert by_num[2]["equations"][0]["fact_equation"] == "- f1"
    assert by_num[3]["equations"][0]["fact_equation"] == "f1"
    assert all(finding.get("code") != "equation_graph_cycle" for finding in findings)


def test_audit_and_rebuild_financial_facts_flags_missing_refs_mismatch_and_period_issues() -> None:
    facts = [
        {
            "fact_num": 1,
            "value": "10",
            "period_type": "instant",
            "period_start": "2024-01-01",
            "path": ["A"],
        },
        {
            "fact_num": 2,
            "value": "9",
            "row_role": "total",
            "period_type": "duration",
            "period_start": "2023-01-01",
            "period_end": "2023-12-31",
            "date": "2024",
            "equation": "10 + 1",
            "fact_equation": "f1 + f99",
            "path": ["A", "total"],
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)
    codes = {str(finding.get("code")) for finding in findings}

    fact_1 = next(fact for fact in rebuilt if fact["fact_num"] == 1)
    assert fact_1["period_start"] is None
    assert "instant_period_start_not_null" in codes
    assert "fact_equation_missing_reference" in codes
    assert "equation_rebuild_unresolved" in codes
    assert "duration_date_year_mismatch" in codes


def test_audit_and_rebuild_financial_facts_warns_on_nonascending_fact_equation_order() -> None:
    facts = [
        {"fact_num": 1, "value": "10", "path": ["A"]},
        {"fact_num": 2, "value": "5", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "15",
            "row_role": "total",
            "fact_equation": "f2 + f1",
            "path": ["A", "total"],
        },
    ]

    _rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)

    warning = next(finding for finding in findings if finding.get("code") == "fact_equation_non_ascending_reference_order")
    assert warning["severity"] == "warning"
    assert warning["fact_num"] == 3


def test_audit_and_rebuild_financial_facts_allows_duration_without_range() -> None:
    facts = [
        {
            "fact_num": 1,
            "value": "10",
            "period_type": "duration",
            "period_start": None,
            "period_end": None,
            "path": ["A"],
        }
    ]

    _rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=False)

    assert all(finding.get("code") != "duration_missing_period_bounds" for finding in findings)


def test_remap_fact_equation_references_maps_resequenced_fact_nums() -> None:
    remapped = remap_fact_equation_references("f1 + f2 - f7", {1: 2, 2: 3, 7: 8})
    assert remapped == "f2 + f3 - f8"


def test_resequence_fact_numbers_and_remap_fact_equations_updates_references() -> None:
    facts = [
        {"fact_num": 4, "value": "900", "path": ["A"]},
        {"fact_num": 1, "value": "100", "path": ["A"]},
        {"fact_num": 2, "value": "20", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "path": ["A", "total"],
        },
    ]

    resequenced = resequence_fact_numbers_and_remap_fact_equations(facts)
    assert [fact["fact_num"] for fact in resequenced] == [1, 2, 3, 4]
    assert resequenced[3]["equations"][0]["fact_equation"] == "f2 + f3"
    assert resequenced[3]["equations"][0]["equation"] == "100 + 20"


def test_resequence_fact_numbers_remaps_saved_equation_variants() -> None:
    facts = [
        {"fact_num": 4, "value": "900", "path": ["A"]},
        {"fact_num": 1, "value": "100", "path": ["A"]},
        {"fact_num": 2, "value": "20", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {
                    "equation": "100 + 20",
                    "fact_equation": "f1 + f2",
                },
                {
                    "equation": "900 - 20",
                    "fact_equation": "f4 - f2",
                },
            ],
            "path": ["A", "total"],
        },
    ]

    resequenced = resequence_fact_numbers_and_remap_fact_equations(facts)
    active_total = resequenced[3]
    assert active_total["equations"][0]["fact_equation"] == "f2 + f3"
    assert active_total["equations"][0]["fact_equation"] == "f2 + f3"
    assert active_total["equations"][1]["fact_equation"] == "f1 - f3"
