from __future__ import annotations

from finetree_annotator.equation_integrity import (
    audit_and_rebuild_financial_facts,
    remap_fact_equation_references,
    resequence_fact_numbers_and_remap_fact_equations,
)


def test_audit_and_rebuild_financial_facts_repairs_roles_and_equation_from_references() -> None:
    facts = [
        {"fact_num": 10, "value": "269968", "natural_sign": "positive", "aggregation_role": None, "path": ["רכוש קבוע", "עלות"]},
        {
            "fact_num": 12,
            "value": "209255",
            "natural_sign": "positive",
            "aggregation_role": None,
            "path": ["רכוש קבוע", "בניכוי - פחת שנצבר"],
        },
        {
            "fact_num": 14,
            "value": "60713",
            "natural_sign": "positive",
            "aggregation_role": None,
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
    assert by_num[10]["aggregation_role"] == "additive"
    assert by_num[12]["row_role"] == "detail"
    assert by_num[12]["aggregation_role"] == "subtractive"
    assert by_num[14]["row_role"] == "total"
    assert by_num[14]["aggregation_role"] == "additive"
    assert by_num[14]["equation"] == "269968 - 209255"
    assert by_num[14]["fact_equation"] == "f10 - f12"
    assert all(finding.get("code") != "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_treats_dash_as_zero_in_equation() -> None:
    facts = [
        {"fact_num": 1, "value": "-", "aggregation_role": "additive", "path": ["A"]},
        {"fact_num": 2, "value": "5", "aggregation_role": "additive", "path": ["A"]},
        {
            "fact_num": 3,
            "value": "5",
            "row_role": "total",
            "aggregation_role": "additive",
            "path": ["A", "total"],
            "equation": "5",
            "fact_equation": "f1 + f2",
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(facts, apply_repairs=True)
    total_fact = next(fact for fact in rebuilt if fact["fact_num"] == 3)
    assert total_fact["equation"] == "0 + 5"
    assert total_fact["fact_equation"] == "f1 + f2"
    assert all(finding.get("code") != "equation_arithmetic_mismatch" for finding in findings)


def test_audit_and_rebuild_financial_facts_preserves_explicit_aggregation_role() -> None:
    facts = [
        {
            "fact_num": 12,
            "value": "209255",
            "natural_sign": "positive",
            "aggregation_role": "additive",
            "path": ["רכוש קבוע", "בניכוי - פחת שנצבר"],
        },
    ]

    rebuilt, findings = audit_and_rebuild_financial_facts(
        facts,
        statement_type="balance_sheet",
        apply_repairs=True,
    )

    assert rebuilt[0]["aggregation_role"] == "additive"
    assert all(
        not (
            finding.get("code") == "aggregation_role_corrected"
            and finding.get("fact_num") == 12
        )
        for finding in findings
    )


def test_audit_and_rebuild_financial_facts_flags_missing_refs_mismatch_and_period_issues() -> None:
    facts = [
        {
            "fact_num": 1,
            "value": "10",
            "aggregation_role": "additive",
            "period_type": "instant",
            "period_start": "2024-01-01",
            "path": ["A"],
        },
        {
            "fact_num": 2,
            "value": "9",
            "row_role": "total",
            "aggregation_role": "additive",
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


def test_remap_fact_equation_references_maps_resequenced_fact_nums() -> None:
    remapped = remap_fact_equation_references("f1 + f2 - f7", {1: 2, 2: 3, 7: 8})
    assert remapped == "f2 + f3 - f8"


def test_resequence_fact_numbers_and_remap_fact_equations_updates_references() -> None:
    facts = [
        {"fact_num": 4, "value": "900", "path": ["A"]},
        {"fact_num": 1, "value": "100", "path": ["A"]},
        {"fact_num": 2, "value": "20", "path": ["A"]},
        {"fact_num": 3, "value": "120", "equation": "100 + 20", "fact_equation": "f1 + f2", "path": ["A", "total"]},
    ]

    resequenced = resequence_fact_numbers_and_remap_fact_equations(facts)
    assert [fact["fact_num"] for fact in resequenced] == [1, 2, 3, 4]
    assert resequenced[3]["fact_equation"] == "f2 + f3"
    assert resequenced[3]["equation"] == "100 + 20"
