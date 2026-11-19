"""Tests for M6 capability categories, evidence levels and compliance evidence.

The failure mode these guard against is a formatting accident being promoted to
a capability claim, so most tests assert that a level is *not* reached.
"""

from __future__ import annotations

from evaluation.m6_capability import (LEVELS, MIN_EXAMPLES_FOR_USEFUL,
                                      assess_levels, categories_from_manifest,
                                      compliance_evidence)

EVALUATED = {"status": "complete", "complete": True,
             "examples_evaluated": 2, "examples_total": 26}


def example(output, checks):
    return {"output": output,
            "metrics": {"compliance": {"denominator": len(checks),
                                       "numerator": sum(checks.values()),
                                       "checks": [
                                           {"rule": rule, "ok": ok}
                                           for rule, ok in checks.items()]}}}


def summary(*, syntax=(1, 20), functional=(0, 0), compliance=(4, 6)):
    def metric(n, d):
        return {"numerator": n, "denominator": d,
                "rate": (n / d) if d else None, "preliminary": True}
    return {"syntax": metric(*syntax), "functional": metric(*functional),
            "compliance": metric(*compliance)}


def test_no_checklist_means_all_categories_unavailable():
    results = categories_from_manifest({"by_task": {}})
    assert len(results) == 10
    assert all(r.status == "UNAVAILABLE" for r in results)
    assert all(r.eligible_examples == 0 for r in results)


def test_scored_category_reports_small_denominator():
    results = categories_from_manifest({"by_task": {
        "code_generation": {"syntax": {"numerator": 1, "denominator": 2,
                                       "rate": 0.5}}}})
    by_name = {r.category: r for r in results}
    assert by_name["code_generation"].status == "SCORED"
    assert by_name["code_generation"].eligible_examples == 2
    assert "preliminary" in by_name["code_generation"].note
    assert by_name["debugging"].status == "UNAVAILABLE"


def test_task_with_only_human_review_is_unavailable_not_zero():
    results = categories_from_manifest({"by_task": {
        "code_explanation": {"human_review": {"numerator": 0,
                                              "denominator": 0,
                                              "rate": None}}}})
    by_name = {r.category: r for r in results}
    assert by_name["code_explanation"].status == "UNAVAILABLE"
    assert by_name["code_explanation"].eligible_examples == 0


def test_eligible_count_ignores_a_leading_human_review_metric():
    # Regression: taking the first metric's denominator reported a scored task
    # as UNAVAILABLE whenever human_review was listed before syntax.
    results = categories_from_manifest({"by_task": {
        "code_completion": {
            "human_review": {"numerator": 0, "denominator": 0, "rate": None},
            "syntax": {"numerator": 1, "denominator": 20, "rate": 0.05},
            "compliance": {"numerator": 3, "denominator": 6, "rate": 0.5}}}})
    by_name = {r.category: r for r in results}
    assert by_name["code_completion"].status == "SCORED"
    assert by_name["code_completion"].eligible_examples == 20


def test_vacuous_rules_never_reach_level_five():
    evidence = compliance_evidence([
        example("some long prose answer", {"non_empty": True, "no_fences": True})
    ])
    assert evidence["substantive_passes"] == 0
    assert evidence["vacuous_passes"] == 2
    result = assess_levels({**EVALUATED, "summary": summary()},
                           compliance_evidence=evidence)
    assert result["highest_level"] == 3
    assert "level 5 not claimed" in result["note"]


def test_stub_answer_is_not_substantive_even_for_real_rule():
    evidence = compliance_evidence([example('\n    """', {"max_lines": True})])
    assert evidence["substantive_passes"] == 0
    assert evidence["vacuous_passes"] == 1


def test_real_rule_on_real_answer_reaches_level_five():
    evidence = compliance_evidence([
        example("def add(a, b):\n    return a + b\n", {"line_starts_with": True,
                                                      "non_empty": True})])
    assert evidence["substantive_passes"] == 1
    result = assess_levels({**EVALUATED, "summary": summary()},
                           compliance_evidence=evidence)
    assert result["highest_level"] == 5


def test_missing_evidence_does_not_credit_level_five():
    result = assess_levels({**EVALUATED, "summary": summary()},
                           compliance_evidence=None)
    assert 5 not in result["reached"]
    assert "uncorroborated" in result["note"]


def test_functional_metric_with_zero_denominator_never_reaches_four():
    result = assess_levels({**EVALUATED, "summary": summary(functional=(0, 0))},
                           compliance_evidence={})
    assert 4 not in result["reached"]


def test_level_six_needs_multiple_passes_and_a_large_benchmark():
    large = {**EVALUATED, "examples_total": MIN_EXAMPLES_FOR_USEFUL}
    single = assess_levels({**large, "summary": summary(functional=(1, 20))},
                           compliance_evidence={})
    assert 6 not in single["reached"]
    double = assess_levels({**large, "summary": summary(functional=(2, 20))},
                           compliance_evidence={})
    assert 6 in double["reached"]
    assert 7 not in double["reached"]


def test_embedding_head_is_capped_at_generation():
    result = assess_levels({**EVALUATED, "summary": summary(compliance=(4, 6))},
                           task_head="embedding",
                           compliance_evidence={"substantive_passes": 4})
    assert result["highest_level"] == 2
    assert "formatting artifacts" in result["note"]


def test_not_evaluated_claims_no_level():
    result = assess_levels({"status": "blocked", "complete": False})
    assert result["highest_level"] == 0
    assert result["highest_name"] == "NOT_EVALUATED"


def test_levels_are_ordered_and_named():
    assert [level.level for level in LEVELS] == list(range(1, 8))
    assert LEVELS[0].name == "RUNTIME_WORKS"
    assert LEVELS[-1].name == "PRODUCT_READINESS_ESTABLISHED"
