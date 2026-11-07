"""Stabilization decision for Taksh v1.2.0 Increment 6.

Pure decision logic over the evidence from the frozen AD-1/C3 suites and
the supplemental behavior suite. The thresholds below are stated here so
they can be reviewed before they are applied; they were chosen to be
strictly stronger than the Increment 5 promote gate:

* ``AD1_BINDING_LOSS_LIMIT = 1``: AD-1 binding is a retained capability of
  the baseline (6/29). Losing more than one case means the candidate traded
  away core name-copying ability for narrow gains. One case of noise is
  tolerated; four (Increment 5) is not.
* ``MAX_BEHAVIOR_REGRESSIONS = 1``: of the eight supplemental behaviours,
  the candidate may regress functionally on at most one relative to the
  baseline. Regressing on two or more means the improvement is not broad
  coverage.
* Functional improvement means strictly greater aggregate functional rate
  AND strictly greater functional accuracy conditional on binding (baseline
  ``None`` counts as 0.0). Binding-only gains (Increment 3 pattern) therefore
  cannot satisfy this rule.
* Mode collapse (candidate, either frozen suite) or an AD-1 parse fall is
  an automatic short-circuit: with a functional regression it is D,
  otherwise it blocks A/B and yields C unless functional improved (then B).

Decisions: A (repeatable functional evidence), B (narrow improvement,
more controlled work needed), C (no meaningful improvement), D (regresses
important capabilities).
"""

from __future__ import annotations

AD1_BINDING_LOSS_LIMIT = 1
MAX_BEHAVIOR_REGRESSIONS = 1
EXPECTED_BEHAVIORS = 8


def decide_stabilization(
    *,
    baseline_functional_rate: float,
    candidate_functional_rate: float,
    baseline_conditional: float | None,
    candidate_conditional: float | None,
    behavior_regressions: int,
    ad1_binding_loss: int,
    ad1_parse_fall: bool,
    candidate_collapsed: bool,
) -> dict:
    baseline_conditional = 0.0 if baseline_conditional is None else baseline_conditional
    candidate_conditional = 0.0 if candidate_conditional is None else candidate_conditional
    functional_improved = candidate_functional_rate > baseline_functional_rate
    conditional_improved = candidate_conditional > baseline_conditional
    functional_regressed = candidate_functional_rate < baseline_functional_rate

    narrow = (
        ad1_binding_loss > AD1_BINDING_LOSS_LIMIT
        or behavior_regressions > MAX_BEHAVIOR_REGRESSIONS
    )

    if functional_regressed or ad1_parse_fall:
        decision = "D"
        reasons = []
        if functional_regressed:
            reasons.append(
                f"functional accuracy regressed: {candidate_functional_rate:.1%} vs {baseline_functional_rate:.1%}"
            )
        if ad1_parse_fall:
            reasons.append("AD-1 parse rate fell below baseline")
    elif not (functional_improved and conditional_improved):
        decision = "C"
        reasons = []
        if not functional_improved:
            reasons.append(
                f"functional accuracy did not improve: {candidate_functional_rate:.1%} vs {baseline_functional_rate:.1%}"
            )
        if not conditional_improved:
            reasons.append(
                "functional accuracy conditional on binding did not improve: "
                f"{candidate_conditional:.1%} vs {baseline_conditional:.1%}"
            )
    elif narrow or candidate_collapsed:
        decision = "B"
        reasons = []
        if ad1_binding_loss > AD1_BINDING_LOSS_LIMIT:
            reasons.append(
                f"AD-1 binding loss {ad1_binding_loss} exceeds the limit of {AD1_BINDING_LOSS_LIMIT}"
            )
        if behavior_regressions > MAX_BEHAVIOR_REGRESSIONS:
            reasons.append(
                f"candidate regresses functionally on {behavior_regressions} behaviours "
                f"(limit {MAX_BEHAVIOR_REGRESSIONS})"
            )
        if candidate_collapsed:
            reasons.append("mode collapse detected in candidate")
    else:
        decision = "A"
        reasons = [
            "functional accuracy and conditional functional accuracy improved, "
            "AD-1 binding loss within limit, per-behaviour coverage held, no collapse"
        ]

    return {
        "decision": decision,
        "reasons": reasons,
        "functional_improved": functional_improved,
        "conditional_improved": conditional_improved,
        "ad1_binding_loss": ad1_binding_loss,
        "behavior_regressions": behavior_regressions,
        "thresholds": {
            "ad1_binding_loss_limit": AD1_BINDING_LOSS_LIMIT,
            "max_behavior_regressions": MAX_BEHAVIOR_REGRESSIONS,
        },
    }


def per_behavior_summary(records: list[dict]) -> dict:
    """Aggregate scored supplemental records by behaviour.

    Records need ``behavior``, ``parses``, ``exact``, ``structural``,
    ``binding``, ``functional`` (``{"passed": bool|None}``).
    """
    by_behavior: dict[str, dict] = {}
    for record in records:
        entry = by_behavior.setdefault(
            record["behavior"],
            {"cases": 0, "parses": 0, "exact": 0, "structural": 0, "binding": 0, "functional_passed": 0, "functional_scored": 0},
        )
        entry["cases"] += 1
        entry["parses"] += bool(record["parses"])
        entry["exact"] += bool(record["exact"])
        entry["structural"] += bool(record["structural"])
        entry["binding"] += bool(record["binding"])
        if record.get("functional", {}).get("passed") is not None:
            entry["functional_scored"] += 1
            entry["functional_passed"] += bool(record["functional"]["passed"])
    for entry in by_behavior.values():
        entry["functional_rate"] = (
            entry["functional_passed"] / entry["functional_scored"] if entry["functional_scored"] else None
        )
    return dict(sorted(by_behavior.items()))


def compare_behaviors(baseline: dict, candidate: dict) -> dict:
    """Count improved/regressed/matched behaviours and per-behavior records."""
    improved, regressed, matched = [], [], []
    details = {}
    for behavior in sorted(set(baseline) | set(candidate)):
        base = baseline.get(behavior, {"functional_rate": None, "functional_scored": 0})
        cand = candidate.get(behavior, {"functional_rate": None, "functional_scored": 0})
        base_rate = base["functional_rate"] or 0.0
        cand_rate = cand["functional_rate"] or 0.0
        details[behavior] = {"baseline": base_rate, "candidate": cand_rate}
        if cand_rate > base_rate:
            improved.append(behavior)
        elif cand_rate < base_rate:
            regressed.append(behavior)
        else:
            matched.append(behavior)
    return {
        "improved": improved,
        "regressed": regressed,
        "matched": matched,
        "details": details,
        "regression_count": len(regressed),
    }
