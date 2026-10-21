"""Metric aggregation for the capability evaluation (Taksh v1.4.0).

Every rate carries its denominator, and a metric that could not be computed
is reported as ``None`` with a reason rather than as zero. A zero would read
as "the model failed every case"; the truth is usually "the measurement was
not possible here".
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _rate(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return numerator / denominator


def aggregate(records: list) -> dict:
    """Aggregate per-case records into the reported metric set."""
    total = len(records)
    excluded = [r for r in records if r.get("outcome") == "excluded"]
    blocked = [r for r in records if r.get("outcome") == "blocked"]
    evaluated = [r for r in records if r.get("outcome") in ("pass", "fail")]

    syntax_records = [
        r for r in records if r.get("outcome") != "excluded" and r.get("syntax_valid") is not None
    ]
    syntax_valid = sum(1 for r in syntax_records if r.get("syntax_valid"))
    functional_records = [r for r in evaluated if r.get("functional") is not None]
    functional_pass = sum(1 for r in functional_records if r.get("functional"))
    format_records = [r for r in evaluated if r.get("format_compliant") is not None]
    format_ok = sum(1 for r in format_records if r.get("format_compliant"))
    completion_records = [r for r in evaluated if r.get("category") in (
        "function_completion",
        "small_algorithm",
        "string_and_list",
        "control_flow",
    )]
    completion_ok = sum(1 for r in completion_records if r.get("outcome") == "pass")

    latencies = [r.get("latency_seconds") for r in evaluated if r.get("latency_seconds") is not None]
    tokens = [r.get("tokens_generated") for r in evaluated if r.get("tokens_generated") is not None]
    timeouts = sum(1 for r in evaluated if r.get("timed_out"))
    refusals = sum(1 for r in evaluated if r.get("refused"))

    by_category = defaultdict(lambda: {"total": 0, "passed": 0, "syntax_valid": 0})
    by_format = defaultdict(lambda: {"total": 0, "passed": 0, "syntax_valid": 0})
    for record in records:
        if record.get("outcome") == "excluded":
            continue
        entry = by_category[record.get("category", "unknown")]
        entry["total"] += 1
        if record.get("outcome") == "pass":
            entry["passed"] += 1
        if record.get("syntax_valid"):
            entry["syntax_valid"] += 1
        fmt = by_format[record.get("prompt_format", "unknown")]
        fmt["total"] += 1
        if record.get("outcome") == "pass":
            fmt["passed"] += 1
        if record.get("syntax_valid"):
            fmt["syntax_valid"] += 1

    failure_categories = Counter(
        r.get("failure_category") for r in evaluated if r.get("outcome") != "pass"
    )

    return {
        "cases_total": total,
        "cases_evaluated": len(evaluated),
        "cases_blocked": len(blocked),
        "cases_excluded": len(excluded),
        "syntax_valid_count": syntax_valid,
        "syntax_valid_denominator": len(syntax_records),
        "syntax_valid_rate": _rate(syntax_valid, len(syntax_records)),
        "functional_pass_count": functional_pass,
        "functional_pass_denominator": len(functional_records),
        "functional_pass_rate": _rate(functional_pass, len(functional_records)),
        "functional_measurement_available": bool(functional_records),
        "completion_success_count": completion_ok,
        "completion_success_denominator": len(completion_records),
        "completion_success_rate": _rate(completion_ok, len(completion_records)),
        "format_compliance_count": format_ok,
        "format_compliance_denominator": len(format_records),
        "format_compliance_rate": _rate(format_ok, len(format_records)),
        "mean_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "total_tokens_generated": sum(tokens) if tokens else 0,
        "timeout_count": timeouts,
        "timeout_rate": _rate(timeouts, len(evaluated)),
        "refusal_count": refusals,
        "refusal_rate": _rate(refusals, len(evaluated)),
        "failure_categories": dict(sorted(failure_categories.items())),
        "by_category": {
            category: {
                **counts,
                "pass_rate": _rate(counts["passed"], counts["total"]),
            }
            for category, counts in sorted(by_category.items())
        },
        "by_prompt_format": {
            name: {
                **counts,
                "pass_rate": _rate(counts["passed"], counts["total"]),
                "syntax_valid_rate": _rate(counts["syntax_valid"], counts["total"]),
            }
            for name, counts in sorted(by_format.items())
        },
    }
