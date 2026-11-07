"""Separate identifier-copy metrics (Increment 9, WP5).

Each metric is reported independently and never combined into one score:
exact identifier-copy, substitution, propagation, and multi-variable
binding accuracy, split by familiar versus unfamiliar identifiers.
Parse, binding, and functional execution remain scored by the existing
evaluators; this module only adds the copy-specific view.
"""

from __future__ import annotations

from evaluation.diagnostics import FAMILIAR_IDENTIFIERS, UNFAMILIAR_IDENTIFIERS, copy_case_bindings

COPY_METRICS = ("exact_copy", "substitution", "propagation", "multi_binding")


def _is_familiar(signature: str) -> bool:
    params = [arg.strip() for arg in signature.split("(", 1)[1].split(")", 1)[0].split(",") if arg.strip()]
    return any(param in FAMILIAR_IDENTIFIERS for param in params)


def score_copy_case(kind: str, signature: str, completion: str) -> dict:
    """Score one completion on copy-specific dimensions."""
    bindings = copy_case_bindings(completion, signature)
    substitution_ok = bool(bindings["used"]) and not bindings["invented_identifiers"] and (
        " = " in completion or ":=" in completion or "=" in completion
    )
    return {
        "kind": kind,
        "familiar": _is_familiar(signature),
        "parses": bindings["parses"],
        "exact_copy": bindings["exact_copy"],
        "substitution": substitution_ok,
        "propagation": any(bindings["used"].values()) and bindings["parses"],
        "multi_binding": sum(bindings["used"].values()) >= 2 and bindings["parses"],
    }


def copy_report(scored: list[dict]) -> dict:
    """Independent copy metrics, split familiar/unfamiliar."""
    report: dict = {}
    for label, subset in (
        ("all", scored),
        ("familiar", [r for r in scored if r["familiar"]]),
        ("unfamiliar", [r for r in scored if not r["familiar"]]),
    ):
        n = len(subset)
        report[label] = {
            "cases": n,
            **{
                metric: (round(sum(bool(r[metric]) for r in subset) / n, 4) if n else None)
                for metric in COPY_METRICS
            },
        }
    return report
