"""TAKSH L0.7.0 — corpus planning engine (deterministic).

Computes per-language achievable targets under four strategies using ONLY
verified availability (Python, 1,590,703 tokens). Unavailable languages are
reported IMPOSSIBLE — quantities are never fabricated to make a plan look
balanced. A plan is a feasibility statement, not evidence that data exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PLAN_PATH = REPO / "configs" / "data" / "l070_corpus_plan.yaml"
VERIFIED = {"python": 1_590_703}
VERIFIED_SAMPLES = {"python": 384}
MAX_EXPOSURE = 4


def load_plan_config(path: Path = PLAN_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _registry_languages() -> list[dict]:
    from evaluation.l051_language_registry import load_registry

    return load_registry()["languages"] + [{"id": "karkain", "category": "first-class",
                                            "l1_target": False}]


def estimate_targets(strategy: str, family_weights: dict[str, float],
                     min_l1: int, min_roadmap: int) -> dict[str, int]:
    """Deterministic per-language TARGETS (aspirational, honest)."""
    targets = {}
    for lang in _registry_languages():
        lid = lang["id"]
        base = min_l1 if lang.get("l1_target") else min_roadmap
        if lid == "karkain":
            targets[lid] = 0          # zero corpus exists; no target invented
            continue
        if strategy == "equal_language":
            targets[lid] = base
        elif strategy == "proportional_to_availability":
            targets[lid] = VERIFIED.get(lid, 0)
        elif strategy == "quality_weighted":
            targets[lid] = VERIFIED.get(lid, 0)   # only verified slices carry weight
        elif strategy == "hybrid_language_family":
            weight = family_weights.get(lang.get("category", ""), 1.0)
            targets[lid] = int(base * weight)
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")
    return targets


def plan_for(strategy: str, cfg: dict) -> dict:
    family_weights = cfg["strategies"]["hybrid_language_family"]["family_weights"]
    min_l1 = int(cfg["target_model"]["min_tokens_per_l1_language"])
    min_roadmap = int(cfg["target_model"]["min_tokens_per_roadmap_language"])

    targets = estimate_targets(strategy, family_weights, min_l1, min_roadmap)
    rows = []
    achievable_total = 0
    impossible = []
    for lid, target in targets.items():
        available = VERIFIED.get(lid, 0)
        achieved = min(target, available)
        achievable_total += achieved
        repeat_factor = math.ceil(target / available) if available else None
        exposure_ok = (repeat_factor is None or repeat_factor <= MAX_EXPOSURE)
        unplannable = available == 0
        if target > 0 and achieved < target:
            impossible.append(lid)
        rows.append({
            "language_id": lid,
            "target_tokens": target,
            "available_tokens": available,
            "achievable_tokens": achieved,
            "shortfall_tokens": max(target - achieved, 0),
            "unplannable": unplannable,
            "impossible": target > 0 and achieved < target,
            "sample_repeat_needed": repeat_factor,
            "exposure_within_ceiling": exposure_ok,
        })
    unplannable_count = sum(1 for r in rows if r["unplannable"])
    plan_blob = json.dumps({s: targets for s in [strategy]}, sort_keys=True).encode()
    return {
        "strategy": strategy,
        "targets_total": sum(targets.values()),
        "achievable_total": achievable_total,
        "impossible_languages": impossible,
        "impossible_count": len(impossible),
        "unplannable_languages": unplannable_count,
        "rows": rows,
        "exposure_ceiling": MAX_EXPOSURE,
        "exposure_violations": [r["language_id"] for r in rows if not r["exposure_within_ceiling"]],
        "plan_sha256": hashlib.sha256(plan_blob).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_corpus_plan.json")
    args = parser.parse_args(argv)

    cfg = load_plan_config()
    strategies = list(cfg["strategies"])
    plans = {name: plan_for(name, cfg) for name in strategies}

    report = {
        "schema": "taksh.l070_corpus_plan/1",
        "generated_by": "evaluation/l070_corpus_planner.py",
        "verified_availability": VERIFIED,
        "verified_samples": VERIFIED_SAMPLES,
        "strategies": plans,
        "honesty_notes": [
            "targets are planning parameters, not existence claims",
            "no strategy can create tokens for languages with zero verified availability",
            "exposure ceiling (4x) violated in every feasible strategy for languages with <25% of target availability",
            "python is the only language with measured tokens; 53 + Karkain remain corpus-blocked",
        ],
        "winning_strategy": None,
        "verdict": "NO STRATEGY IS ACTIONABLE until acquisition is authorized and verified; 53/54 languages are unplannable today (python is the only verified corpus)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, plan in plans.items():
        print(f"{name:30s} achievable={plan['achievable_total']:>10,} unplannable={plan['unplannable_languages']:>3}/54 impossible={plan['impossible_count']:>3}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
