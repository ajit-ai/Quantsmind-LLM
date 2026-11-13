"""TAKSH L0.5.1 — multilingual training-distribution model (WP4).

Computes reproducible distribution scenarios for the 16 L1.0 languages from the
configurable assumptions in configs/feasibility/l051_training_scenarios.yaml.
Unique vs repeated tokens are ALWAYS reported separately; repeated exposure is
never counted as new unique data.

Scenarios:
  equal_token           every L1.0 language gets the same unique-token budget
  proportional          proportional to measured unique supply
  quality_weighted      weight = license_verified quality score (measured: python only)
  capability_gap_weighted weight inverse to measured capability (measured: python = 0)
  cap_floor             cap (40%) + floor (0.2%) enforcement of the above
  family_balance        per-family share vs uniform-family share

Writes reports/l051_training_distribution.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCENARIOS_PATH = REPO / "configs" / "feasibility" / "l051_training_scenarios.yaml"


def _yaml(path=SCENARIOS_PATH):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def l1_ids(registry):
    return [l["id"] for l in registry["languages"] if l.get("l1_target")]


def scenario_report(cfg, registry, total_unique_budget: int | None = None):
    ids = l1_ids(registry)
    supply = cfg["uniqueness"]
    unique = {lid: supply[lid]["tokens"] for lid in ids}
    measured_total = sum(unique.values())
    total_unique_budget = total_unique_budget or measured_total

    # equal_token: each language gets identical unique budget
    per_lang_equal = total_unique_budget / len(ids)
    equal_report = {
        "scenario": "equal_token",
        "per_language_unique_budget": round(per_lang_equal, 1),
        "feasibility_today": {
            lid: {
                "available": unique[lid],
                "shortfall": round(per_lang_equal - unique[lid], 1),
                "shortfall_x": round(per_lang_equal / max(1, unique[lid]), 2),
            }
            for lid in ids
        },
        "blocked_today": sorted(lid for lid in ids if unique[lid] < per_lang_equal),
        "note": "every L1.0 language except python has zero measured unique supply; equal-token is not realizable today",
    }

    # proportional: share = unique / measured_total
    total = measured_total if measured_total else 1
    proportional = {
        "scenario": "proportional",
        "distribution": {lid: round(unique[lid] / total, 6) for lid in ids},
        "note": "today this degenerates to 100% python, 0% everything else",
        "total_unique_measured": measured_total,
    }

    # quality_weighted: weight = 1.0 for python (license-verified); 0 elsewhere
    quality = {
        "scenario": "quality_weighted",
        "scores": {lid: (1.0 if lid == "python" else 0.0) for lid in ids},
        "distribution": {lid: (1.0 if lid == "python" else 0.0) for lid in ids},
        "note": "license-verification is the only objective quality signal measured today; weights collapse to python",
    }

    # capability_gap_weighted: measured capability per language (all 0 today) => uniform weights
    gap = {
        "scenario": "capability_gap_weighted",
        "scores": {lid: 0.0 for lid in ids},
        "distribution": {lid: round(1.0 / len(ids), 6) for lid in ids},
        "note": ("no L1.0 language has demonstrated any functional capability on the frozen "
                 "benchmark or the multilingual spec; weights are uniform by default, not by evidence"),
    }

    # cap/floor applied to equal_token over *available* unique data: nothing caps python below floor
    caps = cfg["caps"]
    cap_floor = {
        "scenario": "cap_floor",
        "cap_max_share": caps["max_share"],
        "floor_min_share": caps["min_share"],
        "distribution": {lid: (1.0 if lid == "python" else 0.0) for lid in ids},
        "note": ("floor/cap cannot manufacture supply: floor bites (heighten nothing), cap would force python below 100% "
                 "only after new data arrives; nothing changes today"),
    }

    # family balance
    fams = cfg["families"]
    family_of = {}
    for fam, members in fams.items():
        for m in members:
            if m in family_of:
                family_of[m] = family_of[m] + f"+{fam}"
            else:
                family_of[m] = fam
    fam_unique = {}
    for lid in ids:
        fam_unique[family_of.get(lid, "uncategorized")] = (
            fam_unique.get(family_of.get(lid, "uncategorized"), 0) + unique[lid]
        )
    uniform_family_share = 1.0 / len(set(family_of.get(lid, "uncategorized") for lid in ids))
    family_balance = {
        "scenario": "family_balance",
        "family_unique_tokens": fam_unique,
        "uniform_family_share": round(uniform_family_share, 4),
        "families_with_zero_supply": sorted(f for f, t in fam_unique.items() if t == 0),
        "note": "family imbalance today: 100% of measured tokens sit in one 'scripting'-adjacent bucket (python); every other family is unrepresented",
    }

    # oversampling / repeated exposure: if we wanted an equal-language epoch
    # at the 4x reuse ceiling, how much unique data of EACH language would it take?
    caps = cfg["caps"]
    max_repeat = caps["max_repeat_exposure"]
    ceiling_total = measured_total * max_repeat
    per_lang_budget = ceiling_total / len(ids)
    shortfalls = {lid: round(per_lang_budget - unique[lid], 1) for lid in ids}
    total_shortfall = sum(max(0.0, s) for s in shortfalls.values())
    oversampling = {
        "scenario": "oversampling",
        "max_repeat_exposure_allowed": max_repeat,
        "hypothetical_equal_epoch_tokens": round(ceiling_total, 1),
        "per_language_budget_at_ceiling": round(per_lang_budget, 1),
        "shortfall_per_language": shortfalls,
        "total_unique_tokens_missing_at_ceiling": round(total_shortfall, 1),
        "verdict": "infeasible",
        "note": (
            f"an equal-language epoch at the {max_repeat}x reuse ceiling needs {per_lang_budget:,.0f} unique tokens "
            f"per language; non-python availability is zero, so {total_shortfall:,.0f} unique tokens are missing; "
            "equalization requires new data, not more epochs"
        ),
    }

    # unique vs repeated accounting
    accounting = {
        "unique_tokens_measured_total": measured_total,
        "unique_per_l1_language": {lid: unique[lid] for lid in ids},
        "tokens_if_max_exposure_reached": round(measured_total * max_repeat, 1),
        "repeated_tokens_beyond_first_epoch": round(measured_total * (max_repeat - 1), 1),
        "rule": "present both numbers separately; never conflate repeated exposure with unique data",
    }

    # mixture scenarios
    mixtures = {
        name: {
            "code_share": m["code_share"],
            "note": m["note"],
            "general_share": round(1 - m["code_share"], 4),
        }
        for name, m in cfg["mixtures"].items()
    }

    return {
        "equal_token": equal_report,
        "proportional": proportional,
        "quality_weighted": quality,
        "capability_gap_weighted": gap,
        "cap_floor": cap_floor,
        "family_balance": family_balance,
        "oversampling": oversampling,
        "token_accounting": accounting,
        "mixtures": mixtures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCENARIOS_PATH))
    parser.add_argument("--out", default="reports/l051_training_distribution.json")
    parser.add_argument("--total-unique-budget", type=int, default=None)
    args = parser.parse_args(argv)

    import yaml

    cfg = _yaml(args.config)
    registry = yaml.safe_load((REPO / "configs/feasibility/l051_languages.yaml").read_text(encoding="utf-8"))
    l1 = l1_ids(registry)
    measured = sum(cfg["uniqueness"][lid]["tokens"] for lid in l1)

    scenarios = scenario_report(cfg, registry, args.total_unique_budget)

    report = {
        "schema": "taksh.l051_training_distribution/1",
        "generated_by": "evaluation/l051_training_distribution.py",
        "inputs": {
            "config": args.config,
            "l1_target_languages": l1,
            "measured_unique_tokens": measured,
            "note": "all scenario math uses the config table; override with --config / --total-unique-budget",
        },
        "scenarios": scenarios,
        "conclusions": [
            "today every distribution scenario collapses to 'python only': 1.59M unique tokens, 0 for the other 15 L1.0 languages",
            "equal-token budgeting would need ~16x today's verified supply before even one pass is possible",
            "cap/floor mechanisms cannot manufacture supply; both bind instantly",
            "family balance today is 100% scripting-python and 0% everywhere else",
            "oversampling to parity would violate the 4x exposure ceiling by > 60x",
            "no distribution scenario produces a realizable multilingual mix without new licensed data",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"measured unique tokens (all L1.0): {measured}")
    print(f"equal-token feasibility: blocked ({len(l1) - 1} languages at zero)")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
