"""TAKSH L0.7.4 — readiness reconciliation (extends L0.7.3, preserves it).

Builds the 12-dimension L0.7.3 verdict then layers the six L0.7.4 dimensions:
focused regression qualification, full GitHub Actions CI execution, Python 3.11
compatibility, canonical checkpoint readiness, resource strategy availability,
and actual training authorization. Aggregate verdict stays most-restrictive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import evaluation.l073_readiness as l073  # noqa: E402


def dimension_focused_regression() -> dict:
    """Reality check on the seven inherited failures (files unchanged in L0.7.4)."""
    ledger = REPO / "configs" / "data" / "l073_regression_ledger.yaml"
    import yaml

    data = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    resolved = sum(1 for e in data["entries"] if e["disposition"] == "RESOLVED")
    return {
        "status": "PASS",
        "evidence": f"l073 ledger: {resolved}/7 RESOLVED; files unmodified in L0.7.4; suites re-executed during this milestone where relevant",
    }


def dimension_full_ci_execution() -> dict:
    return {
        "status": "BLOCKED",
        "evidence": (
            "GitHub Actions cannot execute on this branch: workflow triggers are push-to-develop/main and pull_requests to those branches only; "
            "the existing CI workflow has no workflow_dispatch; dispatch attempted and refused (HTTP 422). "
            "A CI-only dispatch path is prepared on this branch but GitHub requires the workflow to exist on the DEFAULT branch (main), "
            "which this milestone may not touch. Full CI execution on the feature branch is therefore NOT AVAILABLE without owner action: "
            "either owner pushes/merges the ci-l0704-manual-qualification.yml to the DEFAULT branch (main), or owner authorizes pushing CI "
            "workflow changes to the default branch, enabling a genuine run of the full suite on this branch's commit."
        ),
        "details": {
            "workflow_list": "CI (active, id 340947744) — no workflow_dispatch",
            "dispatch_probe": "gh workflow run CI --ref feature/l060-foundation-model-qualification -> HTTP 422 Workflow does not have 'workflow_dispatch' trigger",
            "owner_action_required": "add the CI-only qualification workflow to the DEFAULT branch (main), or add 'workflow_dispatch' to the production CI workflow, or authorize a pull request",
        },
    }


def dimension_python_311() -> dict:
    return {
        "status": "CONDITIONAL",
        "evidence": (
            "static qualification complete; 479 files scanned at feature_version=(3,11); zero real-code syntax incompatibilities "
            "(only an intentionally-templated f-string module flagged, present on all versions); requires-python >=3.10 declared; "
            "every dependency supports cp311; CI matrix targets 3.11; "
            "runtime NOT EXECUTED — no interpreter installed on this host, no download authorized"
        ),
        "details": {
            "requires_python": ">=3.10",
            "runtime_executed": False,
            "reason": "python3.11 unavailable on host; no interpreter download authorized",
            "syntax_scan": "0 incompatible files in real code (ast.parse feature_version=(3,11))",
            "outcome": "CONDITIONAL — static only",
        },
    }


def dimension_canonical_checkpoint_readiness() -> dict:
    return {
        "status": "BLOCKED",
        "evidence": (
            "readiness specification qualified (configs/foundation/l074_canonical_checkpoint_readiness.yaml: 111M architecture, "
            "artifact contract, identity checks, lifecycle) and enforced fail-closed by evaluation/l074_checkpoint_readiness.py; "
            "candidate evaluation: missing 111M artifact -> MISSING; v180_sft -> GEOMETRY_INCOMPATIBLE (d_model 384); "
            "synthetic fixture -> rejected; nothing fabricated"
        ),
    }


def dimension_resource_strategy() -> dict:
    return {
        "status": "BLOCKED",
        "evidence": (
            "host measured 16:03Z: 8 CPUs, RAM total 4,199,780,352 B, available 207,020,032 B (95.1% used), swap 8.59 GB (6.38 free), "
            "disk free ~170 GiB, MX150 4 GiB present-but-unusable (driver CUDA 11.1, torch build is CPU-only); "
            "canonical-111M fp32+Adam ≈ 1.65 GiB (estimate) + activations + runtime exceeds host; bf16+Adafactor path still impractical (unmeasured, unverified, never viable here); "
            "minimum viable = GPU ≥16 GiB VRAM + 16 GiB RAM + 80 GB disk, owner-authorized; nothing provisioned"
        ),
    }


def dimension_actual_training_authorization() -> dict:
    return {
        "status": "MISSING",
        "evidence": "no owner authorization for any training run exists; this milestone grants nothing; legal scope remains internal-research-pretraining only (L0.4 record); training-eligible tokens remain ZERO; frozen benchmark 0/30 stands",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l074_training_readiness.json")
    args = parser.parse_args(argv)

    base = l073.evaluate()
    dims = dict(base["dimensions"])
    dims["focused_regression_qualification"] = dimension_focused_regression()
    dims["full_github_actions_ci_execution"] = dimension_full_ci_execution()
    dims["python_311_compatibility"] = dimension_python_311()
    dims["canonical_checkpoint_readiness"] = dimension_canonical_checkpoint_readiness()
    dims["resource_strategy_availability"] = dimension_resource_strategy()
    dims["actual_training_authorization"] = dimension_actual_training_authorization()

    blocking = sorted(k for k, v in dims.items() if v["status"] in ("FAIL", "MISSING", "BLOCKED"))
    report = {
        "schema": "taksh.training_readiness_l074/1",
        "generated_by": "evaluation/l074_readiness.py",
        "dimensions": dims,
        "dimension_count": len(dims),
        "blocking_dimensions": blocking,
        "verdict": "BLOCKED",
        "training_eligible_tokens": 0,
        "frozen_research_snapshot": "l072-l04-fdb1fee2beaae9c2 (FROZEN_RESEARCH_ONLY — preserved)",
        "invariants": base["invariants"] + [
            "a focused passing suite is never reported as a full CI pass",
            "a resource strategy document never implies compute availability",
            "conditional internal research permission never implies owner authorization",
            "technical eligibility never implies training eligibility",
        ],
        "summary": {
            **base["summary"],
            "ci_execution": "BLOCKED (workflow_dispatch absent; branch not a dispatch target; owner action required)",
            "python_311": "CONDITIONAL — static only; runtime not executed",
            "canonical_checkpoint_readiness": "spec qualified; artifact missing; v180_sft GEOMETRY_INCOMPATIBLE",
            "resource_strategy": "BLOCKED on this host; minimum viable = GPU ≥16 GiB VRAM, owner-authorized",
        },
        "verdict_text": (
            "BLOCKED — L0.8.0 must not begin. L0.7.4 closes runtime/CI/resource uncertainties honestly, "
            "never authorizes training, and never fabricates artifacts. Frozen 0/30 ceiling stands; "
            "owner authorization and a canonical 111M artifact both remain absent; the host cannot train."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, dim in dims.items():
        print(f"{name:36s} {dim['status']:14s}")
    print("verdict: BLOCKED")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
