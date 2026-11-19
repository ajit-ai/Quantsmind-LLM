"""Promote-or-reject gate for v1.2.0 Increment 7 (generalization-first).

Reads the evaluation JSON (frozen AD-1/C3 via evaluate_v120_increment1.py
plus supplemental via analyze_v120_stabilization.py — pass the stabilization
JSON's supplemental block through by using its decision input shape, or an
evaluation JSON containing ``supplemental`` keys), verifies checkpoint and
fixture hashes, and applies :func:`training.v120_experiment7.decide_promotion7`.
Writes nothing to any checkpoint directory. Exit code 0 means the decision
was computed (check ``promote``); 2 means verification failed first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.experiment_artifacts import file_sha256
from training.v120_experiment7 import DEFAULT_CONFIG, decide_promotion7, load_config


def _summaries(payload: dict) -> tuple[dict, dict]:
    """Extract baseline/candidate summaries from an evaluation JSON.

    Expected layout: ``summary.{baseline,candidate}`` as written by
    ``scripts/evaluate_v120_increment1.py``, with each side additionally
    carrying a ``supplemental`` block (``{cases, report, by_behavior}``)
    from ``scripts/analyze_v120_stabilization.py``.
    """
    summary = payload["summary"]
    return summary["baseline"], summary["candidate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide whether an Increment 7 candidate may be promoted")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    baseline_path = Path(payload.get("baseline_checkpoint", config["baseline_checkpoint"]))
    candidate_path = Path(payload["candidate_checkpoint"])

    hashes_ok = True
    verification = {}
    if not baseline_path.is_file():
        hashes_ok = False
        verification["baseline"] = f"missing: {baseline_path}"
    else:
        baseline_sha = file_sha256(baseline_path)
        verification["baseline_sha256"] = baseline_sha
        if baseline_sha != config["baseline_checkpoint_sha256"]:
            hashes_ok = False
            verification["baseline"] = "hash mismatch"
    if not candidate_path.is_file():
        hashes_ok = False
        verification["candidate"] = f"missing: {candidate_path}"
    else:
        verification["candidate_sha256"] = file_sha256(candidate_path)
    for name in ("ad1", "c3"):
        fixture = config["frozen_fixtures"][name]
        path = Path(fixture["path"])
        if not path.is_file() or file_sha256(path) != fixture["sha256"]:
            hashes_ok = False
            verification[f"fixture_{name}"] = "hash mismatch"
        else:
            verification[f"fixture_{name}_sha256"] = fixture["sha256"]

    baseline, candidate = _summaries(payload)
    decision = decide_promotion7(
        baseline,
        candidate,
        hashes_ok=hashes_ok and not any("mismatch" in str(v) or "missing" in str(v) for v in verification.values()),
    )
    output = {
        "evaluation": str(args.evaluation),
        "verification": verification,
        "decision": decision,
    }
    print(json.dumps(output, indent=2))
    if not hashes_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
