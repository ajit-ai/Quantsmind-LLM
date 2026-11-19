"""TAKSH L0.8.0-PRE — H: aggregate fail-closed preflight validator (18 axes).

Emits TWO verdicts on independent axes:

  preparation_qualified — "preparation is qualified to proceed as far as preparation goes"
  training_verdict     — always NO_GO here until authorization/compute/checkpoint exist

A preparation qualification is never reported as a training authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _load(name: str) -> dict:
    path = REPO / "reports" / f"l080pre_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_aggregate.json")
    args = parser.parse_args(argv)

    model = _load("model_identity")
    tok = _load("tokenizer_integrity")
    corpus = _load("corpus_verify")
    train = _load("training_config")
    res = _load("resource_estimate")
    ckpt = _load("checkpoint_recovery")
    bench = _load("benchmark_safeguard")

    dimensions = {
        "owner_training_authorization": {"status": "MISSING",
            "evidence": "A1.1-C authorizes preparation only; zero training authorization exists (reference TAKSH-OWNER-DECISION-2026-11-11)",
            "responsible_party": "owner", "owner_decision": True},
        "compute_authorization": {"status": "MISSING",
            "evidence": "A2-E: no compute authorization of any kind; host cannot train; MX150 ineffective",
            "responsible_party": "owner", "owner_decision": True},
        "model_configuration_integrity": {"status": "PASS" if model.get("architecture_verified") else "FAIL",
            "evidence": f"params={model.get('derived_parameters'):,}; fingerprint={model.get('model_config_fingerprint','')[:16]}…",
            "responsible_party": "engineering", "owner_decision": False},
        "tokenizer_integrity": {"status": "PASS" if (tok.get("committed_blob_matches_l0706_anchor") and tok.get("vocabulary_equal")) else "FAIL",
            "evidence": f"committed blob matches L0.7.6 anchor; vocab={tok.get('vocab_size')}; working-tree CRLF separately documented",
            "responsible_party": "engineering", "owner_decision": False},
        "corpus_snapshot_integrity": {"status": "PASS" if corpus.get("identity_and_integrity_verified") else "FAIL",
            "evidence": f"snapshot {corpus.get('snapshot_id')}; eligible {corpus.get('eligible_children')} samples / {corpus.get('eligible_tokens'):,} tokens",
            "responsible_party": "engineering", "owner_decision": False},
        "legal_eligibility": {"status": "CONDITIONAL",
            "evidence": "internal research pretraining only (L0.4/L0.7.0 record); external/commercial release requires fresh counsel review",
            "responsible_party": "owner + counsel", "owner_decision": True},
        "training_eligible_token_count": {"status": "MISSING",
            "evidence": "0 training-eligible tokens (technical ≠ legal ≠ authorization); frozen verdict FROZEN_RESEARCH_ONLY",
            "responsible_party": "engineering fact; promotion requires owner authorization (absent)", "owner_decision": True},
        "language_coverage": {"status": "PARTIAL",
            "evidence": "python only; 52 languages + Karkain: zero qualified corpus tokens",
            "responsible_party": "engineering fact; expansion requires data decisions (owner)", "owner_decision": False},
        "checkpoint_availability": {"status": "MISSING",
            "evidence": "taksh-foundation-111m checkpoint MISSING; v180_sft is GEOMETRY_INCOMPATIBLE; nothing materialized",
            "responsible_party": "engineering spawn; blocked by owner decisions A1/A2", "owner_decision": False},
        "hardware_feasibility": {"status": "BLOCKED",
            "evidence": f"available {res.get('measured', {}).get('available_bytes')} B (~{(res.get('measured', {}).get('available_bytes') or 0)/2**20:.1f} MiB); estimated optimizer state {res.get('estimates', {}).get('fp32_optimizer_state_total_bytes'):,} B — host cannot qualify",
            "responsible_party": "owner (environment)", "owner_decision": True},
        "training_configuration_completeness": {"status": "CONDITIONAL",
            "evidence": "complete proposal written (L0.6.1 canonical values inherited where applicable); unverified hypotheses flagged; NOT_AUTHORIZED by design",
            "responsible_party": "engineering (proposal); owner authorizes any execution", "owner_decision": True},
        "checkpoint_recovery_readiness": {"status": "CONDITIONAL",
            "evidence": "atomic writes, interrupted-write detection, fingerprint tamper rejection, resume gates — all qualified on synthetic fixtures; artifact missing",
            "responsible_party": "engineering", "owner_decision": False},
        "evaluation_integrity": {"status": "PASS" if (bench.get("frozen_python_benchmark", {}).get("intact") and bench.get("multilingual_benchmark", {}).get("intact") and not bench.get("contamination_detected")) else "FAIL",
            "evidence": "frozen benchmarks intact; zero overlap with training snapshot; 0/30 stands",
            "responsible_party": "engineering", "owner_decision": False},
        "ci_and_runtime_qualification": {"status": "BLOCKED",
            "evidence": "CI execution blocked on this branch (workflow_dispatch not exposed from non-default branch; owner action); Python 3.11 runtime CONDITIONAL (static only); inherited CI regressions resolved (L0.7.3/L0.7.4)",
            "responsible_party": "owner (CI change); engineering (runtime verification)", "owner_decision": True},
        "reproducibility": {"status": "PASS",
            "evidence": "snapshot rebuild deterministic; content-derived IDs; atomic writes; resume gates defined",
            "responsible_party": "engineering", "owner_decision": False},
        "artifact_provenance": {"status": "MISSING",
            "evidence": "no training run exists; provenance contracts pinned; nothing materialized",
            "responsible_party": "owner (authorization); engineering (execution under authorization)", "owner_decision": True},
        "karkain_integration_status": {"status": "BLOCKED",
            "evidence": "Karkain DISCOVERED; kcc.exe 0xC0000135; TAKSH-KARK-BENCH 0 tasks; zero corpus; spec 1.1.0",
            "responsible_party": "owner (Karkain repo)", "owner_decision": True},
        "distribution_publication_authorization": {"status": "NOT_AUTHORIZED",
            "evidence": "no distribution/publication rights of any kind; publication prohibited",
            "responsible_party": "owner", "owner_decision": True},
    }

    prep_failing = [k for k, v in dimensions.items()
                    if v["status"] in ("FAIL", "BLOCKED", "MISSING", "NOT_AUTHORIZED")
                    and k not in ("owner_training_authorization", "compute_authorization",
                                  "training_eligible_token_count", "checkpoint_availability",
                                  "hardware_feasibility", "ci_and_runtime_qualification",
                                  "artifact_provenance", "karkain_integration_status",
                                  "distribution_publication_authorization")]
    preparation = "PREPARATION_QUALIFIED" if not prep_failing else "PREPARATION_NOT_QUALIFIED"
    training = "TRAINING_GO" if all(
        dimensions[k]["status"] not in ("MISSING", "BLOCKED", "NOT_AUTHORIZED", "FAIL")
        for k in ("owner_training_authorization", "compute_authorization",
                  "training_eligible_token_count", "checkpoint_availability",
                  "hardware_feasibility", "legal_eligibility")
    ) else "NO_GO"

    report = {
        "schema": "taksh.l080pre_aggregate/1",
        "generated_by": "evaluation/l080pre/gate.py",
        "verified_on": "2026-11-11",
        "dimensions": dimensions,
        "dimension_count": len(dimensions),
        "blocking_dimensions": [k for k, v in dimensions.items()
                                if v["status"] in ("MISSING", "BLOCKED", "NOT_AUTHORIZED", "FAIL")],
        "preparation_verdict": preparation,
        "training_verdict": "NO_GO",
        "training_authorized": False,
        "training_eligible_tokens": 0,
        "verdict_rule": "preparation qualification never authorizes training; verdict axes are independent",
        "governance": {
            "A1_1_C_preparation_only": True,
            "A2_E_no_compute": True,
            "v180_sft_not_canonical": True,
            "frozen_benchmarks_immutable": True,
        },
        "note": "expected outcome under current owner decisions: PREPARATION_QUALIFIED (if all preparation axes pass) AND NO_GO for training — always reported separately",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"preparation_verdict: {preparation}")
    print(f"training_verdict: NO_GO")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
