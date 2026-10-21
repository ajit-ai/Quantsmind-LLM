"""Read-only corpus readiness report (Taksh v1.5.0).

The command answers "could production training start today?" without changing
anything. It never approves a source, never writes to the registry, and never
treats the absence of evidence as permission. With zero approved sources the
answer is always "not authorized", which is the correct state today.

Usage
-----
    python -m data.python_v1.readiness --contract data/contracts/python_v1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.python_v1.benchmarks import load_benchmark_records, load_registry
from data.python_v1.decision_register import ROADMAP_DECISIONS as _DECISION_STATUS
from data.python_v1.registry import load_contract

DECISION_REGISTER = {
    "D1": "source licensing: unresolved",
    "D2": "permitted use: unresolved",
    "D3": "approved-source allowlist: empty",
    "D4": "provenance evidence: unresolved",
    "D5": "redistribution/notice obligations: unresolved",
    "D6": "tokenizer vocabulary and model geometry: open",
}


def assess(contract_path) -> dict:
    """Produce the readiness report for a contract. Read-only."""
    contract = load_contract(contract_path)
    sources = contract.get("sources", [])
    approved = []
    denied = []
    provenance_complete = 0
    license_complete = 0
    for source in sources:
        licence = source.get("license") or {}
        approval = source.get("approval") or {}
        decided = (
            approval.get("status") == "approved"
            and licence.get("decision") == "approved"
        )
        (approved if decided else denied).append(source.get("id"))
        if source.get("provenance", source.get("location")):
            provenance_complete += 1
        if licence.get("evidence") and licence.get("evidence_sha256"):
            license_complete += 1

    benchmarks = {"status": "not_configured", "checked": [], "not_checked": []}
    registry_path = (contract.get("benchmarks") or {}).get("registry")
    if registry_path:
        base = Path(contract_path).resolve().parent
        candidate = (base / registry_path)
        if not candidate.is_file():
            candidate = Path(registry_path)
        if candidate.is_file():
            registry = load_registry(candidate)
            records = load_benchmark_records(registry, base_dir=candidate.parent)
            benchmarks = {
                "status": "loaded",
                "checked": sorted(
                    p["id"] for p in records.provenance if p.get("availability") == "available"
                ),
                "not_checked": sorted(
                    p["id"] for p in records.provenance if p.get("availability") != "available"
                ),
            }

    eligible = len(approved)
    blocked = len(denied)
    return {
        "schema": "quantsmind.corpus_readiness/1",
        "contract": str(contract_path),
        "contract_version": contract.get("version"),
        "dataset_class": contract.get("dataset_class", "production"),
        "sources": {
            "declared": len(sources),
            "approved": eligible,
            "approved_ids": approved,
            "denied": blocked,
            "denied_ids": denied,
            "provenance_complete": provenance_complete,
            "license_metadata_complete": license_complete,
        },
        "eligible_training_sources": eligible,
        "blocked_or_rejected_candidates": blocked,
        "missing_decisions": sorted(
            identifier
            for identifier, entry in _DECISION_STATUS.items()
            if entry["status"] in ("unresolved", "empty", "open", "blocked",
                                   "partially addressed")
        ),
        "eligible_training_records": 0 if not approved else None,
        "contamination_registry": benchmarks,
        "split_readiness": "unknown: no corpus has been built",
        "tokenizer_compatibility": "checked when a corpus manifest exists",
        "production_training_authorized": bool(approved) and contract.get(
            "dataset_class", "production"
        ) == "production",
        "default_decision": contract.get("policy", {}).get("default_decision", "deny"),
        "decision_register": DECISION_REGISTER,
        "notes": [
            "This report is read-only: it never approves a source.",
            "Zero approved sources means production training is unauthorized.",
            "Historical corpora are not treated as approved under this contract.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Corpus readiness report")
    parser.add_argument("--contract", default="data/contracts/python_v1.yaml")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args(argv)
    report = assess(args.contract)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        sources = report["sources"]
        print(f"contract          : {report['contract']}")
        print(f"dataset class     : {report['dataset_class']}")
        print(f"sources declared  : {sources['declared']}")
        print(f"sources approved  : {sources['approved']}")
        print(f"sources denied    : {sources['denied']}")
        print(f"production authorized: {report['production_training_authorized']}")
        print(f"benchmark registry: {report['contamination_registry']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
