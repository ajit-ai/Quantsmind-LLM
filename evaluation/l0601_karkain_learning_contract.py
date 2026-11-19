"""TAKSH L0.6.1 — H: Karkain learning-integration contract validator.

Verifies the TAKSH-side contract is well-formed: identity, registry linkage,
spec/compiler pinning, corpus manifest schema (zero entries), KCC state, and
evaluation references. Training eligibility fails closed; no Karkain claim of
capability is possible from this milestone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONTRACT_PATH = REPO / "configs" / "foundation" / "l0601_karkain_learning_contract.yaml"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


def load_contract(path=CONTRACT_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_contract(c: dict) -> list[str]:
    problems: list[str] = []
    ident = c.get("identity") or {}
    for key in ("language_id", "name", "extensions", "registry_linkage",
                "spec_reference", "provenance"):
        if not ident.get(key):
            problems.append(f"identity.{key} required")
    if ident.get("language_id") != "karkain":
        problems.append("identity.language_id must be 'karkain'")
    if ".kark" not in ident.get("extensions", []):
        problems.append("karkain contract requires .kark extension")

    linkage = ident.get("registry_linkage") or {}
    if linkage.get("extension_entry") != "karkain":
        problems.append("registry_linkage.extension_entry must be 'karkain'")
    if linkage.get("qualification_status") not in {
        "REGISTERED", "DISCOVERED", "COMPREHENSION_IN_PROGRESS",
    }:
        problems.append("qualification_status must remain a pre-capability state")

    spec = ident.get("spec_reference") or {}
    for key in ("spec_version", "compiler_version", "kcc_engine", "kir_version"):
        if not spec.get(key):
            problems.append(f"spec_reference.{key} required")
    if spec.get("spec_version") != "1.1.0":
        problems.append("spec_version must pin 1.1.0")

    corpus = c.get("corpus_manifest") or {}
    if corpus.get("status") != "NOT_INSTANTIATED":
        problems.append("corpus manifest must remain NOT_INSTANTIATED")
    if corpus.get("zero_entries") is not True:
        problems.append("corpus manifest must declare zero entries")

    kcc = c.get("kcc_state_on_host") or {}
    if not (kcc.get("kcc_binary") and kcc.get("compile_commands")):
        problems.append("kcc_state_on_host must record measured toolchain failure")

    if (REPO / "evaluation" / "l02_python_bench_v1.json").exists():
        digest = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
        if digest != FROZEN_PY_SHA:
            problems.append("frozen python benchmark changed")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_karkain_learning_contract.json")
    args = parser.parse_args(argv)

    contract = load_contract()
    problems = validate_contract(contract)
    if problems:
        for p in problems:
            print("KARK CONTRACT INVALID:", p)
        return 2

    report = {
        "schema": "taksh.l0601_karkain_learning_contract/1",
        "generated_by": "evaluation/l0601_karkain_learning_contract.py",
        "valid": True,
        "language_id": contract["identity"]["language_id"],
        "spec_version": contract["identity"]["spec_reference"]["spec_version"],
        "compiler_version": contract["identity"]["spec_reference"]["compiler_version"],
        "corpus_entries": 0,
        "corpus_status": contract["corpus_manifest"]["status"],
        "training_eligibility": {"currently_met": contract["training_eligibility"]["currently_met"],
                                 "verdict": "NOT ELIGIBLE"},
        "kcc_state": contract["kcc_state_on_host"],
        "frozen_python_benchmark_unchanged": True,
        "verdict": "KARK CONTRACT QUALIFIED (zero capability claims; no Karkain execution performed)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"karkain learning contract valid; eligibility: NOT ELIGIBLE")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
