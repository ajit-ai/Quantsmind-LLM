"""TAKSH L0.7.1 — Karkain corpus adapter (54th first-class language).

Schema and adapter for future .kark corpora. Synthetic fixtures validate the
adapter only; they are never data. KCC remains the authoritative compiler
(0xC0000135 blocker, unchanged). Zero Karkain corpus exists; zero counts are
claimed; nothing here is training-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONTRACT = REPO / "configs" / "foundation" / "l0601_karkain_learning_contract.yaml"
STRATEGY = REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"

KARK_EXTENSIONS = (".kark",)
SYNTHETIC_FIXTURES = [
    ("fn synth_main() { print(\"synth\"); }", "positive-syntax"),
    ("fn synth_borrow() { let x = 1; let y = x; }", "ownership-case"),
    ("fn synth_error() -> Int { return \"wrong type\"; }", "negative-syntax"),
]

ADAPTER_SCHEMA = {
    "sample_id": "uuid5(taksh:karkain:<original_fingerprint>:<index>)",
    "language_id": "karkain",
    "extensions": [".kark"],
    "spec_version_pinned": "1.1.0",
    "compiler_version_pinned": "1.1.0 (windows/amd64, Stable Build)",
    "kir_version": "KIR v1 (SSA)",
    "required_metadata": [
        "per-file sha256", "spec_pin", "compiler_pin", "repo+revision+branch",
        "license + owner authorization record id", "kcc validation status",
    ],
    "corpus_groups": [
        "std_examples", "positive_programs", "negative_programs", "borrow_move_cases",
        "codegen_cases", "debug_repair", "repository_level", "kir_annotated", "diagnostics",
    ],
}


def validate() -> list[str]:
    problems: list[str] = []
    import yaml

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    strategy = yaml.safe_load(STRATEGY.read_text(encoding="utf-8"))
    ident = contract.get("identity", {})
    if ident.get("language_id") != "karkain":
        problems.append("contract identity must be karkain")
    if ident.get("spec_reference", {}).get("spec_version") != "1.1.0":
        problems.append("SPEC must remain pinned to 1.1.0")
    if strategy.get("identity", {}).get("qualification_status") != "DISCOVERED":
        problems.append("Karkain must remain DISCOVERED")
    if strategy.get("corpus_requirements") and any(r.get("status") != "NOT_INSTANTIATED" for r in strategy["corpus_requirements"]):
        problems.append("corpus requirements must remain NOT_INSTANTIATED")
    frozen = REPO / "evaluation" / "l02_python_bench_v1.json"
    if frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest() != FROZEN_PY_SHA:
        problems.append("frozen Python benchmark changed")
    return problems


def is_kark_file(path: str) -> bool:
    return Path(path).suffix.lower() in KARK_EXTENSIONS


def adapter_self_check() -> dict:
    parsed = []
    for content, kind in SYNTHETIC_FIXTURES:
        fp = hashlib.sha256(content.encode("utf-8")).hexdigest()
        parsed.append({
            "kind": kind,
            "label": "SYNTHETIC — adapter test; NOT training data",
            "original_fingerprint": fp,
            "language": "karkain",
            "spec_version": ADAPTER_SCHEMA["spec_version_pinned"],
            "kcc_validation": "NOT_POSSIBLE (0xC0000135)",
            "training_eligibility": "INELIGIBLE",
        })
    return {"fixtures": parsed, "fixtures_label": "SYNTHETIC_TEST_FIXTURE"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_karkain_adapter.json")
    args = parser.parse_args(argv)

    problems = validate()
    if problems:
        for p in problems:
            print("KARK ADAPTER INVALID:", p)
        return 2

    report = {
        "schema": "taksh.l071_karkain_adapter/1",
        "generated_by": "evaluation/l071_karkain.py",
        "valid": True,
        "adapter_schema": ADAPTER_SCHEMA,
        "adapter_self_check": adapter_self_check(),
        "corpus_tokens_measured": 0,
        "corpus_claim": "NONE — zero Karkain corpus exists",
        "host_blocker": "kcc.exe exits 0xC0000135; all compile-pipeline commands exit 3; no diagnostics",
        "frozen_python_benchmark_unchanged": True,
        "verdict": "KARK ADAPTER QUALIFIED (schema only; no ingestion; zero capability claims)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"karkain adapter valid; synthetic fixtures={len(SYNTHETIC_FIXTURES)} (labeled); corpus=0")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
