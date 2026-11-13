"""TAKSH L0.5.1 — multilingual acceptance and promotion gates (WP9).

Defines the per-language capability qualification framework. Python is bound by
the existing frozen benchmark and its stricter bar (unchanged). The remaining
15 L1.0 languages are bound by the new multilingual starter benchmark and a
documented proposed threshold; the Python bar is never lowered to accommodate
them.

No model has been evaluated on the multilingual benchmark; today every gate is
NOT MET. This tool generates the framework and today's status report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

L1_TARGETS = [
    "python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
    "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
]

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"

GATES = [
    ("tokenizer_correctness", "tokenizer_audit.json", "audit exists for this language; roundtrip exact on the sample"),
    ("dataset_qualification", "dataset_feasibility.json", "legal review + in-repo or authorized verified corpus per slice; secrets scrubbed; leakage scan clean"),
    ("training_exposure", "run_manifest.json", "documented steps, tokens, mixture %, seed, final loss + val loss curve"),
    ("syntax_validity", "benchmark_run.jsonl", "at least 75% of generations parse under the language's own parser/compiler"),
    ("compilation_or_interpretation", "benchmark_run.jsonl", "at least 50% of generations execute without environment failure"),
    ("execution_success", "benchmark_run.jsonl", "at least 50% of generations run to completion under timeout"),
    ("functional_correctness", "benchmark_run.jsonl", "at least 50% of tasks pass all unit assertions"),
    ("test_coverage", "benchmark_run.jsonl", "every benchmark task assertion executed and recorded"),
    ("determinism", "repeat_run.jsonl", "100% byte-identical greedy completions across repeated runs"),
    ("safety", "sandbox_log", "zero sandbox escapes, zero unsafe generation incidents"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_acceptance_gates.json")
    args = parser.parse_args(argv)

    import hashlib

    frozen_path = REPO / "evaluation" / "l02_python_bench_v1.json"
    frozen_ok = hashlib.sha256(frozen_path.read_bytes()).hexdigest() == FROZEN_PY_SHA

    per_language = {}
    for lid in L1_TARGETS:
        per_language[lid] = {
            "language": lid,
            "gates": {
                gate_id: {
                    "requirement": requirement,
                    "evidence_source": source,
                    "status_today": "NOT_MET",
                    "note": "no multilingual model has been trained or evaluated; capability is unmeasured and unverified",
                }
                for gate_id, source, requirement in GATES
            },
        }

    report = {
        "schema": "taksh.l051_acceptance_gates/1",
        "generated_by": "evaluation/l051_acceptance_gates.py",
        "frozen_python_bar": {
            "path": "evaluation/l02_python_bench_v1.json",
            "sha256": FROZEN_PY_SHA,
            "hash_matches": frozen_ok,
            "unchanged": True,
            "requirements": {
                "functional_correctness": ">= 15/30 problems",
                "syntax_valid_rate": ">= 0.80",
                "execution_rate": ">= 0.60",
                "category_coverage": "every A/B/C category >= 30%",
                "repeat_consistency": "100% byte-identical greedy",
                "safety": "zero violations",
            },
        },
        "multilingual_proposed_thresholds": {
            "functional_correctness": ">= 8/16 starter tasks per language",
            "syntax_valid_rate": ">= 0.75",
            "execution_rate": ">= 0.50",
            "category_coverage": "all instantiated families >= 25% correct",
            "repeat_consistency": "100% byte-identical greedy",
            "safety": "zero violations",
            "justification": (
                "fixed before any evaluation; floor is 50% of the starter set per language; deliberately stricter than "
                "L0.5's general Python bar proportionally; cannot be lowered after seeing results"
            ),
        },
        "promotion_rules": [
            "a language is promoted to 'supported' only when every gate passes on the multilingual benchmark",
            "tokenizer success or compiler availability alone is NEVER sufficient for 'supported'",
            "Python promotion requires the frozen benchmark pass; the multilingual bar does not replace it",
            "demotion on any regression is mandatory",
            "no soft claims in UI/marketing: supported = benchmark passed; else 'experimental' or 'excluded'",
        ],
        "gates": GATES,
        "per_language": per_language,
        "status_today": "ALL_GATES_NOT_MET (no multilingual model has been trained or evaluated)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"gates defined: {len(GATES)} per language x {len(L1_TARGETS)} languages")
    print(f"frozen python benchmark untouched: {frozen_ok}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
