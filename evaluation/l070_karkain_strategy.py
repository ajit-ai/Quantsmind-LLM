"""TAKSH L0.7.0 — Karkain data-strategy validator.

Verifies the .kark corpus strategy is well-formed and honest: SPEC/compiler
pinned to 1.1.0, zero invented corpus counts, every corpus requirement
NOT_INSTANTIATED, benchmark exclusion stated, kcc blocker recorded. No
Karkain ingestion is authorized; synthetic fixture strings below are labeled
synthetic and are NOT data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

STRATEGY_PATH = REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"

# SYNTHETIC fixtures — schema tests only; NOT Karkain training data.
SYNTHETIC_FIXTURES = [
    'fn main() { print("hello karkain"); }',
    'struct Point { x: Int, y: Int }',
    'fn add(a: Int, b: Int) -> Int { a + b }',
]


def load_strategy(path: Path = STRATEGY_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_strategy(s: dict) -> list[str]:
    problems: list[str] = []
    ident = s.get("identity") or {}
    if ident.get("spec_version_pinned") != "1.1.0":
        problems.append("SPEC must remain pinned to 1.1.0")
    if not str(ident.get("compiler_version_pinned", "")).startswith("1.1.0"):
        problems.append("compiler must remain pinned to 1.1.0")
    if ident.get("qualification_status") != "DISCOVERED":
        problems.append("Karkain must remain DISCOVERED (never a capability state)")

    reqs = s.get("corpus_requirements") or []
    if not reqs:
        problems.append("corpus_requirements must define requirements")
    for req in reqs:
        if req.get("status") != "NOT_INSTANTIATED":
            problems.append(f"requirement {req.get('id')} must remain NOT_INSTANTIATED")
    if len(reqs) > 0:
        # no token/sample counts anywhere in the strategy
        blob = json.dumps(s).lower()
        for banned in ("verified tokens", "qualified tokens", "sample_count:", "token_count:"):
            if banned in blob:
                problems.append(f"strategy contains a count-claim marker: {banned!r}")

    blockers = s.get("current_blockers") or []
    if not any("0xC0000135" in str(b) for b in blockers):
        problems.append("kcc host blocker (0xC0000135) must be recorded")

    frozen = REPO / "evaluation" / "l02_python_bench_v1.json"
    if frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest() != FROZEN_PY_SHA:
        problems.append("frozen Python benchmark changed")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_karkain_strategy.json")
    args = parser.parse_args(argv)

    strategy = load_strategy()
    problems = validate_strategy(strategy)
    if problems:
        for p in problems:
            print("KARK STRATEGY INVALID:", p)
        return 2

    counts = {r.get("kind", "?").split(" (")[0][:48]: r.get("status") for r in strategy["corpus_requirements"]}
    report = {
        "schema": "taksh.l070_karkain_strategy/1",
        "generated_by": "evaluation/l070_karkain_strategy.py",
        "valid": True,
        "spec_version_pinned": strategy["identity"]["spec_version_pinned"],
        "compiler_version_pinned": strategy["identity"]["compiler_version_pinned"],
        "qualification_status": strategy["identity"]["qualification_status"],
        "corpus_requirements": strategy["corpus_requirements"],
        "corpus_tokens_measured": 0,
        "corpus_claim": "NONE — no counts exist; zero training-eligible Karkain tokens",
        "compiler_backed_validation": strategy["compiler_backed_validation_requirements"],
        "synthetic_fixtures": [{"content": f, "label": "SYNTHETIC — schema test fixture, NOT training data"}
                               for f in SYNTHETIC_FIXTURES],
        "host_blocker": "kcc.exe 0xC0000135 / exit 3 (all compile-pipeline commands; reproduced with fresh compiler)",
        "frozen_python_benchmark_unchanged": True,
        "verdict": "KARK STRATEGY QUALIFIED (structure only; no ingestion authorized; no counts exist)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"karkain strategy valid; requirements={len(strategy['corpus_requirements'])} (all NOT_INSTANTIATED)")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
