"""TAKSH L0.5.1 — multilingual dataset feasibility register (WP3).

Per-language assessment for all 53 catalogue languages. Ground truth from this
repository: the only license-verified code supply is Python (1,590,703 measured
tokens in data/l04_code_corpus; 100% of the repo code corpus is .py).

Nothing is downloaded. Every entry records:
  * measured in-repo supply (0 where absent — measured, not assumed)
  * external candidate availability (public corpora exist for major languages)
  * license status and commercial-use notes
  * per-slice legal-review requirements
  * a verdict: qualified-in-repo | conditional-on-legal-review | deferred | blocked

The tool deliberately distinguishes "a public corpus exists" from "we may train
on it": many files are unusable as-is (copyleft, mixed provenance, unknown
provenance, opt-out obligations).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Candidate availability tiers (public corpora with usable Python/other code).
TIER_HIGH = ["python", "javascript", "typescript", "java", "c", "cpp",
             "csharp", "go", "rust", "php", "ruby", "swift", "kotlin", "sql"]
TIER_MEDIUM = ["bash", "fsharp", "scala", "groovy", "perl", "r", "matlab", "julia",
               "haskell", "ocaml", "elixir", "erlang", "lisp", "vbnet", "dart"]
TIER_LOW = ["cobol", "fortran", "delphi", "objectivec", "assembly", "plsql", "tsql",
            "vba", "gdscript", "godot", "mojo", "nim", "crystal", "solidity",
            "prolog", "zig", "ada", "lua", "gleam", "odin", "vlang", "hack",
            "pony", "micropython", "clojure"]

PYTHON_IN_REPO = 1_590_703
REPO_DOC_LANGUAGES = {"python": 241, "javascript": 0, "typescript": 0}


def verdict_for(language_id: str) -> dict:
    measured = PYTHON_IN_REPO if language_id == "python" else 0
    in_repo_docs = REPO_DOC_LANGUAGES.get(language_id, 0)
    if language_id == "python":
        tier = "high"
        verdict = "qualified-in-repo"
        commercial = "per-file MIT/BSD/Apache-2.0 (M3-verified + Apache-2.0 project source); fully cleared for internal research training"
        legal = False
        note = ("only language with verified in-repo supply; The Stack/PyPI candidates exist "
                "for scale-up; legal review required for any external acquisition")
    elif language_id in TIER_HIGH:
        tier = "high"
        verdict = "conditional-on-legal-review"
        commercial = "permissive subsets exist upstream (The Stack filtered variants); commercial use possible with per-file allowlisting; no blanket clearance"
        legal = True
        note = ("major public corpora exist (The Stack v1/v2, CodeSearchNet); "
                "volume is high but per-file license filtering and secret scanning must pass before any ingestion")
    elif language_id in TIER_MEDIUM:
        tier = "medium"
        verdict = "conditional-on-legal-review"
        commercial = "mixed; some nik or niche corpora carry copyleft obligations"
        legal = True
        note = ("usable volume exists in filtered corpora; expect heavy filtering loss; "
                "manual license triage per slice required")
    elif language_id in TIER_LOW:
        tier = "low"
        verdict = "deferred"
        commercial = "unknown/unverified; niche corpora unverified"
        legal = True
        note = ("public corpus scarcity or licensing uncertainty; effort per token is high; "
                "defer until the 16 L1.0 languages are qualified and evidence demands expansion")
    else:
        tier = "unknown"
        verdict = "blocked"
        commercial = "unverified"
        legal = True
        note = "no evidence of usable licensed corpora in-repo; external audit required before any claim"
    return {
        "language": language_id,
        "in_repo_tokens_measured": measured,
        "in_repo_documents": in_repo_docs,
        "candidate_availability": tier,
        "license_status": "clear" if language_id == "python" else "uncertain-requires-review",
        "commercial_use_note": commercial,
        "redistribution_restrictions": "per upstream terms if acquisition is authorized; not redistributable as-is here",
        "attribution_required": language_id != "python",
        "dataset_size_evidence": (
            "measured (L0.4, see data/l04_code_corpus/dataset_manifest.json)"
            if language_id == "python" else
            "none measured; external corpora report sizes but are unverified in this environment"
        ),
        "language_identification_risk": "low" if language_id in TIER_HIGH else "medium-high",
        "near_duplicate_risk": "high (typical of public code)" if language_id != "python" else "measured-and-removed-L0.4",
        "quality_contamination_risk": (
            "low (verified)" if language_id == "python" else
            "medium (public corpora contain dead, vendored, and duplicated code)"
        ),
        "secret_risk": "low (scanned L0.4)" if language_id == "python" else "high (industry-wide, needs scanning)",
        "legal_review_required": legal,
        "verdict": verdict,
        "notes": note,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_dataset_feasibility.json")
    args = parser.parse_args(argv)

    import yaml

    registry = yaml.safe_load((REPO / "configs/feasibility/l051_languages.yaml").read_text(encoding="utf-8"))
    language_ids = [l["id"] for l in registry["languages"]]

    per_language = {lid: verdict_for(lid) for lid in language_ids}
    verdicts = {}
    for entry in per_language.values():
        verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1

    l1_ids = [l["id"] for l in registry["languages"] if l.get("l1_target")]
    l1_verdicts = {lid: per_language[lid]["verdict"] for lid in l1_ids}
    l1_summary = {
        "l1_target_languages": l1_ids,
        "qualified_in_repo": sorted(lid for lid in l1_ids if l1_verdicts[lid] == "qualified-in-repo"),
        "conditional_on_legal_review": sorted(lid for lid in l1_ids if l1_verdicts[lid] == "conditional-on-legal-review"),
        "deferred": sorted(lid for lid in l1_ids if l1_verdicts[lid] == "deferred"),
        "blocked": sorted(lid for lid in l1_ids if l1_verdicts[lid] == "blocked"),
    }

    report = {
        "schema": "taksh.l051_dataset_feasibility/1",
        "generated_by": "evaluation/l051_dataset_feasibility.py",
        "in_repo_ground_truth": {
            "l04_code_tokens": 1_590_703,
            "l04_code_share": 0.4999,
            "repo_corpus_doc_languages": REPO_DOC_LANGUAGES,
            "note": "100% of in-repo code is Python (241 repo files + 193 M3-verified Python); "
                    "zero tokens for every other language are available in-repo today",
        },
        "external_candidates": [
            {
                "corpus": "BigCode The Stack v1/v2 (filtered variants)",
                "scope": "high-tier languages (Python/JS/TS/Java/C/C++/C#/Go/Rust/PHP/Ruby/Swift/Kotlin/SQL)",
                "license": "BigCode OpenRAIL (M); per-file filtering to permissive licenses upstream",
                "commercial_use": "permitted with use restrictions + attribution; per-file obligations inherited",
                "size_evidence": "reported TB-scale raw; per-language extraction reported in upstream docs; unverified here",
                "legal_review_required": True,
                "benchmark_contamination_check": "mandatory (repos contain coding-benchmark solutions at scale)",
            },
            {
                "corpus": "CodeSearchNet",
                "scope": "Python, JavaScript, Java, PHP, Ruby, Go",
                "license": "dataset metadata CC-BY-4.0; code files retain original licenses (mixed)",
                "commercial_use": "ambiguous per-file; cannot be assumed",
                "size_evidence": "~6M functions (reported, unverified here)",
                "legal_review_required": True,
            },
            {
                "corpus": "PyPI permissively-licensed packages",
                "scope": "Python (only)",
                "license": "per-package licenses (MIT/Apache/BSD/PSF majority; copyleft minority)",
                "commercial_use": "mostly permissive; per-package triage required",
                "size_evidence": "500k+ packages (reported, unverified here)",
                "legal_review_required": True,
            },
        ],
        "verdicts": verdicts,
        "l1_target_summary": l1_summary,
        "per_language": per_language,
        "conclusions": [
            "today, in-repo multilingual support reduces to Python only; every other language has zero measured supply",
            "a 16-language product cannot exist without owner-authorized licensed data acquisition and legal review",
            "niche languages (COBOL/VBA/Assembly/Solidity/Lisp..) cannot be qualified at any plausible budget in the near term",
            "dataset work must proceed per-language with per-slice licensing review — no corpus-wide clearance is valid",
            "benchmark-leakage scanning must be part of every ingestion pipeline (this repo already has the scanner)",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"languages assessed: {len(per_language)} | verdicts: {verdicts}")
    print(f"L1.0 qualified-in-repo: {l1_summary['qualified_in_repo']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
