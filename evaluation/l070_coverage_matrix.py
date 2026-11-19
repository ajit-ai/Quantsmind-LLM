"""TAKSH L0.7.0 — multilingual coverage matrix (53 languages + Karkain).

Builds the per-language qualification coverage picture from the committed
53-language registry (configs/feasibility/l051_languages.yaml) and the L0.7.0
source registry. Candidate coverage is reported as REPORTED/UNVERIFIED and is
NEVER counted as training tokens. Only measured in-repo corpus counts are
counted (Python only, 1,590,703 tokens).

Accounting separation:
  qualified_corpus | candidate_corpus | unverified_corpus |
  approved_corpus | acquired_corpus | training_eligible_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l051_language_registry import load_registry  # noqa: E402

_STACK_V1 = ["python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
             "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
             "vbnet", "vba", "cobol", "fortran", "delphi", "perl", "groovy", "scala",
             "objectivec", "assembly", "plsql", "tsql", "r", "matlab", "julia",
             "haskell", "fsharp", "ocaml", "elixir", "erlang", "clojure", "lisp",
             "zig", "ada", "lua", "solidity"]

CANDIDATE_COVERAGE = {
    "src-bigcode-stack-v1": {
        "languages": list(_STACK_V1),
        "evidence": "The Stack v1 card (358 languages; 2026-11-11); conservative subset shown (niche ids not verified); per-file license pass-through; NOT verified",
        "license_status": "HUMAN_LEGAL_REVIEW_REQUIRED",
    },
    "src-bigcode-stack-v2": {
        "languages": list(_STACK_V1),
        "evidence": "The Stack v2 card (658 languages; 2026-11-11); conservative subset shown (niche ids not verified); SWHID metadata only; NOT verified",
        "license_status": "HUMAN_LEGAL_REVIEW_REQUIRED",
    },
    "src-codesearchnet": {
        "languages": ["go", "java", "javascript", "php", "python", "ruby"],
        "evidence": "paper arXiv:1909.09436; Go/Java/JavaScript/PHP/Python/Ruby; ~6M functions; NOT verified",
        "license_status": "HUMAN_LEGAL_REVIEW_REQUIRED",
    },
    "src-pypi-permissive": {
        "languages": ["python"],
        "evidence": "aggregate package ecosystem; license per package; no dataset; NOT verified",
        "license_status": "REJECTED",
    },
    "src-karkain-corpus": {
        "languages": ["karkain"],
        "evidence": "owner-internal; zero corpus; NOT available for ingestion",
        "license_status": "REJECTED",
    },
}

QUALIFIED_IN_REPO = {
    "python": {"tokens": 1_590_703, "samples": 384,
               "source": "src-taksh-l04-code-corpus (internal, licensed)"},
}


def candidate_sources_for(language_id: str) -> list[dict]:
    found = []
    for source_id, body in CANDIDATE_COVERAGE.items():
        if language_id in body["languages"]:
            found.append({"source_id": source_id,
                          "evidence": body["evidence"],
                          "license_status": body["license_status"]})
    return found


def qualification_priority(language: dict, candidates: list[dict]) -> str:
    lang_id = language["id"]
    if lang_id == "karkain":
        return "BLOCKED"
    if language.get("category") == "l1_target":
        return "P1" if candidates else "P1 (no candidate yet)"
    if language.get("category") in ("enterprise", "scientific", "systems"):
        return "P2" if candidates else "P3"
    return "P3"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_coverage_matrix.json")
    args = parser.parse_args(argv)

    registry = load_registry()
    languages = registry["languages"]
    assert len(languages) == 53, f"expected 53 languages, found {len(languages)}"

    entries = []
    totals = {"qualified_tokens": 0, "qualified_samples": 0,
              "candidate_languages": 0, "l1_candidates": 0}
    for lang in languages:
        lid = lang["id"]
        qualified = QUALIFIED_IN_REPO.get(lid, {"tokens": 0, "samples": 0, "source": None})
        candidates = candidate_sources_for(lid)
        acquired = bool(qualified["tokens"])
        totals["qualified_tokens"] += qualified["tokens"]
        totals["qualified_samples"] += qualified["samples"]
        totals["candidate_languages"] += 1 if candidates else 0
        if lang.get("l1_target") and candidates:
            totals["l1_candidates"] += 1
        entries.append({
            "registry_id": lid,
            "name": lang["name"],
            "family": lang["category"],
            "l1_target": bool(lang.get("l1_target")),
            "extensions": lang.get("extensions", []),
            "toolchain_available_here": lang.get("execution_available_here"),
            "existing_qualified_tokens": qualified["tokens"],
            "existing_qualified_samples": qualified["samples"],
            "qualified_source": qualified["source"],
            "candidate_sources": candidates,
            "candidate_quality": "unverified" if candidates else "none",
            "license_status": ("APPROVED_WITH_CONDITIONS (internal)" if acquired else
                               "HUMAN_LEGAL_REVIEW_REQUIRED" if candidates else "NO_CANDIDATE"),
            "provenance_status": "VERIFIED (internal)" if acquired else ("METADATA_REVIEW" if candidates else "NONE"),
            "acquisition_approval_state": "internal_verified" if acquired else "not_requested",
            "corpus_gap": "none (partial coverage: Python only)" if acquired else "full",
            "recommended_priority": qualification_priority(lang, candidates),
            "blockers": ([] if acquired else
                         ["no acquisition authorized", "legal review required"] +
                         ([] if candidates else ["no known candidate source"])),
        })

    karkain = {
        "registry_id": "karkain",
        "name": "Karkain",
        "family": "first-class extension (54th language)",
        "l1_target": False,
        "extensions": [".kark"],
        "toolchain_available_here": False,
        "existing_qualified_tokens": 0,
        "existing_qualified_samples": 0,
        "qualified_source": None,
        "candidate_sources": candidate_sources_for("karkain"),
        "candidate_quality": "unverified",
        "license_status": "REJECTED (owner-internal; not authorized)",
        "provenance_status": "UNVERIFIED",
        "acquisition_approval_state": "not_requested",
        "corpus_gap": "full",
        "recommended_priority": "BLOCKED",
        "blockers": ["spec/compiler pinned 1.1.0; kcc broken on this host (0xC0000135)",
                     "owner authorization absent", "zero corpus; zero tokens"],
    }
    entries.append(karkain)

    by_family = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)

    groups = {
        "initial_16_l1_target": [e["registry_id"] for e in entries if e["l1_target"]],
        "enterprise_legacy": [e["registry_id"] for e in entries if e["family"] == "enterprise"],
        "scientific_functional": [e["registry_id"] for e in entries if e["family"] == "scientific"],
        "systems_embedded_games_blockchain": [e["registry_id"] for e in entries if e["family"] == "systems"],
        "emerging_specialist": [e["registry_id"] for e in entries if e["family"] == "emerging"],
        "karkain_54th": ["karkain"],
    }

    report = {
        "schema": "taksh.l070_coverage_matrix/1",
        "generated_by": "evaluation/l070_coverage_matrix.py",
        "total_languages": len(entries),
        "entries": entries,
        "groups": groups,
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "accounting": {
            "qualified_corpus_tokens": totals["qualified_tokens"],
            "qualified_corpus_samples": totals["qualified_samples"],
            "candidate_corpus_tokens": 0,
            "unverified_corpus_tokens": 0,
            "approved_for_acquisition_tokens": 0,
            "actually_acquired_tokens": totals["qualified_tokens"],
            "training_eligible_tokens": totals["qualified_tokens"],
            "note": "candidate and unverified corpora are never counted; 1,590,703 Python tokens are the only real number",
        },
        "l1_languages_with_unverified_candidates": totals["l1_candidates"],
        "supporting_technologies": "tracked separately in configs/feasibility/l051_languages.yaml (20 entries); zero qualified tokens; candidate coverage reported unverified (stacks, not counted)",
        "verdict": "1/53 languages (+0 Karkain) hold qualified corpora; 52 + 54th unqualified; no acquisition authorized",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"coverage matrix: {len(entries)} entries; qualified tokens={totals['qualified_tokens']:,}")
    print(f"L1 candidates (unverified): {totals['l1_candidates']}/16")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
