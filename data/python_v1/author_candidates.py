"""Author the v1.7.0 Python training-data candidate registry.

Candidates are *research records*, not approvals. Everything in this file was
read from artifacts inside this repository; where a fact could only be
established by inspecting an upstream source, the candidate is marked
``REQUIRES_EXTERNAL_VERIFICATION`` and the gap is recorded. No network access
is used and no dataset is downloaded.

Usage:
    python -m data.python_v1.author_candidates
"""

from __future__ import annotations

import json
from pathlib import Path

from data.python_v1.candidates import CANDIDATE_REGISTRY_SCHEMA

STD_DECISIONS = [
    "D1 licensing and permitted use for training",
    "D2 provenance chain acceptance",
    "D3 allowlist entry",
    "D5 attribution and redistribution obligations",
]


def candidate(candidate_id, category, name, evidence_class, state, gaps,
              decisions, **fields) -> dict:
    record = {
        "candidate_id": candidate_id,
        "category": category,
        "name": name,
        "publisher": "unknown",
        "location": "unknown",
        "revision": "unknown",
        "declared_license": "unknown",
        "license_evidence": "none_available",
        "commercial_use": "not_assessed",
        "training_use": "not_assessed",
        "attribution_obligations": "not_assessed",
        "redistribution": "not_assessed",
        "provenance_quality": "unknown",
        "third_party_rights_risk": "not_assessed",
        "personal_data_risk": "not_assessed",
        "secret_exposure_risk": "not_assessed",
        "contamination_risk": "not_assessed",
        "approximate_volume": "unknown",
        "python_relevance": "unknown",
        "execution_evidence": "none_available",
        "evidence_class": evidence_class,
        "state": state,
        "evidence_gaps": list(gaps),
        "required_human_decisions": list(decisions),
        "local_artifacts": [],
        "scores": {},
        "notes": "",
    }
    record.update(fields)
    return record


def scores(authorization, provenance, relevance, quality, contamination,
           uncertainty, acquisition, processing, learning) -> dict:
    return {
        "authorization_feasibility": authorization,
        "provenance_quality": provenance,
        "python_relevance": relevance,
        "data_quality": quality,
        "contamination_risk": contamination,
        "licensing_uncertainty": uncertainty,
        "acquisition_cost": acquisition,
        "processing_cost": processing,
        "learning_value": learning,
    }


LOCAL = [
    candidate(
        "local-code-corpus-v2", "python_code_datasets",
        "In-repository code corpus v2 (437 Python records)",
        "VERIFIED_LOCAL_EVIDENCE", "ready_for_human_review",
        ["licence verification predates the v1.6.0 source-manifest contract",
         "commercial use and redistribution were never assessed",
         "not screened against the v1.4.0/v1.5.0 benchmark suites"],
        STD_DECISIONS + ["whether the historical evidence may be re-authorized"],
        publisher="unknown; upstream repositories recorded per record",
        location="data/code_corpus_v2.jsonl",
        revision="per-record repo and commit recorded",
        declared_license="per-record spdx identifier",
        license_evidence="data/licence_verification.json (historical run)",
        provenance_quality="per-record repo, commit, content hash and attribution",
        contamination_risk="not screened against the frozen suites",
        approximate_volume="437 records (measured locally)",
        python_relevance="high: Python source files",
        execution_evidence="none attached",
        local_artifacts=["data/code_corpus_v2.jsonl", "data/licence_verification.json"],
        scores=scores(0, 4, 5, 3, 2, 2, 5, 5, 3),
        notes="the only candidate with per-record provenance already in the repository",
    ),
    candidate(
        "local-m3-verified-split", "python_code_datasets",
        "M3 verified code split (193 train / 63 validation)",
        "VERIFIED_LOCAL_EVIDENCE", "evidence_incomplete",
        ["the historical authorization is not expressed as a v1.6.0 source manifest",
         "contamination screening against the frozen suites was not performed"],
        STD_DECISIONS,
        publisher="unknown; upstream repositories recorded per record",
        location="data/m3_code_verified.train.jsonl",
        revision="per-record repo and commit recorded",
        declared_license="per-record spdx identifier",
        license_evidence="data/licence_verification.json (historical run)",
        provenance_quality="per-record repo, commit, content hash and attribution",
        contamination_risk="not screened against the frozen suites",
        approximate_volume="256 records (measured locally)",
        python_relevance="high",
        execution_evidence="none attached",
        local_artifacts=["data/m3_code_verified.train.jsonl", "data/m3_code_verified.val.jsonl"],
        scores=scores(0, 4, 5, 3, 2, 2, 5, 5, 3),
        notes="the split the historical S2 run trained on",
    ),
    candidate(
        "taksh-own-repository-code", "owned_original_python",
        "Python code owned by this project (training, evaluation, serving, data)",
        "VERIFIED_LOCAL_EVIDENCE", "research_only",
        ["no owner rights decision for internal reuse in a trained model",
         "contamination against the frozen suites is not screened"],
        ["rights decision for internal code used in a trained model",
         "whether such code may ship with a released model"],
        publisher="this project",
        location="repository tree (Python files)",
        declared_license="project licence, see LICENSE at the repository root",
        license_evidence="LICENSE at the repository root",
        provenance_quality="first-party authorship with per-file git history",
        contamination_risk="the frozen evaluation suites were authored here and overlap the code",
        approximate_volume="thousands of Python files and lines (locally countable)",
        python_relevance="very high",
        execution_evidence="the repository test suite is executable",
        local_artifacts=["training/", "evaluation/", "serving/", "data/"],
        scores=scores(1, 5, 5, 4, 1, 4, 5, 4, 3),
        notes="safest rights profile, highest self-contamination risk",
    ),
    candidate(
        "local-instruct-v1", "code_documentation_pairs",
        "In-repository instruction pairs (610 train / 21 validation)",
        "VERIFIED_LOCAL_EVIDENCE", "evidence_incomplete",
        ["per-example licence and source fields exist but were never audited",
         "not screened against the frozen instruction surfaces"],
        STD_DECISIONS + ["whether instruction data is in scope for the first experiment"],
        publisher="mixed; per-example source field",
        location="data/instruct_v1.train.jsonl",
        declared_license="per-example licence field",
        license_evidence="per-example fields, unaudited",
        provenance_quality="per-example source and identifier",
        contamination_risk="not screened against the frozen suites",
        approximate_volume="631 records (measured locally)",
        python_relevance="medium",
        local_artifacts=["data/instruct_v1.train.jsonl"],
        scores=scores(0, 2, 3, 3, 2, 2, 5, 5, 3),
    ),
    candidate(
        "local-code-sft-v0", "code_documentation_pairs",
        "In-repository code SFT pairs (442 records)",
        "VERIFIED_LOCAL_EVIDENCE", "not_recommended",
        ["a quarantine marker exists for this file in the historical pipeline",
         "the quarantine reason was not recorded in a machine-readable form"],
        ["whether a quarantined file may be reconsidered at all"],
        publisher="unknown",
        location="data/code_sft_v0.jsonl",
        contamination_risk="historical quarantine marker present",
        approximate_volume="442 records (measured locally)",
        python_relevance="medium",
        local_artifacts=["data/code_sft_v0.jsonl", "data/code_sft_v0.quarantine.json"],
        scores=scores(0, 1, 3, 1, 1, 2, 5, 4, 1),
        notes="recommended against until the quarantine reason is understood",
    ),
    candidate(
        "synthetic-python-fixtures", "synthetic_or_generated",
        "Locally authored synthetic Python fixtures (diagnostic corpus and instruction examples)",
        "VERIFIED_LOCAL_EVIDENCE", "research_only",
        ["no real-world Python is represented, so learning value is diagnostic only"],
        ["decide whether synthetic data may support a first experiment"],
        publisher="this project",
        location="scripts/build_synthetic_diagnostic_corpus.py, training/fixtures/",
        declared_license="project-internal, generated content",
        license_evidence="generator script and fixtures in the repository",
        provenance_quality="first-party generation with fixed seeds",
        contamination_risk="none against the frozen suites",
        approximate_volume="24 modules and 4 instruction examples (measured locally)",
        python_relevance="low: templated families only",
        execution_evidence="the generated modules are syntactically valid",
        local_artifacts=["scripts/build_synthetic_diagnostic_corpus.py"],
        scores=scores(3, 5, 1, 3, 5, 5, 5, 5, 1),
        notes="safe but shows memorisation, not transfer",
    ),
]

#: Category-level candidates. Nothing about a specific upstream project is
#: asserted here: this milestone performed no network access, so licence,
#: revision and volume facts remain unverified by construction.
EXTERNAL = [
    candidate(
        "external-public-python-repositories", "public_python_repositories",
        "Public Python repositories with explicit licence files (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["licence text, revision and volume require upstream inspection",
         "per-repository licence review has not been performed"],
        STD_DECISIONS + ["which repositories enter the D3 allowlist"],
        publisher="various upstream publishers",
        location="to be specified by the owner",
        contamination_risk="unknown until each repository is inspected",
        python_relevance="high when the repository is Python",
        scores=scores(0, 1, 4, 3, 2, 1, 3, 4, 4),
        notes="category placeholder; no specific project is claimed here",
    ),
    candidate(
        "external-python-code-datasets", "python_code_datasets",
        "Aggregated Python code datasets with documented provenance (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["the per-file licence mix of aggregated datasets is not uniform",
         "the provenance of the aggregation cannot be confirmed offline"],
        STD_DECISIONS,
        publisher="various",
        location="to be specified by the owner",
        contamination_risk="high unless per-file provenance is preserved",
        python_relevance="high",
        scores=scores(0, 1, 5, 3, 1, 1, 2, 3, 4),
        notes="category placeholder",
    ),
    candidate(
        "external-code-documentation-pairs", "code_documentation_pairs",
        "Code and documentation pairs from licensed projects (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["pairing rules and licence inheritance need upstream verification"],
        STD_DECISIONS,
        location="to be specified by the owner",
        python_relevance="medium to high",
        scores=scores(0, 1, 3, 3, 2, 1, 3, 3, 3),
        notes="category placeholder",
    ),
    candidate(
        "external-tests-and-implementations", "tests_and_implementations",
        "Python test and implementation pairs (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["licence inheritance from the implementation repository is unverified",
         "execution evidence would have to be produced locally"],
        STD_DECISIONS,
        location="to be specified by the owner",
        python_relevance="high",
        execution_evidence="would require local execution, which is blocked",
        scores=scores(0, 1, 5, 4, 2, 1, 3, 3, 4),
        notes="category placeholder",
    ),
    candidate(
        "external-algorithmic-examples", "algorithmic_examples",
        "Algorithmic programming examples with published problem statements (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["problem-statement rights and contest terms are unverified",
         "solutions are usually community written with unclear provenance"],
        STD_DECISIONS,
        location="to be specified by the owner",
        python_relevance="medium",
        scores=scores(0, 1, 3, 3, 2, 1, 3, 3, 3),
        notes="category placeholder",
    ),
    candidate(
        "external-licensed-educational-material", "licensed_educational_material",
        "Officially licensed educational programming material (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["the exact licence and its attribution requirements need upstream verification"],
        STD_DECISIONS,
        location="to be specified by the owner",
        python_relevance="medium",
        scores=scores(0, 2, 3, 4, 2, 2, 3, 3, 3),
        notes="category placeholder",
    ),
    candidate(
        "external-human-authored-synthetic", "synthetic_or_generated",
        "Human-authored or independently generated synthetic Python examples (category candidate)",
        "REQUIRES_EXTERNAL_VERIFICATION", "research_only",
        ["the generator's own licence and the provenance of its training material are unverified"],
        STD_DECISIONS,
        location="to be specified by the owner",
        python_relevance="unknown",
        scores=scores(0, 2, 3, 3, 3, 2, 3, 3, 3),
        notes="category placeholder",
    ),
]


def main() -> int:
    candidates = LOCAL + EXTERNAL
    payload = {
        "schema": CANDIDATE_REGISTRY_SCHEMA,
        "version": "candidates_v1",
        "methodology": {
            "evidence_rule": (
                "a fact is only recorded as verified when it was read from an "
                "artifact in this repository; anything requiring upstream "
                "inspection is marked REQUIRES_EXTERNAL_VERIFICATION"
            ),
            "network_access": "none used in this milestone",
            "ranking": (
                "the nine criteria are summed and the authorization "
                "feasibility criterion is a hard gate: a candidate whose "
                "authorization is not feasible never ranks above one whose "
                "authorization is feasible, regardless of its other merits"
            ),
            "separation": (
                "candidate discovery is not approval; production eligibility "
                "stays governed by the v1.6.0 source-manifest contract"
            ),
        },
        "candidates": candidates,
    }
    path = Path(__file__).resolve().parent / "contracts" / "python_candidates_v1.json"
    if not path.parent.exists():
        path = Path("data/contracts/python_candidates_v1.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} with {len(candidates)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
