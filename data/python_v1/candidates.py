"""Python training-data candidate registry (Taksh v1.7.0).

Discovery and approval are deliberately separate. This registry records
*candidates* — sources that might support future training — with the evidence
that exists about each one, the gaps that remain, and the human decisions
that would be required. Nothing here approves anything, and no candidate
state implies production eligibility: production eligibility continues to be
governed exclusively by the v1.6.0 source-manifest approval contract
(``quantsmind.source_manifest/2``, state ``approved``).

Evidence classes
----------------
``VERIFIED_LOCAL_EVIDENCE``
    The facts were read from artifacts that exist in this repository.
``REQUIRES_EXTERNAL_VERIFICATION``
    The fact can only be established by inspecting the upstream source, which
    this milestone is not permitted to do.
``UNKNOWN``
    No evidence is available, and none is invented.

No network access is used anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CANDIDATE_REGISTRY_SCHEMA = "quantsmind.candidate_registry/1"
REVIEW_PACKET_SCHEMA = "quantsmind.candidate_review_packet/1"

#: Candidate states. None of them means "approved for production".
CANDIDATE_STATES = (
    "research_only",
    "evidence_incomplete",
    "ready_for_human_review",
    "not_recommended",
)

EVIDENCE_CLASSES = (
    "VERIFIED_LOCAL_EVIDENCE",
    "REQUIRES_EXTERNAL_VERIFICATION",
    "UNKNOWN",
)

CATEGORIES = (
    "public_python_repositories",
    "python_code_datasets",
    "code_documentation_pairs",
    "tests_and_implementations",
    "algorithmic_examples",
    "licensed_educational_material",
    "owned_original_python",
    "synthetic_or_generated",
)

#: Ranking criteria. A high score never overrides an authorization blocker.
RANKING_CRITERIA = (
    "authorization_feasibility",
    "provenance_quality",
    "python_relevance",
    "data_quality",
    "contamination_risk",
    "licensing_uncertainty",
    "acquisition_cost",
    "processing_cost",
    "learning_value",
)


@dataclass
class Candidate:
    """One training-data candidate and the evidence known about it."""

    candidate_id: str
    category: str
    name: str
    publisher: str = "unknown"
    location: str = "unknown"
    revision: str = "unknown"
    declared_license: str = "unknown"
    license_evidence: str = "none_available"
    commercial_use: str = "not_assessed"
    training_use: str = "not_assessed"
    attribution_obligations: str = "not_assessed"
    redistribution: str = "not_assessed"
    provenance_quality: str = "unknown"
    third_party_rights_risk: str = "not_assessed"
    personal_data_risk: str = "not_assessed"
    secret_exposure_risk: str = "not_assessed"
    contamination_risk: str = "not_assessed"
    approximate_volume: str = "unknown"
    python_relevance: str = "unknown"
    execution_evidence: str = "none_available"
    evidence_class: str = "UNKNOWN"
    state: str = "research_only"
    evidence_gaps: list = field(default_factory=list)
    required_human_decisions: list = field(default_factory=list)
    local_artifacts: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "category": self.category,
            "name": self.name,
            "publisher": self.publisher,
            "location": self.location,
            "revision": self.revision,
            "declared_license": self.declared_license,
            "license_evidence": self.license_evidence,
            "commercial_use": self.commercial_use,
            "training_use": self.training_use,
            "attribution_obligations": self.attribution_obligations,
            "redistribution": self.redistribution,
            "provenance_quality": self.provenance_quality,
            "third_party_rights_risk": self.third_party_rights_risk,
            "personal_data_risk": self.personal_data_risk,
            "secret_exposure_risk": self.secret_exposure_risk,
            "contamination_risk": self.contamination_risk,
            "approximate_volume": self.approximate_volume,
            "python_relevance": self.python_relevance,
            "execution_evidence": self.execution_evidence,
            "evidence_class": self.evidence_class,
            "state": self.state,
            "evidence_gaps": list(self.evidence_gaps),
            "required_human_decisions": list(self.required_human_decisions),
            "local_artifacts": list(self.local_artifacts),
            "scores": dict(self.scores),
            "notes": self.notes,
        }


def validate_candidate(candidate: dict) -> list:
    """Return every problem with a candidate record."""
    problems: list = []
    if not isinstance(candidate, dict):
        return ["candidate must be a mapping"]
    for field in (
        "candidate_id",
        "category",
        "name",
        "evidence_class",
        "state",
        "evidence_gaps",
        "required_human_decisions",
    ):
        if field not in candidate:
            problems.append(f"missing '{field}'")
    if candidate.get("category") not in CATEGORIES:
        problems.append(f"category must be one of {list(CATEGORIES)}")
    if candidate.get("evidence_class") not in EVIDENCE_CLASSES:
        problems.append(f"evidence_class must be one of {list(EVIDENCE_CLASSES)}")
    if candidate.get("state") not in CANDIDATE_STATES:
        problems.append(f"state must be one of {list(CANDIDATE_STATES)}")
    for key in ("evidence_gaps", "required_human_decisions"):
        if not isinstance(candidate.get(key), list):
            problems.append(f"{key} must be a list")
    return problems


def load_registry(path) -> dict:
    """Load and validate a candidate registry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != CANDIDATE_REGISTRY_SCHEMA:
        raise ValueError(
            f"{path}: schema must be {CANDIDATE_REGISTRY_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )
    candidates = data.get("candidates") or []
    problems: list = []
    seen = set()
    for index, candidate in enumerate(candidates):
        problems.extend(validate_candidate(candidate))
        candidate_id = candidate.get("candidate_id")
        if candidate_id in seen:
            problems.append(f"duplicate candidate_id {candidate_id!r}")
        seen.add(candidate_id)
    if problems:
        raise ValueError(f"{path}: invalid registry: " + "; ".join(problems))
    return data


def rank_candidates(candidates: list) -> list:
    """Rank candidates transparently.

    The score is the sum of the per-criterion values (each 0-5, higher is
    better) minus the licensing-uncertainty penalty. Authorization
    feasibility is a **hard gate**: a candidate whose authorization is not
    feasible is ranked last regardless of its other merits, because a high
    score must never override an authorization blocker.
    """
    ranked = []
    for candidate in candidates:
        scores = candidate.get("scores") or {}
        total = sum(float(scores.get(name, 0)) for name in RANKING_CRITERIA)
        blocked = float(scores.get("authorization_feasibility", 0)) <= 0
        ranked.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "score": total,
                "authorization_blocked": blocked,
                "evidence_class": candidate.get("evidence_class"),
                "state": candidate.get("state"),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["authorization_blocked"],
            -row["score"],
            row["candidate_id"],
        )
    )
    return ranked


def build_review_packet(candidate: dict) -> dict:
    """Build the packet a human reviewer needs for one candidate.

    The packet is a request. It states the unresolved questions and the next
    action, and it never implies that the candidate is approved.
    """
    problems = validate_candidate(candidate)
    gaps = list(candidate.get("evidence_gaps") or [])
    decisions = list(candidate.get("required_human_decisions") or [])
    if candidate.get("evidence_class") == "REQUIRES_EXTERNAL_VERIFICATION":
        gaps.append(
            "licence, revision and volume can only be confirmed by inspecting "
            "the upstream source, which this milestone did not do"
        )
    return {
        "schema": REVIEW_PACKET_SCHEMA,
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("name"),
        "category": candidate.get("category"),
        "publisher": candidate.get("publisher"),
        "location": candidate.get("location"),
        "revision": candidate.get("revision"),
        "evidence_references": {
            "license_evidence": candidate.get("license_evidence"),
            "local_artifacts": list(candidate.get("local_artifacts") or []),
            "evidence_class": candidate.get("evidence_class"),
        },
        "rights_and_licensing_questions": {
            "declared_license": candidate.get("declared_license"),
            "commercial_use": candidate.get("commercial_use"),
            "training_use": candidate.get("training_use"),
            "attribution_obligations": candidate.get("attribution_obligations"),
            "redistribution": candidate.get("redistribution"),
            "third_party_rights_risk": candidate.get("third_party_rights_risk"),
            "personal_data_risk": candidate.get("personal_data_risk"),
            "secret_exposure_risk": candidate.get("secret_exposure_risk"),
        },
        "contamination_concerns": candidate.get("contamination_risk"),
        "intended_use": candidate.get("training_use"),
        "unresolved_questions": gaps,
        "required_human_decisions": decisions,
        "explicit_blockers": gaps or ["no decision has been taken yet"],
        "current_state": candidate.get("state"),
        "recommended_next_action": (
            "assemble a source manifest for the v1.6.0 contract and record a "
            "human review decision"
        ),
        "registry_problems": problems,
        "note": (
            "A candidate review packet is a request for a decision. It never "
            "implies approval, and no candidate state makes a source eligible "
            "for production ingestion."
        ),
    }
