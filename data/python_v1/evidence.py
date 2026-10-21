"""Corpus evidence qualification (Taksh v1.8.0).

Turns the v1.7.0 candidate registry into *evidence*: what is verifiable from
local artifacts, what is missing, what contradicts itself, and what an owner
must decide. Discovery is not authorization, and a high evidence-completeness
score never makes a source eligible.

Evidence classes: ``VERIFIED_LOCAL_EVIDENCE``,
``REQUIRES_EXTERNAL_VERIFICATION``, ``UNKNOWN``, ``CONFLICTING_EVIDENCE``.

Only structural metadata is inspected for corpora already present in this
repository (record counts, licences, revisions, splits, hashes), plus the
project's own synthetic fixtures. Source text of unauthorized material is not
read for analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EVIDENCE_CLASSES = (
    "VERIFIED_LOCAL_EVIDENCE",
    "REQUIRES_EXTERNAL_VERIFICATION",
    "UNKNOWN",
    "CONFLICTING_EVIDENCE",
)

DISPOSITIONS = (
    "REQUEST_EXTERNAL_EVIDENCE",
    "READY_FOR_OWNER_REVIEW",
    "REJECT_CANDIDATE",
    "RETAIN_FOR_RESEARCH_ONLY",
)

EXPECTED_RECORD_FIELDS = (
    "text",
    "content_sha1",
    "source_id",
    "repo",
    "commit",
    "spdx",
    "attribution",
    "file_path",
    "split",
)


@dataclass
class RecordAudit:
    """Result of checking one corpus file's record-level metadata."""

    path: str
    records: int = 0
    missing_fields: dict = field(default_factory=dict)
    hash_mismatches: int = 0
    conflicting: list = field(default_factory=list)
    licences: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    repositories: int = 0
    duplicate_content: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "records": self.records,
            "missing_fields": self.missing_fields,
            "hash_mismatches": self.hash_mismatches,
            "conflicting": self.conflicting[:20],
            "licences": self.licences,
            "splits": self.splits,
            "repositories": self.repositories,
            "duplicate_content": self.duplicate_content,
            "notes": self.notes,
        }


def audit_corpus_file(path, *, check_hashes: bool = True) -> RecordAudit:
    """Check a JSONL corpus file's record-level provenance consistency."""
    audit = RecordAudit(path=str(path))
    file_path = Path(path)
    if not file_path.is_file():
        audit.notes.append("file not found")
        return audit
    seen_hashes: dict = {}
    licences = Counter()
    splits = Counter()
    repos = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                audit.notes.append(f"line {number}: invalid JSON ({exc.msg})")
                continue
            audit.records += 1
            for field in EXPECTED_RECORD_FIELDS:
                if field not in record or record[field] in (None, ""):
                    audit.missing_fields[field] = (
                        audit.missing_fields.get(field, 0) + 1
                    )
            licences[str(record.get("spdx", "unknown"))] += 1
            splits[str(record.get("split", "unknown"))] += 1
            if record.get("repo"):
                repos.add(record["repo"])
            digest = record.get("content_sha1")
            if digest and check_hashes and record.get("text") is not None:
                actual = hashlib.sha1(
                    str(record["text"]).encode("utf-8")
                ).hexdigest()
                if actual != digest:
                    audit.hash_mismatches += 1
                    if len(audit.conflicting) < 20:
                        audit.conflicting.append(
                            f"line {number}: content_sha1 does not match the text"
                        )
            if digest:
                if digest in seen_hashes:
                    audit.duplicate_content += 1
                else:
                    seen_hashes[digest] = number
    audit.licences = dict(sorted(licences.items()))
    audit.splits = dict(sorted(splits.items()))
    audit.repositories = len(repos)
    return audit


def completeness_score(audit: RecordAudit) -> dict:
    """Transparent, separated scores. None of them is an approval.

    * ``evidence_completeness`` -- how much of the expected metadata exists.
    * ``data_quality`` -- hash consistency, duplicates, malformed records.
    * ``authorization_eligibility`` -- always 0 until an owner approves.
    * ``expected_learning_utility`` -- a qualitative planning hint.
    """
    completeness = 0.0
    quality = 0.0
    if audit.records:
        expected = len(EXPECTED_RECORD_FIELDS) * audit.records
        absent = sum(audit.missing_fields.values())
        completeness = max(0.0, 1.0 - (absent / expected))
        penalties = (
            audit.hash_mismatches
            + audit.duplicate_content
            + sum(audit.missing_fields.values())
        )
        quality = max(0.0, 1.0 - (penalties / audit.records))
    return {
        "evidence_completeness": round(completeness, 4),
        "data_quality": round(quality, 4),
        "authorization_eligibility": 0.0,
        "expected_learning_utility": "not_assessed",
        "note": (
            "authorization_eligibility is 0 by construction: no candidate is "
            "approved, and a complete or high-quality corpus does not change "
            "that. Only an owner decision can move it."
        ),
    }


def build_dossier(candidate: dict, audit: RecordAudit | None = None) -> dict:
    """A human-reviewable dossier for one candidate.

    The dossier is a recommendation, not an authorization state, and it
    separates verified facts from unknowns.
    """
    scores = completeness_score(audit) if audit is not None else {
        "evidence_completeness": 0.0,
        "data_quality": 0.0,
        "authorization_eligibility": 0.0,
        "expected_learning_utility": "not_assessed",
    }
    verified = []
    unknown = []
    if audit is not None and audit.records:
        verified.append(f"{audit.records} records with per-record metadata")
        verified.append(f"licence identifiers: {sorted(audit.licences)}")
        verified.append(f"partitions: {sorted(audit.splits)}")
    if candidate.get("evidence_class") == "REQUIRES_EXTERNAL_VERIFICATION":
        unknown.append(
            "licence terms, revision and volume require upstream inspection"
        )
    unknown.extend(candidate.get("evidence_gaps") or [])
    if audit is not None and (audit.hash_mismatches or audit.conflicting):
        unknown.append(
            f"{audit.hash_mismatches} records have a content hash that does not "
            "match their text"
        )
    if candidate.get("state") == "not_recommended":
        disposition = "REJECT_CANDIDATE"
    elif candidate.get("evidence_class") == "REQUIRES_EXTERNAL_VERIFICATION":
        disposition = "REQUEST_EXTERNAL_EVIDENCE"
    elif unknown:
        disposition = "REQUEST_EXTERNAL_EVIDENCE"
    else:
        disposition = "READY_FOR_OWNER_REVIEW"
    return {
        "schema": "quantsmind.candidate_dossier/1",
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("name"),
        "category": candidate.get("category"),
        "verified_facts": verified,
        "unknown_facts": unknown,
        "rights_questions": {
            "declared_license": candidate.get("declared_license"),
            "commercial_use": candidate.get("commercial_use"),
            "training_use": candidate.get("training_use"),
            "attribution_obligations": candidate.get("attribution_obligations"),
            "redistribution": candidate.get("redistribution"),
        },
        "provenance_risks": {
            "third_party_rights": candidate.get("third_party_rights_risk"),
            "personal_data": candidate.get("personal_data_risk"),
            "secrets": candidate.get("secret_exposure_risk"),
            "contamination": candidate.get("contamination_risk"),
        },
        "contamination_findings": (
            audit.splits if audit is not None else {}
        ),
        "scores": scores,
        "recommended_disposition": disposition,
        "required_reviewer": "the repository owner (rights decision)",
        "exact_decision_required": (
            "accept or reject the licence and its evidence, the intended-use "
            "scope, the provenance chain and the contamination assessment, and "
            "record the decision with a reviewer and a rationale"
        ),
        "record_audit": audit.to_dict() if audit is not None else None,
        "note": (
            "A dossier is a recommendation. It does not approve a source, and "
            "no candidate state makes a source eligible for production "
            "ingestion."
        ),
    }


def diversity_report(audit: RecordAudit) -> dict:
    """Metadata-only diversity diagnostics for a corpus.

    No source text is read: the report uses record counts, licence spread,
    partition spread, repository coverage and the structural hash signal.
    """
    if not audit.records:
        return {"records": 0, "note": "no records"}
    licences = audit.licences or {}
    dominant_licence = max(licences.items(), key=lambda item: item[1])[0] if licences else "unknown"
    dominant_share = (
        licences.get(dominant_licence, 0) / audit.records if licences else 0.0
    )
    splits = audit.splits or {}
    distinct_splits = len([name for name in splits if name not in ("unknown", "none")])
    return {
        "records": audit.records,
        "repositories": audit.repositories,
        "records_per_repository": (
            round(audit.records / audit.repositories, 2) if audit.repositories else None
        ),
        "licence_count": len(licences),
        "dominant_licence": dominant_licence,
        "dominant_licence_share": round(dominant_share, 4),
        "partitions": splits,
        "distinct_partitions": distinct_splits,
        "duplicate_content_records": audit.duplicate_content,
        "duplicate_content_share": round(audit.duplicate_content / audit.records, 4),
        "hash_mismatches": audit.hash_mismatches,
        "narrowness_signal": (
            "narrow: one repository or one licence dominates"
            if (audit.repositories <= 2 or dominant_share >= 0.9)
            else "not narrow by these indicators"
        ),
        "note": (
            "metadata-only inspection; the content of the corpus was not read "
            "for this report"
        ),
    }


def owner_decision_packet(dossiers: list) -> dict:
    """The owner's decision packet: the questions that actually need answers."""
    questions = []
    for dossier in dossiers:
        for question in dossier.get("unknown_facts", [])[:3]:
            questions.append(
                {
                    "candidate_id": dossier["candidate_id"],
                    "question": question,
                    "decision_maker": "the repository owner",
                    "artifact_required": "a source manifest with state 'approved'",
                }
            )
    return {
        "schema": "quantsmind.owner_decision_packet/1",
        "questions": questions,
        "approved_sources": 0,
        "eligible_sources": 0,
        "production_training_authorized": False,
        "note": (
            "Generating this packet grants nothing. An approval requires an "
            "explicit owner decision recorded in a source manifest, and a "
            "training run requires a separate authorization."
        ),
    }
