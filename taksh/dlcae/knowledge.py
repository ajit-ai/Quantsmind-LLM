"""Knowledge Formation Engine — structured, versioned language knowledge.

Knowledge states (exactly five): OBSERVED, INFERRED, VERIFIED, CONFLICTED,
INVALIDATED. Knowledge rules (mandatory):

- retrieved documentation or model output is never VERIFIED by virtue of
  having been retrieved — verification requires an independent validation
  verdict (Validation Engine);
- conflicting observations are kept, both marked CONFLICTED — never silently
  merged;
- invalidation is explicit and versioned; records are never deleted;
- provenance and confidence are mandatory on every record;
- content-derived fingerprints make exact duplicates idempotent.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable

from taksh.dlcae.types import Provenance, VerificationState, fingerprint, utcnow_iso

KNOWLEDGE_KINDS = frozenset({
    "concept", "syntax_structure", "semantic_relationship", "type_information",
    "dependency_information", "language_rule", "example", "counterexample",
})


@dataclass
class KnowledgeRecord:
    record_id: str
    language_id: str
    kind: str
    name: str
    content: dict[str, Any]
    provenance: Provenance
    state: VerificationState = VerificationState.OBSERVED
    confidence: float = 0.5
    version: int = 1
    supersedes: str | None = None
    conflicts_with: tuple[str, ...] = ()
    verification_evidence: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=utcnow_iso)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = fingerprint(
                {"language_id": self.language_id, "kind": self.kind,
                 "name": self.name, "content": self.content,
                 "version": self.version})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.knowledge_record/1",
            "record_id": self.record_id,
            "language_id": self.language_id,
            "kind": self.kind,
            "name": self.name,
            "content": self.content,
            "provenance": self.provenance.to_dict(),
            "state": self.state.value,
            "confidence": self.confidence,
            "version": self.version,
            "supersedes": self.supersedes,
            "conflicts_with": list(self.conflicts_with),
            "verification_evidence": self.verification_evidence,
            "created": self.created,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeRecord":
        prov = Provenance.from_dict(data["provenance"])
        return cls(
            record_id=data["record_id"], language_id=data["language_id"],
            kind=data["kind"], name=data["name"], content=dict(data.get("content", {})),
            provenance=prov,
            state=VerificationState(data.get("state", "OBSERVED")),
            confidence=float(data.get("confidence", 0.5)),
            version=int(data.get("version", 1)),
            supersedes=data.get("supersedes"),
            conflicts_with=tuple(data.get("conflicts_with", ())),
            verification_evidence=dict(data.get("verification_evidence", {})),
            created=data.get("created", utcnow_iso()),
            fingerprint=data.get("fingerprint", ""),
        )


class KnowledgeFormationEngine:
    """Append-oriented, versioned store of knowledge records (in memory).

    Persistence goes through :class:`taksh.dlcae.persistence.PersistenceStore`;
    this engine owns *semantics*, not files.
    """

    def __init__(self) -> None:
        self._records: dict[str, KnowledgeRecord] = {}
        self._by_fingerprint: dict[str, str] = {}
        self._counter = itertools.count(1)

    # ------------------------------------------------------------------ #
    def add(self, language_id: str, kind: str, name: str, content: dict[str, Any], *,
            provenance: Provenance, state: VerificationState = VerificationState.OBSERVED,
            confidence: float = 0.5) -> KnowledgeRecord:
        if kind not in KNOWLEDGE_KINDS:
            raise ValueError(f"unknown knowledge kind: {kind!r}; "
                             f"allowed: {sorted(KNOWLEDGE_KINDS)}")
        candidate = KnowledgeRecord(
            record_id=f"kr-{fingerprint({'k': next(self._counter)})[:12]}",
            language_id=language_id, kind=kind, name=name, content=dict(content),
            provenance=provenance, state=state, confidence=confidence)
        existing_id = self._by_fingerprint.get(candidate.fingerprint)
        if existing_id is not None:
            return self._records[existing_id]          # exact duplicate: idempotent
        self._detect_conflicts(candidate)
        self._records[candidate.record_id] = candidate
        self._by_fingerprint[candidate.fingerprint] = candidate.record_id
        return candidate

    def _detect_conflicts(self, candidate: KnowledgeRecord) -> None:
        for record in self._records.values():
            if record.state is VerificationState.INVALIDATED:
                continue
            if (record.language_id == candidate.language_id
                    and record.kind == candidate.kind
                    and record.name == candidate.name):
                record.state = VerificationState.CONFLICTED
                candidate.state = VerificationState.CONFLICTED
                record.conflicts_with = tuple(sorted(set(record.conflicts_with) | {candidate.record_id}))
                candidate.conflicts_with = tuple(sorted(set(candidate.conflicts_with) | {record.record_id}))

    def verify(self, record_id: str, *, validation_verdict: str,
               evidence: dict[str, Any] | None = None) -> KnowledgeRecord:
        record = self._get(record_id)
        if validation_verdict != "PASS":
            # verification requires a real PASS from the Validation Engine;
            # anything else leaves the record unverified (never fabricated).
            record.verification_evidence = {
                **(evidence or {}),
                "validation_verdict": validation_verdict,
                "admitted": False,
                "reason": "verification requires a Validation Engine PASS verdict",
            }
            return record
        record.state = VerificationState.VERIFIED
        record.confidence = max(record.confidence, 0.9)
        record.verification_evidence = {
            **(evidence or {}),
            "validation_verdict": "PASS",
            "admitted": True,
            "verified_at": utcnow_iso(),
        }
        return record

    def contest(self, record_id: str, reason: str) -> KnowledgeRecord:
        record = self._get(record_id)
        record.state = VerificationState.CONFLICTED
        record.verification_evidence = {**record.verification_evidence,
                                        "contested_reason": reason}
        return record

    def invalidate(self, record_id: str, reason: str) -> KnowledgeRecord:
        """Explicit invalidation; the record is kept, never deleted."""
        record = self._get(record_id)
        record.state = VerificationState.INVALIDATED
        record.verification_evidence = {**record.verification_evidence,
                                        "invalidated_reason": reason,
                                        "invalidated_at": utcnow_iso()}
        return record

    def new_version(self, record_id: str, content: dict[str, Any], *,
                    provenance: Provenance | None = None) -> KnowledgeRecord:
        previous = self._get(record_id)
        prov = provenance or previous.provenance
        updated = self.add(previous.language_id, previous.kind, previous.name, content,
                           provenance=prov, state=VerificationState.OBSERVED,
                           confidence=previous.confidence)
        updated.version = previous.version + 1
        updated.supersedes = previous.record_id
        updated.fingerprint = fingerprint(
            {"language_id": updated.language_id, "kind": updated.kind,
             "name": updated.name, "content": updated.content,
             "version": updated.version})
        self._by_fingerprint[updated.fingerprint] = updated.record_id
        return updated

    # ------------------------------------------------------------------ #
    def query(self, *, language_id: str | None = None, kind: str | None = None,
              state: VerificationState | str | None = None,
              name_contains: str | None = None) -> list[KnowledgeRecord]:
        state_value = VerificationState(state) if state else None
        results = []
        for record in self._records.values():
            if language_id and record.language_id != language_id:
                continue
            if kind and record.kind != kind:
                continue
            if state_value and record.state is not state_value:
                continue
            if name_contains and name_contains.lower() not in record.name.lower():
                continue
            results.append(record)
        return sorted(results, key=lambda r: (r.created, r.record_id))

    def all_records(self) -> list[KnowledgeRecord]:
        return sorted(self._records.values(), key=lambda r: (r.created, r.record_id))

    def _get(self, record_id: str) -> KnowledgeRecord:
        try:
            return self._records[record_id]
        except KeyError:
            raise KeyError(f"unknown knowledge record: {record_id}") from None

    # ------------------------------------------------------------------ #
    def export(self) -> dict[str, Any]:
        records = [r.to_dict() for r in self.all_records()]
        return {
            "schema": "taksh.dlcae.knowledge_store/1",
            "exported": utcnow_iso(),
            "record_count": len(records),
            "records": records,
            "snapshot_fingerprint": fingerprint(records),
        }

    def import_store(self, data: dict[str, Any]) -> dict[str, Any]:
        """Import a store snapshot; integrity is verified, never assumed."""
        problems: list[str] = []
        records = data.get("records", [])
        expected = data.get("snapshot_fingerprint")
        if expected and fingerprint(records) != expected:
            problems.append("snapshot fingerprint mismatch — store is corrupt or tampered")
        seen: set[str] = set()
        imported = 0
        for raw in records:
            record = KnowledgeRecord.from_dict(raw)
            if record.fingerprint and fingerprint(
                    {"language_id": record.language_id, "kind": record.kind,
                     "name": record.name, "content": record.content,
                     "version": record.version}) != record.fingerprint:
                problems.append(f"record {record.record_id}: fingerprint mismatch — rejected")
                continue
            if record.record_id in seen or record.record_id in self._records:
                continue  # idempotent re-import
            seen.add(record.record_id)
            self._records[record.record_id] = record
            self._by_fingerprint[record.fingerprint] = record.record_id
            imported += 1
        return {"imported": imported, "problems": problems,
                "ok": not problems}


def provenance_from_source(source: str, *, license_spdx: str = "UNKNOWN",
                           source_kind: str = "synthetic-fixture",
                           record_id: str | None = None) -> Provenance:
    """Convenience builder for deterministic test fixtures."""
    return Provenance(record_id=record_id or fingerprint({"source": source})[:16],
                      source=source, source_kind=source_kind,
                      license_spdx=license_spdx)
