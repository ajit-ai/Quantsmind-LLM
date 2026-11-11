"""Source-evidence ledger — invalidation tied to source identity.

Every promoted knowledge item is traceable to a source hash and the
validation run that verified it. When the source artifact changes (different
hash), all evidence bound to the previous hash is invalidated — never
silently retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.knowledge import KnowledgeFormationEngine
from taksh.dlcae.types import fingerprint, utcnow_iso


@dataclass
class EvidenceBinding:
    source_sha256: str
    record_id: str
    validation_verdict: str
    bound_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {"source_sha256": self.source_sha256, "record_id": self.record_id,
                "validation_verdict": self.validation_verdict,
                "bound_at": self.bound_at}


class SourceEvidenceLedger:
    """Binds knowledge records to source hashes; invalidates on change."""

    def __init__(self, knowledge: KnowledgeFormationEngine) -> None:
        self._knowledge = knowledge
        self._bindings: list[EvidenceBinding] = []

    def bind(self, source_sha256: str, record_id: str,
             validation_verdict: str) -> EvidenceBinding:
        if validation_verdict != "PASS":
            raise ValueError("evidence may be bound only against a Validation Engine PASS verdict")
        binding = EvidenceBinding(source_sha256=source_sha256, record_id=record_id,
                                  validation_verdict=validation_verdict)
        self._bindings.append(binding)
        return binding

    def verify_source(self, source_sha256: str, *,
                      reason: str = "source identity changed") -> dict[str, Any]:
        """Invalidate every binding to any other source hash.

        Returns a report of records invalidated; idempotent. Never re-promotes
        anything automatically.
        """
        invalidated: list[str] = []
        keep: list[EvidenceBinding] = []
        for binding in self._bindings:
            if binding.source_sha256 != source_sha256:
                record = self._knowledge.invalidate(
                    binding.record_id,
                    reason=f"{reason} ({binding.source_sha256[:12]}… -> "
                           f"{source_sha256[:12]}…)")
                invalidated.append(record.record_id)
            else:
                keep.append(binding)
        self._bindings = keep
        return {
            "schema": "taksh.dlcae.invalidation_report/1",
            "at": utcnow_iso(),
            "source_sha256": source_sha256,
            "invalidated_records": sorted(set(invalidated)),
            "invalidated_count": len(set(invalidated)),
            "retained_bindings": len(keep),
            "rule": "evidence tied to a source hash cannot outlive that source identity",
        }

    def bindings_for(self, source_sha256: str) -> list[EvidenceBinding]:
        return [b for b in self._bindings if b.source_sha256 == source_sha256]

    def snapshot(self) -> dict[str, Any]:
        items = [b.to_dict() for b in self._bindings]
        return {"schema": "taksh.dlcae.evidence_ledger/1",
                "bindings": items,
                "ledger_fingerprint": fingerprint(items),
                "count": len(items)}
