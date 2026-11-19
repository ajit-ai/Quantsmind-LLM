"""Shared types, canonical JSON serialization and fingerprints.

Every persisted or compared artifact in DLCAE is produced through
``canonical_json``/``fingerprint`` so that hashes are stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, tight separators, stable defaults."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """Content-derived fingerprint of a JSON-serializable object."""
    return sha256_text(canonical_json(obj))


class VerificationState(str, Enum):
    """Knowledge/observation verification states (exactly five)."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    VERIFIED = "VERIFIED"
    CONFLICTED = "CONFLICTED"
    INVALIDATED = "INVALIDATED"


class ValidationVerdict(str, Enum):
    """Validation outcomes. No verdict is ever coerced to PASS."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNVERIFIED = "UNVERIFIED"


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


class QualificationStatus(str, Enum):
    """Language qualification statuses (exactly six, plus never-invent states)."""

    REGISTERED = "REGISTERED"
    DISCOVERED = "DISCOVERED"
    COMPREHENSION_IN_PROGRESS = "COMPREHENSION_IN_PROGRESS"
    FUNCTIONALLY_QUALIFIED = "FUNCTIONALLY_QUALIFIED"
    ENGINEERING_QUALIFIED = "ENGINEERING_QUALIFIED"
    BLOCKED = "BLOCKED"


CAPABILITY_STATES = frozenset(
    {QualificationStatus.FUNCTIONALLY_QUALIFIED,
     QualificationStatus.ENGINEERING_QUALIFIED}
)
NON_CAPABILITY_STATES = frozenset(QualificationStatus) - CAPABILITY_STATES


@dataclass
class Provenance:
    """Provenance attached to every record DLCAE produces."""

    record_id: str
    source: str
    source_kind: str = "unknown"          # synthetic-fixture | repository | document | toolchain | owner-decision
    license_spdx: str = "UNKNOWN"
    retrieved: str = field(default_factory=utcnow_iso)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "source_kind": self.source_kind,
            "license_spdx": self.license_spdx,
            "retrieved": self.retrieved,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(
            record_id=str(data["record_id"]),
            source=str(data["source"]),
            source_kind=str(data.get("source_kind", "unknown")),
            license_spdx=str(data.get("license_spdx", "UNKNOWN")),
            retrieved=str(data.get("retrieved", utcnow_iso())),
            note=str(data.get("note", "")),
        )


@dataclass
class Observation:
    """A single comprehension-level observation.

    Offsets are byte offsets where available; records may mark uncertainty
    and conflicting observation ids instead of being dropped.
    """

    level: int
    kind: str
    detail: dict[str, Any]
    provenance: Provenance
    confidence: float = 1.0
    state: VerificationState = VerificationState.OBSERVED
    uncertain: bool = False
    conflict_ids: tuple[str, ...] = ()
    offsets: tuple[int, int] | None = None
    observation_id: str = ""

    def __post_init__(self) -> None:
        if not self.observation_id:
            self.observation_id = fingerprint(
                {"level": self.level, "kind": self.kind,
                 "detail": self.detail, "state": self.state.value,
                 "offsets": list(self.offsets) if self.offsets else None})[:16]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "level": self.level,
            "kind": self.kind,
            "detail": self.detail,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "state": self.state.value,
            "uncertain": self.uncertain,
            "conflict_ids": list(self.conflict_ids),
            "offsets": list(self.offsets) if self.offsets else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        offsets = data.get("offsets")
        return cls(
            level=int(data["level"]),
            kind=str(data["kind"]),
            detail=dict(data.get("detail", {})),
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=float(data.get("confidence", 1.0)),
            state=VerificationState(data.get("state", "OBSERVED")),
            uncertain=bool(data.get("uncertain", False)),
            conflict_ids=tuple(data.get("conflict_ids", ())),
            offsets=(int(offsets[0]), int(offsets[1])) if offsets else None,
            observation_id=str(data.get("observation_id", "")),
        )
