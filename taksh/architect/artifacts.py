"""Structured artifacts produced by Design Architect.

Artifacts are inert data. They are not applied to a repository, compiled or
executed by this package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from taksh.architect.contract import ARTIFACT_SCHEMA


def _fingerprint(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Adr:
    id: str
    title: str
    status: str
    decision: str
    rationale: str
    consequences: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Adr":
        return Adr(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagram:
    kind: str
    type: str
    source: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Diagram":
        return Diagram(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Risk:
    category: str
    severity: str
    likelihood: str
    mitigation: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Risk":
        return Risk(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewFinding:
    category: str
    severity: str
    location: str
    message: str
    evidence: str = "MODEL_GENERATED"

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ReviewFinding":
        return ReviewFinding(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectArtifacts:
    summary: str
    source_request_sha256: str
    schema: str = ARTIFACT_SCHEMA
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decisions: list[Adr] = field(default_factory=list)
    diagrams: list[Diagram] = field(default_factory=list)
    api_specs: list[dict[str, Any]] = field(default_factory=list)
    data_models: list[dict[str, Any]] = field(default_factory=list)
    migration: dict[str, Any] = field(default_factory=dict)
    risks: list[Risk] = field(default_factory=list)
    review: list[ReviewFinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError(f"schema must be {ARTIFACT_SCHEMA!r}, got {self.schema!r}")
        if self.source_request_sha256 and len(self.source_request_sha256) != 64:
            raise ValueError("source_request_sha256 must be a 64-character SHA-256 hex digest")

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArchitectArtifacts":
        payload = dict(data)
        for collection, item_type in (
            ("decisions", Adr),
            ("diagrams", Diagram),
            ("risks", Risk),
            ("review", ReviewFinding),
        ):
            payload[collection] = [item_type.from_dict(item) for item in payload.get(collection, [])]
        return ArchitectArtifacts(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = self.schema
        return data

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())
