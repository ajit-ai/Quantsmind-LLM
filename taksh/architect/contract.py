"""Design Architect task contract and schema identities.

A contract is a statement of intent. It does not authorize execution,
training, file mutation or deployment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

CONTRACT_ID = "qm.architect_design/1"
ARTIFACT_SCHEMA = "taksh.architect_artifacts/1"
BENCH_SCHEMA = "taksh.arch_bench/1"


class ArtifactKind(str, Enum):
    OVERVIEW = "overview"
    ADR = "adrs"
    DIAGRAMS = "diagrams"
    API_SPECS = "api_specs"
    DATA_MODELS = "data_models"
    MIGRATION = "migration"
    RISKS = "risks"
    REVIEW = "review"


class ArchitectTaskKind(str, Enum):
    DESIGN = "design"
    REVIEW = "review"
    MIGRATION = "migration"
    ADR = "adr"
    DIAGRAM = "diagram"
    API_CONTRACT = "api_contract"
    DATA_MODEL = "data_model"


@dataclass
class ArchitectContract:
    """Input contract for a Design Architect task."""

    requirements: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    repository_context: dict[str, Any] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    non_functional: dict[str, Any] = field(default_factory=dict)
    requested_artifacts: list[str] = field(default_factory=lambda: [
        ArtifactKind.OVERVIEW.value,
        ArtifactKind.ADR.value,
        ArtifactKind.DIAGRAMS.value,
        ArtifactKind.API_SPECS.value,
        ArtifactKind.RISKS.value,
    ])
    review_level: str = "draft"
    task_kind: ArchitectTaskKind = ArchitectTaskKind.DESIGN
    schema: str = CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema != CONTRACT_ID:
            raise ValueError(f"schema must be {CONTRACT_ID!r}, got {self.schema!r}")
        if self.review_level not in {"draft", "review", "approval_candidate"}:
            raise ValueError("review_level must be draft|review|approval_candidate")
        if not isinstance(self.task_kind, ArchitectTaskKind):
            self.task_kind = ArchitectTaskKind(self.task_kind)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_kind"] = self.task_kind.value
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArchitectContract":
        payload = dict(data)
        payload["task_kind"] = ArchitectTaskKind(
            payload.get("task_kind", ArchitectTaskKind.DESIGN.value)
        )
        return ArchitectContract(**payload)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_schema_fingerprint(payload: dict[str, Any]) -> str:
    """Fingerprint a payload under the artifact schema."""
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"artifact payload schema must be {ARTIFACT_SCHEMA!r}")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
