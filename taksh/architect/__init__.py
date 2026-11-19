"""TAKSH Design Architect — coding-adjacent software architecture specialist.

Engineering-qualified building blocks only. No model weights exist, no
capability is claimed, and generated artifacts are never applied to a
repository by this package.
"""

from taksh.architect.contract import (
    ARTIFACT_SCHEMA,
    BENCH_SCHEMA,
    CONTRACT_ID,
    ArchitectContract,
    ArtifactKind,
    ArchitectTaskKind,
    artifact_schema_fingerprint,
)
from taksh.architect.artifacts import (
    Adr,
    ArchitectArtifacts,
    Diagram,
    ReviewFinding,
    Risk,
)
from taksh.architect.render import render_architect_prompt
from taksh.architect.validate import ValidationIssue, validate_artifacts
from taksh.architect.review import review_design

ARCHITECT_PACKAGE_VERSION = "0.1.0"

__all__ = [
    "ARCHITECT_PACKAGE_VERSION",
    "ArchitectContract",
    "ArchitectArtifacts",
    "ArchitectTaskKind",
    "ArtifactKind",
    "Adr",
    "Diagram",
    "ReviewFinding",
    "Risk",
    "ValidationIssue",
    "CONTRACT_ID",
    "ARTIFACT_SCHEMA",
    "BENCH_SCHEMA",
    "render_architect_prompt",
    "validate_artifacts",
    "review_design",
    "artifact_schema_fingerprint",
]
