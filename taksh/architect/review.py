"""Human-review heuristics for software architecture artifacts.

Findings are advisory and model-generated. They are never evidence of
correctness and never replace human review.
"""

from __future__ import annotations

from typing import Any

from taksh.architect.artifacts import ArchitectArtifacts, ReviewFinding


def review_design(
    artifacts: ArchitectArtifacts,
    *,
    requirements: list[dict[str, Any]] | list[str] | None = None,
) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []

    corpus = " ".join([
        artifacts.summary,
        " ".join(decision.title + " " + decision.decision for decision in artifacts.decisions),
        " ".join(diagram.source for diagram in artifacts.diagrams),
        str(artifacts.api_specs),
        str(artifacts.data_models),
        str(artifacts.migration),
        " ".join(risk.category + " " + risk.mitigation for risk in artifacts.risks),
    ]).lower()

    if not artifacts.risks:
        findings.append(ReviewFinding(
            "operability",
            "HIGH",
            "risks",
            "No explicit risks identified; production-facing designs must address failure and rollback.",
        ))
    if not any(term in corpus for term in ("rollback", "roll back", "revert")):
        findings.append(ReviewFinding(
            "operability",
            "MEDIUM",
            "migration",
            "No rollback or revert strategy is visible in the artifacts.",
        ))
    if not any(term in corpus for term in ("observ", "monitor", "alert", "metric", "log")):
        findings.append(ReviewFinding(
            "operability",
            "MEDIUM",
            "summary",
            "No observability, monitoring or alerting strategy is visible.",
        ))
    if artifacts.api_specs and not any(term in corpus for term in ("auth", "token", "permission", "scope", "tls")):
        findings.append(ReviewFinding(
            "security",
            "HIGH",
            "api_specs",
            "API surface has no visible authentication, authorization or transport-security decision.",
        ))
    if artifacts.api_specs and not any(term in corpus for term in ("secret", "credential", "key")):
        findings.append(ReviewFinding(
            "security",
            "MEDIUM",
            "api_specs",
            "Secret handling is not explicitly addressed for the API surface.",
        ))
    if not artifacts.open_questions and requirements:
        ambiguous = [
            item for item in requirements
            if any(word in str(item).lower() for word in ("maybe", "could", "should", "unclear", "optional"))
        ]
        if ambiguous:
            findings.append(ReviewFinding(
                "requirements",
                "MEDIUM",
                "open_questions",
                "Ambiguous requirements produced no clarifying open questions.",
            ))
    if not artifacts.decisions:
        findings.append(ReviewFinding(
            "design_rationale",
            "MEDIUM",
            "decisions",
            "No ADR or decision rationale is recorded; reviewers cannot audit why this design was chosen.",
        ))
    if not artifacts.diagrams:
        findings.append(ReviewFinding(
            "structure",
            "LOW",
            "diagrams",
            "No architecture diagram is included; maintainability review will be harder.",
        ))
    return findings
