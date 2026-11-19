"""Structural validation for Design Architect artifacts.

Validation checks structure, references and safety invariants. It never
executes generated artifacts and never writes repository files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from taksh.architect.artifacts import ArchitectArtifacts


@dataclass
class ValidationIssue:
    category: str
    severity: str
    location: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
        }


def _sha256_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value or ""))


def _mermaid_ok(source: str) -> bool:
    stripped = (source or "").lstrip()
    allowed = ("graph ", "flowchart ", "sequenceDiagram", "C4Context", "C4Container", "C4Component")
    return bool(stripped) and stripped.startswith(allowed)


def _dependency_graph(artifacts: ArchitectArtifacts) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for item in artifacts.data_models:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        graph.setdefault(name, set())
        for dependency in item.get("depends_on", []) or []:
            graph[name].add(str(dependency))
    migration_dependencies = artifacts.migration.get("dependencies", []) if isinstance(artifacts.migration, dict) else []
    for item in migration_dependencies:
        source = str(item.get("from", "")).strip()
        target = str(item.get("to", "")).strip()
        if source and target:
            graph.setdefault(source, set()).add(target)
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    colour: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        colour[node] = 1  # visiting
        for child in graph.get(node, set()):
            if colour.get(child) == 1:
                if child in path:
                    cycle = path[path.index(child):] + [child]
                    key = tuple(cycle)
                    if list(cycle) not in cycles:
                        cycles.append(cycle)
            elif colour.get(child) != 2:
                visit(child, path + [child])
        colour[node] = 2

    for node in list(graph):
        if colour.get(node, 0) == 0:
            visit(node, [node])
    return cycles


def validate_artifacts(
    artifacts: ArchitectArtifacts,
    *,
    requirements: list[dict[str, Any]] | list[str] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not artifacts.summary.strip():
        issues.append(ValidationIssue("missing_summary", "HIGH", "summary", "Summary is empty"))
    if not _sha256_hex(artifacts.source_request_sha256):
        issues.append(ValidationIssue(
            "request_hash",
            "HIGH",
            "source_request_sha256",
            "source_request_sha256 must be a 64-character SHA-256 hex digest",
        ))

    ids = [decision.id for decision in artifacts.decisions]
    if len(ids) != len(set(ids)):
        seen, duplicates = set(), set()
        for identifier in ids:
            if identifier in seen:
                duplicates.add(identifier)
            seen.add(identifier)
        issues.append(ValidationIssue(
            "duplicate_adr",
            "HIGH",
            "decisions",
            f"Duplicate ADR identifiers: {sorted(duplicates)}",
        ))
    for index, decision in enumerate(artifacts.decisions):
        if not decision.title.strip() or not decision.decision.strip() or not decision.rationale.strip():
            issues.append(ValidationIssue(
                "adr_fields",
                "HIGH",
                f"decisions[{index}]",
                f"ADR {decision.id or index} is missing title, decision or rationale",
            ))
        if decision.status not in {"proposed", "accepted", "rejected", "superseded"}:
            issues.append(ValidationIssue(
                "adr_status",
                "MEDIUM",
                f"decisions[{index}]",
                f"ADR status must be proposed|accepted|rejected|superseded, got {decision.status!r}",
            ))

    for index, diagram in enumerate(artifacts.diagrams):
        if not _mermaid_ok(diagram.source):
            issues.append(ValidationIssue(
                "diagram_syntax",
                "MEDIUM",
                f"diagrams[{index}]",
                f"Mermaid diagram {index} does not start with a supported diagram declaration",
            ))

    for index, spec in enumerate(artifacts.api_specs):
        if not isinstance(spec, dict) or not spec:
            issues.append(ValidationIssue(
                "api_spec",
                "MEDIUM",
                f"api_specs[{index}]",
                "API specification is empty",
            ))
            continue
        if not any(key in spec for key in ("openapi", "async_api", "info", "paths", "channels")):
            issues.append(ValidationIssue(
                "api_spec",
                "MEDIUM",
                f"api_specs[{index}]",
                "API specification lacks OpenAPI/AsyncAPI structure markers",
            ))

    for index, risk in enumerate(artifacts.risks):
        if risk.severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            issues.append(ValidationIssue(
                "risk_severity",
                "MEDIUM",
                f"risks[{index}]",
                f"Risk severity must be LOW|MEDIUM|HIGH|CRITICAL, got {risk.severity!r}",
            ))

    for cycle in _cycles(_dependency_graph(artifacts)):
        issues.append(ValidationIssue(
            "dependency_cycle",
            "HIGH",
            "data_models",
            f"Uncontrolled dependency cycle: {' -> '.join(cycle)}",
        ))

    if requirements:
        combined = " ".join([
            artifacts.summary,
            " ".join(decision.decision + " " + decision.rationale for decision in artifacts.decisions),
            " ".join(diagram.source for diagram in artifacts.diagrams),
            json_render(artifacts.api_specs),
            json_render(artifacts.data_models),
            json_render(artifacts.migration),
            " ".join(risk.category + " " + risk.mitigation for risk in artifacts.risks),
        ]).lower()
        for index, requirement in enumerate(requirements):
            requirement_id = None
            if isinstance(requirement, dict):
                requirement_id = str(requirement.get("id", "")).strip().lower()
                requirement_text = str(requirement.get("text", requirement)).lower()
            else:
                requirement_text = str(requirement).lower()
            if requirement_id and requirement_id in combined:
                continue
            if len(requirement_text) > 24 and all(token not in combined for token in requirement_text.split()[:3]):
                issues.append(ValidationIssue(
                    "requirement_coverage",
                    "MEDIUM",
                    f"requirements[{index}]",
                    "Requirement does not appear to be covered by any artifact",
                ))
            if any(word in requirement_text for word in ("maybe", "could", "should", "unclear")):
                if not artifacts.open_questions:
                    issues.append(ValidationIssue(
                        "ambiguity_unresolved",
                        "MEDIUM",
                        f"requirements[{index}]",
                        "Ambiguous requirement produced no open question",
                    ))

    return issues


def json_render(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str)
