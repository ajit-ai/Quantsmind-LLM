"""Canonical prompt rendering for Design Architect tasks."""

from __future__ import annotations

from taksh.architect.contract import CONTRACT_ID, ArchitectContract


def render_architect_prompt(contract: ArchitectContract) -> str:
    """Render the single canonical prompt used for every architect task.

    The renderer is deterministic: identical contracts produce identical bytes.
    """
    if contract.schema != CONTRACT_ID:
        raise ValueError(f"unexpected contract schema: {contract.schema!r}")

    lines: list[str] = [
        f"Contract: {CONTRACT_ID}",
        f"Task kind: {contract.task_kind.value}",
        f"Review level: {contract.review_level}",
        "",
        "Requirements:",
    ]
    for requirement in contract.requirements:
        lines.append(f"- {requirement}")
    if contract.constraints:
        lines.append("")
        lines.append("Constraints:")
        lines.extend(f"- {constraint}" for constraint in contract.constraints)
    if contract.languages:
        lines.append("")
        lines.append("Languages:")
        lines.extend(f"- {language}" for language in contract.languages)
    if contract.non_functional:
        lines.append("")
        lines.append("Non-functional requirements:")
        for key, value in sorted(contract.non_functional.items()):
            lines.append(f"- {key}: {value}")
    if contract.repository_context:
        lines.append("")
        lines.append("Repository context:")
        for key, value in sorted(contract.repository_context.items()):
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Requested artifacts:")
    lines.extend(f"- {artifact}" for artifact in contract.requested_artifacts)
    lines.append("")
    lines.append(
        "Output rules: explicit assumptions; open questions when information is missing; "
        "unique ADR identifiers; no repository mutation; no executable code; "
        "model-generated output is not verified output."
    )
    return "\n".join(lines) + "\n"
