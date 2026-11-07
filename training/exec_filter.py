"""Execution-backed filtering helpers for code-training candidates.

C# candidates are compiled locally. Execution is always opt-in and is disabled
by default. This module does not make generated code safe to run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation.csharp.runner import compile_csharp


@dataclass(frozen=True)
class ExecutionFilterResult:
    language: str
    outcome: str
    passed: bool
    execution_authorized: bool
    diagnostics: list[dict[str, Any]]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_language(language: str | None) -> str:
    normalized = (language or "").strip().lower()
    aliases = {"cs": "csharp", "c#": "csharp", "dotnet": "csharp", ".net": "csharp"}
    return aliases.get(normalized, normalized)


def evaluate_candidate(
    language: str | None,
    source: str,
    *,
    allow_execution: bool = False,
    target_framework: str = "net10.0",
    timeout: int = 90,
) -> ExecutionFilterResult:
    normalized = _normalize_language(language)
    if normalized != "csharp":
        raise ValueError(f"unsupported execution language: {language!r}; supported: csharp")
    result = compile_csharp(
        source,
        target_framework=target_framework,
        timeout=timeout,
        allow_execution=allow_execution,
    )
    passed = result.outcome == ("executed" if allow_execution else "compiled")
    if allow_execution:
        passed = passed and result.exit_code == 0
    diagnostics = [*result.diagnostics, *result.source_diagnostics]
    detail = (
        f"outcome={result.outcome}; exit_code={result.exit_code}; "
        f"diagnostics={len(diagnostics)}"
    )
    return ExecutionFilterResult(
        language=normalized,
        outcome=result.outcome,
        passed=passed,
        execution_authorized=allow_execution,
        diagnostics=diagnostics,
        detail=detail,
    )


def filter_candidates(
    candidates,
    *,
    allow_execution: bool = False,
    target_framework: str = "net10.0",
    timeout: int = 90,
) -> list[ExecutionFilterResult]:
    results = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("each candidate must be a dict with language and source")
        results.append(
            evaluate_candidate(
                candidate.get("language"),
                candidate.get("source", ""),
                allow_execution=allow_execution,
                target_framework=target_framework,
                timeout=timeout,
            )
        )
    return results
