"""Semantic Analysis Contract — versioned, language-neutral.

Schema ``taksh.dlcae.semantic_contract/1``. A SemanticResult records:

- language identity + registry version;
- source artifact identity (content hash, module, file);
- comprehension observations;
- parser/AST observations when available;
- dependency and symbol context;
- toolchain identity and version;
- validation configuration;
- semantic status, diagnostics, symbols, references, type relations,
  dependency relations, locations, confidence, provenance, toolchain
  execution status, and unresolved/unsupported constructs.

Statuses (exactly six): PASS, FAIL, PARTIAL, BLOCKED, UNVERIFIED,
NOT_APPLICABLE. A missing toolchain never produces PASS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.types import fingerprint, utcnow_iso

SEMANTIC_CONTRACT_VERSION = 1
SEMANTIC_SCHEMA = "taksh.dlcae.semantic_contract/1"

SEMANTIC_STATUSES = frozenset({
    "PASS", "FAIL", "PARTIAL", "BLOCKED", "UNVERIFIED", "NOT_APPLICABLE",
})
STATUS_ORDER = {"PASS": 5, "FAIL": 0, "PARTIAL": 2, "BLOCKED": 1,
                "UNVERIFIED": 3, "NOT_APPLICABLE": 4}

DIAGNOSTIC_CATEGORIES = frozenset({
    "syntax_error", "type_error", "unresolved_symbol", "missing_dependency",
    "compiler_error", "runtime_exception", "test_failure", "warning",
    "timeout", "resource_limit", "blocked", "style",
})
SEVERITIES = frozenset({"ERROR", "WARNING", "INFO", "HINT"})


class SemanticStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ContractError(Exception):
    """Raised when a semantic result violates the contract."""


@dataclass
class Diagnostic:
    category: str
    message: str
    severity: str = "ERROR"
    line: int | None = None
    column: int | None = None
    file: str | None = None
    symbol: str | None = None
    source: str = "unspecified"     # adapter id, e.g. "python-mypy"
    tool_version: str = "unknown"
    evidence_classification: str = "HEURISTIC"  # TOOL_VERIFIED | MODEL_INFERRED | HEURISTIC | UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "message": self.message,
                "severity": self.severity, "line": self.line, "column": self.column,
                "file": self.file, "symbol": self.symbol, "source": self.source,
                "tool_version": self.tool_version,
                "evidence_classification": self.evidence_classification}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Diagnostic":
        return cls(category=data["category"], message=data["message"],
                   severity=data.get("severity", "ERROR"),
                   line=data.get("line"), column=data.get("column"),
                   file=data.get("file"), symbol=data.get("symbol"),
                   source=data.get("source", "unspecified"),
                   tool_version=data.get("tool_version", "unknown"),
                   evidence_classification=data.get("evidence_classification", "HEURISTIC"))


def validate_diagnostic(diagnostic: dict[str, Any]) -> list[str]:
    problems = []
    if diagnostic.get("category") not in DIAGNOSTIC_CATEGORIES:
        problems.append(f"unknown category: {diagnostic.get('category')!r}")
    if diagnostic.get("severity") not in SEVERITIES:
        problems.append(f"unknown severity: {diagnostic.get('severity')!r}")
    if not isinstance(diagnostic.get("message"), str) or not diagnostic["message"].strip():
        problems.append("message must be a non-empty string")
    for key in ("line", "column"):
        value = diagnostic.get(key)
        if value is not None and (not isinstance(value, int) or value < 1):
            problems.append(f"{key} must be a positive integer or null")
    return problems


def validate_result(result: dict[str, Any]) -> list[str]:
    """Contract validation. Returns a list of problems (empty == valid)."""
    problems: list[str] = []
    if result.get("schema") != SEMANTIC_SCHEMA:
        problems.append(f"schema mismatch: {result.get('schema')!r} != {SEMANTIC_SCHEMA!r}")
    if result.get("contract_version") != SEMANTIC_CONTRACT_VERSION:
        problems.append(f"contract_version mismatch: {result.get('contract_version')!r}")
    status = result.get("status")
    if status not in SEMANTIC_STATUSES:
        problems.append(f"invalid status: {status!r}")
    for key in ("language_id", "source_sha256", "file", "module"):
        value = result.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key} must be a non-empty string")
    toolchain = result.get("toolchain")
    if not isinstance(toolchain, dict) or "available" not in toolchain:
        problems.append("toolchain must record availability")
    elif toolchain.get("available") is False and status == SemanticStatus.PASS:
        problems.append("missing toolchain must never produce PASS")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, list):
        problems.append("diagnostics must be a list")
    else:
        for i, diagnostic in enumerate(diagnostics):
            problems.extend(f"diagnostics[{i}]: {p}" for p in validate_diagnostic(diagnostic))
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source_sha256"):
        problems.append("provenance must include source_sha256")
    return problems


def build_result(*, language_id: str, source_sha256: str, file: str, module: str,
                 registry_version: str = "dynreg/1",
                 produced_by: str = "taksh.dlcae.semantic/1.0.0") -> dict[str, Any]:
    return {
        "schema": SEMANTIC_SCHEMA,
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "language_id": language_id,
        "registry_version": registry_version,
        "source_sha256": source_sha256,
        "file": file,
        "module": module,
        "status": SemanticStatus.UNVERIFIED,
        "diagnostics": [],
        "symbols": [],
        "references": [],
        "type_relationships": [],
        "dependency_relationships": [],
        "comprehension": {},
        "toolchain": {"id": "none", "name": "none", "version": "unknown",
                      "available": False, "execution_status": "NOT_RUN"},
        "provenance": {"source_sha256": source_sha256, "produced_by": produced_by,
                       "produced_at": utcnow_iso()},
        "unsupported_constructs": [],
        "unresolved": [],
        "result_id": fingerprint({"lang": language_id, "sha": source_sha256,
                                  "file": file})[:16],
    }


def merge_status(stages: dict[str, str]) -> str:
    """Deterministic stage-status merge.

    Any ALARM (FAIL/BLOCKED) blocks the aggregate; partial results degrade to
    PARTIAL; anything unverified stays UNVERIFIED; else PASS.
    """
    values = list(stages.values())
    if not values:
        return SemanticStatus.NOT_APPLICABLE
    if any(v == SemanticStatus.FAIL for v in values):
        return SemanticStatus.FAIL
    if any(v == SemanticStatus.BLOCKED for v in values):
        return SemanticStatus.PARTIAL if any(v in (SemanticStatus.PASS, SemanticStatus.PARTIAL)
                                              for v in values) else SemanticStatus.BLOCKED
    if any(v == SemanticStatus.PARTIAL for v in values):
        return SemanticStatus.PARTIAL
    if any(v == SemanticStatus.UNVERIFIED for v in values):
        return SemanticStatus.UNVERIFIED
    return SemanticStatus.PASS


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
