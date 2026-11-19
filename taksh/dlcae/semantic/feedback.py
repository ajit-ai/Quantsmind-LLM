"""Compiler/interpreter feedback records, correlation and failure localization.

Feedback records are structured, hash-pinned and provenance-complete. The
correlation layer groups diagnostics by location, links downstream failures
to upstream observations, separates primary from cascading failures, and
preserves uncertainty when the root cause is not established.

Repair candidates are suggestions only — generated text, never applied. No
source file is ever modified by this module; there is no code path that
writes sources.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.types import fingerprint, utcnow_iso

# Lower number = earlier / more causal in a failure chain.
_CATEGORY_RANK = {
    "syntax_error": 0,
    "missing_dependency": 1,
    "compiler_error": 2,
    "unresolved_symbol": 3,
    "type_error": 4,
    "runtime_exception": 5,
    "test_failure": 6,
    "resource_limit": 7,
    "timeout": 7,
    "warning": 9,
    "style": 10,
    "blocked": 2,
}

EVIDENCE_CLASSIFICATIONS = frozenset({"TOOL_VERIFIED", "MODEL_INFERRED",
                                      "HEURISTIC", "UNVERIFIED"})


@dataclass
class FeedbackRecord:
    feedback_id: str
    source_sha256: str
    language_id: str
    adapter: str
    adapter_version: str
    toolchain_name: str
    toolchain_version: str
    category: str
    message: str
    line: int | None = None
    column: int | None = None
    file: str | None = None
    execution_context: str = "not executed"
    reproducibility: dict[str, Any] = field(default_factory=dict)
    evidence_classification: str = "UNVERIFIED"
    context: dict[str, Any] = field(default_factory=dict)
    primary: bool | None = None          # set by correlation
    cascade_of: list[str] = field(default_factory=list)
    created: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.feedback_record/1",
            "feedback_id": self.feedback_id,
            "source_sha256": self.source_sha256,
            "language_id": self.language_id,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "toolchain_name": self.toolchain_name,
            "toolchain_version": self.toolchain_version,
            "category": self.category,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "file": self.file,
            "execution_context": self.execution_context,
            "reproducibility": self.reproducibility,
            "evidence_classification": self.evidence_classification,
            "context": self.context,
            "primary": self.primary,
            "cascade_of": self.cascade_of,
            "created": self.created,
        }

    @classmethod
    def build(cls, *, source_sha256: str, language_id: str, adapter: str,
              adapter_version: str, toolchain_name: str, toolchain_version: str,
              category: str, message: str, line: int | None = None,
              column: int | None = None, file: str | None = None,
              execution_context: str = "not executed",
              reproducibility: dict[str, Any] | None = None,
              evidence_classification: str = "UNVERIFIED",
              context: dict[str, Any] | None = None) -> "FeedbackRecord":
        if evidence_classification not in EVIDENCE_CLASSIFICATIONS:
            raise ValueError(f"unknown evidence classification: {evidence_classification!r}")
        record = cls(
            feedback_id=fingerprint({"src": source_sha256, "adapter": adapter,
                                     "cat": category, "line": line,
                                     "message": message})[:16],
            source_sha256=source_sha256, language_id=language_id,
            adapter=adapter, adapter_version=adapter_version,
            toolchain_name=toolchain_name, toolchain_version=toolchain_version,
            category=category, message=message, line=line, column=column, file=file,
            execution_context=execution_context,
            reproducibility=reproducibility or {
                "tool": toolchain_name,
                "tool_version": toolchain_version,
                "adapter": adapter,
                "adapter_version": adapter_version,
                "source_sha256": source_sha256,
                "platform": sys.platform,
            },
            evidence_classification=evidence_classification,
            context=context or {})
        return record


def from_diagnostic(diagnostic: dict[str, Any], *, source_sha256: str,
                    language_id: str, adapter: str, adapter_version: str,
                    toolchain_name: str, toolchain_version: str,
                    file: str) -> FeedbackRecord:
    return FeedbackRecord.build(
        source_sha256=source_sha256, language_id=language_id,
        adapter=adapter, adapter_version=adapter_version,
        toolchain_name=toolchain_name, toolchain_version=toolchain_version,
        category=diagnostic.get("category", "compiler_error"),
        message=diagnostic.get("message", "unspecified diagnostic"),
        line=diagnostic.get("line"), column=diagnostic.get("column"), file=file,
        execution_context=f"{toolchain_name} ({toolchain_version})",
        reproducibility={"tool": toolchain_name, "tool_version": toolchain_version,
                         "adapter": adapter, "adapter_version": adapter_version,
                         "source_sha256": source_sha256},
        evidence_classification=diagnostic.get("evidence_classification", "UNVERIFIED"))


def _rank(record: FeedbackRecord) -> tuple[int, int]:
    return (_CATEGORY_RANK.get(record.category, 8), record.line or 1_000_000)


def correlate(records: list[FeedbackRecord]) -> dict[str, Any]:
    """Group and classify primary vs cascading failures — deterministic."""
    if not records:
        return {"schema": "taksh.dlcae.failure_correlation/1",
                "groups": [], "primary": None, "cascades": [], "uncertain": True,
                "verdict": "NO_EVIDENCE"}
    groups: dict[str, list[FeedbackRecord]] = {}
    for record in records:
        key = f"{record.file or '<unknown>'}:{record.line or 0}"
        groups.setdefault(key, []).append(record)
    flattened = sorted(records, key=_rank)
    primary = flattened[0]
    primary_rank = _rank(primary)
    cascades: list[dict[str, Any]] = []
    for record in flattened[1:]:
        ranked_up = _rank(record) >= primary_rank
        same_scope = record.file == primary.file
        downstream = record.line is not None and primary.line is not None \
            and record.line >= primary.line
        if ranked_up and same_scope and downstream and record is not primary:
            record.cascade_of = [primary.feedback_id]
            record.primary = False
            cascades.append({"cascading": record.feedback_id,
                             "cascade_of": primary.feedback_id,
                             "reason": "downstream of the earliest high-rank failure "
                                       "in the same file"})
    primary.primary = True
    for record in records:
        if record.primary is None:
            record.primary = False
    earliest_chain = [r.feedback_id for r in flattened]
    primary_rank = _rank(primary)
    peers = [r for r in records if _rank(r) == primary_rank]
    uncertain = len(peers) > 1
    return {
        "schema": "taksh.dlcae.failure_correlation/1",
        "source_sha256": records[0].source_sha256,
        "groups": [{"key": key, "count": len(group),
                    "categories": sorted({r.category for r in group})}
                   for key, group in sorted(groups.items())],
        "primary": primary.feedback_id,
        "primary_category": primary.category,
        "primary_line": primary.line,
        "cascades": cascades,
        "earliest_known_failure": earliest_chain[0],
        "ordered_chain": earliest_chain,
        "uncertain": uncertain,
        "verdict": ("ROOT_CAUSE_IDENTIFIED" if not uncertain else "ROOT_CAUSE_UNCERTAIN"),
        "note": ("correlation is a heuristic over recorded diagnostics only; "
                 "an unresolved correlation preserves uncertainty — never a guess"),
    }


# --------------------------------------------------------------------------- #
# Repair candidates — suggestions, never applied, never verified
# --------------------------------------------------------------------------- #
_REPAIR_TEMPLATES = {
    "syntax_error": "inspect and rebalance the reported bracket/quote/indentation at the recorded location",
    "unresolved_symbol": "declare the symbol, fix the spelling, or import its defining module",
    "missing_dependency": "add the dependency to the local environment declarations or remove the import",
    "type_error": "reconcile the declared annotation with the inferred use site; change the annotation or the use",
    "compiler_error": "address the compiler diagnostic at the reported location before interpreting downstream failures",
    "runtime_exception": "route the exception to its handler; add the missing guard or fix the precondition",
    "test_failure": "treat the failing assertion as a behavioral requirement; reconcile implementation with the test",
    "timeout": "investigate the unbounded loop or external dependency before widening the limit",
    "resource_limit": "reduce fixture size or allocation; limits are not to be raised silently",
    "blocked": "unblock the toolchain/license/authorization barrier before attempting repair",
}


def repair_candidates(diagnostics: list[dict[str, Any]], *,
                      source_sha256: str) -> dict[str, Any]:
    """Structured repair *candidates*. Never modifies source; never verified."""
    candidates = []
    for i, diagnostic in enumerate(diagnostics):
        category = str(diagnostic.get("category", "compiler_error"))
        suggestion = _REPAIR_TEMPLATES.get(category,
                                           "inspect the diagnostic context; no automatic repair is authorized")
        candidates.append({
            "suggestion_id": fingerprint({"src": source_sha256, "cat": category,
                                          "i": i, "msg": diagnostic.get("message")})[:16],
            "for_diagnostic": {"category": category,
                               "line": diagnostic.get("line"),
                               "column": diagnostic.get("column"),
                               "symbol": diagnostic.get("symbol")},
            "suggestion": suggestion,
            "origin": "suggestion-not-verified-fix",
            "verified": False,
            "source_mutation": False,
            "confidence": 0.2,
        })
    return {
        "schema": "taksh.dlcae.repair_candidates/1",
        "source_sha256": source_sha256,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "note": ("candidates are suggestions only; no source is modified; "
                 "verified fixes require revalidation through the Validation Engine "
                 "after explicit caller request"),
    }
