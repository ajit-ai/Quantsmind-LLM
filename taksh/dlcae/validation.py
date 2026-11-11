"""Validation Engine — fail-closed validation contracts per comprehension stage.

Two strictly separated modes:

- ``model_only``    model output scored without any tool assistance;
- ``tool_assisted`` output scored by an available toolchain oracle.

Rules:

- a toolchain that is not available yields ``BLOCKED``/``NOT_APPLICABLE``,
  never ``PASS``;
- model-only heuristics may return model-only ``PASS``/``UNVERIFIED`` but are
  never treated as verified capability;
- Python has a real in-process oracle on this host: ``compile()``/``ast``
  (stdlib, offline). Everything else is explicit and limited.
- Karkain: KCC is unavailable on this host (0xC0000135); every tool-assisted
  Karkain verdict is BLOCKED — bypassing the compiler is not possible here.
"""

from __future__ import annotations

import ast
from typing import Any

from taksh.dlcae.types import ValidationVerdict

VALIDATION_STAGES = (
    "lexical", "structural", "parser", "semantic", "execution", "tests", "regression",
)


def _python_compiles(source: str) -> dict[str, Any]:
    try:
        compile(source, "<dlcae-validation>", "exec")
    except SyntaxError as exc:
        return {"ok": False, "error": f"{exc.msg} (line {exc.lineno})"}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _python_ast(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _balanced(source: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairs.items()}
    stack: list[str] = []
    in_str: str | None = None
    for ch in source:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch in pairs:
            stack.append(ch)
        elif ch in closing and (not stack or stack.pop() != closing[ch]):
            return False
    return not stack


class ValidationEngine:
    """Stage-gated validation with recorded, immutable verdicts."""

    def __init__(self, *, host_capabilities: dict[str, Any] | None = None) -> None:
        self.ledger: list[dict[str, Any]] = []
        self.host_capabilities = host_capabilities or {
            "python": {"compile": True, "ast": True, "type_checker": False,
                       "test_runner": True},
            "karkain": {"kcc": False,
                        "reason": "kcc.exe fails with 0xC0000135 (missing dependency)"},
        }

    # ------------------------------------------------------------------ #
    def validate(self, stage: str, payload: dict[str, Any], *,
                 language_id: str, mode: str = "model_only") -> dict[str, Any]:
        if stage not in VALIDATION_STAGES:
            verdict = {"verdict": ValidationVerdict.NOT_APPLICABLE.value,
                       "reason": f"unknown validation stage: {stage!r}"}
            return self._record(stage, language_id, mode, payload, verdict)
        if mode not in ("model_only", "tool_assisted"):
            return self._record(stage, language_id, mode, payload,
                                {"verdict": ValidationVerdict.NOT_APPLICABLE.value,
                                 "reason": f"unknown mode: {mode!r}"})
        source = str(payload.get("source", ""))
        if mode == "model_only":
            verdict = self._model_only(stage, source, language_id)
        else:
            verdict = self._tool_assisted(stage, source, language_id)
        return self._record(stage, language_id, mode, payload, verdict)

    # ------------------------------------------------------------------ #
    def _model_only(self, stage: str, source: str, language_id: str) -> dict[str, Any]:
        if not source.strip():
            return {"verdict": ValidationVerdict.FAIL.value,
                    "reason": "empty payload",
                    "mode_note": "model_only heuristics; never capability evidence"}
        if stage in ("lexical", "structural"):
            ok = _balanced(source)
            return {"verdict": ValidationVerdict.PASS.value if ok else ValidationVerdict.FAIL.value,
                    "reason": "bracket-balance heuristic (model_only)"
                    if ok else "unbalanced brackets (model_only)",
                    "mode_note": "model_only heuristics; never capability evidence"}
        return {"verdict": ValidationVerdict.UNVERIFIED.value,
                "reason": f"no model_only oracle for stage {stage!r}; "
                          f"UNVERIFIED — never coerced to PASS",
                "mode_note": "model_only heuristics; never capability evidence"}

    # ------------------------------------------------------------------ #
    def _tool_assisted(self, stage: str, source: str, language_id: str) -> dict[str, Any]:
        lane = self.host_capabilities.get(language_id.lower(), {})
        if language_id.lower() == "python":
            if stage in ("lexical", "structural", "parser"):
                result = _python_compiles(source)
                return ({"verdict": ValidationVerdict.PASS.value,
                         "reason": "python compile() accepted the source",
                         "oracle": "python compile() (stdlib, offline)"}
                        if result["ok"] else
                        {"verdict": ValidationVerdict.FAIL.value, "reason": result["error"],
                         "oracle": "python compile() (stdlib, offline)"})
            if stage == "semantic":
                if lane.get("type_checker"):
                    return {"verdict": ValidationVerdict.UNVERIFIED.value,
                            "reason": "type checker declared available but not wired"}
                return {"verdict": ValidationVerdict.BLOCKED.value,
                        "reason": "no type checker qualified on this host; semantic not verified"}
            if stage in ("execution", "tests"):
                return {"verdict": ValidationVerdict.BLOCKED.value,
                        "reason": "execution/tests require an explicit sandboxed harness; "
                                  "not engaged in this engineering milestone"}
            if stage == "regression":
                return {"verdict": ValidationVerdict.NOT_APPLICABLE.value,
                        "reason": "no regression baseline supplied; refusing to invent one"}
        if language_id.lower() == "karkain":
            return {"verdict": ValidationVerdict.BLOCKED.value,
                    "reason": ("KCC unavailable on this host "
                               f"({lane.get('reason', 'kcc.exe 0xC0000135')}); "
                               "every Karkain tool-assisted verdict is BLOCKED"),
                    "oracle": "KCC (unavailable)"}
        return {"verdict": ValidationVerdict.BLOCKED.value,
                "reason": f"no toolchain qualified for {language_id!r} on this host"}

    # ------------------------------------------------------------------ #
    def _record(self, stage: str, language_id: str, mode: str,
                payload: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        record = {
            "schema": "taksh.dlcae.validation/1",
            "stage": stage,
            "language_id": language_id,
            "mode": mode,
            **verdict,
        }
        self.ledger.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.ledger:
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
        return {"ledger_entries": len(self.ledger),
                "verdict_counts": counts,
                "model_only_entries": sum(1 for r in self.ledger if r["mode"] == "model_only"),
                "tool_assisted_entries": sum(1 for r in self.ledger if r["mode"] == "tool_assisted"),
                "note": "verdicts are evidence about a stage, never language qualification"}
