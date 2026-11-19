"""SemanticAnalysisEngine — source → comprehension → symbols → types → deps →
validation → knowledge → feedback, with stage status, partial analysis,
interruption and resume, per-stage evidence and dependency-aware invalidation.

No source file is ever modified. Repair candidates are suggestions only.
The training handoff stays disabled; knowledge promotion still requires a
Validation Engine PASS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.comprehension import IncrementalComprehensionEngine
from taksh.dlcae.knowledge import KnowledgeFormationEngine, provenance_from_source
from taksh.dlcae.persistence import PersistenceStore
from taksh.dlcae.semantic.contract import (
    SemanticStatus, build_result, merge_status, source_sha256, validate_result,
)
from taksh.dlcae.semantic.dependencies import analyze_python_dependencies, impact_analysis
from taksh.dlcae.semantic.evidence import SourceEvidenceLedger
from taksh.dlcae.semantic.feedback import (
    FeedbackRecord, correlate, from_diagnostic, repair_candidates,
)
from taksh.dlcae.semantic.symbols import SymbolAnalysisEngine
from taksh.dlcae.semantic.tools import ToolchainRegistry
from taksh.dlcae.semantic.typesys import TypeAnalysisEngine
from taksh.dlcae.types import VerificationState
from taksh.dlcae.validation import ValidationEngine

STAGE_ORDER = ("comprehension", "symbols", "types", "dependencies", "validation", "knowledge", "feedback")


class SemanticEngineError(Exception):
    pass


@dataclass
class StageOutcome:
    status: str
    artifact: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class SemanticAnalysisEngine:
    def __init__(self, *,
                 knowledge: KnowledgeFormationEngine | None = None,
                 validation: ValidationEngine | None = None,
                 toolchains: ToolchainRegistry | None = None,
                 state_store: PersistenceStore | None = None) -> None:
        self.knowledge = knowledge or KnowledgeFormationEngine()
        self.validation = validation or ValidationEngine()
        self.toolchains = toolchains or ToolchainRegistry()
        self.state_store = state_store
        self.comprehension = IncrementalComprehensionEngine()
        self.symbols = SymbolAnalysisEngine()
        self.types = TypeAnalysisEngine()
        self.evidence = SourceEvidenceLedger(self.knowledge)
        self.feedback_records: list[FeedbackRecord] = []
        self._interrupted: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def analyze(self, document_id: str, language_id: str, text: str, *,
                file: str | None = None, module: str | None = None,
                mode: str = "model_only", stop_after: str | None = None,
                resume_token: str | None = None,
                extra_sources: dict[str, str] | None = None,
                changed_modules: list[str] | None = None) -> dict[str, Any]:
        if mode not in ("model_only", "tool_assisted"):
            raise SemanticEngineError(f"unknown mode: {mode!r}")
        if stop_after is not None and stop_after not in STAGE_ORDER:
            raise SemanticEngineError(f"stop_after must be one of {STAGE_ORDER}")
        if resume_token and stop_after:
            raise SemanticEngineError("use either resume_token or stop_after, not both")

        file = file or f"{document_id}.{self._extension_for(language_id)}"
        module = module or _module_of(file)
        digest = source_sha256(text)
        resume_state = self._interrupted.pop(resume_token, None) if resume_token else None
        if resume_token and resume_state is None and self.state_store is not None:
            # crashed-process resume: the interrupted record was persisted before return
            try:
                loaded = self.state_store.load_json(f"semantic/interrupted/{resume_token}.json")
            except Exception:  # noqa: BLE001
                loaded = None
            if isinstance(loaded, dict):
                resume_state = loaded
        if resume_token and resume_state is None:
            raise SemanticEngineError(f"unknown resume token: {resume_token!r} (never fabricated)")
        if resume_state is not None and resume_state["source_sha256"] != digest:
            raise SemanticEngineError(
                "resume token was issued for a different source hash; refusing to mix evidence")

        result = build_result(language_id=language_id, source_sha256=digest,
                              file=file, module=module)
        result["mode"] = mode
        stage_status: dict[str, str] = {}
        completed = list(resume_state["completed"]) if resume_state else []

        def run_stage(name: str, fn):
            if name in completed:
                stage_status[name] = resume_state["stage_status"][name]
                result[name] = resume_state[name]
                return
            if stop_after and STAGE_ORDER.index(name) > STAGE_ORDER.index(stop_after):
                result["status"] = "INCOMPLETE"
                result["resume_token"] = self._issue_resume_token(result, stage_status, completed)
                return
            t0 = time.perf_counter()
            outcome = fn()
            stage_status[name] = outcome.status
            result[name] = outcome.artifact
            completed.append(name)
            stage_status[f"{name}_duration_ms"] = str(round((time.perf_counter() - t0) * 1000, 3))

        # -- comprehension
        def _comprehension():
            bundle = self.comprehension.comprehend(document_id, text, language_id)
            bundle_dict = bundle.to_dict()
            failure = any(o.kind == "parse_failure" for o in bundle.observations())
            status = SemanticStatus.FAIL if failure else SemanticStatus.PASS
            return StageOutcome(status, {"levels": bundle_dict["levels"],
                                          "level_status": bundle_dict["level_status"]})
        run_stage("comprehension", _comprehension)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- symbols
        def _symbols():
            analysis = self.symbols.analyze(text, language_id=language_id, file=file, module=module)
            artifact = analysis.to_dict()
            artifact["fingerprint"] = analysis.adapter_version
            self._emit_diagnostics(artifact.get("diagnostics", []), digest,
                                   language_id, analysis.adapter, analysis.adapter_version,
                                   route_tool=analysis.adapter)
            return StageOutcome(artifact["status"], artifact)
        run_stage("symbols", _symbols)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- types
        def _types():
            if stage_status.get("symbols") == SemanticStatus.FAIL:
                return StageOutcome(SemanticStatus.BLOCKED,
                                    {"relations": [],
                                     "reason": "syntax failure upstream — type analysis cannot proceed"})
            analysis = self.types.analyze(text, language_id=language_id, file=file, module=module)
            artifact = analysis.to_dict()
            return StageOutcome(artifact["status"], artifact)
        run_stage("types", _types)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- dependencies
        def _dependencies():
            sources = dict(extra_sources) if extra_sources else {module: text}
            if language_id.lower() != "python":
                return StageOutcome(SemanticStatus.NOT_APPLICABLE,
                                    {"edges": [],
                                     "reason": f"no dependency adapter qualified for {language_id!r}"})
            graph = analyze_python_dependencies(sources)
            artifact = graph.to_dict()
            for diagnostic in graph.diagnostics:
                from taksh.dlcae.semantic.contract import Diagnostic

                mapped = Diagnostic(
                    category=str(diagnostic.get("category", "compiler_error")),
                    message=str(diagnostic.get("message", "dependency diagnostic")),
                    severity="ERROR", line=diagnostic.get("line"), file=file,
                    source="dependency-graph", tool_version="1.0.0",
                    evidence_classification="TOOL_VERIFIED")
                self.feedback_records.append(from_diagnostic(
                    mapped.to_dict(), source_sha256=digest, language_id=language_id,
                    adapter="dependency-graph", adapter_version="1.0.0",
                    toolchain_name="dependency-graph", toolchain_version="1.0.0",
                    file=file))
            if changed_modules:
                artifact["impact"] = impact_analysis(graph, changed_modules)
            return StageOutcome(SemanticStatus.PASS if not graph.missing else SemanticStatus.PARTIAL,
                                artifact)
        run_stage("dependencies", _dependencies)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- validation (toolchain or model-only, honest BLOCKED)
        def _validation():
            artifact: dict[str, Any] = {"mode": mode, "checks": []}
            toolchain = self.toolchains.for_language(language_id)
            if mode == "tool_assisted":
                if toolchain is None or not toolchain.available():
                    name = language_id.lower()
                    detail = ("TOOLCHAIN_UNAVAILABLE" if toolchain is None
                              else "toolchain missing on this host")
                    return StageOutcome(SemanticStatus.BLOCKED,
                                        {"mode": mode, "checks": [{"tool": name, "available": False,
                                                                   "reason": detail}]})
                checks: list[dict[str, Any]] = []
                syntax = toolchain.syntax_check(text, filename=file, timeout_s=30.0)
                checks.append(syntax.to_dict())
                from taksh.dlcae.semantic.contract import Diagnostic

                if syntax.ok:
                    checks.append({"tool": syntax.tool, "verdict": "PASS",
                                   "note": "syntax accepted by the local toolchain"})
                else:
                    mapped = Diagnostic(category="syntax_error",
                                        message=(syntax.stderr or syntax.stdout or "syntax check failed").strip()[:400],
                                        severity="ERROR", file=file,
                                        source=syntax.tool, tool_version=syntax.tool_version,
                                        evidence_classification="TOOL_VERIFIED")
                    checks.append(mapped.to_dict())
                    self.feedback_records.append(from_diagnostic(
                        mapped.to_dict(), source_sha256=digest, language_id=language_id,
                        adapter=syntax.tool, adapter_version=syntax.tool_version,
                        toolchain_name=syntax.tool, toolchain_version=syntax.tool_version,
                        file=file))
                semantic_result = toolchain.semantic_check(text, filename=file, timeout_s=60.0)
                checks.append(semantic_result.to_dict())
                verdict = SemanticStatus.PASS
                if not syntax.ok:
                    verdict = SemanticStatus.FAIL
                elif not semantic_result.ok:
                    if semantic_result.stderr.startswith("TOOLCHAIN_UNAVAILABLE"):
                        verdict = SemanticStatus.PARTIAL     # syntax ok, types unverified
                    else:
                        verdict = SemanticStatus.PARTIAL
                artifact["checks"] = checks
                artifact["toolchain"] = {"id": toolchain.name,
                                         "name": toolchain.name,
                                         "version": toolchain.tool_version(),
                                         "available": True,
                                         "execution_status": "EXECUTED",
                                         "env_fingerprint": toolchain.env_fingerprint()}
                return StageOutcome(verdict, artifact)
            # model_only
            verdict = self.validation.validate("structural", {"source": text},
                                               language_id=language_id, mode="model_only")
            artifact["checks"] = [verdict]
            artifact["toolchain"] = {"id": "model-only", "name": "model-only",
                                     "version": "heuristic", "available": False,
                                     "execution_status": "NOT_EXECUTED",
                                     "env_fingerprint": ""}
            return StageOutcome(
                SemanticStatus.PASS if verdict["verdict"] == "PASS" else SemanticStatus.UNVERIFIED,
                artifact)
        run_stage("validation", _validation)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- knowledge formation (promotion requires Validation PASS)
        def _knowledge():
            created = []
            # Promotion is gated on the validation stage outcome — never on
            # model confidence, never on syntax alone for tool_assisted mode.
            tool_pass = stage_status.get("validation") == SemanticStatus.PASS
            provenance = provenance_from_source(
                f"document:{document_id}", license_spdx="owner-internal-research",
                record_id=f"doc-{digest[:16]}", source_kind="session")
            symbols = result.get("symbols", {})
            for symbol in symbols.get("symbols", [])[:64]:   # bounded: fixture-scale
                record = self.knowledge.add(
                    language_id, kind="language_rule",
                    name=f"symbol:{symbol['qualified_name']}",
                    content={"symbol": symbol, "source_sha256": digest},
                    provenance=provenance,
                    state=VerificationState.OBSERVED,
                    confidence=0.6 if tool_pass else 0.5)
                if tool_pass:
                    record = self.knowledge.verify(
                        record.record_id, validation_verdict="PASS",
                        evidence={"stage": "validation",
                                  "mode": mode,
                                  "source_sha256": digest,
                                  "note": "bound to the validation run, not to the absence of diagnostics alone"})
                    self.evidence.bind(digest, record.record_id, "PASS")
                created.append(record.record_id)
            return StageOutcome(SemanticStatus.PASS if created else SemanticStatus.UNVERIFIED,
                                {"records_created": created,
                                 "promotion": ("VERIFIED-bound-to-source" if tool_pass
                                               else "OBSERVED-only (no Validation PASS)")})
        run_stage("knowledge", _knowledge)
        if result.get("status") == "INCOMPLETE":
            return self._finalize(result, stage_status, digest)

        # -- feedback correlation + repair candidates (suggestions only)
        def _feedback():
            source_blocked = language_id.lower() == "karkain"
            diags: list[dict[str, Any]] = []
            for check in result.get("validation", {}).get("checks", []):
                if isinstance(check, dict) and "category" in check:
                    diags.append(check)
            correlated = correlate(self.feedback_records) if self.feedback_records else {
                "verdict": "NO_EVIDENCE", "cascades": [], "uncertain": True}
            candidates = repair_candidates(diags, source_sha256=digest)
            return StageOutcome(SemanticStatus.PASS if not source_blocked else SemanticStatus.BLOCKED,
                                {"correlation": correlated,
                                 "repair_candidates": candidates,
                                 "feedback_records": len(self.feedback_records),
                                 "training_handoff": "DISABLED (A1.1-C / A2-E)"})
        run_stage("feedback", _feedback)

        return self._finalize(result, stage_status, digest)

    def verify_source(self, source_sha256: str, *,
                      reason: str = "source identity changed") -> dict[str, Any]:
        """Public invalidation path for the runtime integration (ENG2)."""
        report = self.evidence.verify_source(source_sha256)
        return {**report, "reason": reason}

    # ------------------------------------------------------------------ #
    def _issue_resume_token(self, result, stage_status, completed) -> str:
        token = f"resume-{self._interrupt_digest(result)}"
        self._interrupted[token] = {
            "source_sha256": result["source_sha256"],
            "completed": list(completed),
            "stage_status": {k: v for k, v in stage_status.items()
                             if not k.endswith("_duration_ms")},
        }
        for stage in completed:
            self._interrupted[token][stage] = result.get(stage)
        if self.state_store is not None:
            try:
                self.state_store.save_json(f"semantic/interrupted/{token}.json",
                                           self._interrupted[token])
            except Exception:  # noqa: BLE001
                result.setdefault("warnings", []).append(
                    "interrupted state could not be persisted; in-memory token only")
        return token

    @staticmethod
    def _interrupt_digest(result) -> str:
        import hashlib

        return hashlib.sha256(repr(sorted(result.items())).encode()).hexdigest()[:12]

    def _emit_diagnostics(self, diagnostics, digest, language_id, adapter, adapter_version,
                          *, route_tool):
        for diagnostic in diagnostics:
            self.feedback_records.append(from_diagnostic(
                diagnostic, source_sha256=digest, language_id=language_id,
                adapter=adapter, adapter_version=adapter_version,
                toolchain_name=route_tool, toolchain_version=adapter_version,
                file=None))

    @staticmethod
    def _extension_for(language_id: str) -> str:
        return {"python": "py", "javascript": "js", "bash": "sh", "go": "go",
                "java": "java", "csharp": "cs", "karkain": "kark"}.get(language_id.lower(), "txt")

    def _finalize(self, result, stage_status, digest) -> dict[str, Any]:
        if result.get("status") == "INCOMPLETE":
            problems = validate_result({**result, "status": SemanticStatus.UNVERIFIED})
            result["status"] = "INCOMPLETE"
            result["stage_status"] = stage_status
            result["contract_problems"] = problems
            return result
        result["source_sha256"] = digest
        result["stage_status"] = stage_status
        result["status"] = merge_status(
            {k: v for k, v in ((k, v) for k, v in stage_status.items()
                               if k in STAGE_ORDER)})
        result["diagnostics"] = list(result.get("symbols", {}).get("diagnostics", []))
        result["provenance"] = {**result["provenance"], "source_sha256": digest,
                                "run_id": result["result_id"]}
        result["evidence"] = self.evidence.snapshot()
        problems = validate_result(result)
        result["contract_problems"] = problems
        if self.state_store is not None:
            try:
                self.state_store.save_json_idempotent(
                    f"semantic/{result['result_id']}.json", result)
            except Exception:  # noqa: BLE001
                result.setdefault("warnings", []).append("state store write failed; result in memory only")
        return result


def _module_of(file: str) -> str:
    base = file.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or "module"
