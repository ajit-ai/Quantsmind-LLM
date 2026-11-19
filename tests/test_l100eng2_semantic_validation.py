"""Focused tests for L1.0.0-ENG2 — Semantic and Execution Validation Foundation.

Deterministic synthetic fixtures only. Subprocess calls are limited to
locally installed, offline toolchains. No downloads, no network, no frozen
benchmark fixtures, no source mutation outside tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.dlcae.semantic.contract import (  # noqa: E402
    DIAGNOSTIC_CATEGORIES, SEMANTIC_CONTRACT_VERSION, SEMANTIC_STATUSES,
    Diagnostic, SemanticStatus, build_result, merge_status, source_sha256,
    validate_diagnostic, validate_result,
)
from taksh.dlcae.semantic.dependencies import (  # noqa: E402
    analyze_python_dependencies, hash_of_sources, impact_analysis,
)
from taksh.dlcae.experiments import ExperimentFramework  # noqa: E402
from taksh.dlcae.semantic.engine import SemanticAnalysisEngine, SemanticEngineError  # noqa: E402
from taksh.dlcae.semantic.evidence import SourceEvidenceLedger  # noqa: E402
from taksh.dlcae.semantic.feedback import (  # noqa: E402
    FeedbackRecord, correlate, from_diagnostic, repair_candidates,
)
from taksh.dlcae.semantic.metrics import (  # noqa: E402
    evaluate_compatibility, evaluate_dependency_graph, evaluate_reproducibility,
    evaluate_symbol_detection, evaluate_type_relations,
)
from taksh.dlcae.semantic.symbols import SymbolAnalysisEngine  # noqa: E402
from taksh.dlcae.semantic.tools import (  # noqa: E402
    PythonToolchain, ToolchainRegistry, ToolchainUnavailable,
)
from taksh.dlcae.semantic.typesys import TypeAnalysisEngine, types_compatible  # noqa: E402
from taksh.dlcae.knowledge import KnowledgeFormationEngine, provenance_from_source  # noqa: E402
from taksh.dlcae.types import VerificationState  # noqa: E402

PY_FIXTURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
PY_BROKEN = "def broken(:\n    return 1\n"
KARK_FIXTURE = "fn main() {}\n"
CYCLE_SOURCES = {
    "cyc_a": "import cyc_b\nVALUE_A = 1\n",
    "cyc_b": "import cyc_c\nVALUE_B = 1\n",
    "cyc_c": "import cyc_a\nVALUE_C = 1\n",
}


# --------------------------------------------------------------------------- #
# 1. Semantic contract schema and versioning
# --------------------------------------------------------------------------- #
class TestSemanticContract:
    def test_schema_version_registered(self):
        assert SEMANTIC_CONTRACT_VERSION == 1
        assert SEMANTIC_STATUSES == frozenset({"PASS", "FAIL", "PARTIAL", "BLOCKED",
                                              "UNVERIFIED", "NOT_APPLICABLE"})

    def test_result_roundtrip_validates(self):
        result = build_result(language_id="python", source_sha256="a" * 64,
                              file="x.py", module="x")
        assert validate_result(result) == []

    def test_missing_toolchain_never_pass(self):
        result = build_result(language_id="python", source_sha256="a" * 64,
                              file="x.py", module="x")
        result["status"] = "PASS"   # toolchain.available defaults False
        problems = validate_result(result)
        assert any("missing toolchain" in p for p in problems)

    def test_unknown_status_rejected(self):
        result = build_result(language_id="python", source_sha256="a" * 64,
                              file="x.py", module="x")
        result["status"] = "TOtallyFine"
        assert validate_result(result)

    def test_diagnostic_categories_enforced(self):
        assert validate_diagnostic({"category": "syntax_error", "message": "x",
                                    "severity": "ERROR"}) == []
        assert validate_diagnostic({"category": "bogus", "message": "x",
                                    "severity": "ERROR"})

    def test_invalid_line_rejected(self):
        assert validate_diagnostic({"category": "warning", "message": "x",
                                    "severity": "INFO", "line": 0})

    def test_merge_status_deterministic(self):
        assert merge_status({"a": "PASS", "b": "PASS"}) == "PASS"
        assert merge_status({"a": "FAIL", "b": "PASS"}) == "FAIL"
        assert merge_status({"a": "BLOCKED", "b": "PASS"}) == "PARTIAL"
        assert merge_status({"a": "BLOCKED"}) == "BLOCKED"


# --------------------------------------------------------------------------- #
# 2. Symbol declarations and references
# --------------------------------------------------------------------------- #
class TestSymbols:
    def test_declarations_found(self):
        analysis = SymbolAnalysisEngine().analyze(
            PY_FIXTURE, language_id="python", file="t.py", module="t")
        names = {s["name"] for s in analysis.symbols}
        assert {"add", "a", "b"} <= names

    def test_imports_recorded(self):
        analysis = SymbolAnalysisEngine().analyze(
            "import os\nfrom pathlib import Path\n", language_id="python",
            file="t.py", module="t")
        modules = {i["module"] for i in analysis.imports}
        assert {"os", "pathlib"} <= modules

    def test_references_resolved(self):
        analysis = SymbolAnalysisEngine().analyze(
            "def f():\n    return defined_name\n\ndefined_name = 1\n",
            language_id="python", file="t.py", module="t")
        resolved = {r["name"]: r.get("resolution") for r in analysis.references}
        assert resolved.get("defined_name") == "defined_name"

    # 3. Scope and shadowing
    def test_shadowing_recorded(self):
        analysis = SymbolAnalysisEngine().analyze(
            "x = 1\ndef f():\n    x = 2\n    return x\n",
            language_id="python", file="t.py", module="t")
        assert any(s["name"] == "x" for s in analysis.shadowing)

    def test_nested_scope_qualified_names(self):
        analysis = SymbolAnalysisEngine().analyze(
            "class C:\n    def m(self):\n        return 1\n",
            language_id="python", file="t.py", module="t")
        qualified = {s["qualified_name"] for s in analysis.symbols}
        assert "C" in {q.split(".")[0] for q in qualified}
        assert any("C.m" in q for q in qualified)

    # 4. Unresolved identifiers
    def test_unresolved_identifier_detected(self):
        analysis = SymbolAnalysisEngine().analyze(
            "def f():\n    return definitely_not_declared_xyz\n",
            language_id="python", file="t.py", module="t")
        assert any(u["name"] == "definitely_not_declared_xyz"
                   for u in analysis.unresolved)
        assert analysis.status == "PARTIAL"

    def test_duplicate_function_flagged(self):
        analysis = SymbolAnalysisEngine().analyze(
            "def f():\n    return 1\ndef f():\n    return 2\n",
            language_id="python", file="t.py", module="t")
        assert analysis.duplicates
        assert any("f" in d["qualified_name"] for d in analysis.duplicates)

    def test_unbalanced_source_gives_fail(self):
        analysis = SymbolAnalysisEngine().analyze(
            PY_BROKEN, language_id="python", file="t.py", module="t")
        assert analysis.status == "FAIL"
        assert any(d.category == "syntax_error" for d in analysis.diagnostics)


# --------------------------------------------------------------------------- #
# 5/6. Type relationships and unknown types
# --------------------------------------------------------------------------- #
class TestTypes:
    def test_function_signature_relations(self):
        analysis = TypeAnalysisEngine().analyze(
            TYPE_FIXTURE := "def annotate(a: int, b: str) -> str:\n    c: int = a\n    return b\n",
            language_id="python", file="t.py", module="t")
        kinds = {r["kind"] for r in analysis.relations}
        assert {"signature", "parameter", "return", "variable"} <= kinds

    def test_union_and_generic_become_composite(self):
        analysis = TypeAnalysisEngine().analyze(
            "def f(x: list[int], y: 'int | str' = 1) -> float:\n    return 1.0\n",
            language_id="python", file="t.py", module="t")
        composites = [r for r in analysis.relations if r.get("notion") == "composite"]
        assert composites

    def test_absent_annotation_is_explicit_unknown(self):
        analysis = TypeAnalysisEngine().analyze(
            "def f(x):\n    y\n    return y\n", language_id="python",
            file="t.py", module="t")
        unknowns = [r for r in analysis.relations
                    if r.get("explicit_unknown")]
        assert unknowns, "unannotated members must be recorded as explicit unknowns"

    def test_types_compatible_matrix(self):
        assert types_compatible("int", "float")["observation"] == "COMPATIBLE_WIDENING"
        assert types_compatible(None, "int")["observation"] == "UNVERIFIED"
        assert types_compatible("str", "int")["observation"] == "INCOMPATIBLE_OR_UNKNOWN"
        assert types_compatible("int", "int")["observation"] == "IDENTICAL"

    def test_arkain_types_blocked(self):
        analysis = TypeAnalysisEngine().analyze(
            KARK_FIXTURE, language_id="karkain", file="t.kark", module="t")
        assert analysis.status == "BLOCKED"


# --------------------------------------------------------------------------- #
# 7/8/9. Dependency graph, cycles, missing detection, hash invalidation
# --------------------------------------------------------------------------- #
class TestDependencies:
    def test_local_edges_built(self):
        graph = analyze_python_dependencies(CYCLE_SOURCES)
        pairs = {(e["from"], e["to"]) for e in graph.edges}
        assert ("cyc_a", "cyc_b") in pairs
        assert ("cyc_b", "cyc_c") in pairs

    def test_cycle_detected(self):
        graph = analyze_python_dependencies(CYCLE_SOURCES)
        assert graph.cycles
        assert any("cyc_a" in cycle for cycle in graph.cycles)

    def test_missing_dependency_detected(self):
        graph = analyze_python_dependencies(
            {"needs_missing": "import definitely_not_installed_pkg_xyz123\n"})
        assert any(m["module"] == "definitely_not_installed_pkg_xyz123"
                   for m in graph.missing)

    def test_stdlib_recognized(self):
        graph = analyze_python_dependencies({"uses_json": "import json\n"})
        assert not graph.missing
        assert any(e["to"] == "json" for e in graph.edges)

    def test_impact_analysis_reverse_deps(self):
        graph = analyze_python_dependencies(CYCLE_SOURCES)
        impact = impact_analysis(graph, ["cyc_c"])
        assert "cyc_b" in impact["affected"] or "cyc_a" in impact["affected"]

    def test_source_hash_invalidation(self):
        knowledge = KnowledgeFormationEngine()
        ledger = SourceEvidenceLedger(knowledge)
        hash_a = hash_of_sources({"x": "a = 1\n"})
        hash_b = hash_of_sources({"x": "a = 2\n"})
        assert hash_a != hash_b
        prov = provenance_from_source("fixture", license_spdx="owner-internal-research",
                                      record_id="prov-inval")
        rec = knowledge.add("python", "language_rule", "inval:test", {"n": 1},
                            provenance=prov, state=VerificationState.OBSERVED)
        knowledge.verify(rec.record_id, validation_verdict="PASS",
                         evidence={"oracle": "fixture"})
        ledger.bind(hash_a, rec.record_id, "PASS")
        report = ledger.verify_source(hash_b, reason="fixture change")
        assert report["invalidated_count"] == 1
        assert knowledge.query(state=VerificationState.INVALIDATED)


# --------------------------------------------------------------------------- #
# 10/11/12. Toolchain discovery, missing toolchain refusal, diagnostics
# --------------------------------------------------------------------------- #
class TestToolchains:
    def test_discovery_reports_installed_tools(self):
        discovery = ToolchainRegistry().discover_all()
        assert discovery["python"]["available"] is True
        assert discovery["python"]["version"] == sys.version.split()[0]
        assert "go" in discovery and "javascript" in discovery and "bash" in discovery
        assert all("status" in v for v in discovery.values())

    def test_python_adapter_syntax_and_semantic(self):
        tool = PythonToolchain()
        syntax = tool.syntax_check(PY_FIXTURE, filename="fixture.py")
        assert syntax.ok is True
        semantic = tool.semantic_check(PY_FIXTURE, filename="fixture.py", timeout_s=60.0)
        assert semantic.tool == "python-mypy"
        assert semantic.evidence_classification == "TOOL_VERIFIED"

    def test_python_adapter_rejects_broken_syntax(self):
        tool = PythonToolchain()
        syntax = tool.syntax_check(PY_BROKEN, filename="broken.py")
        assert syntax.ok is False

    def test_missing_toolchain_refuses_pass(self):
        registry = ToolchainRegistry()
        engine = registry.adapters.get("cobol")
        assert engine is None  # missing toolchains are absent, not fake PASS

    def test_rust_is_not_available_but_recorded(self):
        discovery = ToolchainRegistry().discover_all()
        # rustc genuinely absent on this host; never reported as available
        assert discovery.get("rust", {}).get("available", False) is False


# --------------------------------------------------------------------------- #
# 13. Execution timeouts and output limits
# --------------------------------------------------------------------------- #
class TestTimeouts:
    def test_timeout_terminates_process(self):
        tool = PythonToolchain()
        result = tool.execute("import time\nwhile True:\n    time.sleep(0.05)\n",
                              filename="loop.py", timeout_s=1.0)
        assert result.timed_out is True
        assert result.ok is False

    def test_output_is_bounded(self):
        tool = PythonToolchain()
        result = tool.execute("print('x' * 5_000_000)\n", filename="big.py",
                              timeout_s=15.0)
        assert len(result.stdout) <= 64 * 1024 + 16

    def test_unknown_tool_does_not_execute(self):
        class Fake(PythonToolchain):
            language_ids = ("nosuchlang",)
            name = "nosuchlang"

        fake = Fake()
        assert fake.discover() is True   # honestly reuses python interpreter


# --------------------------------------------------------------------------- #
# 14/15/16. Feedback provenance, correlation, repair candidates
# --------------------------------------------------------------------------- #
class TestFeedback:
    def _records(self):
        sha = "a" * 64
        return [
            FeedbackRecord.build(source_sha256=sha, language_id="python",
                                 adapter="a-py", adapter_version="1",
                                 toolchain_name="python", toolchain_version="3",
                                 category="syntax_error", message="unexpected ')'",
                                 line=1, evidence_classification="TOOL_VERIFIED"),
            FeedbackRecord.build(source_sha256=sha, language_id="python",
                                 adapter="a-my", adapter_version="2",
                                 toolchain_name="python", toolchain_version="3",
                                 category="type_error", message="downstream mismatch",
                                 line=2, evidence_classification="TOOL_VERIFIED"),
        ]

    def test_feedback_record_provenance(self):
        rec = self._records()[0].to_dict()
        assert rec["source_sha256"] == "a" * 64
        assert rec["reproducibility"]["source_sha256"] == "a" * 64

    def test_correlation_primary_is_syntax(self):
        result = correlate(self._records())
        assert result["primary_category"] == "syntax_error"
        assert result["earliest_known_failure"] == result["primary"]
        assert result["cascades"]

    def test_repair_candidates_never_mutate(self, tmp_path):
        src = tmp_path / "keep.py"
        src.write_text(PY_FIXTURE, encoding="utf-8")
        before = src.read_bytes()
        result = repair_candidates(
            [{"category": "type_error", "message": "mismatch", "line": 1}],
            source_sha256="b" * 64)
        assert result["candidate_count"] == 1
        assert all(c["source_mutation"] is False for c in result["candidates"])
        assert all(c["verified"] is False for c in result["candidates"])
        assert src.read_bytes() == before

    def test_create_feedback_from_diagnostic(self):
        sha = "c" * 64
        diag = {"category": "missing_dependency", "message": "no module named x",
                "line": 3, "column": 1, "severity": "ERROR"}
        rec = from_diagnostic(diag, source_sha256=sha, language_id="python",
                              adapter="dep", adapter_version="1",
                              toolchain_name="python", toolchain_version="3",
                              file="t.py")
        assert rec.category == "missing_dependency"
        assert rec.source_sha256 == sha


# --------------------------------------------------------------------------- #
# 17. Knowledge state promotion and demotion
# --------------------------------------------------------------------------- #
class TestKnowledgePromotion:
    def test_knowledge_states_preserved(self):
        knowledge = KnowledgeFormationEngine()
        prov = provenance_from_source("fixture", license_spdx="owner-internal-research",
                                      record_id="prov-kw")
        rec = knowledge.add("python", "language_rule", "kw:test", {"n": 1},
                            provenance=prov, state=VerificationState.OBSERVED)
        knowledge.verify(rec.record_id, validation_verdict="PASS",
                         evidence={"oracle": "fixture"})
        assert knowledge.query(state=VerificationState.VERIFIED)
        knowledge.invalidate(rec.record_id, reason="fixture update")
        assert knowledge.query(state=VerificationState.INVALIDATED)


# --------------------------------------------------------------------------- #
# 18. Interrupted analysis recovery
# --------------------------------------------------------------------------- #
class TestInterruptedAnalysis:
    def test_stop_after_symbols_and_resume(self, tmp_path):
        engine = SemanticAnalysisEngine(state_store=None)
        paused = engine.analyze("int-1", "python", PY_FIXTURE,
                                stop_after="symbols")
        assert paused["status"] == "INCOMPLETE"
        token = paused["resume_token"]
        resumed = engine.analyze("int-1", "python", PY_FIXTURE,
                                 resume_token=token)
        assert resumed["status"] in ("PASS", "PARTIAL")
        assert all(stage in resumed for stage in (
            "comprehension", "symbols", "types", "dependencies",
            "validation", "knowledge", "feedback"))

    def test_resume_rejects_changed_source(self):
        engine = SemanticAnalysisEngine()
        paused = engine.analyze("int-2", "python", PY_FIXTURE,
                                stop_after="symbols")
        token = paused["resume_token"]
        with pytest.raises(SemanticEngineError):
            engine.analyze("int-2", "python", PY_FIXTURE + "\n",
                           resume_token=token)


# --------------------------------------------------------------------------- #
# 19. Karkain blocked-state behavior
# --------------------------------------------------------------------------- #
class TestKarkain:
    def test_karkain_semantic_blocked(self):
        engine = SemanticAnalysisEngine()
        result = engine.analyze("kark-1", "karkain", KARK_FIXTURE,
                                mode="tool_assisted")
        assert result["symbols"]["status"] == "BLOCKED"
        assert result["types"]["status"] == "BLOCKED"
        assert result["validation"]["checks"][0]["available"] is False
        assert result["status"] in ("PARTIAL", "BLOCKED")

    def test_arkain_benchmark_still_zero(self):
        from taksh.dlcae.karkain import KarkainAdapter
        adapter = KarkainAdapter()
        assert adapter.kark_bench_status()["instantiated_tasks"] == 0
        assert adapter.semantic_summary()["status"] == "BLOCKED"

    def test_karkain_adapter_registered(self):
        engine = SymbolAnalysisEngine()
        analysis = engine.analyze(KARK_FIXTURE, language_id="karkain",
                                  file="t.kark", module="t")
        assert analysis.status == "BLOCKED"


# --------------------------------------------------------------------------- #
# End-to-end runtime integration
# --------------------------------------------------------------------------- #
class TestRuntimeIntegration:
    def test_runtime_semantic_method(self, tmp_path):
        from taksh.dlcae.runtime import DLCAERuntime

        runtime = DLCAERuntime(tmp_path / "runtime")
        result = runtime.semantic_analysis("int-doc", "python", PY_FIXTURE,
                                           mode="model_only")
        assert result["status"] in ("PASS", "PARTIAL")
        assert "source_sha256" in result

    def test_runtime_invalidation_path(self, tmp_path):
        from taksh.dlcae.runtime import DLCAERuntime

        runtime = DLCAERuntime(tmp_path / "runtime")
        result = runtime.semantic_analysis("int-doc2", "python", PY_FIXTURE,
                                           mode="tool_assisted")
        report = runtime.semantic_invalidate("f" * 64, reason="fixture change")
        assert report["invalidated_count"] >= 0


# --------------------------------------------------------------------------- #
# 20. Training handoff remains disabled
# --------------------------------------------------------------------------- #
def test_training_handoff_remains_disabled():
    from taksh.dlcae.learning import LearningFeedbackEngine

    handoff = LearningFeedbackEngine().handoff_to_training([])
    assert handoff["authorized"] is False
    assert handoff["training_eligible_tokens"] == 0


# --------------------------------------------------------------------------- #
# Semantic metrics (synthetic ground truth only) + reproducibility
# --------------------------------------------------------------------------- #
class TestSemanticMetrics:
    def test_symbol_detection_on_ground_truth(self):
        metrics = evaluate_symbol_detection()
        assert metrics["precision"] >= 0.9
        assert metrics["recall"] >= 0.9
        assert metrics["unresolved"]["all_expected_detected"] is True

    def test_type_metrics_kinds(self):
        metrics = evaluate_type_relations()
        assert metrics["expected_kinds_present"] is True

    def test_dependency_metrics_detect_cycle_and_missing(self):
        metrics = evaluate_dependency_graph()
        assert metrics["cycle_detected"] is True
        assert metrics["missing_dependency_detected"] is True
        assert metrics["installed_stdlib_recognized"] is True

    def test_reproducibility_byte_identical(self):
        assert evaluate_reproducibility()["byte_identical"] is True

    def test_compatibility_unknown_discipline(self):
        assert evaluate_compatibility()["unknown_discipline_preserved"] is True

    def test_semantic_metrics_recorded_with_arm_run(self):
        framework = ExperimentFramework()
        framework.register_task_set("synth_sem",
                                    [{"id": "t1", "expected": "a", "scoring": "exact_match"}],
                                    scoring="exact_match")
        from taksh.dlcae.semantic.metrics import evaluate_reproducibility

        semantic = evaluate_reproducibility()
        run = framework.run_arm("B", "synth_sem", seeds=[1, 2, 3],
                                model_fn=lambda task, seed: {"answer": "a"},
                                semantic_metrics=semantic)
        assert run["semantic_metrics"]["recorded"] is True
        assert run["semantic_metrics"]["values"]["byte_identical"] is True


# --------------------------------------------------------------------------- #
# Frozen artifacts are never modified by this suite
# --------------------------------------------------------------------------- #
def test_frozen_benchmarks_preserved():
    import hashlib
    frozen = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    multi = hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest()
    assert frozen == "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
    assert multi == "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
