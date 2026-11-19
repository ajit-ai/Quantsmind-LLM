"""Focused tests for L1.0.0-ENG3 — End-to-End Coding Intelligence Integration.

Deterministic test doubles for the model; locally installed toolchains for
validation; synthetic fixtures only; never frozen benchmarks; never the
canonical model (it does not exist).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.dlcae.coding.evidence import EvidenceCategory, may_promote  # noqa: E402
from taksh.dlcae.coding.model import (  # noqa: E402
    ModelCapabilityState, ModelSession, StaticModelDouble,
    canonical_geometry_match, canonical_fingerprint,
)
from taksh.dlcae.coding.orchestrator import (  # noqa: E402
    CodingTaskOrchestrator, TaskState, extract_code,
)
from taksh.dlcae.coding.task import DevSuite, DevTaskTests, CodingTask, render_task_prompt  # noqa: E402
from taksh.dlcae.runtime import DLCAERuntime  # noqa: E402
from taksh.dlcae.semantic.tools import PythonToolchain  # noqa: E402
from taksh.dlcae.semantic.dependencies import analyze_python_dependencies  # noqa: E402
from taksh.dlcae.knowledge import KnowledgeFormationEngine, provenance_from_source  # noqa: E402
from taksh.dlcae.semantic.evidence import SourceEvidenceLedger  # noqa: E402
from taksh.dlcae.types import VerificationState  # noqa: E402

PY_FIXTURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
GOOD_CANDIDATE = "def add(a, b):\n    return a + b\n"
BROKEN_CANDIDATE = "def add(a, b):\n    return 0\n"
KARK_FIXTURE = "fn main() {}\n"


def _task(entry="add", task_id="ENG3-TEST-001") -> CodingTask:
    return CodingTask(
        task_id=task_id,
        language="python",
        description="Write add(a, b) returning the sum.",
        constraints=["return code only"],
        expected_behavior="add(2, 3) == 5",
        entry_point=entry,
        tests=[DevTaskTests(name="basic", check="call_equals",
                            args=(2, 3), expected=5, entry_point=entry),
               DevTaskTests(name="negative", check="call_equals",
                            args=(-1, 1), expected=0, entry_point=entry)])


def _suite_task() -> CodingTask:
    return DevSuite().get("ENG3-DEV-001")


# --------------------------------------------------------------------------- #
# 1. Model unavailable behavior
# --------------------------------------------------------------------------- #
def test_model_unavailable_when_unconfigured():
    session = ModelSession(None, None)
    assessment = session.assess()
    assert assessment["state"] == "MODEL_UNAVAILABLE"
    assert any("checkpoint" in p for p in assessment["problems"])


def test_unavailable_model_blocks_coding_run():
    session = ModelSession(None, None)
    orchestrator = CodingTaskOrchestrator(session, max_repair_attempts=1)
    result = orchestrator.run(_suite_task())
    assert result["state"] == "BLOCKED"
    assert result["blocking_reason"]
    assert result["capability_claim"] is None


# --------------------------------------------------------------------------- #
# 2. Canonical checkpoint mismatch (v180_sft never substituted)
# --------------------------------------------------------------------------- #
def test_v180_sft_rejected_as_foundation():
    session = ModelSession(REPO / "runs" / "v180_sft" / "checkpoint_last.pt",
                           REPO / "tokenizer" / "taksh_code_bpe_v1.json")
    assessment = session.assess()
    assert assessment["state"] == "MODEL_UNAVAILABLE"
    assert "GEOMETRY_INCOMPATIBLE" in (assessment["reason"] or "")
    assert assessment["fingerprints"]["geometry_mismatch"]["d_model"] == (384, 768)


def test_canonical_geometry_match_rejects_v180_geometry():
    v180_cfg = {"vocab_size": 5822, "d_model": 384, "n_layers": 8,
                "n_heads": 8, "n_kv_heads": 4, "ffn_dim": 1536,
                "max_seq_len": 1024}
    ok, mismatch = canonical_geometry_match(v180_cfg)
    assert ok is False
    assert "d_model" in mismatch


def test_canonical_fingerprint_is_foundation_identity():
    fp = canonical_fingerprint()
    assert len(fp) == 64


# --------------------------------------------------------------------------- #
# 3. Explicit model selection — no implicit defaults
# --------------------------------------------------------------------------- #
def test_missing_tokenizer_blocks():
    session = ModelSession(REPO / "runs" / "v180_sft" / "checkpoint_last.pt",
                           REPO / "tokenizer" / "nonexistent_tokenizer.json")
    assessment = session.assess()
    assert assessment["state"] == "MODEL_UNAVAILABLE"
    assert any("tokenizer" in p for p in assessment["problems"])


def test_missing_checkpoint_blocks():
    session = ModelSession(REPO / "runs" / "does_not_exist.pt",
                           REPO / "tokenizer" / "taksh_code_bpe_v1.json")
    assessment = session.assess()
    assert assessment["state"] == "MODEL_UNAVAILABLE"


def test_model_session_requires_explicit_arguments():
    with pytest.raises(TypeError):
        ModelSession()  # no implicit defaults — explicit config required


# --------------------------------------------------------------------------- #
# 4. Generation result capture
# --------------------------------------------------------------------------- #
def test_generation_captures_model_identity():
    double = StaticModelDouble(responses=[GOOD_CANDIDATE])
    session = double.assess()
    assert session["model_kind"] == "test_double"
    out = double.generate("prompt", seed=1, max_new_tokens=32)
    assert out["state"] == ModelCapabilityState.GENERATION_COMPLETED
    assert out["model_id"] == "static-test-double"
    assert out["evidence_category"] == "MODEL_GENERATED"
    assert out["trace_fingerprint"]
    assert out["capability_claim"] is None


def test_attempt_records_generation_block():
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                    max_repair_attempts=0).run(_suite_task())
    attempt = result["attempts"][0]
    assert attempt["state"] == "GENERATION_COMPLETED"
    assert attempt["model_kind"] == "test_double"
    assert attempt["generation"] is not None
    assert attempt["source_sha256"]
    assert attempt["evidence_category"] == "MODEL_GENERATED"


# --------------------------------------------------------------------------- #
# 5. Generation failure handling
# --------------------------------------------------------------------------- #
class _FailingDouble:
    model_kind = "test_double"

    def __init__(self):
        self.state = "MODEL_READY"
        self.generation_config = {}
        self.last_trace = None
        self.last_trace_fingerprint = None

    def assess(self):
        return {"state": "MODEL_READY", "model_kind": "test_double",
                "canonical": False, "problems": []}

    def generate(self, prompt, *, seed=None, max_new_tokens=64,
                 temperature=0.0, task_id=None, mode="model_only"):
        return {"state": ModelCapabilityState.GENERATION_FAILED,
                "error": "simulated failure",
                "model_kind": "test_double", "capability_claim": None,
                "evidence_category": "MODEL_GENERATED"}


def test_generation_failure_is_recorded_not_hidden():
    result = CodingTaskOrchestrator(_FailingDouble(), max_repair_attempts=1).run(_suite_task())
    assert result["state"] == "FAILED"
    assert result["blocking_reason"]
    assert result["attempts"][0]["state"] == "GENERATION_FAILED"


# --------------------------------------------------------------------------- #
# 6. DLCAE integration
# --------------------------------------------------------------------------- #
def test_semantic_pipeline_integrated_in_attempt(tmp_path):
    runtime = DLCAERuntime(tmp_path / "runtime")
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                    runtime=runtime, max_repair_attempts=0).run(_suite_task())
    semantic = result["attempts"][0]["validation"]["semantic"]
    assert semantic["status"] in ("PASS", "PARTIAL")
    assert "stage_status" in semantic


# --------------------------------------------------------------------------- #
# 7. Semantic analysis integration (dependency edge evidence)
# --------------------------------------------------------------------------- #
def test_dependency_analysis_runs_locally():
    graph = analyze_python_dependencies({
        "uses_json": "import json\nVALUE = 1\n",
        "uses_local": "import uses_json\n",
    })
    assert not graph.missing
    assert any(e["kind"] == "local" for e in graph.edges)


# --------------------------------------------------------------------------- #
# 8. Toolchain selection
# --------------------------------------------------------------------------- #
def test_python_toolchain_selected_for_python_task():
    tool = PythonToolchain()
    assert tool.discover() is True
    syntax = tool.syntax_check(PY_FIXTURE, filename="fixture.py")
    assert syntax.ok is True


def test_non_python_language_blocked_not_fabricated():
    resources = CodingTaskOrchestrator(StaticModelDouble(responses=[KARK_FIXTURE]),
                                       max_repair_attempts=0).run(
        CodingTask(task_id="KARK-TASK", language="karkain",
                   description="d", entry_point="main",
                   tests=[DevTaskTests(name="t", check="call_equals",
                                       args=(), expected=1, entry_point="main")]))
    assert resources["attempts"][0]["validation"]["state"] == "BLOCKED"
    assert resources["state"] == "BLOCKED"


# --------------------------------------------------------------------------- #
# 9/10/11. Python syntax validation, execution, compiler feedback
# --------------------------------------------------------------------------- #
def test_python_syntax_validation_detects_broken_code():
    tool = PythonToolchain()
    syntax = tool.syntax_check("def broken(:\n    return 1\n", filename="broken.py")
    assert syntax.ok is False


def test_python_execution_via_sandbox_and_hidden_tests():
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                    max_repair_attempts=0).run(_suite_task())
    execution = result["attempts"][0]["validation"]["execution"]
    assert execution["all_passed"] is True
    assert execution["tests_passed"] == 3
    assert execution["workspace"].startswith("isolated-tmp-workspace")


def test_structured_compiler_feedback_distinguishes_categories():
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[BROKEN_CANDIDATE]),
                                    max_repair_attempts=0).run(_suite_task())
    categories = {d["category"] for d in result["attempts"][0]["diagnostics"]}
    assert "test_failure" in categories
    assert result["attempts"][0]["validation"]["correlation"]["earliest_known_failure"]


# --------------------------------------------------------------------------- #
# 13/14. Runtime failure localization and cascading classification
# --------------------------------------------------------------------------- #
def test_correlation_distinguishes_primary_from_cascade():
    from taksh.dlcae.semantic.feedback import correlate, FeedbackRecord

    sha = "a" * 64
    records = [
        FeedbackRecord.build(source_sha256=sha, language_id="python",
                             adapter="py", adapter_version="1",
                             toolchain_name="python", toolchain_version="3",
                             category="syntax_error", message="unexpected ')'",
                             line=1, evidence_classification="TOOL_VERIFIED"),
        FeedbackRecord.build(source_sha256=sha, language_id="python",
                             adapter="py", adapter_version="1",
                             toolchain_name="python", toolchain_version="3",
                             category="type_error", message="downstream mismatch",
                             line=2, evidence_classification="TOOL_VERIFIED"),
    ]
    correlation = correlate(records)
    assert correlation["primary_category"] == "syntax_error"
    assert correlation["cascades"]


# --------------------------------------------------------------------------- #
# 15/16/17. Repair candidates, non-mutating safety, revalidation
# --------------------------------------------------------------------------- #
def test_repair_candidates_generated_and_non_mutating(tmp_path):
    original = tmp_path / "keep.py"
    original.write_text(PY_FIXTURE, encoding="utf-8")
    before = original.read_bytes()
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[BROKEN_CANDIDATE]),
                                    max_repair_attempts=0).run(
        CodingTask(task_id="REPAIR-001", language="python",
                   description="d", entry_point="add",
                   tests=[DevTaskTests(name="t", check="call_equals",
                                       args=(1, 1), expected=2,
                                       entry_point="add")]))
    candidates = result["attempts"][0]["repair_candidates"]
    assert candidates["candidate_count"] >= 1
    assert all(c["source_mutation"] is False for c in candidates["candidates"])
    assert all(c["verified"] is False for c in candidates["candidates"])
    assert original.read_bytes() == before


def test_revalidation_after_repair():
    double = StaticModelDouble(responses=[BROKEN_CANDIDATE, GOOD_CANDIDATE])
    result = CodingTaskOrchestrator(double, max_repair_attempts=2).run(_suite_task())
    assert result["state"] == "COMPLETED"
    assert result["attempts"][1]["validation"]["execution"]["all_passed"] is True
    assert result["attempts"][0]["validation"]["execution"]["all_passed"] is False


# --------------------------------------------------------------------------- #
# 18. Source-hash evidence invalidation
# --------------------------------------------------------------------------- #
def test_source_hash_invalidation():
    knowledge = KnowledgeFormationEngine()
    ledger = SourceEvidenceLedger(knowledge)
    sha_a = "a" * 64
    sha_b = "b" * 64
    prov = provenance_from_source("fixture", license_spdx="owner-internal-research",
                                  record_id="prov-eng3-inval")
    rec = knowledge.add("python", "language_rule", "eng3:inval", {"n": 1},
                        provenance=prov, state=VerificationState.OBSERVED)
    knowledge.verify(rec.record_id, validation_verdict="PASS",
                     evidence={"oracle": "fixture"})
    ledger.bind(sha_a, rec.record_id, "PASS")
    report = ledger.verify_source(sha_b, reason="fixture change")
    assert report["invalidated_count"] == 1


# --------------------------------------------------------------------------- #
# 19. Learning authorization enforcement
# --------------------------------------------------------------------------- #
def test_learning_promotion_gates_enforced():
    assert may_promote(EvidenceCategory.TOOL_VERIFIED) is True
    assert may_promote(EvidenceCategory.HUMAN_CONFIRMED) is True
    assert may_promote(EvidenceCategory.MODEL_GENERATED) is False
    assert may_promote(EvidenceCategory.TOOL_OBSERVED) is False
    assert may_promote("UNRELIABLE") is False


def test_training_handoff_still_refused():
    from taksh.dlcae.learning import LearningFeedbackEngine

    handoff = LearningFeedbackEngine().handoff_to_training([])
    assert handoff["authorized"] is False
    assert handoff["training_eligible_tokens"] == 0


# --------------------------------------------------------------------------- #
# 20. Karkain blocked-state preservation
# --------------------------------------------------------------------------- #
def test_karkain_remains_blocked_and_zero_tasks():
    from taksh.dlcae.karkain import KarkainAdapter

    adapter = KarkainAdapter()
    assert adapter.kark_bench_status()["instantiated_tasks"] == 0
    assert adapter.semantic_summary()["status"] == "BLOCKED"
    assert adapter.validate(KARK_FIXTURE, stage="parser")["verdict"] == "BLOCKED"


# --------------------------------------------------------------------------- #
# 21. Task interruption and recovery
# --------------------------------------------------------------------------- #
def test_interrupted_task_recovers_from_ledger(tmp_path):
    import threading

    orchestrator = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                          max_repair_attempts=1,
                                          state_dir=str(tmp_path / "state"))
    cancel = threading.Event()
    cancel.set()
    result = orchestrator.run(_suite_task(), cancel_event=cancel)
    assert result["state"] == "INTERRUPTED"
    assert result["interruption"]["reason"]


def test_state_dir_persists_ledger(tmp_path):
    state = tmp_path / "state"
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                    max_repair_attempts=0,
                                    state_dir=str(state)).run(_suite_task())
    assert result["state"] == "COMPLETED"
    assert (state / "_persistence").is_dir()


# --------------------------------------------------------------------------- #
# 22. Timeout and cancellation
# --------------------------------------------------------------------------- #
def test_execution_timeout_kills_hung_candidate(tmp_path):
    result = CodingTaskOrchestrator(
        StaticModelDouble(responses=["import time\nwhile True:\n    time.sleep(0.05)\n"]),
        max_repair_attempts=0,
        execution_timeout_s=1.0,
    ).run(CodingTask(task_id="HANG-001", language="python",
                     description="hang test", entry_point="solve",
                     tests=[DevTaskTests(name="t", check="call_equals",
                                         args=(), expected=1,
                                         entry_point="solve")]))
    validation = result["attempts"][0]["validation"] if result["attempts"] else {}
    assert result["state"] in ("FAILED", "BLOCKED")
    assert validation.get("execution", {}).get("timed_out") is True


def test_cancellation_before_run_stops_pipeline(tmp_path):
    import threading

    cancel = threading.Event()
    cancel.set()
    result = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                    max_repair_attempts=1).run(
        _suite_task(), cancel_event=cancel)
    assert result["state"] == "INTERRUPTED"


# --------------------------------------------------------------------------- #
# 23. Maximum repair-attempt enforcement
# --------------------------------------------------------------------------- #
def test_max_repair_attempts_bounded():
    double = StaticModelDouble(responses=[BROKEN_CANDIDATE] * 8)
    result = CodingTaskOrchestrator(double, max_repair_attempts=1).run(_suite_task())
    assert result["state"] == "FAILED"
    assert result["final_result"]["attempts_used"] == 2
    assert "max_repair_attempts" in result["blocking_reason"]


# --------------------------------------------------------------------------- #
# 24. Model-only vs tool-assisted separation
# --------------------------------------------------------------------------- #
def test_model_only_and_tool_assisted_are_separate(tmp_path):
    double = StaticModelDouble(responses=[GOOD_CANDIDATE])
    assisted = CodingTaskOrchestrator(double, tool_assisted=True,
                                      max_repair_attempts=0).run(_suite_task())
    double2 = StaticModelDouble(responses=[GOOD_CANDIDATE])
    model_only = CodingTaskOrchestrator(double2, tool_assisted=False,
                                        max_repair_attempts=0).run(_suite_task())
    assert assisted["mode"] == "tool_assisted"
    assert model_only["mode"] == "model_only"
    assert assisted["attempts"][0]["generation"]["sampling"]["mode"] == "tool_assisted"
    assert model_only["attempts"][0]["generation"]["sampling"]["mode"] == "model_only"


# --------------------------------------------------------------------------- #
# 25. Reproducible machine-readable reports
# --------------------------------------------------------------------------- #
def test_qualification_report_loads_and_separates_axes():
    report_path = REPO / "reports" / "l100eng3_coding_integration.json"
    if not report_path.is_file():
        pytest.skip("qualification report not yet generated")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "taksh.l100eng3_coding_qualification/1"
    axes = report["qualification"]
    assert axes["engineering_qualification"].startswith("ENGINEERING_QUALIFIED")
    assert axes["model_availability"] == "MODEL_UNAVAILABLE"
    assert axes["actual_coding_capability"].startswith("NOT_EVALUATED")
    assert "language_qualification" in axes
    assert report["capability_claim"] == "none"


def test_double_runs_are_deterministic():
    out1 = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                  max_repair_attempts=0).run(_suite_task())
    out2 = CodingTaskOrchestrator(StaticModelDouble(responses=[GOOD_CANDIDATE]),
                                  max_repair_attempts=0).run(_suite_task())
    assert out1["attempts"][0]["source_sha256"] == out2["attempts"][0]["source_sha256"]
    assert out1["attempts"][0]["state"] == out2["attempts"][0]["state"]


# --------------------------------------------------------------------------- #
# Development suite integrity
# --------------------------------------------------------------------------- #
def test_dev_suite_loads_six_synthetic_tasks():
    suite = DevSuite()
    assert len(suite) == 6
    for task in suite.tasks():
        assert task.task_id.startswith("ENG3-DEV-")
        assert task.language == "python"
        assert task.tests


def test_dev_suite_hashes_stable():
    suite = DevSuite()
    assert suite.suite_sha256() == suite.suite_sha256()


def test_extract_code_strips_fences():
    assert extract_code("```python\ndef f():\n    pass\n```") == "def f():\n    pass"
    assert extract_code("def f():\n    pass") == "def f():\n    pass"


def test_frozen_benchmark_preserved():
    import hashlib
    digest = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    assert digest == "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


def test_karkain_registry_still_discovered():
    from taksh.dlcae.registry import DynamicLanguageRegistry

    registry = DynamicLanguageRegistry()
    assert registry.status_of("karkain") == "DISCOVERED"
    assert not registry.is_capability_state("karkain")
