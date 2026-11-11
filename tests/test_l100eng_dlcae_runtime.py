"""Focused tests for TAKSH L1.0.0-ENG — DLCAE Runtime Foundation.

Deterministic test doubles only. No training, no model execution, no
external data, no downloads, no fixtures outside tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.dlcae.adaptation import LanguageAdaptationEngine  # noqa: E402
from taksh.dlcae.comprehension import IncrementalComprehensionEngine  # noqa: E402
from taksh.dlcae.experiments import ExperimentFramework, frozen_benchmark_problem_ids  # noqa: E402
from taksh.dlcae.karkain import KarkainAdapter  # noqa: E402
from taksh.dlcae.knowledge import KnowledgeFormationEngine, provenance_from_source  # noqa: E402
from taksh.dlcae.learning import LearningFeedbackEngine, read_owner_decision  # noqa: E402
from taksh.dlcae.persistence import CorruptionError, PersistenceStore  # noqa: E402
from taksh.dlcae.registry import (  # noqa: E402
    DynamicLanguageRegistry, RegistryError, extension_entry_template,
)
from taksh.dlcae.runtime import DLCAERuntime  # noqa: E402
from taksh.dlcae.sequences import (  # noqa: E402
    CycleError, DependencyError, SequenceOrchestrationEngine,
)
from taksh.dlcae.types import (  # noqa: E402
    QualificationStatus, TaskState, ValidationVerdict, VerificationState,
)
from taksh.dlcae.validation import ValidationEngine  # noqa: E402

PY_FIXTURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"


# --------------------------------------------------------------------------- #
# A. Incremental Comprehension Engine
# --------------------------------------------------------------------------- #
class TestIncrementalComprehension:
    def test_levels_one_through_seven_all_present(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", PY_FIXTURE, "python")
        assert sorted(bundle.levels) == [1, 2, 3, 4, 5, 6, 7]
        assert bundle.level_status[4] == "COMPLETE"
        assert len(bundle.observations()) >= 7

    def test_offsets_preserved(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", PY_FIXTURE, "python")
        sample = bundle.observations(2)[0].detail["sample_offsets"]
        assert sample and all(len(p) == 2 for p in sample)

    def test_syntax_failure_halts_deeper_levels(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", "def broken(:\n    return 1\n", "python")
        assert bundle.level_status[4] == "FAILED"
        assert bundle.level_status[5] == "BLOCKED"
        assert bundle.level_status[7] == "BLOCKED"
        assert "halted" in bundle.note

    def test_verified_observations_survive_incremental_update(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", PY_FIXTURE, "python")
        proven = next(o for o in bundle.observations() if o.level == 1)
        proven.state = VerificationState.VERIFIED
        original_id = proven.observation_id
        bundle = engine.resume_from(bundle, PY_FIXTURE, "python", levels=[1, 4])
        flat = {o.observation_id for o in bundle.observations()}
        assert original_id in flat

    def test_uncertain_and_conflicted_observations_are_kept(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", "def f():\n    return True\n", "karkain")
        level4 = bundle.level_status[4]
        assert level4 in ("UNCERTAIN", "COMPLETE")
        assert all(o.detail for o in bundle.observations(4))

    def test_tokenization_never_implies_understanding(self):
        engine = IncrementalComprehensionEngine()
        bundle = engine.comprehend("doc", "??? weird content(", "klingon_plan9")
        assert bundle.verification_status != "VERIFIED"
        assert all(o.kind != "execution_verification" or "UNVERIFIED" in json.dumps(o.detail)
                   for o in bundle.observations())

    def test_granularity_is_configurable(self):
        with pytest.raises(ValueError):
            IncrementalComprehensionEngine(granularity="quantum")
        for granularity in ("char", "token", "line"):
            engine = IncrementalComprehensionEngine(granularity=granularity)
            assert engine.comprehend("d", "x = 1", "python").granularity == granularity


# --------------------------------------------------------------------------- #
# B. Sequence Orchestration Engine
# --------------------------------------------------------------------------- #
def _sequence_engine() -> SequenceOrchestrationEngine:
    return SequenceOrchestrationEngine(handlers={
        "ok": lambda payload: {"echo": payload},
        "fail": lambda payload: (_ for _ in ()).throw(RuntimeError("boom")),
        "slow-double": lambda payload: {"value": payload["n"] * 2},
    })


class TestSequenceOrchestration:
    def test_c_and_d_sequences_are_independent(self):
        eng = _sequence_engine()
        c = eng.create_sequence("C-seq", [
            {"task_id": "C", "action": "ok"},
            {"task_id": "C1", "action": "ok", "depends_on": ["C"]},
            {"task_id": "C2", "action": "ok", "depends_on": ["C1"]},
        ])
        d = eng.create_sequence("D-seq", [
            {"task_id": "D1", "action": "fail"},
            {"task_id": "D2", "action": "ok", "depends_on": ["D1"]},
        ])
        res_c = eng.run(c)
        res_d = eng.run(d)
        assert res_c["state"] == "COMPLETED"
        assert res_d["state"] == "FAILED"
        assert eng.get(c).state == "COMPLETED"     # failure in D never corrupts C
        assert eng.get(d).tasks["D2"].state is TaskState.BLOCKED

    def test_cycle_detection(self):
        eng = _sequence_engine()
        with pytest.raises(CycleError):
            eng.create_sequence("loopy", [
                {"task_id": "A", "action": "ok", "depends_on": ["B"]},
                {"task_id": "B", "action": "ok", "depends_on": ["A"]},
            ])

    def test_unknown_dependency_rejected(self):
        eng = _sequence_engine()
        with pytest.raises(DependencyError):
            eng.create_sequence("bad-dep", [{"task_id": "A", "action": "ok",
                                              "depends_on": ["ZZZ"]}])

    def test_retry_policy_bounded(self):
        attempts = {"count": 0}

        def flaky(payload):
            attempts["count"] += 1
            raise RuntimeError("always fails")

        eng = SequenceOrchestrationEngine(handlers={"flaky": flaky})
        sid = eng.create_sequence("retry", [{"task_id": "A", "action": "flaky",
                                             "retry": {"max_attempts": 3}}])
        eng.run(sid)
        assert attempts["count"] == 3
        assert eng.get(sid).tasks["A"].state is TaskState.FAILED

    def test_checkpoint_resume_roundtrip(self, tmp_path):
        eng = _sequence_engine()
        sid = eng.create_sequence("resume", [
            {"task_id": "C", "action": "ok"},
            {"task_id": "C1", "action": "ok", "depends_on": ["C"]},
        ])
        eng.run(sid)
        ledger = eng.checkpoint(sid)
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        eng2 = _sequence_engine()
        restored = eng2.restore(json.loads(ledger_path.read_text(encoding="utf-8")))
        assert restored == sid
        result = eng2.run(restored)
        assert result["state"] == "COMPLETED"

    def test_interrupted_sequence_resumes_as_pending(self):
        eng = _sequence_engine()
        sid = eng.create_sequence("interrupt", [{"task_id": "A", "action": "ok"}])
        eng.mark_interrupted(sid, reason="simulated interruption")
        ledger = eng.checkpoint(sid)
        eng2 = _sequence_engine()
        restored = eng2.restore(ledger)
        assert eng2.run(restored)["state"] == "COMPLETED"

    def test_cancelled_sequence_cannot_resume(self):
        eng = _sequence_engine()
        sid = eng.create_sequence("cancel", [{"task_id": "A", "action": "ok"}])
        eng.cancel(sid, reason="owner abort")
        with pytest.raises(Exception):
            eng.restore(eng.checkpoint(sid))

    def test_idempotent_task_not_reexecuted_on_resume(self):
        calls = {"n": 0}

        def counting(payload):
            calls["n"] += 1
            return {"n": calls["n"]}

        eng = SequenceOrchestrationEngine(handlers={"count": counting})
        sid = eng.create_sequence("idem", [{"task_id": "A", "action": "count"}])
        eng.run(sid)
        first_result = eng.get(sid).tasks["A"].result
        eng.restore(eng.checkpoint(sid))
        eng2 = SequenceOrchestrationEngine(handlers={"count": counting})
        eng2.restore(eng.checkpoint(sid))
        eng2.run(sid)
        assert eng2.get(sid).tasks["A"].result == first_result
        assert calls["n"] == 1  # never re-executed

    def test_bounded_execution(self):
        eng = SequenceOrchestrationEngine(max_tasks_per_run=1, handlers={"ok": lambda p: {}})
        sid = eng.create_sequence("bounded", [
            {"task_id": f"T{i}", "action": "ok"} for i in range(5)
        ])
        result = eng.run(sid)
        assert result.get("interrupted") is True
        assert len(result["completed"]) == 1


# --------------------------------------------------------------------------- #
# C. Knowledge Formation Engine
# --------------------------------------------------------------------------- #
class TestKnowledgeFormation:
    def _prov(self):
        return provenance_from_source("synthetic-fixture",
                                      license_spdx="owner-internal-research",
                                      record_id="prov-test")

    def test_observed_inferred_verified_states(self):
        engine = KnowledgeFormationEngine()
        rec = engine.add("python", "language_rule", "rule-1", {"n": 1},
                         provenance=self._prov(),
                         state=VerificationState.OBSERVED)
        assert rec.state is VerificationState.OBSERVED
        verified = engine.verify(rec.record_id, validation_verdict="PASS",
                                 evidence={"oracle": "probe"})
        assert verified.state is VerificationState.VERIFIED

    def test_verification_requires_pass_verdict(self):
        engine = KnowledgeFormationEngine()
        rec = engine.add("python", "language_rule", "rule-2", {"n": 2},
                         provenance=self._prov(),
                         state=VerificationState.OBSERVED)
        result = engine.verify(rec.record_id, validation_verdict="BLOCKED")
        assert result.state is not VerificationState.VERIFIED
        assert result.verification_evidence["admitted"] is False

    def test_exact_duplicates_are_idempotent(self):
        engine = KnowledgeFormationEngine()
        a = engine.add("python", "concept", "dup", {"n": 1}, provenance=self._prov())
        b = engine.add("python", "concept", "dup", {"n": 1}, provenance=self._prov())
        assert a.record_id == b.record_id

    def test_conflicting_records_both_marked_conflicted(self):
        engine = KnowledgeFormationEngine()
        a = engine.add("python", "language_rule", "same-name", {"v": "A"}, provenance=self._prov())
        b = engine.add("python", "language_rule", "same-name", {"v": "B"}, provenance=self._prov())
        assert a.state is VerificationState.CONFLICTED
        assert b.state is VerificationState.CONFLICTED
        assert set(a.conflicts_with) == {b.record_id}

    def test_invalidation_is_explicit_not_deletion(self):
        engine = KnowledgeFormationEngine()
        rec = engine.add("python", "concept", "bad-fact", {"n": 3}, provenance=self._prov())
        engine.invalidate(rec.record_id, reason="counterexample observed")
        again = engine.query(state=VerificationState.INVALIDATED)
        assert len(again) == 1
        assert again[0].record_id == rec.record_id

    def test_versioning_supersedes(self):
        engine = KnowledgeFormationEngine()
        v1 = engine.add("python", "concept", "versioned", {"n": 1}, provenance=self._prov())
        v2 = engine.new_version(v1.record_id, {"n": 2})
        assert v2.version == 2
        assert v2.supersedes == v1.record_id

    def test_export_import_integrity(self):
        engine = KnowledgeFormationEngine()
        engine.add("python", "concept", "export-me", {"n": 9}, provenance=self._prov())
        dump = engine.export()
        dump["records"][0]["content"]["n"] = 999   # tamper
        result = KnowledgeFormationEngine().import_store(dump)
        assert result["ok"] is False
        assert any("fingerprint" in p for p in result["problems"])


# --------------------------------------------------------------------------- #
# D. Language Adaptation + registry integration
# --------------------------------------------------------------------------- #
class TestLanguageAdaptation:
    def _adapter(self, tmp_path):
        registry = DynamicLanguageRegistry()
        validation = ValidationEngine()
        return LanguageAdaptationEngine(registry, validation), registry

    def test_synthetic_language_registration_no_ceiling(self):
        _, registry = self._adapter(None)
        before = len(registry.all_ids())
        entry = extension_entry_template("synthetic_lang_x", "Synthetic X", [".synthx"],
                                         source="synthetic fixture",
                                         license_spdx="owner-internal-research")
        registry.register(entry)
        assert len(registry.all_ids()) == before + 1
        assert registry.status_of("synthetic_lang_x") == "REGISTERED"

    def test_true_discovery_flow_for_unknown_language(self):
        _, registry = self._adapter(None)
        entry = extension_entry_template("totally_new_lang", "Totally New Lang", [".tnl"],
                                         source="synthetic fixture",
                                         license_spdx="owner-internal-research")
        registry.register(entry)
        promoted = registry.promote("totally_new_lang", "DISCOVERED")
        assert promoted["qualification_status"] == "DISCOVERED"

    def test_capability_status_refused_without_evaluation_evidence(self):
        _, registry = self._adapter(None)
        registry.register(extension_entry_template(
            "nocap_lang", "NoCap", [".nocap"],
            source="synthetic fixture", license_spdx="owner-internal-research"))
        with pytest.raises(RegistryError):
            registry.promote("nocap_lang", "FUNCTIONALLY_QUALIFIED",
                             evidence={"evaluation_passed": False,
                                       "benchmark_bar": "bar", "regression_passed": False})

    def test_regression_demotes_language(self):
        adaptation, registry = self._adapter(None)
        registry.register(extension_entry_template(
            "regress_lang", "Regress", [".reg"],
            source="synthetic fixture", license_spdx="owner-internal-research"))
        registry.promote("regress_lang", "DISCOVERED")
        registry.promote("regress_lang", "COMPREHENSION_IN_PROGRESS")
        demoted = adaptation.record_regression("regress_lang",
                                               reason="synthetic regression")
        assert demoted["qualification_status"] == "DISCOVERED"

    def test_blocked_requires_owner_unblock(self):
        adaptation, registry = self._adapter(None)
        registry.register(extension_entry_template(
            "blocked_lang", "Blocked", [".blk"],
            source="synthetic fixture", license_spdx="UNKNOWN"))
        adaptation.block("blocked_lang", reason="license UNKNOWN")
        with pytest.raises(RegistryError):
            registry.promote("blocked_lang", "DISCOVERED")
        with pytest.raises(RegistryError):
            adaptation.unblock("blocked_lang", owner_reference="")
        unblocked = adaptation.unblock("blocked_lang", owner_reference="OWNER-123")
        assert unblocked["qualification_status"] == "DISCOVERED"

    def test_karkain_first_class_and_blocked(self):
        _, registry = self._adapter(None)
        karkain = registry.lookup("karkain")
        assert karkain is not None
        assert ".kark" in karkain["extensions"]
        assert karkain["qualification_status"] == "DISCOVERED"
        assert not registry.is_capability_state("karkain")

    def test_missing_toolchains_recorded(self):
        adaptation, _ = self._adapter(None)
        assert adaptation.missing_toolchains("karkain")

    def test_python_registered_in_base(self):
        _, registry = self._adapter(None)
        assert registry.status_of("python") == "REGISTERED"
        karkain_blocked = registry.lookup("python")
        assert karkain_blocked["origin"] == "base-53"

    def test_collision_with_base_id_rejected(self):
        _, registry = self._adapter(None)
        entry = extension_entry_template("python", "Imposter", [".py3"],
                                         source="synthetic fixture",
                                         license_spdx="owner-internal-research")
        with pytest.raises(RegistryError):
            registry.register(entry)

    def test_karkain_contract_forbids_bypass(self):
        from taksh.dlcae.karkain import KarkainAdapter, KCC_STARTUP_FAILURE
        adapter = KarkainAdapter()
        summary = adapter.blocked_summary()
        assert KCC_STARTUP_FAILURE == "0xC0000135"
        assert "bypass KCC" in summary["actions_prohibited"]
        assert adapter.validate("fn main() {}", stage="execution")["verdict"] == "BLOCKED"


# --------------------------------------------------------------------------- #
# E. Validation Engine
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_model_only_and_tool_assisted_are_separate(self):
        engine = ValidationEngine()
        m = engine.validate("structural", {"source": "x = 1"}, language_id="python",
                            mode="model_only")
        t = engine.validate("parser", {"source": "x = 1"}, language_id="python",
                            mode="tool_assisted")
        assert m["mode"] == "model_only" and t["mode"] == "tool_assisted"
        assert "mode_note" in m

    def test_karkain_tool_assisted_is_blocked_never_pass(self):
        engine = ValidationEngine()
        result = engine.validate("parser", {"source": "fn main() {}"},
                                 language_id="karkain", mode="tool_assisted")
        assert result["verdict"] == "BLOCKED"

    def test_unknown_language_tool_assisted_blocked(self):
        engine = ValidationEngine()
        result = engine.validate("parser", {"source": "whatever"},
                                 language_id="cobol74", mode="tool_assisted")
        assert result["verdict"] == "BLOCKED"

    def test_semantic_stage_blocked_without_type_checker(self):
        engine = ValidationEngine()
        result = engine.validate("semantic", {"source": "x = 1"},
                                 language_id="python", mode="tool_assisted")
        assert result["verdict"] == "BLOCKED"

    def test_unverified_never_coerced_to_pass(self):
        engine = ValidationEngine()
        result = engine.validate("execution", {"source": "x = 1"},
                                 language_id="python", mode="model_only")
        assert result["verdict"] == "UNVERIFIED"

    def test_ledger_records_all_verdicts(self):
        engine = ValidationEngine()
        engine.validate("lexical", {"source": "x = 1"}, language_id="python",
                        mode="model_only")
        engine.validate("parser", {"source": "x = 1"}, language_id="python",
                        mode="tool_assisted")
        summary = engine.summary()
        assert summary["ledger_entries"] == 2
        assert summary["model_only_entries"] == 1
        assert summary["tool_assisted_entries"] == 1


# --------------------------------------------------------------------------- #
# F. Learning Feedback Engine
# --------------------------------------------------------------------------- #
class TestLearningFeedback:
    def test_training_handoff_refused_under_a1_1_c(self):
        engine = LearningFeedbackEngine()
        result = engine.handoff_to_training([])
        assert result["authorized"] is False
        assert "A1.1-A" in result["reason"]
        assert result["training_eligible_tokens"] == 0

    def test_owner_decision_record_fail_closed(self, tmp_path):
        missing = tmp_path / "nonexistent.md"
        decision = read_owner_decision(missing)
        assert decision["checked"] == []
        assert decision["ok"] is False
        assert decision["training_authorized"] is False

    def test_owner_decision_record_only_allows_known_checks(self):
        decision = read_owner_decision()
        assert "A1.1-C" in decision["checked"]
        assert "A2-E" in decision["checked"]
        assert "A1.1-A" not in decision["checked"]

    def test_candidate_pipeline_fail_closed(self):
        engine = LearningFeedbackEngine()
        observation = {"observation_id": "obs-1", "state": "INFERRED",
                       "kind": "task_intent"}
        prov = provenance_from_source("synthetic", license_spdx="owner-internal-research")
        candidate = engine.from_observation(observation, language_id="python",
                                            source_provenance=prov,
                                            content_ref="doc#obs-1")
        result = engine.eligibility_check(candidate, validation_verdict="BLOCKED")
        assert result["state"] == "REJECTED"

    def test_duplicate_candidate_detected(self):
        engine = LearningFeedbackEngine()
        observation = {"observation_id": "obs-dup", "state": "VERIFIED"}
        prov = provenance_from_source("synthetic", license_spdx="owner-internal-research",
                                      record_id="prov-dup")
        c1 = engine.from_observation(observation, language_id="python",
                                     source_provenance=prov, content_ref="r#1")
        c2 = engine.from_observation(observation, language_id="python",
                                     source_provenance=prov, content_ref="r#1")
        assert c2["state"] == "DUPLICATE"

    def test_replay_manifest_executes_nothing(self):
        engine = LearningFeedbackEngine()
        manifest = engine.replay_manifest()
        assert manifest["executes_training"] is False
        assert manifest["candidate_count"] == 0


# --------------------------------------------------------------------------- #
# Persistence and recovery
# --------------------------------------------------------------------------- #
class TestPersistence:
    def test_roundtrip_and_integrity(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        store.save_json("seq/one.json", {"a": 1})
        assert store.load_json("seq/one.json") == {"a": 1}
        assert store.verify_integrity()["ok"] is True

    def test_tamper_detected(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        store.save_json("seq/one.json", {"a": 1})
        target = tmp_path / "store" / "seq" / "one.json"
        target.write_text('{"a": 999}', encoding="utf-8")
        with pytest.raises(CorruptionError):
            store.load_json("seq/one.json")

    def test_unrecorded_file_flagged(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        store.save_json("seq/one.json", {"a": 1})
        (tmp_path / "store" / "seq" / "planted.json").write_text("{}", encoding="utf-8")
        result = store.verify_integrity()
        assert result["ok"] is False
        assert any("unrecorded" in p for p in result["problems"])

    def test_interrupted_write_detected(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        store.save_json("seq/one.json", {"a": 1})
        store.interrupt("seq/one.json", b"partial-bytes")
        result = store.verify_integrity()
        assert any("interrupted" in p for p in result["problems"])

    def test_path_escape_rejected(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        with pytest.raises(Exception):
            store.save_json("../../escape.json", {"bad": True})

    def test_schema_version_gate(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        store.save_json("seq/one.json", {"a": 1})
        manifest = json.loads((tmp_path / "store" / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["schema_version"] = 999
        (tmp_path / "store" / "MANIFEST.json").write_text(json.dumps(manifest),
                                                          encoding="utf-8")
        with pytest.raises(Exception):
            store.load_json("seq/one.json")

    def test_idempotent_save(self, tmp_path):
        store = PersistenceStore(tmp_path / "store")
        first = store.save_json_idempotent("seq/one.json", {"a": 1})
        second = store.save_json_idempotent("seq/one.json", {"a": 1})
        assert first["written"] is True
        assert second["written"] is False


# --------------------------------------------------------------------------- #
# Karkain first-class adapter
# --------------------------------------------------------------------------- #
class TestKarkainAdapter:
    def test_kark_extension_recognised(self):
        adapter = KarkainAdapter()
        assert adapter.is_karkain_source("kernel.kark")
        assert not adapter.is_karkain_source("kernel.py")

    def test_validation_always_blocked(self):
        adapter = KarkainAdapter()
        for stage in ("lexical", "parser", "execution"):
            assert adapter.validate("fn main() {}", stage=stage)["verdict"] == "BLOCKED"

    def test_spec_pin_recorded(self):
        adapter = KarkainAdapter()
        assert adapter.spec_pin()["karkain_spec_version"] == "1.1.0"

    def test_zero_tasks_and_no_claim(self):
        adapter = KarkainAdapter()
        status = adapter.kark_bench_status()
        assert status["instantiated_tasks"] == 0
        assert status["capability_claim"] == "none"

    def test_contract_documents_kcc_failure(self):
        adapter = KarkainAdapter()
        availability = adapter.toolchain_availability()
        assert availability["kcc"]["available"] is False
        assert "0xC0000135" in availability["kcc"]["error"]


# --------------------------------------------------------------------------- #
# Experiment framework
# --------------------------------------------------------------------------- #
class TestExperimentFramework:
    def test_requires_three_seeds(self):
        framework = ExperimentFramework()
        framework.register_task_set("synth", [{"id": "t1", "expected": "a"}],
                                    scoring="exact_match")
        with pytest.raises(ValueError):
            framework.run_arm("B", "synth", seeds=[1, 2],
                              model_fn=lambda task, seed: {"answer": "a"})

    def test_arms_use_same_taskset_and_scoring(self):
        framework = ExperimentFramework()
        framework.register_task_set("synth",
                                    [{"id": f"t{i}", "expected": str(i)} for i in range(3)],
                                    scoring="exact_match")
        for arm in ("A", "B", "C"):
            framework.run_arm(arm, "synth", seeds=[1, 2, 3],
                              model_fn=lambda task, seed: {"answer": task["expected"]})
        runs = framework.runs
        assert runs[0]["taskset_sha256"] == runs[1]["taskset_sha256"] == runs[2]["taskset_sha256"]
        assert runs[0]["scoring"] == runs[1]["scoring"] == runs[2]["scoring"]

    def test_frozen_benchmark_id_rejected(self):
        framework = ExperimentFramework()
        bench_ids = frozen_benchmark_problem_ids()
        if bench_ids:
            with pytest.raises(ValueError):
                framework.register_task_set(
                    "contaminated",
                    [{"id": sorted(bench_ids)[0], "expected": "x"}],
                    scoring="exact_match")

    def test_comparison_not_established_without_all_arms(self):
        framework = ExperimentFramework()
        framework.register_task_set("synth",
                                    [{"id": "t1", "expected": "a"}],
                                    scoring="exact_match")
        framework.run_arm("B", "synth", seeds=[1, 2, 3],
                          model_fn=lambda task, seed: {"answer": "a"})
        assert framework.comparison()["verdict"] == "NOT_ESTABLISHED"

    def test_comparison_separated_model_only_tool_assisted(self):
        framework = ExperimentFramework()
        framework.register_task_set("synth",
                                    [{"id": "t1", "expected": "a"}],
                                    scoring="exact_match")
        for arm in ("A", "B", "C"):
            framework.run_arm(arm, "synth", seeds=[1, 2, 3],
                              model_fn=lambda task, seed: {"answer": "a"},
                              tool_assisted=(arm == "C"))
        comparison = framework.comparison()
        for run in framework.runs:
            for result in run["results"]:
                assert result["mode"] in ("model_only", "tool_assisted")


# --------------------------------------------------------------------------- #
# Runtime integration
# --------------------------------------------------------------------------- #
class TestRuntimeIntegration:
    def test_full_session_flow(self, tmp_path):
        runtime = DLCAERuntime(tmp_path / "runtime")
        session_id = runtime.open_session("doc1", "python", PY_FIXTURE,
                                          source="synthetic-fixture")
        result = runtime.run_session(session_id)
        assert result["state"] == "COMPLETED"
        status = runtime.status()
        assert status["canonical_checkpoint"] == "MISSING"
        assert status["training_handoff"] is False
        assert status["persistence_integrity"]["ok"] is True

    def test_learning_candidates_do_not_release_training(self, tmp_path):
        runtime = DLCAERuntime(tmp_path / "runtime")
        session_id = runtime.open_session("doc2", "python", PY_FIXTURE,
                                          source="synthetic-fixture")
        runtime.run_session(session_id)
        candidates = runtime.make_learning_candidates(session_id)
        assert candidates
        handoff = runtime.handoff([c["candidate_id"] for c in candidates])
        assert handoff["authorized"] is False
        assert handoff["released"] == []

    def test_subsystem_flow_demo_deterministic(self, tmp_path):
        runtime = DLCAERuntime(tmp_path / "runtime")
        flow1 = runtime.subsystem_flow_demo("doc3", "python", PY_FIXTURE)
        flow2 = runtime.subsystem_flow_demo("doc4", "python", PY_FIXTURE)
        assert flow1["dlcae"]["observations"] == flow2["dlcae"]["observations"]
        assert flow1["training"] == "DISABLED — no weights touched"
        assert flow1["learning_engine"].startswith("handoff DISABLED")

    def test_sequence_isolated_between_sessions(self, tmp_path):
        runtime = DLCAERuntime(tmp_path / "runtime")
        a = runtime.open_session("iso-a", "python", PY_FIXTURE, source="synthetic-fixture")
        b = runtime.open_session("iso-b", "python", "def broken(:\n", source="synthetic-fixture")
        runtime.run_session(a)
        runtime.run_session(b)
        assert runtime.sequences.status_map()[runtime._require_session(a)["sequence_id"]] == "COMPLETED"
        # the broken document is honestly recorded, not crashed; deeper levels stay BLOCKED
        bundle_b = runtime._require_session(b)["bundle"]
        assert bundle_b.level_status[4] == "FAILED"
        assert bundle_b.level_status[5] == "BLOCKED"
        assert runtime.sequences.status_map()[runtime._require_session(b)["sequence_id"]] == "COMPLETED"


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_benchmark_preserved():
    import hashlib
    digest = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    assert digest == FROZEN_PY_SHA


def test_v180_sft_checkpoint_preserved():
    import hashlib
    digest = hashlib.sha256()
    with (REPO / "runs" / "v180_sft" / "checkpoint_last.pt").open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_roadmap_not_regressed():
    roadmap = (REPO / "configs" / "roadmap" / "taksh-v3-roadmap.yaml").read_text(encoding="utf-8")
    assert "DLCAE" in roadmap


def test_package_imports_cleanly():
    import importlib
    for module in ("taksh.dlcae", "taksh.dlcae.comprehension", "taksh.dlcae.sequences",
                   "taksh.dlcae.knowledge", "taksh.dlcae.registry",
                   "taksh.dlcae.adaptation", "taksh.dlcae.validation",
                   "taksh.dlcae.learning", "taksh.dlcae.persistence",
                   "taksh.dlcae.karkain", "taksh.dlcae.experiments",
                   "taksh.dlcae.runtime"):
        importlib.import_module(module)
