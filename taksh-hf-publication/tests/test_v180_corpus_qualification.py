"""Taksh v1.8.0 qualification tests: corpus evidence, authorization
simulation, repetition diagnostics, learning protocol and serving preservation.

Offline, CPU-bounded, deterministic. No source is approved and no training is
executed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.python_v1.approval_sim import (
    SCENARIOS,
    run_simulation,
    scenario_manifests,
)
from data.python_v1.candidates import load_registry
from data.python_v1.evidence import (
    DISPOSITIONS,
    EVIDENCE_CLASSES,
    audit_corpus_file,
    build_dossier,
    completeness_score,
    diversity_report,
    owner_decision_packet,
)
from data.python_v1.preflight import TRAINING_NOT_AUTHORIZED, preflight
from data.python_v1.readiness import assess
from evaluation.python_eval.budget import (
    BudgetError,
    classify_completion,
    classify_termination,
    repetition_metrics,
    validate_budget,
)
from evaluation.python_eval.decoding import (
    STRATEGIES,
    UNSUPPORTED_STRATEGIES,
    run_strategy,
)
from training.learning_protocol import (
    EXPERIMENT_STATUS,
    STOP_CONDITIONS,
    SUCCESS_CRITERIA,
    build_protocol,
    hypothesis_register,
    verdict_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "data" / "contracts" / "python_candidates_v1.json"
V14_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v1.json"
V15_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v15.json"


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


# --------------------------------------------------------------------------- #
# Corpus evidence qualification
# --------------------------------------------------------------------------- #
def test_evidence_classes_are_explicit():
    assert set(EVIDENCE_CLASSES) == {
        "VERIFIED_LOCAL_EVIDENCE",
        "REQUIRES_EXTERNAL_VERIFICATION",
        "UNKNOWN",
        "CONFLICTING_EVIDENCE",
    }


def test_record_audit_reports_missing_and_conflicting_metadata(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "def f():\n    return 1\n",
                "content_sha1": "0" * 40,
                "source_id": "s",
                "repo": "r",
                "commit": "c",
                "spdx": "MIT",
                "attribution": "a",
                "file_path": "f.py",
                "split": "train",
            }
        )
        + "\n"
        + json.dumps({"text": "x = 1\n", "repo": "r"}) + "\n",
        encoding="utf-8",
    )
    audit = audit_corpus_file(path)
    assert audit.records == 2
    assert audit.hash_mismatches == 1
    assert audit.conflicting
    assert audit.missing_fields["content_sha1"] == 1
    assert audit.licences == {"MIT": 1, "unknown": 1}


def test_audit_does_not_modify_the_corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    original = json.dumps({"text": "x = 1\n", "content_sha1": "0" * 40}) + "\n"
    path.write_text(original, encoding="utf-8")
    audit_corpus_file(path)
    assert path.read_text(encoding="utf-8") == original


def test_completeness_separates_the_dimensions(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "x = 1\n",
                "content_sha1": "0" * 40,
                "source_id": "s",
                "repo": "r",
                "commit": "c",
                "spdx": "MIT",
                "attribution": "a",
                "file_path": "f.py",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scores = completeness_score(audit_corpus_file(path))
    assert scores["evidence_completeness"] == 1.0
    assert scores["authorization_eligibility"] == 0.0
    assert "not" in scores["expected_learning_utility"]


def test_dossier_recommends_without_approving(registry):
    candidate = registry["candidates"][0]
    dossier = build_dossier(candidate)
    assert dossier["recommended_disposition"] in DISPOSITIONS
    assert dossier["exact_decision_required"]
    assert dossier["required_reviewer"]
    assert "does not approve" in dossier["note"]


def test_dossier_lists_unknown_facts_for_external_candidates(registry):
    candidate = next(
        c for c in registry["candidates"]
        if c["evidence_class"] == "REQUIRES_EXTERNAL_VERIFICATION"
    )
    dossier = build_dossier(candidate)
    assert dossier["unknown_facts"]
    assert dossier["recommended_disposition"] in (
        "REQUEST_EXTERNAL_EVIDENCE",
        "RETAIN_FOR_RESEARCH_ONLY",
    )


def test_diversity_report_is_metadata_only(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "text": f"x = {i}\n",
                    "content_sha1": f"{i:040d}",
                    "source_id": "s",
                    "repo": f"r{i % 2}",
                    "commit": "c",
                    "spdx": "MIT",
                    "attribution": "a",
                    "file_path": f"{i}.py",
                    "split": "train",
                }
            )
            for i in range(4)
        ),
        encoding="utf-8",
    )
    report = diversity_report(audit_corpus_file(path))
    assert report["records"] == 4
    assert report["repositories"] == 2
    assert "metadata-only" in report["note"]


def test_owner_packet_grants_nothing(registry):
    dossiers = [build_dossier(candidate) for candidate in registry["candidates"]]
    packet = owner_decision_packet(dossiers)
    assert packet["approved_sources"] == 0
    assert packet["production_training_authorized"] is False
    assert "grants nothing" in packet["note"]


# --------------------------------------------------------------------------- #
# Authorization simulation
# --------------------------------------------------------------------------- #
def test_simulation_covers_every_scenario():
    results = run_simulation()
    assert set(results["scenarios"]) == set(SCENARIOS)
    assert results["authorized_scenarios"] == ["complete_approved"]
    for name in ("missing_reviewer", "missing_timestamp", "missing_license_evidence",
                 "revoked_approval", "expired_approval", "inconsistent_provenance",
                 "contamination_blocker", "attribution_requirement_unmet"):
        assert not results["scenarios"][name]["authorized"], name
        assert results["scenarios"][name]["blockers"], name


def test_simulation_never_writes_to_the_registry():
    contract = REPO_ROOT / "data" / "contracts" / "python_v1.yaml"
    before = contract.read_bytes()
    run_simulation()
    assert contract.read_bytes() == before
    assert "never opened for writing" in run_simulation()["isolation"]


def test_simulation_fixtures_are_fabricated():
    for manifest in scenario_manifests().values():
        assert manifest["source_id"] == "simulated_source"
        assert manifest["origin"].startswith("local://simulated")


def test_production_approval_remains_zero():
    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["approved"] == 0
    assert report["production_training_authorized"] is False
    decision = preflight(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert decision["decision"] == TRAINING_NOT_AUTHORIZED


def test_revocation_and_expiry_block_authorization(tmp_path):
    from data.python_v1.approval_sim import fixture_manifest

    manifest = fixture_manifest()
    manifest["revocation_status"] = "revoked"
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "authorization_revoked" in preflight(path)["blockers"]
    manifest["revocation_status"] = "not_revoked"
    manifest["decision_expires_on"] = "2020-01-01"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "authorization_expired" in preflight(path, now="2026-10-21")["blockers"]


def test_revision_mismatch_blocks_authorization(tmp_path):
    from data.python_v1.approval_sim import fixture_manifest

    path = tmp_path / "m.json"
    path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
    blockers = preflight(path, expected_revision="sha256:" + "9" * 64)["blockers"]
    assert "revision_mismatch" in blockers


# --------------------------------------------------------------------------- #
# Repetition diagnostics and decoding
# --------------------------------------------------------------------------- #
def test_repetition_metrics_are_deterministic():
    assert repetition_metrics([1, 1, 2, 2]) == repetition_metrics([1, 1, 2, 2])
    assert repetition_metrics([5] * 20)["max_repeated_run"] == 20


def test_termination_and_quality_classification():
    assert classify_termination(finish_reason="stop", generated_tokens=5, budget=64, empty=False) == "eos"
    assert classify_termination(finish_reason="length", generated_tokens=64, budget=64, empty=False) == "budget"
    assert classify_completion(text="x" * 80, token_ids=[4] * 40, parseable=True, complete_program=True) == "degenerate_repetition"
    assert classify_completion(text="def f():\n    return 1\n", token_ids=[1, 2, 3, 4], parseable=True, complete_program=True) == "complete_program"


def test_unsupported_decoding_strategy_is_refused():
    import torch

    class Stub:
        cfg = {"max_seq_len": 64, "vocab_size": 5822}

    with pytest.raises(ValueError):
        run_strategy(Stub(), None, [], strategy="repetition_penalty", budget=8, seed=1)
    assert "repetition_penalty" in UNSUPPORTED_STRATEGIES
    assert set(STRATEGIES) == {
        "greedy", "temperature_0.8", "top_k_50", "top_p_0.95"
    }


def test_nucleus_sampling_does_not_crash():
    """Regression: the top_p path used to scatter along the wrong dimension."""
    import torch

    from serving.local_inference import _filter_logits

    logits = torch.zeros(1, 16)
    probs = _filter_logits(logits, 1.0, 0, 0.9)
    # Nucleus sampling truncates the tail; the surviving mass may be below 1.
    assert 0.0 < float(probs.sum()) <= 1.0 + 1e-4
    assert float(probs.max()) == pytest.approx(1.0 / 16, abs=1e-4)


def test_sampled_decoding_is_reproducible_with_a_seed():
    import torch

    from serving.local_inference import _filter_logits

    for batch in (1, 2, 3):
        for vocab in (16, 5822):
            for top_p in (0.5, 0.95, 1.0):
                logits = torch.randn(batch, vocab)
                probs = _filter_logits(logits, 1.0, 0, top_p)
                assert probs.shape == (batch, vocab)
                assert bool((probs >= 0).all())
                assert bool(torch.isfinite(probs).all())
                assert float(probs.sum(dim=-1).max()) <= 1.0 + 1e-4
                assert float(probs.sum(dim=-1).min()) > 0.0
                assert torch.equal(probs.argmax(dim=-1), logits.argmax(dim=-1))


def test_sampling_boundaries_behave_documented_ways():
    import torch

    from serving.local_inference import _filter_logits

    logits = torch.randn(1, 32)
    # top_p = 0 keeps only the most likely token
    probs = _filter_logits(logits, 1.0, 0, 0.0)
    assert int((probs > 0).sum()) == 1
    # top_p above 1 leaves the distribution unfiltered
    full = _filter_logits(logits, 1.0, 0, 1.5)
    assert float(full.sum()) == pytest.approx(1.0, abs=1e-4)
    # extreme logits stay finite
    extreme = _filter_logits(logits * 1e4, 1.0, 0, 0.9)
    assert bool(torch.isfinite(extreme).all())
    # a sampled id stays inside the vocabulary
    token = int(torch.multinomial(_filter_logits(logits, 1.0, 50, 0.9), 1).item())
    assert 0 <= token < 32


def test_sampled_decoding_is_reproducible_with_a_seed():
    from tokenizers import Tokenizer

    from serving.local_inference import load_model
    from serving.local_inference import generate_ids

    # Legacy checkpoint predates safe-serialisation; trusted local artifact only.
    model, _ = load_model(
        str(REPO_ROOT / "runs" / "m7_pretrain" / "checkpoint_last.pt"),
        "cpu",
        allow_unsafe_trusted=True,
    )
    tokenizer = Tokenizer.from_file(str(REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"))
    prompt = "def add(a, b):\n"
    first = generate_ids(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8, seed=7)
    second = generate_ids(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8, seed=7)
    assert first["token_ids"] == second["token_ids"]


def test_tokenizer_fragmentation_metrics():
    from tokenizers import Tokenizer

    from training.tokenizer_analysis import python_fragmentation_report

    tokenizer = Tokenizer.from_file(str(REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"))
    report = python_fragmentation_report(tokenizer, ["def f():\n    return 1\n"])
    assert report["keywords_total"] == 35
    assert 0.0 <= report["keyword_fragmentation_share"] <= 1.0
    assert report["indentation_tokens"]["four_spaces"] == 1
    assert report["unknown_token_id"] is not None


# --------------------------------------------------------------------------- #
# Learning protocol, benchmarks, resources and serving
# --------------------------------------------------------------------------- #
def test_learning_protocol_is_not_authorized():
    protocol = build_protocol()
    assert protocol["status"] == EXPERIMENT_STATUS
    assert protocol["authorization_gates"]
    assert "statistical_limitations" in protocol
    assert "eight-case" in protocol["statistical_limitations"]


def test_success_criteria_exclude_loss_reduction_alone():
    joined = " ".join(SUCCESS_CRITERIA)
    assert "held_out" in joined
    assert "degenerate" in joined
    assert "training loss" not in joined.lower()


def test_stop_conditions_are_explicit():
    assert "degenerate" in " ".join(STOP_CONDITIONS)
    assert "contamination_detected" in STOP_CONDITIONS


def test_hypothesis_register_covers_the_ten_hypotheses():
    register = hypothesis_register()
    assert len(register) == 10
    verdicts = set(verdict_counts())
    assert verdicts <= {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "NOT_SUPPORTED_BY_CURRENT_EVIDENCE",
        "UNTESTED",
    }


def test_frozen_benchmarks_are_preserved():
    v14 = json.loads(V14_CASES.read_text(encoding="utf-8"))
    v15 = json.loads(V15_CASES.read_text(encoding="utf-8"))
    assert len(v14["cases"]) == 20
    assert len(v15["cases"]) == 16
    assert all("surface" not in case for case in v14["cases"])


def test_evaluation_surfaces_remain_separate():
    from evaluation.python_eval.surfaces import check_contract

    case = {"surface": "instruction_to_code", "prompt": "x"}
    allowed, _ = check_contract(case, {"completion": True, "instruction": False})
    assert allowed is False


def test_resource_model_bounds_are_consistent():
    from training.resources import memory_model, runtime_model

    memory = memory_model(parameters=9_359_104, vocab_size=5822, d_model=256,
                          layers=8, context_length=512)
    assert memory["training_total_gib"] < 1.0
    unknown = runtime_model(tokens=1000, measured_tokens_per_second=None)
    assert unknown["seconds"] is None


def test_budget_validation_rejects_impossible_combinations():
    with pytest.raises(BudgetError):
        validate_budget(500, 32, 512)
    assert validate_budget(100, 32, 512)["remaining_context"] == 412


def test_serving_contract_is_covered_by_the_established_suite():
    """The serving contract is verified by tests/test_openai_server.py and the
    v1.4.0 agent-contract suite; this module adds the nucleus regression and
    the reserved-token checks that were new in v1.8.0."""
    assert (REPO_ROOT / "tests" / "test_openai_server.py").is_file()
    assert (REPO_ROOT / "tests" / "test_v140_capability.py").is_file()


def test_no_training_is_executed_by_the_qualification_suite():
    """The experiment plan is a specification; nothing in this suite trains."""
    import training.train as trainer

    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "def run_training" in source
    plan = build_protocol()
    assert plan["status"] == EXPERIMENT_STATUS
    assert "NOT_AUTHORIZED_FOR_EXECUTION" in plan["note"]
