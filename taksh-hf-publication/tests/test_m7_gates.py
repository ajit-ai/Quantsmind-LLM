"""Negative-path tests for M7 gates and evidence handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import m7_gates


THRESHOLDS = {
    "syntax_rate_min": 0.50,
    "min_locked_generation": 20,
    "functional_pass_min": 0.50,
    "min_functional_tasks": 8,
    "min_absolute_gain": 0.10,
    "category_count_for_level4": 2,
    "independent_review_required": True,
}


def _summary(candidate_syntax=1.0, candidate_functional=1.0):
    candidate = {
        "model": "m7_candidate",
        "partitions": {
            "locked": {
                "tasks": {
                    "code_generation": {
                        "attempted": 40,
                        "syntax_rate": candidate_syntax,
                        "functional_attempted": 8,
                        "functional_passed": int(candidate_functional * 8),
                        "functional_rate": candidate_functional,
                    },
                }
            }
        },
    }
    baseline = {
        "model": "m3_base",
        "partitions": {
            "locked": {
                "tasks": {
                    "code_generation": {
                        "attempted": 40,
                        "syntax_rate": 0.0,
                        "functional_attempted": 8,
                        "functional_passed": 0,
                        "functional_rate": 0.0,
                    },
                }
            }
        },
    }
    return {"models": [candidate, baseline]}


def _losses(candidate=6.0, m3=5.0, m4=5.1, untrained=8.7):
    return {
        "losses": {
            "untrained": {"loss": untrained},
            "m3_base": {"loss": m3},
            "m4_tuned": {"loss": m4},
            "m7_candidate": {"loss": candidate},
        }
    }


def test_gate_g_missing_summary_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = m7_gates.gate_g_evidence({"thresholds": THRESHOLDS})

    assert result["ok"] is False
    assert result["status"] == "NOT_RUN"
    assert result["evidence_level"] == 0
    assert "evaluation summary missing" in result["problems"]


def test_gate_g_green_rates_cannot_override_worse_validation_loss(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_dir = Path("runs/m7_eval")
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(_summary(candidate_syntax=1.0, candidate_functional=1.0)),
        encoding="utf-8",
    )
    monkeypatch.setattr(m7_gates, "common_validation_losses",
                        lambda protocol: _losses(candidate=6.0))

    result = m7_gates.gate_g_evidence({"thresholds": THRESHOLDS})

    assert result["ok"] is False
    assert result["evidence_level"] == 0
    assert result["status"] == "TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED"
    assert "did not beat every control" in " ".join(result["rationale"])


def test_gate_g_does_not_claim_level_five_without_human_review(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = _summary()
    summary["models"][0]["partitions"]["locked"]["tasks"].update({
        "sql": {"sql_rate": 1.0, "sql_attempted": 4},
        "json_output": {"structured_rate": 1.0, "structured_attempted": 4},
    })
    summary_dir = Path("runs/m7_eval")
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(m7_gates, "common_validation_losses",
                        lambda protocol: _losses(candidate=4.0))

    result = m7_gates.gate_g_evidence({"thresholds": THRESHOLDS})

    assert result["evidence_level"] == 4
    assert "level 5" in " ".join(result["rationale"])


def test_gate_f_missing_manifests_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protocol = {"historical_checkpoint_sha256": {}, "checkpoints": {}}

    result = m7_gates.gate_f_training(protocol)

    assert result["ok"] is False
    assert any("missing run manifest" in problem for problem in result["problems"])


def test_gate_f_failed_run_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failed = {
        "status": "failed",
        "success": False,
        "training": {
            "steps_completed": 3,
            "planned_steps": 10,
            "metrics": {"final_train_loss": 9.0},
            "history": [{"train_loss": 9.0}],
        },
        "checkpoint": {"path": "runs/missing.pt"},
        "dataset": {},
    }
    monkeypatch.setattr(
        m7_gates, "_run_manifest",
        lambda name: (failed, Path(f"runs/{name}/run_manifest.json")),
    )
    protocol = {
        "historical_checkpoint_sha256": {},
        "checkpoints": {},
        "dataset": {"manifest": "data/m7/m7-code-supervised-v0.manifest.json"},
    }

    result = m7_gates.gate_f_training(protocol)

    assert result["ok"] is False
    assert any("status=failed" in problem for problem in result["problems"])
    assert any("steps did not complete" in problem
               for problem in result["problems"])


def test_resource_gate_rejects_blocked_preflight(tmp_path, monkeypatch):
    report = {
        "verdict": "BLOCKED",
        "hard_gate_failures": ["train artifact missing"],
        "runtime_estimate_seconds": 10_000,
        "memory_required_bytes": 2_000,
        "memory_budget_bytes": 1_000,
        "memory_total_budget_bytes": 1_000,
        "system": {"disk": {"free_bytes": 10 << 30}},
        "model": {"params": 9_359_104},
        "probe": {"peak_rss_bytes": 2_000, "tokens_per_second": 500},
    }
    monkeypatch.setattr("training.preflight.run_preflight",
                        lambda *args, **kwargs: report)

    result = m7_gates.gate_d_resources({}, tmp_path / "M7_PREFLIGHT.json")

    assert result["ok"] is False
    assert any("verdict: BLOCKED" in problem for problem in result["problems"])
    assert any("exceeds 90 min" in problem for problem in result["problems"])
    assert any("exceeds stable machine budget" in problem
               for problem in result["problems"])


def test_m5_benchmark_is_unchanged():
    protocol = m7_gates.load_protocol()
    if not Path(protocol["benchmark_integrity"]["m5_benchmark"]).exists():
        pytest.skip("M5 benchmark data is not part of the source release")

    problems, m5 = m7_gates.m5_integrity(protocol)

    assert problems == []
    assert m5["unchanged"] is True
    assert m5["computed_hash"] == m5["expected_hash"]


def test_changed_m5_hash_is_rejected():
    protocol = m7_gates.load_protocol()
    if not Path(protocol["benchmark_integrity"]["m5_benchmark"]).exists():
        pytest.skip("M5 benchmark data is not part of the source release")
    protocol["benchmark_integrity"]["m5_expected_sha256"] = "0" * 64

    problems, m5 = m7_gates.m5_integrity(protocol)

    assert m5["unchanged"] is False
    assert any("M5 benchmark hash changed" in problem for problem in problems)
