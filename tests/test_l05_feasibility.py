"""Focused tests for the TAKSH L0.5 feasibility gate tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402
    l05_architecture_audit as arch,
    l05_benchmark_audit as bench,
    l05_compute_estimator as comp,
    l05_data_feasibility as data,
    l05_training_strategy as strat,
)
from evaluation.l05_feasibility import preservation_block  # noqa: E402


# --------------------------------------------------------------------------- #
# Architecture audit
# --------------------------------------------------------------------------- #
def test_count_parameters_matches_measured_baseline():
    import yaml

    cfg = yaml.safe_load(open(REPO / "configs/model/code_v160_15m.yaml"))["model"]
    counts = arch.count_parameters(cfg)
    assert counts["total_params"] == 28_787_328
    assert counts["embedding"] == 2_235_648
    assert counts["lm_head"] == 0
    assert counts["per_layer"]["total"] == 2_212_608


def test_count_parameters_grows_with_width_and_depth():
    import yaml

    cfg = yaml.safe_load(open(REPO / "configs/feasibility/l05_scenarios.yaml"))["architecture_scenarios"]
    a = arch.count_parameters({**cfg["A"], "vocab_size": 5822, "tie_embeddings": True, "rope_theta": 10000.0, "norm_eps": 1e-5, "max_seq_len": 2048})
    b = arch.count_parameters({**cfg["B"], "vocab_size": 5822, "tie_embeddings": True, "rope_theta": 10000.0, "norm_eps": 1e-5, "max_seq_len": 2048})
    c = arch.count_parameters({**cfg["C"], "vocab_size": 5822, "tie_embeddings": True, "rope_theta": 10000.0, "norm_eps": 1e-5, "max_seq_len": 2048})
    assert a["total_params"] == 110_658_816
    assert b["total_params"] == 289_115_136
    assert c["total_params"] == 1_018_624_000
    assert a["total_params"] < b["total_params"] < c["total_params"]


def test_analytic_matches_live_model_exactly():
    from training.model import QMModel
    import yaml

    cfg_dict = yaml.safe_load(open(REPO / "configs/model/code_v160_15m.yaml"))["model"]
    live = QMModel(cfg_dict).num_params()
    assert arch.count_parameters(cfg_dict)["total_params"] == live == 28_787_328


def test_scenarios_yaml_declares_required_blocks():
    import yaml

    cfg = yaml.safe_load(open(REPO / "configs/feasibility/l05_scenarios.yaml"))
    for key in ("baseline", "architecture_scenarios", "token_ratios",
                "hardware_classes", "qualified_token_supply", "benchmark"):
        assert key in cfg
    assert cfg["benchmark"]["sha256"] == "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


def test_describe_family_has_expected_keys():
    fam = arch.describe_family({
        "vocab_size": 5822, "d_model": 384, "n_layers": 12, "n_heads": 8,
        "n_kv_heads": 4, "ffn_dim": 1536, "max_seq_len": 2048,
        "rope_theta": 10000.0, "norm_eps": 1e-5,
    })
    for key in ("architecture", "gqa", "ffn_activation", "attention_impl",
                "positional", "loss", "residuals", "initialization"):
        assert key in fam


# --------------------------------------------------------------------------- #
# Data feasibility
# --------------------------------------------------------------------------- #
def test_data_grid_dimensions_and_math():
    scenarios = data._yaml()
    rows = data.grid(scenarios, scenarios["qualified_token_supply"])
    assert len(rows) == 16  # 4 models x 4 ratios
    for r in rows:
        assert r["required_unique_tokens"] == r["params"] * r["tokens_per_param"]
        assert 0.0 < r["code_supply_coverage"] < 1.0
        assert r["epochs_code_only"] >= 1.0
        assert r["storage_bytes_uint16"] == r["required_unique_tokens"] * 2


def test_dataset_register_is_complete_and_nothing_ingested():
    reg = data.DATASET_REGISTER
    assert len(reg) >= 8
    for entry in reg:
        assert "legal_review_required" in entry
        assert entry["ingested_in_l05"] is False
        assert entry["verdict"]
    assert any("EXCLUDED BY DESIGN" in e["verdict"] for e in reg)
    assert any("STRONG candidate" in e["verdict"] for e in reg)


def test_code_supply_coverage_reported_honestly():
    scenarios = data._yaml()
    report = json.loads(open(REPO / "reports" / "l05_data_feasibility.json").read())
    worst = [r for r in report["requirements_grid"] if r["scenario"] == "C"]
    assert all(r["code_supply_coverage"] < 0.001 for r in worst)
    assert report["register_summary"]["zero_ingested_in_l05"] is True
    assert scenarios["qualified_token_supply"]["code_tokens"] == 1_590_703


# --------------------------------------------------------------------------- #
# Training strategy
# --------------------------------------------------------------------------- #
def test_strategy_report_has_all_four_and_verdicts():
    strategies = strat._strategies()
    assert {s["id"] for s in strategies} == {"A", "B", "C", "D"}
    for s in strategies:
        assert s["verdict"] and s["missing_items"] and s["measured_evidence"]


def test_strategy_c_identified_as_only_route_and_blocked():
    out = json.loads(open(REPO / "reports" / "l05_training_strategy.json").read())
    assert "VIABLE ROUTE" in next(s["verdict"] for s in out["strategies"] if s["id"] == "C")
    assert out["comparison_matrix"]


# --------------------------------------------------------------------------- #
# Compute estimator
# --------------------------------------------------------------------------- #
def test_flops_zero_and_positive():
    assert comp.flops(0, 100) == 0.0
    assert comp.flops(1000, 1000) == 6_000_000.0


def test_memory_breakdown_fp32_working_set():
    mem = comp.memory_breakdown(1_000_000, 256, 2, 512, 1)
    assert mem["training_working_set_fp32_bytes"] == 16_000_000
    assert mem["adam_m_v_bytes"] == 8_000_000
    assert mem["checkpoint_model_only_bytes"] == 4_000_000
    assert mem["activation_estimate_bytes"] > 0


def test_hardware_verdict_differs_by_class():
    mem = comp.memory_breakdown(1_018_624_000, 2048, 16, 2048, 1)
    fp32_set = mem["training_working_set_fp32_bytes"]
    act = mem["activation_estimate_bytes"]
    small = comp.hardware_feasibility(fp32_set, act, {"label": "4GB", "vram_gb": 4}, 1_018_624_000)
    large = comp.hardware_feasibility(fp32_set, act, {"label": "80GB", "vram_gb": 80}, 1_018_624_000)
    assert small["fits_training"] is False
    assert large["fits_training"] is True


def test_cpu_duration_estimate_positive_and_order_of_magnitude():
    d = comp.duration_estimates(28_787_328, 1000)
    assert d["cpu_seconds_central"] == round(1000 / 215)
    assert d["cpu_seconds_slow"] > d["cpu_seconds_central"] > d["cpu_seconds_fast"]
    assert d["gpu_duration_uncertainty"]


def test_gpu_price_not_invented_by_default():
    out = json.loads(open(REPO / "reports" / "l05_compute_estimate.json").read())
    assert out["gpu_price_per_hour"] is None


# --------------------------------------------------------------------------- #
# Benchmark audit
# --------------------------------------------------------------------------- #
def test_benchmark_hash_matches_frozen_value():
    out = json.loads(open(REPO / "reports" / "l05_benchmark_audit.json").read())
    assert out["frozen_benchmark"]["hash_matches"] is True
    assert out["frozen_benchmark"]["actual_sha256"] == out["frozen_benchmark"]["expected_sha256"]


def test_structural_audit_passes_and_bar_not_lowered():
    out = json.loads(open(REPO / "reports" / "l05_benchmark_audit.json").read())
    assert out["structural_audit"]["verdict"] == "PASS"
    bar = out["acceptance_criteria_definition"]["bar_for_claiming_coding_capability"]
    assert "functional_correctness" in bar
    assert "syntax_valid_rate" in bar
    assert "repeat_consistency" in bar
    # the bar is not watered down
    assert bar["syntax_valid_rate"].startswith(">=")


def test_serving_separation_is_fully_identical():
    out = json.loads(open(REPO / "reports" / "l05_benchmark_audit.json").read())
    sep = out["serving_separation"]
    assert sep["records_compared"] > 0
    assert sep["identical_raw_vs_serving"] == sep["records_compared"]


def test_sandbox_audit_note_lists_enforcing_tests():
    out = json.loads(open(REPO / "reports" / "l05_benchmark_audit.json").read())
    assert out["sandbox"]["verified_by_tests"] is True
    assert len(out["sandbox"]["enforced_by"]) >= 3


# --------------------------------------------------------------------------- #
# Aggregate report + preservation
# --------------------------------------------------------------------------- #
def test_feasibility_report_aggregates_everything():
    out = json.loads(open(REPO / "reports" / "l05_feasibility_report.json").read())
    for section in ("architecture_audit", "data_feasibility", "training_strategy",
                      "compute_estimate", "benchmark_audit", "preservation", "decision"):
        assert section in out
    assert out["decision"]["verdict"] == "CONDITIONAL_GO"
    assert out["preservation"]["branch"] in {"main", "feature/l060-foundation-model-qualification"}
    assert out["preservation"]["frozen_benchmark_sha256"] == out["preservation"]["frozen_benchmark_expected"]


def test_preservation_block_records_all_ten_checkpoints():
    block = preservation_block()
    for ckpt in (
        "runs/v160_pretrain/checkpoint_last.pt", "runs/v180_sft/checkpoint_last.pt",
        "runs/l02_sft/checkpoint_last.pt", "runs/l02_sft_hp/checkpoint_last.pt",
        "runs/l02_overfit/checkpoint_last.pt", "runs/l03_pretrain/checkpoint_last.pt",
        "runs/l03_pretrain/checkpoint_best.pt", "runs/l04_e1/checkpoint_last.pt",
        "runs/l04_e2/checkpoint_last.pt", "runs/l04_e3/checkpoint_last.pt",
    ):
        assert ckpt in block["checkpoints_preserved_sha256"]
        assert len(block["checkpoints_preserved_sha256"][ckpt]) == 64
    assert block["branch"] in {"main", "feature/l060-foundation-model-qualification"}
    assert block["frozen_benchmark_sha256"] == block["frozen_benchmark_expected"]


def test_decision_lists_missing_authorizations():
    out = json.loads(open(REPO / "reports" / "l05_feasibility_report.json").read())
    assert out["decision"]["missing_authorizations"]
    assert out["decision"]["explicit_non_authorizations"]
    assert out["decision"]["what_would_flip_to_GO"]
    assert out["decision"]["what_would_flip_to_NO_GO"]
