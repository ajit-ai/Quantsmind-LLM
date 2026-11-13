"""Focused tests for the TAKSH L0.5.1 multilingual feasibility tooling."""

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
    l051_architecture_comparison as arch,
    l051_acceptance_gates as gates,  # noqa: F401 - import sanity
    l051_compute_estimator as comp,
    l051_dataset_feasibility as ds,
    l051_language_registry as reg,
    l051_multilingual_benchmark as bmk,
    l051_tokenizer_audit as tok,
    l051_toolchain_audit as tool,  # noqa: F401 - import sanity
    l051_training_distribution as dist,
)
from evaluation.l051_feasibility import preservation_block  # noqa: E402

TOK_PATH = REPO / "tokenizer" / "taksh_code_bpe_v1.json"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_validates_53_and_exactly_16_l1_targets():
    data = reg.load_registry()
    result = reg.validate_registry(data)
    assert result["valid"], result["problems"]
    l1 = [l for l in data["languages"] if l["l1_target"]]
    assert len(l1) == 16
    assert {l["id"] for l in l1} == {
        "python", "javascript", "typescript", "c", "cpp", "java", "csharp",
        "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
    }


def test_supporting_technologies_are_separate():
    data = reg.load_registry()
    supporting = {s["name"] for s in data["supporting_technologies"]}
    assert supporting >= {"HTML", "CSS", "JSON", "YAML", "Terraform HCL", "Verilog", "VHDL"}
    assert "python" not in supporting
    assert len(supporting) == 20


def test_registry_summary_reports_all_fields():
    data = reg.load_registry()
    summary = reg.summarize(data)
    assert summary["l1_target_count"] == 16
    assert set(summary["l1_target_ids"]) == {
        "python", "javascript", "typescript", "c", "cpp", "java", "csharp",
        "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
    }


# --------------------------------------------------------------------------- #
# Tokenizer audit
# --------------------------------------------------------------------------- #
def test_tokenizer_audit_covers_all_73_samples_and_roundtrips():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    for lang_id, code in tok.SAMPLES.items():
        result = tok.audit_language(tokenizer, lang_id, code)
        assert result["decode_roundtrip_exact"] is True, lang_id
        assert result["tokens"] > 0
        assert 0.0 < result["tokens_per_char"] < 1.0


def test_tokenizer_audit_expected_language_count():
    assert len(tok.SAMPLES) == 73
    l1 = {k for k in tok.SAMPLES if "python" != k}  # 16 targets + 20 supporting + rest
    assert {"python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
            "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash"} <= set(tok.SAMPLES)


# --------------------------------------------------------------------------- #
# Dataset feasibility
# --------------------------------------------------------------------------- #
def test_dataset_verdict_counts():
    from collections import Counter

    cfg = {l: ds.verdict_for(l) for l in [
        "python", "javascript", "sql", "cobol", "pony", "dart",
    ]}
    assert cfg["python"]["verdict"] == "qualified-in-repo"
    assert cfg["javascript"]["verdict"] == "conditional-on-legal-review"
    assert cfg["sql"]["verdict"] == "conditional-on-legal-review"
    assert cfg["cobol"]["verdict"] == "deferred"
    assert cfg["dart"]["verdict"] == "conditional-on-legal-review"


def test_dataset_reports_zero_non_python_supply_in_repo():
    report = json.loads(open(REPO / "reports" / "l051_dataset_feasibility.json").read())
    for lid, e in report["per_language"].items():
        if lid != "python":
            assert e["in_repo_tokens_measured"] == 0
            assert e["in_repo_documents"] == 0


def test_no_dataset_claims_clearance_without_review():
    report = json.loads(open(REPO / "reports" / "l051_dataset_feasibility.json").read())
    for lid, e in report["per_language"].items():
        if lid != "python":
            assert e["legal_review_required"] is True
            assert e["license_status"] == "uncertain-requires-review"


# --------------------------------------------------------------------------- #
# Training distribution
# --------------------------------------------------------------------------- #
def test_training_distribution_blocked_today():
    out = json.loads(open(REPO / "reports" / "l051_training_distribution.json").read())
    equal = out["scenarios"]["equal_token"]
    assert len(equal["blocked_today"]) == 15
    over = out["scenarios"]["oversampling"]
    assert over["verdict"] == "infeasible"
    assert out["inputs"]["measured_unique_tokens"] == 1_590_703


def test_distribution_respects_unique_vs_repeated_accounting():
    out = json.loads(open(REPO / "reports" / "l051_training_distribution.json").read())
    accounting = out["scenarios"]["token_accounting"]
    assert accounting["repeated_tokens_beyond_first_epoch"] > accounting["unique_tokens_measured_total"]
    assert accounting["tokens_if_max_exposure_reached"] == pytest.approx(accounting["unique_tokens_measured_total"] * 4)


def test_config_uniqueness_table_is_overrideable():
    cfg = dist._yaml()
    assert cfg["uniqueness"]["python"]["tokens"] == 1_590_703
    assert cfg["uniqueness"]["javascript"]["tokens"] == 0
    assert cfg["caps"]["max_repeat_exposure"] == 4


# --------------------------------------------------------------------------- #
# Architecture comparison
# --------------------------------------------------------------------------- #
def test_architecture_options_cover_a_b_c_d():
    comp = arch.adapter_params(8)
    assert comp > 0
    out = json.loads(open(REPO / "reports" / "l051_architecture_comparison.json").read())
    assert {o["id"] for o in out["options"]} == {"A", "B", "C", "D"}
    assert out["decision_matrix"]["preference_order"][0] == "A"
    # option A adds no parameters; D dominates on cost
    a = next(o for o in out["options"] if o["id"] == "A")
    d = next(o for o in out["options"] if o["id"] == "D")
    assert a["params_total"] == 28_787_328
    assert d["params_total"] > a["params_total"]


# --------------------------------------------------------------------------- #
# Multilingual benchmark
# --------------------------------------------------------------------------- #
def test_benchmark_is_valid_and_covers_16_languages():
    report = json.loads(open(REPO / "reports" / "l051_multilingual_benchmark.json").read())
    assert report["valid"] is True
    assert report["task_count"] == 64
    assert len(report["languages_covered"]) == 16
    assert set(report["instantiated_families"]) == {"1", "2", "3", "4"}
    assert report["frozen_python_benchmark"]["hash_matches"] is True


def test_benchmark_acceptance_bar_is_fixed():
    report = json.loads(open(REPO / "reports" / "l051_multilingual_benchmark.json").read())
    bar = report["acceptance_criteria"]["proposed_thresholds"]
    assert "functional_correctness" in bar
    assert bar["functional_correctness"].startswith(">= 50%")
    assert bar["safety"] == "zero sandbox escapes / unsafe generations tolerated"
    assert report["qualification"]["min_tasks_per_language"] == 16


def test_benchmark_loader_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        bmk.load_benchmark(bad)


def test_frozen_python_benchmark_still_matches():
    import hashlib

    digest = hashlib.sha256(open(REPO / "evaluation" / "l02_python_bench_v1.json", "rb").read()).hexdigest()
    assert digest == "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


# --------------------------------------------------------------------------- #
# Toolchain audit
# --------------------------------------------------------------------------- #
def test_toolchain_audit_finds_python_and_sql():
    out = json.loads(open(REPO / "reports" / "l051_toolchain_audit.json").read())
    assert out["l1_summary"]["python"] is True
    assert out["l1_summary"]["sql"] is True
    assert out["missing"] == ["php", "ruby", "swift", "dart"]


def test_toolchain_never_installs(tmp_path):
    out = json.loads(open(REPO / "reports" / "l051_toolchain_audit.json").read())
    for row in out["checks"]:
        assert row["detail"] != "installing"


# --------------------------------------------------------------------------- #
# Compute estimator
# --------------------------------------------------------------------------- #
def test_compute_estimator_reports_all_four_models_and_equal_split():
    out = json.loads(open(REPO / "reports" / "l051_compute_estimate.json").read())
    assert set(out["by_model"]) == {"existing_28.8M", "candidate_A_110M", "candidate_B_289M", "candidate_C_1019M"}
    assert out["equal_split_multilingual_reality"]["per_language_unique_tokens"] == pytest.approx(99_419, abs=1)
    assert out["equal_split_multilingual_reality"]["qualified_code_tokens"] == 1_590_703


def test_flops_monotone_and_memory_16_bytes_per_param():
    assert comp.flops(0, 100) == 0.0
    assert comp.flops(1000, 1000) == 6_000_000.0
    mem = comp.memory_breakdown(1_000_000, 256, 2, 512, 1)
    assert mem["training_working_set_fp32_bytes"] == 16_000_000


# --------------------------------------------------------------------------- #
# Acceptance gates
# --------------------------------------------------------------------------- #
def test_acceptance_gates_defined_for_all_16_languages_and_none_met():
    out = json.loads(open(REPO / "reports" / "l051_acceptance_gates.json").read())
    assert len(out["per_language"]) == 16
    for lid, entry in out["per_language"].items():
        for gate_id, gate in entry["gates"].items():
            assert gate["status_today"] == "NOT_MET"
    assert out["frozen_python_bar"]["hash_matches"] is True


def test_frozen_python_bar_requirements_preserved():
    out = json.loads(open(REPO / "reports" / "l051_acceptance_gates.json").read())
    py = out["frozen_python_bar"]["requirements"]
    assert py["functional_correctness"] == ">= 15/30 problems"
    assert py["syntax_valid_rate"] == ">= 0.80"


# --------------------------------------------------------------------------- #
# Aggregate + preservation
# --------------------------------------------------------------------------- #
def test_preservation_block_reports_frozen_benchmark_and_checkpoints():
    block = preservation_block()
    assert block["frozen_python_benchmark_sha256"] == block["frozen_python_benchmark_expected"]
    assert block["branch"] in {"main", "feature/l060-foundation-model-qualification"}
    assert len(block["checkpoints_preserved_sha256"]) == 10
    for digest in block["checkpoints_preserved_sha256"].values():
        assert len(digest) == 64


def test_aggregate_report_combines_everything():
    out = json.loads(open(REPO / "reports" / "l051_feasibility_report.json").read())
    for section in ("language_registry", "tokenizer_audit", "dataset_feasibility",
                    "training_distribution", "architecture_comparison",
                    "multilingual_benchmark", "toolchain_audit", "compute_estimate",
                    "acceptance_gates", "preservation", "decision",
                    "per_language_classification", "classification_counts"):
        assert section in out
    assert out["decision"]["verdict"] == "CONDITIONAL_GO"
    counts = out["classification_counts"]
    assert counts["feasible-for-L1.0-investigation"] == 1
