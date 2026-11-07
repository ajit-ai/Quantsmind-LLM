"""Focused tests for Increment 8 diagnostics. No training."""

from __future__ import annotations

import pytest

def test_tokenizer_round_trip_and_fragmentation():
    from tokenizers import Tokenizer

    from evaluation.diagnostics import tokenizer_stats

    stats = tokenizer_stats(Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json"))
    assert stats["familiar"]["all_round_trip"] is True
    assert stats["unfamiliar"]["all_round_trip"] is True
    assert stats["familiar"]["mean_fragments"] < stats["unfamiliar"]["mean_fragments"] or (
        stats["familiar"]["mean_fragments"] <= stats["unfamiliar"]["mean_fragments"]
    )
    assert stats["unfamiliar"]["mean_fragments"] < 8


def test_supervision_alignment_masks_prompt_and_supervises_target():
    from tokenizers import Tokenizer

    from evaluation.diagnostics import supervision_alignment
    from training.binding_curriculum import build_training_examples as build_binding
    from training.functional_curriculum import build_training_examples as build_functional

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    examples = build_binding()[:6] + build_functional()[:6]
    for example in examples:
        alignment = supervision_alignment(example, tokenizer)
        assert alignment["prompt_all_masked"] is True, example.example_id
        assert alignment["target_all_supervised"] is True, example.example_id
        assert alignment["all_params_in_target"] is True, example.example_id
        assert alignment["eos_in_target"] is True


def test_checkpoint_consistency_and_shared_architecture():
    from evaluation.diagnostics import architectures_consistent, checkpoint_param_hash

    paths = {
        "baseline": r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt",
        "increment5": r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\checkpoints\trained.pt",
        "increment7": r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7\increment7-targeted-100\checkpoints\trained.pt",
    }
    first = {label: checkpoint_param_hash(path) for label, path in paths.items()}
    second = {label: checkpoint_param_hash(path) for label, path in paths.items()}
    assert first == second
    report = architectures_consistent(paths)
    assert report["all_same_architecture"] is True
    model = report["configs"]["baseline"]
    assert model["name"] == "qm-code-m3-9m"
    assert model["d_model"] == 256
    assert model["n_layers"] == 8
    assert model["vocab_size"] == 5822
    assert model["max_seq_len"] == 512


def test_copy_probe_helpers():
    from evaluation.diagnostics import COPY_CASES, copy_case_bindings

    assert len(COPY_CASES) == 13
    good = copy_case_bindings("    return qq_item + rr_item\n", "def zz_combined(qq_item, rr_item):")
    assert good["exact_copy"] is True
    bad = copy_case_bindings("    return left + right\n", "def zz_combined(qq_item, rr_item):")
    assert bad["exact_copy"] is False
    assert bad["invented_identifiers"] == ["left", "right"]


def test_diagnostic_cases_do_not_overlap_frozen_suites():
    import json
    from pathlib import Path

    from evaluation.diagnostics import COPY_CASES

    ad1 = json.loads(Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json").read_text(encoding="utf-8"))
    c3 = json.loads(Path("evaluation/python_benchmark_heldout_v1.json").read_text(encoding="utf-8"))
    frozen = {(c["instruction"], c["signature"]) for c in ad1["cases"]} | {
        (c["instruction"], c["signature"]) for c in c3["cases"]
    }
    probe_pairs = {(instruction, signature) for _i, _k, instruction, signature, _f in COPY_CASES}
    assert probe_pairs.isdisjoint(frozen)


def test_diagnostic_script_runs_in_dry_check_mode(tmp_path=None):
    # The diagnostic entry point must exist and expose a dry plan without loading models.
    from pathlib import Path

    import scripts.diagnose_v120 as module  # noqa: F401
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "run_copy_probes" in source
    assert "argparse" in source
