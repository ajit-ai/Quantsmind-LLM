"""Focused tests for Increment 13/14 synthesis and v1.2.3 qualification."""

from __future__ import annotations


def test_operation_synth_covers_new_operations():
    from evaluation.operation_synth import synthesize

    checks = [
        ("Keep the doubled value in a local and return it.", "def d(value_kq):", "propagation_double"),
        ("Filter the evens, then halve their total.", "def h(values_kq):", "filter_sum"),
        ("Return the product modulo three.", "def m(first_kq, second_kq):", "product_modulo"),
        ("Subtract the second value from the first, then double it.", "def b(first_kq, second_kq):", "compose_subtract_double"),
        ("Return the longest word list below.", "def w(words_kq):", "keep_long_words"),
        ("Return True only when both values are positive.", "def b2(first_kq, second_kq):", "both_positive"),
    ]
    for instruction, signature, want in checks:
        result = synthesize(instruction, signature)
        assert result["action"] == "synthesized", instruction
        assert want in (result["operation"] or ""), (instruction, result)


def test_acceptance_suites_deterministic_and_frozen():
    from evaluation.acceptance_suite import build_cases as build_v121
    from evaluation.acceptance_suite_v122 import build_cases as build_v122, suite_digest as digest_v122
    from evaluation.acceptance_suite_v123 import build_cases as build_v123, suite_digest as digest_v123

    assert build_v121() == build_v121()
    assert build_v122() == build_v122()
    assert build_v123() == build_v123()
    assert digest_v123() == digest_v123()
    assert len(build_v123()) == 46
    assert len({case["case_id"] for case in build_v123()}) == 46


def test_acceptance_v123_cases_separate_from_frozen():
    import json
    from pathlib import Path

    from evaluation.acceptance_suite_v123 import build_cases

    ad1 = json.loads(Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json").read_text(encoding="utf-8"))
    c3 = json.loads(Path("evaluation/python_benchmark_heldout_v1.json").read_text(encoding="utf-8"))
    frozen = {(c["instruction"], c["signature"]) for c in ad1["cases"]} | {
        (c["instruction"], c["signature"]) for c in c3["cases"]
    }
    pairs = {(c["instruction"], c["signature"]) for c in build_cases()}
    assert pairs.isdisjoint(frozen)


def test_constrained_decoder_deterministic_and_safe():
    from tokenizers import Tokenizer

    from evaluation.constrained import generate_constrained
    from serving.local_inference import load_model

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    model, _ = load_model(
        r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt", "cpu"
    )
    prompt = "Return the sum of two numbers.\ndef add_values(left_value, right_value):\n"
    first = generate_constrained(model, tokenizer, prompt, enabled=True)
    second = generate_constrained(model, tokenizer, prompt, enabled=True)
    assert first["completion"] == second["completion"]
    assert "interventions" in first
    del model
