"""Focused tests for Increment 13/14 synthesis and constrained decoding."""

from __future__ import annotations


def test_synthesis_covers_new_operations():
    from evaluation.operation_synth import synthesize

    checks = [
        ("Store the amount, then return it doubled.", "def p(amount_kq):", "propagation_double"),
        ("Return True only when both values are positive.", "def b(first_kq, second_kq):", "both_positive"),
        ("Filter the evens, then halve their total.", "def h(values_kq):", "filter_sum"),
        ("Return the product modulo three.", "def m(first_kq, second_kq):", "product_modulo"),
        ("Sort the list in ascending order.", "def f(values_kq):", "sort_values"),
        ("Return the second entry of the list.", "def s(values_kq):", "second_entry"),
    ]
    for instruction, signature, want in checks:
        result = synthesize(instruction, signature)
        assert result["action"] == "synthesized", instruction
        assert want in (result["operation"] or ""), (instruction, result)


def test_synthesis_abstains_on_unsupported():
    from evaluation.operation_synth import synthesize

    result = synthesize("Cryptographically hash the input.", "def h(value_kq):")
    assert result["action"] == "abstain"


def test_constrained_state_balance_and_suppression():
    from evaluation.constrained import ConstraintState, apply_constraints

    import torch

    state = ConstraintState()
    state.update("([")
    assert state.open_depth == 2
    logits = torch.zeros(1, 10)
    logits[0, 7] = 5.0  # eos id stand-in
    filtered, interventions = apply_constraints(logits, state, eos_id=7)
    assert interventions == ["eos_suppressed_unbalanced"]
    assert filtered[7] == float("-inf")


def test_constrained_decoder_deterministic_and_safe():
    from tokenizers import Tokenizer

    from evaluation.constrained import generate_constrained
    from serving.local_inference import load_model

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    model, _ = load_model(
        r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt", "cpu"
    )
    prompt = "Return the sum of two numbers.\ndef add_values(left_value, right_value):\n"
    first = generate_constrained(model, tokenizer, prompt, max_new_tokens=24, enabled=True)
    second = generate_constrained(model, tokenizer, prompt, max_new_tokens=24, enabled=True)
    assert first["completion"] == second["completion"]
    assert first["constrained"] is True
    del model


def test_constrained_disabled_matches_unconstrained_baseline():
    from tokenizers import Tokenizer

    from evaluation.constrained import generate_constrained
    from serving.local_inference import generate_ids, load_model

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    model, _ = load_model(
        r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt", "cpu"
    )
    prompt = "Return the sum of two numbers.\ndef add_values(left_value, right_value):\n"
    constrained_off = generate_constrained(model, tokenizer, prompt, max_new_tokens=24, enabled=False)
    raw = generate_ids(
        model, tokenizer, prompt, max_new_tokens=24, temperature=0.0,
        top_k=1, top_p=1.0, device="cpu", seed=20261023, binding_repair=False,
    )
    assert constrained_off["completion"] == raw["text"]
    del model
