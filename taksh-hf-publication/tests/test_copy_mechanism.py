"""Focused tests for Increment 13 copy mechanism. No training."""

from __future__ import annotations


def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")


def test_extract_source_identifiers_familiar_and_unfamiliar():
    from evaluation.copy_mechanism import extract_source_identifiers

    tokenizer = _tokenizer()
    familiar = extract_source_identifiers("Return the sum.\ndef add_values(left_value, right_value):", tokenizer)
    assert [c.name for c in familiar] == ["left_value", "right_value"]
    assert all(c.valid and c.token_ids for c in familiar)
    unfamiliar = extract_source_identifiers("Copy the input.\ndef zz_copy(probe_00):", tokenizer)
    assert [c.name for c in unfamiliar] == ["probe_00"]
    assert unfamiliar[0].token_ids
    assert unfamiliar[0].token_count if hasattr(unfamiliar[0], "token_count") else len(unfamiliar[0].token_ids) >= 2


def test_extraction_excludes_keywords_builtins_literals():
    from evaluation.copy_mechanism import extract_source_identifiers

    tokenizer = _tokenizer()
    candidates = extract_source_identifiers("def f(len):", tokenizer)
    assert "len" not in [c.name for c in candidates] or not candidates[0].valid
    none = extract_source_identifiers("Return the sum.\n", tokenizer)
    assert none == []


def test_familiar_copy():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    gate = CopyGate()
    result = copy_generate(
        gate, "Return the sum of two numbers.",
        "def add_values(left_value, right_value):",
        "    return left + right\n", tokenizer,
    )
    assert result["decision"]["action"] == "substitute"
    assert result["rewritten"] == "    return left_value + right_value\n"


def test_unfamiliar_copy():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(), "Copy the input back.",
        "def zz_copy(probe_00):",
        "    return value\n", tokenizer,
    )
    assert result["rewritten"] == "    return probe_00\n"


def test_two_variable_selection():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(), "Return the difference between the two values.",
        "def sub_values(customer_total, invoice_total):",
        "    return alpha - beta\n", tokenizer,
    )
    assert result["decision"]["mapping"] == {"alpha": "customer_total", "beta": "invoice_total"}


def test_substitution_hint_first():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(), "Pick the first input.",
        "def pick_it(alpha_value, beta_value):",
        "    return chosen\n", tokenizer,
    )
    assert result["decision"]["mapping"] == {"chosen": "alpha_value"}


def test_propagation_action():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(), "Store then return.",
        "def carry_it(source_value):",
        "    result = source_value\n    return result\n", tokenizer,
    )
    assert result["decision"]["action"] == "propagate"
    assert result["changed"] is False


def test_repeated_occurrences_rewrite_all():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(), "Add them together.",
        "def adds_pair(probe_00, segment_00):",
        "    return a + b + a\n", tokenizer,
    )
    assert result["rewritten"] == "    return probe_00 + segment_00 + probe_00\n"


def test_new_identifier_generation_abstains_safely():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    # No source params available in signature context for this call shape.
    result = copy_generate(
        CopyGate(), "Return something new.", "def make_it():",
        "    return novel_thing\n", tokenizer,
    )
    assert result["changed"] is False
    assert result["decision"]["action"] in ("generate", "abstain")


def test_invalid_candidate_rejection_and_ambiguity_abstain():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    # One invented name, two unused params, no relation hint: positional
    # mapping applies only when counts match, so this abstains.
    result = copy_generate(
        CopyGate(), "Do the computation.",
        "def compute_it(alpha_value, beta_value):",
        "    return gamma\n", tokenizer,
    )
    assert result["changed"] is False
    assert result["decision"]["action"] == "abstain"


def test_gate_can_be_disabled():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    result = copy_generate(
        CopyGate(enabled=False), "Return the sum.",
        "def add_values(left_value, right_value):",
        "    return left + right\n", tokenizer,
    )
    assert result["changed"] is False
    assert result["decision"]["action"] == "generate"


def test_adversarial_plausible_identifier_not_substituted():
    from evaluation.copy_mechanism import CopyGate, copy_generate

    tokenizer = _tokenizer()
    # A plausible completion already correct: it uses source names, so the
    # gate must not rename anything.
    result = copy_generate(
        CopyGate(), "Return the sum.",
        "def add_values(left_value, right_value):",
        "    return left_value + right_value\n", tokenizer,
    )
    assert result["changed"] is False
    assert result["decision"]["action"] == "copy"
