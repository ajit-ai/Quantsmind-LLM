"""Focused tests for Increment 13 integrated generation. No training."""

from __future__ import annotations


def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")


def test_infer_relation_positional_unique():
    from evaluation.hybrid import infer_relation

    relation = infer_relation("Return the sum.", "return a + b", ["a", "b"], ["left_value", "right_value"])
    assert relation["mapping"] == {"a": "left_value", "b": "right_value"}


def test_infer_relation_first_hint():
    from evaluation.hybrid import infer_relation

    relation = infer_relation("Pick the first input.", "return chosen", ["chosen"], ["alpha_value", "beta_value"])
    assert relation["mapping"] == {"chosen": "alpha_value"}


def test_infer_relation_ambiguous_abstains():
    from evaluation.hybrid import infer_relation

    relation = infer_relation("Do something.", "return x", ["x"], ["alpha_value", "beta_value"])
    assert relation["mapping"] == {}


def test_hybrid_generator_disabled_is_model_only():
    class FakeModel:
        def eval(self):
            pass

    class FakeTok:
        def __init__(self):
            from tokenizers import Tokenizer

            self.inner = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")

        def encode(self, text, add_special_tokens=True):
            return self.inner.encode(text, add_special_tokens=False)

        def id_to_token(self, ident):
            return self.inner.id_to_token(ident)

        def decode(self, ids):
            return self.inner.decode(ids)

    from evaluation.hybrid import HybridGenerator

    # Disabled path must still produce text without invoking copy logic.
    result = HybridGenerator(FakeModel(), FakeTok(), enabled=False)
    # generate() needs model/torch; exercise the pure pieces instead.
    assert result.enabled is False
    from evaluation.hybrid import analyse_names, rewrite_body

    analysis = analyse_names("def add_values(left_value, right_value):", "    return left + right\n")
    assert analysis["invented"] == ["left", "right"]
    assert rewrite_body("    return left + right\n", {"left": "left_value", "right": "right_value"}) == "    return left_value + right_value\n"


def test_generalization_suite_deterministic():
    from evaluation.generalization_suite import build_cases, suite_digest

    assert build_cases() == build_cases()
    assert suite_digest() == suite_digest()
    assert len(build_cases()) >= 20
    behaviors = {case["kind"] for case in build_cases()}
    assert len(behaviors) >= 8


def test_generalization_reference_answers_correct():
    from evaluation.functional_eval import evaluate_functional
    from evaluation.generalization_suite import build_cases

    for case in build_cases():
        result = evaluate_functional(
            case["signature"] + "\n" + case["reference_body"] + "\n",
            case["function_name"],
            case["tests"],
        )
        assert result.passed is True, (case["case_id"], result)


def test_hybrid_scores_on_reference_completions():
    from evaluation.hybrid import analyse_names, infer_relation, rewrite_body

    instruction = "Return the sum of two numbers."
    signature = "def add_values(left_value, right_value):"
    body = "    return left + right\n"
    analysis = analyse_names(signature, body)
    relation = infer_relation(instruction, body, analysis["invented"], ["left_value", "right_value"])
    assert rewrite_body(body, relation["mapping"]) == "    return left_value + right_value\n"
