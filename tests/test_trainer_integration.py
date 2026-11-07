"""Focused tests for Increment 10 trainer integration (WP1/WP2). No training."""

from __future__ import annotations

import pytest
import torch


def _example():
    from training.supervised import build_example

    return build_example(
        example_id="TEST-copy-000",
        instruction="Return the sum of two numbers.",
        signature="def add_values(left_value, right_value):",
        body="    total = left_value + right_value\n    return total\n",
        operation="add",
        family="arithmetic",
        split="train",
        seed=1,
    )


def test_build_batches_without_config_has_no_weights():
    from tokenizers import Tokenizer

    from training.supervised_train import build_batches

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    batches = build_batches(tokenizer, [_example()])
    assert len(batches) == 1
    assert batches[0]["weights"] is None


def test_build_batches_with_config_produces_aligned_weights():
    from tokenizers import Tokenizer

    from training.identifier_supervision import IdentifierSupervisionConfig
    from training.supervised_train import build_batches

    config = IdentifierSupervisionConfig(copy_weight=3.0, propagation_weight=1.5)
    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    batches = build_batches(tokenizer, [_example()], identifier_config=config)
    weights = batches[0]["weights"]
    assert weights is not None
    assert len(weights) == len(batches[0]["labels"])
    assert max(weights) > 1.0
    # Mean-normalized: supervised weights average to 1.0.
    labels = batches[0]["labels"]
    mask = labels != -100
    assert abs(float(weights[mask].mean()) - 1.0) < 1e-5


def test_weights_deterministic_and_single_application():
    from tokenizers import Tokenizer

    from training.identifier_supervision import IdentifierSupervisionConfig
    from training.supervised_train import build_batches

    config = IdentifierSupervisionConfig()
    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    first = build_batches(tokenizer, [_example()], identifier_config=config)[0]["weights"]
    second = build_batches(tokenizer, [_example()], identifier_config=config)[0]["weights"]
    assert torch.equal(first, second)


def test_disabling_weights_reproduces_previous_objective():
    from tokenizers import Tokenizer

    from training.objective import token_mean_loss
    from training.supervised_train import build_batches, weighted_mean_loss

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    plain = build_batches(tokenizer, [_example()])[0]
    logits = torch.randn(1, len(plain["labels"]), 5822)
    mean_plain, count_plain = weighted_mean_loss(logits, plain["labels"], None)
    summed, count = token_mean_loss(logits, plain["labels"])
    assert count == count_plain
    assert torch.allclose(mean_plain, summed / count, atol=1e-6)


def test_weighting_changes_only_copy_positions():
    from tokenizers import Tokenizer

    from training.identifier_supervision import IdentifierSupervisionConfig
    from training.supervised_train import build_batches, weighted_mean_loss

    config = IdentifierSupervisionConfig(copy_weight=4.0)
    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    plain = build_batches(tokenizer, [_example()])[0]
    weighted = build_batches(tokenizer, [_example()], identifier_config=config)[0]
    # Identical ids/labels: only the weights sidecar differs.
    assert torch.equal(plain["labels"], weighted["labels"])
    torch.manual_seed(0)
    logits = torch.randn(1, len(plain["labels"]), 5822)
    mean_plain, _ = weighted_mean_loss(logits, plain["labels"], plain["weights"])
    mean_weighted, _ = weighted_mean_loss(logits, weighted["labels"], weighted["weights"])
    assert mean_plain is not None and mean_weighted is not None
    assert not torch.allclose(mean_plain, mean_weighted)


def test_config_validation():
    from training.identifier_supervision import IdentifierSupervisionConfig

    with pytest.raises(ValueError):
        IdentifierSupervisionConfig(copy_weight=0.5)
    IdentifierSupervisionConfig()


def test_trainer_accepts_identifier_config_parameter():
    import inspect

    from training.continued_training import train_from_checkpoint
    from training.supervised_train import train_supervised

    assert "identifier_config" in inspect.signature(train_from_checkpoint).parameters
    assert "identifier_config" in inspect.signature(train_supervised).parameters
