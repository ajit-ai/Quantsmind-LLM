"""Focused tests for Increment 11 learning-signal audit. No full training."""

from __future__ import annotations

import pytest
import torch


def test_prediction_position_alignment():
    """Labels predict ids[i+1]; prompt labels masked, target supervised."""
    from tokenizers import Tokenizer

    from training.objective import IGNORE_INDEX
    from training.supervised import build_example, supervise

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    example = build_example(
        example_id="TRACE-1",
        instruction="Return the sum of two numbers.",
        signature="def add_values(left_value, right_value):",
        body="    total = left_value + right_value\n    return total\n",
        operation="add",
        family="arithmetic",
        split="train",
        seed=1,
    )
    supervision = supervise(tokenizer, example)
    ids = supervision["ids"]
    labels = supervision["labels"]
    assert len(labels) == len(ids) - 1
    # Labels equal shifted ids exactly at every supervised position.
    assert all(label == ids[i + 1] for i, label in enumerate(labels) if label != IGNORE_INDEX)
    prompt_len = supervision["prompt_len"]
    # Masked labels: indices 0..prompt_len-2; first supervised predicts first target token.
    assert all(labels[i] == IGNORE_INDEX for i in range(max(0, prompt_len - 1)))
    assert labels[prompt_len - 1] == ids[prompt_len]
    assert supervision["eos_position"] == len(ids) - 1


def test_identifier_weight_positions():
    from tokenizers import Tokenizer

    from training.identifier_supervision import classify_identifiers, loss_weights
    from training.objective import IGNORE_INDEX
    from training.supervised import build_example, supervise

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    example = build_example(
        example_id="TRACE-1",
        instruction="Return the sum of two numbers.",
        signature="def add_values(left_value, right_value):",
        body="    total = left_value + right_value\n    return total\n",
        operation="add",
        family="arithmetic",
        split="train",
        seed=1,
    )
    supervision = supervise(tokenizer, example)
    weighted = loss_weights(supervision, classify_identifiers(example.signature, example.body), tokenizer=tokenizer)
    weights = weighted["weights"]
    prompt_len = supervision["prompt_len"]
    # Weighted positions are all inside the supervised target region.
    boosted = [i for i, label in enumerate(supervision["labels"]) if label != IGNORE_INDEX and weights[i] > 1.0]
    assert boosted
    assert all(i >= prompt_len - 1 for i in boosted)
    assert max(weights) <= 4.0 * max(1.0, weighted["mean_weight"]) + 1e-6
    assert weighted["weighted_tokens"] >= 2


def test_weighted_gradients_finite_and_different():
    from tokenizers import Tokenizer

    from training.identifier_supervision import IdentifierSupervisionConfig
    from training.supervised import build_example
    from training.supervised_train import _batch_tensor, build_batches, weighted_mean_loss
    from serving.local_inference import load_model

    torch.set_num_threads(1)
    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    examples = [
        build_example(
            example_id="G-1",
            instruction="Return the sum of two numbers.",
            signature="def add_values(left_value, right_value):",
            body="    return left_value + right_value\n",
            operation="add",
            family="arithmetic",
            split="train",
            seed=1,
        )
    ]
    plain = build_batches(tokenizer, examples)
    weighted = build_batches(tokenizer, examples, identifier_config=IdentifierSupervisionConfig())
    model, _ = load_model(
        r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt", "cpu"
    )
    model.train()

    def grad_norm(batches, use_weights):
        model.zero_grad()
        for item in batches:
            batch = _batch_tensor([item])
            logits, _ = model(batch.input_ids)
            mean, supervised = weighted_mean_loss(
                logits, batch.labels, item["weights"] if use_weights else None
            )
            mean.backward()
        return sum(
            float((p.grad ** 2).sum()) if p.grad is not None else 0.0
            for p in model.parameters()
        )

    plain_norm = grad_norm(plain, False)
    weighted_norm = grad_norm(weighted, True)
    assert plain_norm > 0 and weighted_norm > 0
    assert weighted_norm != plain_norm
    del model


def test_tiny_experiment_artifacts_exist():
    import json
    from pathlib import Path

    result = json.loads(
        Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment11\tiny-copy-30\metrics\training.json").read_text(encoding="utf-8")
    )
    assert result["steps"] == 30
    assert result["final_loss"] < result["initial_loss"]
    from training.experiment_artifacts import file_sha256

    assert file_sha256(
        Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment11\tiny-copy-30\checkpoints\trained.pt")
    ) == "3f732c0c61c4151e50d425e4a28c6f189246c17e7897b8b3c020980692489de3"
