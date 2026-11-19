"""Tests for M3 validation, checkpoint resume and manifest behaviour."""

from __future__ import annotations

import json
import math

import pytest
import torch
from tokenizers import Tokenizer

from training.manifest import build_run_manifest, sha256_file
from training.validation import assert_separate, evaluate, perplexity_from_loss

TINY_MODEL = {
    "name": "tiny", "vocab_size": 64, "d_model": 32, "n_layers": 2,
    "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 32,
    "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True,
}


def make_tokens(tmp_path, name="t.tokens.bin", n=4096):
    import numpy as np

    path = tmp_path / name
    rng = np.random.default_rng(0)
    np.asarray(rng.integers(0, 64, size=n), dtype=np.uint16).tofile(path)
    path.with_suffix(".meta.json").write_text(json.dumps({
        "schema": "quantsmind.token_stream/1", "vocab_size": 64,
        "num_tokens": n, "dtype": "uint16",
    }), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Perplexity honesty
# --------------------------------------------------------------------------- #
def test_perplexity_is_null_above_the_uniform_baseline():
    """exp(loss) above ln(vocab) would imply a model better than chance."""
    ppl, note = perplexity_from_loss(9.0, 5822)
    assert ppl is None
    assert "uniform-random baseline" in note


def test_perplexity_is_null_for_non_finite_loss():
    ppl, note = perplexity_from_loss(float("nan"), 100)
    assert ppl is None
    assert "not finite" in note


def test_perplexity_is_reported_when_meaningful():
    ppl, note = perplexity_from_loss(3.0, 5822)
    assert ppl == pytest.approx(math.exp(3.0))
    assert note is None


# --------------------------------------------------------------------------- #
# Validation must not touch weights
# --------------------------------------------------------------------------- #
def test_evaluate_does_not_change_weights(tmp_path):
    from training.data import TokenDataset
    from training.model import QMModel

    dataset = TokenDataset(make_tokens(tmp_path), seed=1)
    model = QMModel(TINY_MODEL)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    result = evaluate(model, dataset, 32, batches=2, seed=7, vocab_size=64)

    assert result.loss is not None and math.isfinite(result.loss)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key]), f"{key} changed during validation"


def test_evaluate_is_deterministic_for_a_fixed_seed(tmp_path):
    from training.data import TokenDataset
    from training.model import QMModel

    dataset = TokenDataset(make_tokens(tmp_path), seed=1)
    model = QMModel(TINY_MODEL)
    first = evaluate(model, dataset, 32, batches=3, seed=11, vocab_size=64)
    second = evaluate(model, dataset, 32, batches=3, seed=11, vocab_size=64)
    assert first.loss == pytest.approx(second.loss)
    assert first.windows == second.windows


def test_evaluate_restores_training_mode(tmp_path):
    from training.data import TokenDataset
    from training.model import QMModel

    dataset = TokenDataset(make_tokens(tmp_path), seed=1)
    model = QMModel(TINY_MODEL)
    model.train()
    evaluate(model, dataset, 32, batches=1, seed=3, vocab_size=64)
    assert model.training is True


def test_evaluate_reports_an_error_instead_of_raising(tmp_path):
    from training.data import TokenDataset
    from training.model import QMModel

    dataset = TokenDataset(make_tokens(tmp_path, n=64), seed=1)
    model = QMModel(TINY_MODEL)
    result = evaluate(model, dataset, 4096, batches=1, seed=3, vocab_size=64)
    assert result.errors
    assert result.loss is None


# --------------------------------------------------------------------------- #
# Validation must not be the training data
# --------------------------------------------------------------------------- #
def test_validation_on_the_training_stream_is_refused(tmp_path):
    path = make_tokens(tmp_path)
    with pytest.raises(ValueError, match="meaningless"):
        assert_separate(path, path)


def test_distinct_validation_streams_are_accepted(tmp_path):
    assert assert_separate(make_tokens(tmp_path, "a.tokens.bin"),
                           make_tokens(tmp_path, "b.tokens.bin")) is None


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def build_ckpt(path, model_cfg=None, step=5, weights=None, **extra):
    from training.model import QMModel

    model = QMModel(model_cfg or TINY_MODEL)
    payload = {
        "model": weights if weights is not None else model.state_dict(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict(),
        "step": step,
        "schema": "quantsmind.checkpoint/1",
        "config": {"run": {"model_config": None}},
    }
    payload.update(extra)
    torch.save(payload, path)
    return path


def test_checkpoint_roundtrip_restores_identical_weights(tmp_path):
    from training.checkpoint import load_model_weights
    from training.model import QMModel

    torch.manual_seed(0)
    source = QMModel(TINY_MODEL)
    path = build_ckpt(tmp_path / "ck.pt", weights=source.state_dict())

    torch.manual_seed(99)
    target = QMModel(TINY_MODEL)
    assert not torch.equal(
        target.state_dict()["tok_emb.weight"], source.state_dict()["tok_emb.weight"]
    )
    load_model_weights(target, torch.load(path, weights_only=False)["model"])
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor), key


def test_shape_mismatch_is_rejected(tmp_path):
    from training.checkpoint import CheckpointIncompatible, load_model_weights
    from training.model import QMModel

    state = torch.load(build_ckpt(tmp_path / "ck.pt"), weights_only=False)["model"]
    with pytest.raises(CheckpointIncompatible, match="Shape mismatches"):
        load_model_weights(QMModel(dict(TINY_MODEL, d_model=48)), state)


def test_missing_parameter_is_fatal(tmp_path):
    from training.checkpoint import CheckpointIncompatible, load_model_weights
    from training.model import QMModel

    state = dict(torch.load(build_ckpt(tmp_path / "ck.pt"), weights_only=False)["model"])
    state.pop(next(iter(state)))
    with pytest.raises(CheckpointIncompatible):
        load_model_weights(QMModel(TINY_MODEL), state)


def test_corrupt_checkpoint_is_rejected_with_a_reason(tmp_path):
    from training.checkpoint import CheckpointIncompatible
    from training.model import QMModel
    from training.pretrain import load_checkpoint_full

    path = tmp_path / "broken.pt"
    path.write_bytes(b"this is not a torch file")
    model = QMModel(TINY_MODEL)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(CheckpointIncompatible, match="corrupt or incomplete"):
        load_checkpoint_full(path, model, optim)


def test_checkpoint_without_optimizer_state_is_rejected(tmp_path):
    from training.checkpoint import CheckpointIncompatible
    from training.model import QMModel
    from training.pretrain import load_checkpoint_full

    torch.save({"model": QMModel(TINY_MODEL).state_dict(), "step": 3,
                "schema": "quantsmind.checkpoint/1"}, tmp_path / "c.pt")
    model = QMModel(TINY_MODEL)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(CheckpointIncompatible, match="optimizer"):
        load_checkpoint_full(tmp_path / "c.pt", model, optim)


def test_wrong_schema_is_rejected(tmp_path):
    from training.checkpoint import CheckpointIncompatible
    from training.model import QMModel
    from training.pretrain import load_checkpoint_full

    path = build_ckpt(tmp_path / "ck.pt")
    payload = torch.load(path, weights_only=False)
    payload["schema"] = "something/else"
    torch.save(payload, path)
    model = QMModel(TINY_MODEL)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(CheckpointIncompatible, match="schema"):
        load_checkpoint_full(path, model, optim)


def test_missing_checkpoint_is_rejected(tmp_path):
    from training.checkpoint import CheckpointIncompatible
    from training.model import QMModel
    from training.pretrain import load_checkpoint_full

    model = QMModel(TINY_MODEL)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(CheckpointIncompatible, match="not found"):
        load_checkpoint_full(tmp_path / "nope.pt", model, optim)


# --------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------- #
def test_rng_state_roundtrip():
    import numpy as np

    from training.pretrain import capture_rng_state, restore_rng_state

    torch.manual_seed(1234)
    np.random.seed(1234)
    state = capture_rng_state()
    expected = torch.rand(3).tolist()

    torch.manual_seed(999)
    np.random.seed(999)
    assert torch.rand(3).tolist() != expected

    assert restore_rng_state(state) is True
    assert torch.rand(3).tolist() == expected


def test_missing_rng_state_is_reported_not_fatal():
    from training.pretrain import restore_rng_state

    assert restore_rng_state(None) is False


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #
def test_manifest_marks_only_complete_runs_successful():
    for status, expected in (("complete", True), ("interrupted", False),
                             ("failed", False), ("stopped_early", False),
                             ("smoke", False)):
        manifest = build_run_manifest(
            run_id="r", status=status, config={}, config_path=None,
            model_cfg=TINY_MODEL, model_config_path=None, params=1,
        )
        assert manifest["success"] is expected, status


def test_manifest_records_checkpoint_hash_and_identity(tmp_path):
    ckpt = build_ckpt(tmp_path / "ck.pt")
    manifest = build_run_manifest(
        run_id="r", status="complete", config={}, config_path=None,
        model_cfg=TINY_MODEL, model_config_path=None, params=123,
        checkpoint_path=str(ckpt), steps_completed=5, tokens_processed=2048,
    )
    assert manifest["checkpoint"]["sha256"] == sha256_file(ckpt)
    assert manifest["checkpoint"]["bytes"] == ckpt.stat().st_size
    assert manifest["training"]["tokens_processed"] == 2048
    assert manifest["model"]["initialization"].startswith("random")
    assert manifest["success_rule"]


def test_manifest_records_warnings_and_interruptions():
    manifest = build_run_manifest(
        run_id="r", status="interrupted", config={}, config_path=None,
        model_cfg=TINY_MODEL, model_config_path=None, params=1,
        warnings=["host memory pressure"], deviations=["smaller model"],
        interruptions=[{"at_step": 5}],
    )
    assert manifest["success"] is False
    assert manifest["warnings"] == ["host memory pressure"]
    assert manifest["deviations"] == ["smaller model"]
    assert manifest["interruptions"] == [{"at_step": 5}]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def tiny_tokenizer(tmp_path, vocab_size=64):
    """A real tokenizer file. Its final vocabulary is returned with the path.

    ``initial_alphabet`` adds the 256 byte tokens regardless of the requested
    size, so the achieved vocabulary is larger than ``vocab_size``. Callers
    build their model from the returned vocabulary rather than assuming one.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    corpus = tmp_path / "tok_corpus.txt"
    corpus.write_text("\n".join(
        f"def f{i}(a, b):\n    return a + b * {i}\n" for i in range(64)
    ), encoding="utf-8")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<unk>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(corpus)], trainer)
    path = tmp_path / "tiny_tokenizer.json"
    tok.save(str(path))
    return path, tok.get_vocab_size()


def tiny_model_for(tmp_path, **overrides):
    """A QMModel whose vocab_size equals the tiny tokenizer's real vocabulary."""
    from training.model import QMModel

    path, vocab = tiny_tokenizer(tmp_path)
    cfg = dict(TINY_MODEL, vocab_size=vocab)
    cfg.update(overrides)
    return QMModel(cfg), Tokenizer.from_file(str(path)), vocab


def test_top_p_sampling_does_not_crash_on_batched_logits(tmp_path):
    """Regression: nucleus filtering scattered along dim 0, the batch axis.

    probs has shape (batch, vocab); scatter_(0, ...) raised
    "index 202 is out of bounds for dimension 0 with size 1", so no sampled
    output was ever produced.
    """
    from training.generate import generate_text

    model, tok, _vocab = tiny_model_for(tmp_path)
    result = generate_text(model, tok, "def", max_new_tokens=4,
                           temperature=0.9, top_k=10, top_p=0.9, seed=3)
    assert result["completion_tokens"] >= 1
    assert result["finish_reason"] in ("stop", "length")


def test_greedy_generation_is_deterministic(tmp_path):
    from training.generate import generate_text

    model, tok, _vocab = tiny_model_for(tmp_path)
    first = generate_text(model, tok, "def", max_new_tokens=6, greedy=True)
    second = generate_text(model, tok, "def", max_new_tokens=6, greedy=True)
    assert first["token_ids"] == second["token_ids"]


def test_generation_records_the_checkpoint_it_used(tmp_path):
    from training.generate import GenerationError, run_generation

    with pytest.raises(GenerationError, match="checkpoint not found"):
        run_generation(tmp_path / "nope.pt", "hi")


def test_generation_rejects_a_mismatched_tokenizer():
    from training.generate import GenerationError, load_tokenizer

    with pytest.raises(GenerationError, match="vocab"):
        load_tokenizer("tokenizer/qm_bpe_v0.json", 999)


def test_generation_rejects_an_oversized_prompt(tmp_path):
    from training.generate import GenerationError, generate_text

    model, tok, _vocab = tiny_model_for(tmp_path)
    with pytest.raises(GenerationError, match="does not silently"):
        generate_text(model, tok, "def " * 500, max_new_tokens=1)


def test_missing_tokenizer_file_is_reported(tmp_path):
    from training.generate import GenerationError, load_tokenizer

    with pytest.raises(GenerationError, match="tokenizer not found"):
        load_tokenizer(tmp_path / "absent.json", 64)


def test_corrupt_checkpoint_is_rejected_by_generation(tmp_path):
    from training.generate import GenerationError, load_checkpoint_for_generation

    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"definitely not a checkpoint")
    with pytest.raises(GenerationError, match="corrupt or incomplete"):
        load_checkpoint_for_generation(path)
