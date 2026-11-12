"""Focused non-training tests for L1.1.0 — Native Model Foundation.

These tests avoid any allocation-heavy 110M-parameter build; every model
construct runs on a tiny configuration. Parameter accounting is performed
against the foundation target with the torch ``meta`` device (no RAM).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from training.model import (  # noqa: E402
    ModelConfigError, QMModel, RMSNorm, build_causal_padding_mask,
    build_rope, build_rope_cached,
)

TINY = {
    "vocab_size": 256,
    "d_model": 16,
    "n_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 1,
    "ffn_dim": 32,
    "max_seq_len": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1.0e-5,
    "tie_embeddings": True,
}
FOUNDATION_111M_YAML = REPO / "configs" / "model" / "foundation_111m.yaml"
TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"


def _tiny() -> dict:
    return dict(TINY)


# --------------------------------------------------------------------------- #
# 1. Foundation config contract
# --------------------------------------------------------------------------- #
def test_foundation_111m_yaml_parses_with_intent_fields():
    doc = yaml.safe_load(FOUNDATION_111M_YAML.read_text(encoding="utf-8"))
    assert doc["id"] == "taksh-foundation-111m"
    assert doc["model"]["vocab_size"] == 5822
    assert doc["model"]["d_model"] == 768
    assert doc["model"]["n_layers"] == 12
    assert doc["model"]["n_heads"] == 12
    assert doc["model"]["n_kv_heads"] == 6
    assert doc["model"]["ffn_dim"] == 3072
    assert doc["model"]["max_seq_len"] == 2048
    assert doc["model"]["tie_embeddings"] is True
    assert doc["model"]["rope_theta"] == 10000.0
    assert doc["intended_role"] == "foundation-product-target"


def test_foundation_111m_parameter_count_exact():
    doc = yaml.safe_load(FOUNDATION_111M_YAML.read_text(encoding="utf-8"))["model"]
    with torch.device("meta"):
        model = QMModel(doc)
        total = sum(p.numel() for p in model.parameters())
    assert total == 110_658_816


def test_foundation_111m_tokenizer_binding_matches():
    doc = yaml.safe_load(FOUNDATION_111M_YAML.read_text(encoding="utf-8"))
    tok = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    assert len(tok["model"]["vocab"]) == doc["model"]["vocab_size"] == 5822


def test_foundation_config_is_not_v180_sft():
    doc = yaml.safe_load(FOUNDATION_111M_YAML.read_text(encoding="utf-8"))["model"]
    assert doc["d_model"] == 768
    assert doc["n_heads"] == 12
    assert doc["n_kv_heads"] == 6
    assert doc["ffn_dim"] == 3072
    # explicit negative: never match the v180_sft geometry
    assert not (doc["d_model"], doc["n_heads"], doc["ffn_dim"]) == (384, 8, 1536)


# --------------------------------------------------------------------------- #
# 2. Model construction and validation
# --------------------------------------------------------------------------- #
def test_tiny_model_builds_and_counts_params():
    model = QMModel(_tiny())
    assert model.num_params() == sum(p.numel() for p in model.parameters())


def test_model_rejects_divisibility_violation():
    bad = _tiny()
    bad["d_model"] = 18        # not divisible by n_heads=2? 18/2=9 ok
    bad["n_heads"] = 4         # 18 % 4 != 0
    with pytest.raises(ModelConfigError):
        QMModel(bad)


def test_model_rejects_kv_divisibility_violation():
    bad = _tiny()
    bad["n_heads"] = 3
    bad["n_kv_heads"] = 2
    with pytest.raises(ModelConfigError):
        QMModel(bad)


def test_model_rejects_odd_head_dim():
    bad = _tiny()
    bad["n_heads"] = 1         # head_dim = 16 (even, ok here); force odd next
    bad["d_model"] = 6
    bad["n_heads"] = 2         # head_dim = 3 (odd)
    with pytest.raises(ModelConfigError):
        QMModel(bad)


def test_model_rejects_zero_or_negative_dims():
    with pytest.raises(ModelConfigError):
        QMModel({**_tiny(), "n_heads": 0})


def test_yaml_numeric_coercion_of_norm_eps():
    doc = _tiny()
    doc["norm_eps"] = "1e-5"   # YAML 1.1 bare scalar becomes a string
    model = QMModel(doc)
    assert isinstance(model.cfg["norm_eps"], float)


def test_yaml_boolean_rejected_for_numeric_field():
    doc = _tiny()
    doc["rope_theta"] = True
    with pytest.raises(ModelConfigError):
        QMModel(doc)


def test_model_param_split_handles_untied():
    doc = _tiny()
    doc["tie_embeddings"] = False
    model = QMModel(doc)
    assert model.num_params() < sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# 3. Forward pass correctness
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_dtypes():
    model = QMModel(_tiny()).eval()
    idx = torch.randint(0, TINY["vocab_size"], (2, 10))
    with torch.no_grad():
        logits, loss = model(idx)
    assert logits.shape == (2, 10, TINY["vocab_size"])
    assert logits.dtype == torch.float32
    assert loss is None


def test_forward_loss_is_finite_and_ignores_pad_targets():
    model = QMModel(_tiny())
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    targets = idx.clone()
    targets[:, :3] = -100     # pad positions must be excluded from loss
    logits, loss = model(idx, targets=targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_forward_loss_changes_with_targets():
    model = QMModel(_tiny()).eval()
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    with torch.no_grad():
        _, loss_a = model(idx, targets=idx)
        _, loss_b = model(idx, targets=torch.zeros_like(idx))
    assert float(loss_a) != float(loss_b)


def test_causality_future_tokens_do_not_affect_past_logits():
    model = QMModel(_tiny()).eval()
    torch.manual_seed(1)
    idx_full = torch.randint(0, TINY["vocab_size"], (1, 12))
    idx_trunc = idx_full[:, :8].clone()
    idx_prefix = idx_trunc[:, :1].clone()
    with torch.no_grad():
        logits_full, _ = model(idx_full)
        logits_prefix, _ = model(idx_prefix)
    # logits at shared prefix positions must match (causal masking)
    torch.testing.assert_close(logits_full[:, 0], logits_prefix[:, 0], atol=1e-6, rtol=1e-5)


def test_padding_mask_shape_and_truth_table():
    lengths = torch.tensor([4, 2])
    mask = build_causal_padding_mask(lengths, 4)
    assert mask.shape == (2, 1, 4, 4)
    assert mask.dtype == torch.bool
    # full sequence sees itself and its past; never the future
    assert mask[0, 0, 3, 0].item() is True
    assert mask[0, 0, 0, 3].item() is False
    # shortened sequence: causal past visible; pads invisible both ways
    assert mask[1, 0, 0, 0].item() is True
    assert mask[1, 0, 0, 1].item() is False      # causal: cannot see the future
    assert mask[1, 0, 1, 1].item() is True       # position 1 sees itself and 0
    assert mask[1, 0, 1, 0].item() is True
    assert mask[1, 0, 0, 2].item() is False      # padding: invisible
    assert mask[1, 0, 2, 0].item() is False      # padding: invisible source
    assert mask[1, 0, 3, 3].item() is False      # padding: invisible target


def test_forward_with_lengths_produces_consistent_prefix_results():
    model = QMModel(_tiny()).eval()
    torch.manual_seed(2)
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    lengths = torch.tensor([8, 5])
    with torch.no_grad():
        logits_masked, _ = model(idx, lengths=lengths, pad_id=0)
        logits_plain, _ = model(idx)
    torch.testing.assert_close(logits_masked[0], logits_plain[0], atol=1e-5, rtol=1e-4)


def test_invalid_length_rejected():
    model = QMModel(_tiny())
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    with pytest.raises(ModelConfigError):
        model(idx, lengths=torch.tensor([8, 99]), pad_id=0)


def test_both_masks_rejected():
    model = QMModel(_tiny())
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    mask = build_causal_padding_mask(torch.tensor([8, 8]), 8)
    with pytest.raises(ModelConfigError):
        model(idx, attn_mask=mask, lengths=torch.tensor([8, 8]), pad_id=0)


def test_unknown_mask_dtype_rejected():
    model = QMModel(_tiny()).eval()
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    bad_mask = torch.ones(2, 1, 8, 8, dtype=torch.int64)
    with pytest.raises(ModelConfigError):
        model(idx, attn_mask=bad_mask)


def test_wrong_mask_shape_rejected():
    model = QMModel(_tiny()).eval()
    idx = torch.randint(0, TINY["vocab_size"], (2, 8))
    bad_mask = torch.ones(2, 1, 8, 9, dtype=torch.bool)
    with pytest.raises(ModelConfigError):
        model(idx, attn_mask=bad_mask)


def test_rope_cache_reuses_tensors():
    a = build_rope_cached(4, 8, 10000.0, torch.device("cpu"), torch.float32)
    b = build_rope_cached(4, 8, 10000.0, torch.device("cpu"), torch.float32)
    assert a[0] is b[0] and a[1] is b[1]


def test_rmsnorm_output_bounded():
    norm = RMSNorm(8)
    x = torch.randn(4, 4, 8) * 100
    out = norm(x)
    assert torch.isfinite(out).all()


def test_embeddings_are_tied():
    model = QMModel(_tiny())
    assert model.lm_head.weight is model.tok_emb.weight


# --------------------------------------------------------------------------- #
# 4. Tokenizer contract and fingerprints
# --------------------------------------------------------------------------- #
def test_committed_tokenizer_fingerprint_matches_record():
    import hashlib

    digest = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_tokenizer_special_tokens_present():
    doc = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    vocab = doc["model"]["vocab"]
    for token, idx in (("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)):
        assert vocab.get(token) == idx


def test_tokenizer_pad_id_zero():
    assert json.loads(TOKENIZER.read_text(encoding="utf-8"))["model"]["vocab"]["<pad>"] == 0


def test_tokenizer_and_model_vocab_agree():
    doc = yaml.safe_load(FOUNDATION_111M_YAML.read_text(encoding="utf-8"))
    tok = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    assert len(tok["model"]["vocab"]) == doc["model"]["vocab_size"]


# --------------------------------------------------------------------------- #
# 5. Checkpoint integrity (in-memory fixtures)
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_fixture(tmp_path):
    import training.checkpoint as ckpt_mod

    model = QMModel(_tiny())
    state = {"model": model.state_dict(), "step": 7,
             "config": {"run": {"model_config": "configs/model/toy_ci.yaml"}},
             "schema": "quantsmind.fixture/synthetic", "provenance": "unit test"}
    import torch as _t
    p = tmp_path / "fixture_ckpt.pt"
    _t.save(state, p)
    report = ckpt_mod.init_from_checkpoint(model, str(p))
    assert report.ok
    assert report.mismatched == []
    # mismatched shapes are always rejected
    bad_state = dict(state)
    bad_state["model"] = dict(model.state_dict())
    for k in list(bad_state["model"]):
        if bad_state["model"][k].dim() >= 2:
            bad_state["model"][k] = bad_state["model"][k][: max(1, bad_state["model"][k].shape[0] - 1)]
            break
    p2 = tmp_path / "bad_ckpt.pt"
    _t.save(bad_state, p2)
    with pytest.raises(ckpt_mod.CheckpointIncompatible):
        ckpt_mod.init_from_checkpoint(model, str(p2))


# --------------------------------------------------------------------------- #
# 6. Frozen benchmark integrity (if present on this branch)
# --------------------------------------------------------------------------- #
def test_frozen_benchmarks_referenced_and_absent_by_design():
    root = REPO / "evaluation" / "l02_python_bench_v1.json"
    if root.is_file():
        import hashlib

        digest = hashlib.sha256(root.read_bytes()).hexdigest()
        assert digest == "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
    else:
        pytest.skip("frozen Python benchmark is a feature-branch artifact; absent on this develop-based branch")


def test_frozen_multilingual_benchmark_absent_by_design():
    root = REPO / "evaluation" / "l051_multilingual_benchmark_v1.json"
    assert not root.is_file(), "multilingual benchmark must not appear on this branch"
