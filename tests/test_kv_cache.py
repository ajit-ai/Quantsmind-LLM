"""Tests for the KV cache (serving/taksh kv_cache module).

These tests verify that the KV cache:
1. Stores and retrieves key/value tensors correctly per layer.
2. Produces generation output identical to the no-cache path (within float tolerance).
3. Clears state between prompts.
4. Handles the seq_len tracking correctly.

Run with:
    python -m pytest tests/test_kv_cache.py -v
"""

from __future__ import annotations

import torch

from taksh.kv_cache import KVCache


def test_kv_cache_basic_store_and_retrieve():
    """Each layer can store and retrieve its own key/value pair.

    Tensor layout: [batch, n_kv_heads, seq, head_dim] (matches Attention output).
    """
    n_layers = 4
    cache = KVCache(n_layers=n_layers)

    # Initially all layers are empty
    for i in range(n_layers):
        assert cache.get(i) is None

    # Populate layer 0: [batch=2, n_kv_heads=4, seq=5, head_dim=64]
    k0 = torch.randn(2, 4, 5, 64)
    v0 = torch.randn(2, 4, 5, 64)
    full_k, full_v = cache.update(0, k0, v0)
    entry = cache.get(0)
    assert entry is not None
    assert torch.equal(full_k, k0)
    assert torch.equal(full_v, v0)
    assert torch.equal(entry.key, k0)
    assert torch.equal(entry.value, v0)
    assert cache.seq_len == 5

    # Other layers still empty
    assert cache.get(1) is None
    assert cache.get(2) is None
    assert cache.get(3) is None


def test_kv_cache_append_grows_sequence():
    """Appending to the same layer concatenates along seq dim (dim=2)."""
    cache = KVCache(n_layers=2)

    k1 = torch.randn(1, 2, 3, 32)   # [batch, n_kv_heads=2, seq=3, head_dim=32]
    v1 = torch.randn(1, 2, 3, 32)
    cache.update(0, k1, v1)
    assert cache.seq_len == 3

    k2 = torch.randn(1, 2, 2, 32)   # append 2 more tokens
    v2 = torch.randn(1, 2, 2, 32)
    full_k, full_v = cache.update(0, k2, v2)
    # Shape: [batch, n_kv_heads, seq, head_dim]
    assert full_k.shape == (1, 2, 5, 32)
    assert full_v.shape == (1, 2, 5, 32)
    assert torch.equal(full_k[:, :, :3], k1)
    assert torch.equal(full_k[:, :, 3:], k2)
    assert cache.seq_len == 5


def test_kv_cache_clear_resets():
    """Clear removes all cached entries and resets seq_len."""
    cache = KVCache(n_layers=2)
    k = torch.randn(1, 2, 3, 32)
    v = torch.randn(1, 2, 3, 32)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert cache.seq_len == 3

    cache.clear()
    for i in range(2):
        assert cache.get(i) is None
    assert cache.seq_len == 0


def test_kv_cache_independent_layers():
    """Layers do not interfere with each other."""
    cache = KVCache(n_layers=3)

    k0 = torch.full((1, 2, 1, 16), 1.0)
    v0 = torch.full((1, 2, 1, 16), 2.0)
    k1 = torch.full((1, 2, 1, 16), 3.0)
    v1 = torch.full((1, 2, 1, 16), 4.0)

    cache.update(0, k0, v0)
    cache.update(1, k1, v1)

    entry0 = cache.get(0)
    entry1 = cache.get(1)
    entry2 = cache.get(2)

    assert entry0 is not None and entry0.key[0, 0, 0, 0] == 1.0
    assert entry1 is not None and entry1.key[0, 0, 0, 0] == 3.0
    assert entry2 is None


def test_kv_cache_repr():
    """__repr__ works without error."""
    cache = KVCache(n_layers=2)
    r = repr(cache)
    assert "KVCache" in r
    assert "n_layers=2" in r


def test_kv_cache_generation_matches_no_cache():
    """End-to-end: KV-cache generation matches no-cache generation exactly
    when sampling is deterministic (temperature=0, argmax).

    This requires a real model, so we use a small synthetic QMModel.
    """
    from training.model import QMModel
    from tokenizers import Tokenizer
    from pathlib import Path

    cfg = {
        "vocab_size": 100,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 64,
        "max_seq_len": 64,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": False,
    }
    model = QMModel(cfg).eval()

    # Use a dummy tokenizer-like object with encode/decode for a tiny vocab.
    class _DummyTok:
        def encode(self, s):
            class _Enc:
                ids = [1, 2, 3, 4]
            return _Enc()
        def decode(self, ids):
            return "".join(chr(97 + (i % 26)) for i in ids)
        def token_to_id(self, t):
            return None
        def get_vocab_size(self):
            return 100

    tokenizer = _DummyTok()
    prompt_ids = [1, 2, 3, 4]
    device = "cpu"

    # Run with KV cache
    torch.manual_seed(42)
    from serving.local_inference import generate_ids
    result_cache = generate_ids(
        model, tokenizer,
        "prompt",
        max_new_tokens=8,
        temperature=0.0,  # deterministic argmax
        top_k=0,
        device=device,
        use_kv_cache=True,
    )

    # Run without KV cache
    torch.manual_seed(42)
    result_no_cache = generate_ids(
        model, tokenizer,
        "prompt",
        max_new_tokens=8,
        temperature=0.0,
        top_k=0,
        device=device,
        use_kv_cache=False,
    )

    assert result_cache["text"] == result_no_cache["text"], (
        f"KV cache output differs from no-cache output:\n"
        f"  cache:   {result_cache['text']!r}\n"
        f"  no-cache: {result_no_cache['text']!r}"
    )
    assert result_cache["finish_reason"] == result_no_cache["finish_reason"]
    assert result_cache["completion_tokens"] == result_no_cache["completion_tokens"]
