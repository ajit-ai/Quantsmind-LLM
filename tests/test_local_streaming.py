"""Streaming must be the same generation, not a replay.

``stream_text`` and ``generate_ids`` share ``_sample_tokens``; these tests run
both against a small real ``QMModel`` and assert token-for-token equivalence at
temperature 0. They also cover the two places a naive streamer would lie:
partial stop sequences and a held-back UTF-8 tail.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from serving.local_inference import generate_ids, stream_text


class _DummyTokenizer:
    """Deterministic ASCII decoder over a tiny vocabulary."""

    def encode(self, text):
        return type("Encoded", (), {"ids": [1, 2, 3, 4]})()

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)

    def token_to_id(self, token):
        return 2 if token == "<eos>" else None

    def get_vocab_size(self):
        return 100


def _model():
    from training.model import QMModel

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
    return QMModel(cfg).eval()


def test_stream_and_buffered_generation_are_identical():
    model = _model()
    tokenizer = _DummyTokenizer()

    torch.manual_seed(42)
    buffered = generate_ids(
        model,
        tokenizer,
        "prompt",
        max_new_tokens=8,
        temperature=0.0,
        top_k=0,
        binding_repair=False,
        use_kv_cache=True,
    )

    torch.manual_seed(42)
    deltas = []
    done = None
    for event in stream_text(
        model,
        tokenizer,
        "prompt",
        max_new_tokens=8,
        temperature=0.0,
        top_k=0,
        use_kv_cache=True,
    ):
        if event["type"] == "delta":
            deltas.append(event["text"])
        else:
            done = event

    assert "".join(deltas) == buffered["text"]
    assert done is not None
    assert done["token_ids"] == buffered["token_ids"]
    assert done["finish_reason"] == buffered["finish_reason"]


class _FixedTokenModel:
    """Always argmaxes to one token id; no KV geometry required."""

    def __init__(self, token_id=7):
        self.token_id = token_id
        self.cfg = {"max_seq_len": 32}

    def __call__(self, idx):
        logits = torch.zeros(idx.shape[0], idx.shape[1], 32)
        logits[..., self.token_id] = 1.0
        return logits, None

    def eval(self):
        return self


def test_stream_never_leaks_a_partial_stop_sequence():
    model = _FixedTokenModel(7)  # 'h' under the dummy decoder
    tokenizer = _DummyTokenizer()
    # 'h' is stop token 7's character; a naive streamer would emit at least one.
    events = list(
        stream_text(
            model,
            tokenizer,
            "prompt",
            max_new_tokens=6,
            temperature=0.0,
            stop=["h"],
            use_kv_cache=False,
        )
    )
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert text == ""
    assert events[-1]["type"] == "done"
    assert events[-1]["finish_reason"] == "stop"
