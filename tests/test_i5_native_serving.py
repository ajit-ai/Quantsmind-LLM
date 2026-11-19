"""Native serving qualification tests (I5).

Builds tiny synthetic artifacts only — never the real 111M checkpoint, never a
benchmark mutation, never a network call.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l060_inference_qualification import build_tiny_artifact  # noqa: E402

from serving.foundation_inference import (  # noqa: E402
    ConfigurationError,
    CheckpointIncompatibleError,
    GenerationFailedError,
    generate_bounded,
    load_foundation,
)


@pytest.fixture()
def tiny(tmp_path):
    return build_tiny_artifact(tmp_path)


def test_load_foundation_returns_identity_hashes(tiny):
    ckpt, tok = tiny
    model, tokenizer, meta = load_foundation(str(ckpt), str(tok), "cpu")
    assert len(meta["checkpoint_sha256"]) == 64
    assert len(meta["tokenizer_sha256"]) == 64
    assert meta["checkpoint_sha256"] == hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert meta["tokenizer_sha256"] == hashlib.sha256(Path(tok).read_bytes()).hexdigest()
    assert meta["model_name"] == "taksh-native" or meta["model_name"] == "qm-code-v160-15m"
    assert meta["max_seq_len"] > 0


def test_missing_checkpoint_fails_closed(tiny, tmp_path):
    _, tok = tiny
    with pytest.raises(CheckpointIncompatibleError):
        load_foundation(str(tmp_path / "nope.pt"), str(tok), "cpu")


def test_missing_tokenizer_fails_closed(tiny, tmp_path):
    ckpt, _ = tiny
    with pytest.raises(CheckpointIncompatibleError):
        load_foundation(str(ckpt), str(tmp_path / "nope.json"), "cpu")


def test_truncated_checkpoint_fails_closed(tiny, tmp_path):
    ckpt, tok = tiny
    broken = tmp_path / "broken.pt"
    broken.write_bytes(Path(ckpt).read_bytes()[:32])
    with pytest.raises(CheckpointIncompatibleError):
        load_foundation(str(broken), str(tok), "cpu")


def test_bad_tokenizer_file_fails_closed(tiny, tmp_path):
    ckpt, _ = tiny
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointIncompatibleError):
        load_foundation(str(ckpt), str(bad), "cpu")


def test_generate_bounded_deterministic(tiny):
    ckpt, tok = tiny
    model, tokenizer, _ = load_foundation(str(ckpt), str(tok), "cpu")
    first = generate_bounded(model, tokenizer, "a b c d", max_new_tokens=4, temperature=0.0, seed=42, device="cpu")
    second = generate_bounded(model, tokenizer, "a b c d", max_new_tokens=4, temperature=0.0, seed=42, device="cpu")
    assert first["token_ids"] == second["token_ids"]
    assert first["text"] == second["text"]


def test_generate_bounded_rejects_prompt_over_context(tiny):
    ckpt, tok = tiny
    model, tokenizer, meta = load_foundation(str(ckpt), str(tok), "cpu")
    with pytest.raises(ConfigurationError):
        generate_bounded(model, tokenizer, "a " * (meta["max_seq_len"] + 1), max_new_tokens=1, device="cpu")


def test_generate_bounded_rejects_budget_exceeding_context(tiny):
    ckpt, tok = tiny
    model, tokenizer, meta = load_foundation(str(ckpt), str(tok), "cpu")
    with pytest.raises(ConfigurationError):
        generate_bounded(model, tokenizer, "a b c d", max_new_tokens=meta["max_seq_len"] + 1, device="cpu")


def test_generate_bounded_rejects_empty_budget(tiny):
    ckpt, tok = tiny
    model, tokenizer, _ = load_foundation(str(ckpt), str(tok), "cpu")
    with pytest.raises(ConfigurationError):
        generate_bounded(model, tokenizer, "a b c", max_new_tokens=0, device="cpu")


def test_generate_bounded_rejects_bad_temperature(tiny):
    ckpt, tok = tiny
    model, tokenizer, _ = load_foundation(str(ckpt), str(tok), "cpu")
    with pytest.raises(ConfigurationError):
        generate_bounded(model, tokenizer, "a b c", temperature=-1.0, device="cpu")
