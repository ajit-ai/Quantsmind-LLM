"""Focused regression tests for native Taksh v160-family inference compatibility.

These tests use tiny synthetic artifacts and never copy the full checkpoint.
An optional real-checkpoint test runs only when the operator-supplied or known
artifact is present; it is evidence of execution, not model capability.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.foundation_inference import (  # noqa: E402
    CheckpointIncompatibleError,
    generate_bounded,
    load_foundation,
)
from serving.native_provider import NativeTakshProvider  # noqa: E402
from serving.providers import GenerationConfig, GenerationRequest  # noqa: E402
from training.checkpoint import load_model_weights  # noqa: E402
from training.model import QMModel  # noqa: E402

V160_CONFIG = REPO / "configs" / "model" / "code_v160_15m.yaml"
FOUNDATION_CONFIG = REPO / "configs" / "foundation" / "l060_foundation.yaml"
TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
REAL_CKPT_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"

TINY_V160 = {
    "name": "qm-code-v160-15m-tiny",
    "vocab_size": 32,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 64,
    "max_seq_len": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "tie_embeddings": True,
}


def _load_yaml_model_cfg(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("model", data)


def _tiny_tokenizer(tmp_path: Path, vocab_size: int = 32):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "<bos>": 1, "<eos>": 2, "[UNK]": 3}
    for i in range(4, vocab_size):
        vocab[f"t{i}"] = i
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tok.json"
    tok.save(str(path))
    return tok, path


def _tiny_v160_artifact(tmp_path: Path, step: int = 7):
    tok, tok_path = _tiny_tokenizer(tmp_path, TINY_V160["vocab_size"])
    torch.manual_seed(11)
    model = QMModel(TINY_V160)
    ckpt_path = tmp_path / "tiny_v160.pt"
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "run": {},
            "model": dict(TINY_V160),
        },
        "step": step,
        "schema": "taksh.checkpoint/test",
    }
    torch.save(payload, ckpt_path)
    return ckpt_path, tok_path, tok, model


def _real_checkpoint_path() -> Path:
    override = os.environ.get("TAKSH_V180_SFT_CHECKPOINT")
    if override:
        return Path(override)
    return Path(r"F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt")


def test_v160_config_recovered_and_matches_expected_geometry():
    cfg = _load_yaml_model_cfg(V160_CONFIG)
    assert cfg["name"] == "qm-code-v160-15m"
    assert cfg["vocab_size"] == 5822
    assert cfg["d_model"] == 384
    assert cfg["n_layers"] == 12
    assert cfg["n_heads"] == 8
    assert cfg["n_kv_heads"] == 4
    assert cfg["ffn_dim"] == 1536
    assert cfg["max_seq_len"] == 2048
    assert cfg["rope_theta"] == 10000.0
    assert cfg["tie_embeddings"] is True
    # parameter count: tied embeddings => 28,787,328 non-embedding/unique-weight total in this family
    model = QMModel(cfg)
    assert sum(p.numel() for p in model.parameters()) == 28787328


def test_canonical_111m_config_preserved():
    data = yaml.safe_load(FOUNDATION_CONFIG.read_text(encoding="utf-8"))
    cfg = data.get("model", data)
    assert data.get("id") == "taksh-foundation-111m"
    assert cfg["vocab_size"] == 5822
    assert cfg["d_model"] == 768
    assert cfg["n_layers"] == 12
    assert cfg["n_heads"] == 12
    assert cfg["n_kv_heads"] == 6
    assert cfg["ffn_dim"] == 3072
    assert cfg["max_seq_len"] == 2048


def test_v160_state_dict_compatible_and_loadable(tmp_path):
    ckpt_path, tok_path, _, _ = _tiny_v160_artifact(tmp_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = QMModel(TINY_V160)
    report = load_model_weights(model, ckpt["model"], source=str(ckpt_path))
    assert len(report.missing) == 0
    assert len(report.unexpected) == 0
    assert len(report.loaded) == len(ckpt["model"])


def test_tokenizer_vocab_compatible_with_v160_config():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    cfg = _load_yaml_model_cfg(V160_CONFIG)
    assert tok.get_vocab_size() == cfg["vocab_size"] == 5822
    assert tok.token_to_id("<bos>") == 1
    assert tok.token_to_id("<eos>") == 2


def test_foundation_loads_v160_family_and_generates_bounded(tmp_path):
    ckpt_path, tok_path, _, _ = _tiny_v160_artifact(tmp_path)
    model, tok, meta = load_foundation(str(ckpt_path), str(tok_path), "cpu")
    assert meta["model_name"] == "qm-code-v160-15m-tiny"
    assert meta["vocab_size"] == 32
    assert meta["max_seq_len"] == 64
    assert meta["parameters"] == model.num_params()

    result = generate_bounded(model, tok, "t4 t5", max_new_tokens=8, temperature=0.0, device="cpu", seed=1)
    assert result["finish_reason"] in {"stop", "length"}
    assert result["completion_tokens"] <= 8
    assert result["prompt_tokens"] >= 1


def test_generation_respects_eos_when_reached(tmp_path):
    ckpt_path, tok_path, _, _ = _tiny_v160_artifact(tmp_path)
    model, tokenizer, _ = load_foundation(str(ckpt_path), str(tok_path), "cpu")

    class EosModel:
        def __init__(self, base):
            self.base = base
            self.cfg = base.cfg

        def eval(self):
            self.base.eval()
            return self

        def num_params(self):
            return self.base.num_params()

        def to(self, device):
            self.base.to(device)
            return self

        def __call__(self, *args, **kwargs):
            logits, loss = self.base(*args, **kwargs)
            logits = torch.full_like(logits, -1e9)
            logits[..., 2] = 1e9  # force EOS token id
            return logits, loss

    eos_model = EosModel(model)
    result = generate_bounded(eos_model, tokenizer, "t4", max_new_tokens=8, temperature=0.0, device="cpu", seed=1)
    assert result["finish_reason"] == "stop"
    assert result["completion_tokens"] == 0
    assert result["eos_id"] == 2


def test_invalid_checkpoint_diagnostics(tmp_path):
    _, tok_path = _tiny_tokenizer(tmp_path, TINY_V160["vocab_size"])
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointIncompatibleError):
        load_foundation(str(bad), str(tok_path), "cpu")


def test_tokenizer_vocab_mismatch_is_rejected(tmp_path):
    ckpt_path, _, _, _ = _tiny_v160_artifact(tmp_path)
    _, wrong_tok = _tiny_tokenizer(tmp_path, vocab_size=16)
    with pytest.raises(CheckpointIncompatibleError, match="tokenizer vocab"):
        load_foundation(str(ckpt_path), str(wrong_tok), "cpu")


def test_native_provider_reports_v160_identity_after_probe(tmp_path):
    ckpt_path, tok_path, _, _ = _tiny_v160_artifact(tmp_path)
    provider = NativeTakshProvider(checkpoint_path=ckpt_path, tokenizer_path=tok_path)
    assert provider.artifact_status()["status"] == "configured_unverified"
    probe = provider.probe()
    assert probe["status"] == "compatible"
    assert provider.identity().model_id == "qm-code-v160-15m-tiny"
    caps = provider.capabilities()
    assert caps.status == "compatible"
    assert caps.available is True
    assert caps.max_context_tokens == 64


def test_native_provider_generates_with_v160_checkpoint(tmp_path):
    ckpt_path, tok_path, _, _ = _tiny_v160_artifact(tmp_path)
    provider = NativeTakshProvider(checkpoint_path=ckpt_path, tokenizer_path=tok_path)
    result = provider.generate(GenerationRequest("t4 t5", GenerationConfig(max_new_tokens=4, temperature=0.0, seed=1)))
    assert result.provider == "native-taksh"
    assert result.model_id == "qm-code-v160-15m-tiny"
    assert result.finish_reason in {"stop", "length"}
    assert result.completion_tokens <= 4


@pytest.mark.skipif(not _real_checkpoint_path().is_file(), reason="real v180_sft checkpoint not present")
def test_real_v180_sft_checkpoint_native_inference_if_present():
    path = _real_checkpoint_path()
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
        assert digest.hexdigest() == REAL_CKPT_SHA
    model, tok, meta = load_foundation(str(path), str(TOKENIZER), "cpu")
    assert meta["model_name"] == "qm-code-v160-15m"
    assert meta["vocab_size"] == 5822
    assert meta["total_parameters"] == 28787328
    result = generate_bounded(
        model, tok, "def add(a, b):\n    ", max_new_tokens=8, temperature=0.0, device="cpu", seed=1
    )
    assert isinstance(result["text"], str)
    assert result["completion_tokens"] >= 0
    assert result["finish_reason"] in {"stop", "length"}
    # Genuine model-generated response captured (content quality not asserted).
    assert "text" in result and "raw_text" in result
