"""Smoke test for the locally staged Taksh S2 experimental artifact."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from serving.local_inference import generate_ids, load_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "taksh-s2-experimental"
MODEL = ARTIFACT / "taksh-s2.pt"
TOKENIZER = ARTIFACT / "tokenizer" / "qm_code_bpe_v0.json"
MODEL_SHA = "02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac"
TOKENIZER_SHA = "90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988"
PROMPT = "Write a function that returns the sum of two integers.\ndef compute_total(x, y):\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(not MODEL.is_file() or not TOKENIZER.is_file(), reason="experimental artifact not staged")
class TestTakshS2ExperimentalArtifact:
    def test_artifact_hashes(self):
        assert _sha256(MODEL) == MODEL_SHA
        assert _sha256(TOKENIZER) == TOKENIZER_SHA

    def test_load_generate_repeat_and_parse(self):
        torch.set_num_threads(4)
        model, config = load_model(str(MODEL), "cpu")
        tokenizer = Tokenizer.from_file(str(TOKENIZER))
        assert int(model.num_params()) == 9359104
        assert int(config["max_seq_len"]) == 512

        first = generate_ids(
            model,
            tokenizer,
            PROMPT,
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            device="cpu",
            seed=20261023,
            binding_repair=False,
        )
        second = generate_ids(
            model,
            tokenizer,
            PROMPT,
            max_new_tokens=32,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            device="cpu",
            seed=20261023,
            binding_repair=False,
        )

        assert first["finish_reason"] == "stop"
        assert first["text"] == second["text"]
        # Known binding failure: the model does not use the requested x/y.
        assert first["text"] == "    return left + right\n"
        ast.parse("def compute_total(x, y):\n" + first["text"])
