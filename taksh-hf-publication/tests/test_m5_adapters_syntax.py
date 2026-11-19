"""Tests for M5 adapters and syntax scoring."""

from __future__ import annotations

import json

import pytest
import torch

from evaluation.m5_adapters import (
    AdapterUnavailable,
    ExternalHFAdapter,
    QMCkptAdapter,
)
from evaluation.m5_scorers import (
    NO_EXECUTABLE_CODE,
    SYNTAX_INVALID,
    SYNTAX_VALID,
    score_syntax,
)

FENCE = chr(96) * 3


def tiny_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(f"def f{i}(a):\n    return a+{i}"
                                for i in range(64)), encoding="utf-8")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train([str(corpus)], trainers.BpeTrainer(
        vocab_size=64, special_tokens=["<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    path = tmp_path / "tok.json"
    tok.save(str(path))
    return path, tok.get_vocab_size()


def tiny_checkpoint(path, vocab_size=128):
    from training.model import QMModel

    cfg = {"vocab_size": vocab_size, "d_model": 32, "n_layers": 1,
           "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 128,
           "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True}
    model = QMModel(cfg)
    torch.save({
        "model": model.state_dict(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict(),
        "step": 1, "schema": "quantsmind.checkpoint/1",
        # The architecture is recorded inline so the loader can rebuild the
        # model without depending on a config file on disk.
        "config": {"run": {}, "model": cfg},
    }, path)


def loaded_adapter(tmp_path):
    tok_path, vocab = tiny_tokenizer(tmp_path)
    tiny_checkpoint(tmp_path / "c.pt", vocab_size=vocab)
    return QMCkptAdapter(str(tmp_path / "c.pt"), tokenizer_path=str(tok_path))


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_refused_before_loading():
    adapter = QMCkptAdapter("does/not/exist.pt")
    ok, reason = adapter.preflight()
    assert ok is False
    assert "not found" in reason
    with pytest.raises(AdapterUnavailable):
        adapter.load()


def test_corrupt_checkpoint_refused_with_a_reason(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch file")
    adapter = QMCkptAdapter(str(path))
    assert adapter.preflight()[0] is True
    with pytest.raises(AdapterUnavailable, match="corrupt or incomplete"):
        adapter.load()


def test_generation_before_load_is_refused():
    with pytest.raises(AdapterUnavailable, match="not loaded"):
        QMCkptAdapter("whatever.pt").generate("hi")


def test_adapter_load_generate_release_cycle(tmp_path):
    adapter = loaded_adapter(tmp_path)
    adapter.load()
    info = adapter.info()
    assert info.adapter == "qm_ckpt"
    assert info.parameters > 0
    assert info.parameters_source == "measured from the loaded state_dict"
    assert info.artifact_sha256
    assert info.vocab_size > 0

    result = adapter.generate("def f(", max_new_tokens=4)
    assert result["completion_tokens"] >= 1
    assert result["seconds"] >= 0

    adapter.release()
    adapter.release()
    with pytest.raises(AdapterUnavailable):
        adapter.generate("hi")


def test_mismatched_tokenizer_is_refused(tmp_path):
    tok_path, _vocab = tiny_tokenizer(tmp_path)
    tiny_checkpoint(tmp_path / "c.pt", vocab_size=999)
    adapter = QMCkptAdapter(str(tmp_path / "c.pt"), tokenizer_path=str(tok_path))
    with pytest.raises(AdapterUnavailable, match="vocab"):
        adapter.load()


def test_external_adapter_refuses_without_licence_record():
    adapter = ExternalHFAdapter("some/model", license_record={})
    ok, reason = adapter.preflight()
    assert ok is False
    assert reason


def test_external_adapter_does_not_download_by_default():
    adapter = ExternalHFAdapter(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        license_record={"license_id": "apache-2.0", "verified": True})
    ok, reason = adapter.preflight()
    assert ok is False
    assert "nothing was fetched" in reason
    assert adapter._model is None


def test_external_adapter_reports_missing_transformers(monkeypatch):
    adapter = ExternalHFAdapter("some/model",
                                license_record={"license_id": "x"})
    monkeypatch.setattr(adapter, "_transformers_available", lambda: None)
    ok, reason = adapter.preflight()
    assert ok is False
    assert "transformers" in reason


def test_blocked_run_is_recorded_not_silently_empty(tmp_path):
    from evaluation.m5_runner import write_blocked_run

    manifest = write_blocked_run(tmp_path / "run", "some/model",
                                 "no weights and download not permitted")
    assert manifest["status"] == "blocked"
    assert manifest["complete"] is False
    on_disk = json.loads(
        (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "blocked"
    assert "summary" not in on_disk


# --------------------------------------------------------------------------- #
# Syntax scoring
# --------------------------------------------------------------------------- #
def test_syntax_success_and_failure():
    good = f"{FENCE}python\ndef f():\n    return 1\n{FENCE}"
    bad = f"{FENCE}python\ndef f(:\n{FENCE}"
    assert score_syntax(good)["status"] == SYNTAX_VALID
    assert score_syntax(bad)["status"] == SYNTAX_INVALID


def test_comment_only_output_is_not_executable_code():
    """Regression guard for the M3 defect that scored noise as valid Python."""
    result = score_syntax("# just a comment\n# and another\n#")
    assert result["status"] == NO_EXECUTABLE_CODE
    assert result["score"] == 0.0


def test_empty_output_rejected():
    result = score_syntax("")
    assert result["denominator"] == 1
    assert result["numerator"] == 0


def test_multiple_blocks_are_ambiguous_not_guessed():
    result = score_syntax(
        f"{FENCE}python\na=1\n{FENCE}\n{FENCE}python\nb=2\n{FENCE}")
    assert result["status"] == "EXTRACTION_AMBIGUOUS"


def test_unfenced_real_code_accepted():
    assert score_syntax("def f():\n    return 1")["status"] == SYNTAX_VALID


def test_syntax_not_claimed_for_other_languages():
    result = score_syntax("SELECT 1", language="sql")
    assert result["status"] == "not_checked"
    assert result["score"] is None
    assert result["denominator"] == 0