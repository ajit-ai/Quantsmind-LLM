"""Focused tests for the TAKSH Local Coding Assistant (L0.1).

Scope (deliberately narrow, per the L0.1 brief):
1. checkpoint loading (hardened loader, incl. the NumPy-RNG allowlist case)
2. tokenizer compatibility and tokenization/decoding
3. deterministic generation smoke test (greedy, tiny token budget)
4. application startup and local-only (loopback) host configuration
5. no external model-provider fallback
6. missing or incompatible checkpoint handling

The real checkpoint tests are skipped (not failed) when the artifact is not
present locally. Nothing here performs network access.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from serving import local_assistant as la  # noqa: E402
from serving import local_inference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "runs" / "v180_sft" / "checkpoint_last.pt"
TOK = ROOT / "tokenizer" / "taksh_code_bpe_v1.json"
requires_checkpoint = pytest.mark.skipif(
    not (CKPT.exists() and TOK.exists()),
    reason="local Taksh checkpoint/tokenizer not present",
)


class _NonAllowlistedGlobal:
    """Module-level so it is picklable; deliberately absent from the allowlist."""

    pass


# --------------------------------------------------------------------------- #
# 1. Checkpoint loading
# --------------------------------------------------------------------------- #
def test_numpy_rng_checkpoint_loads_through_hardened_loader(tmp_path):
    """Taksh training checkpoints carry NumPy RNG state; the narrow allowlist
    must admit exactly that, still under weights_only=True."""
    blob = {
        "model": {},
        "step": 1,
        "rng_state": {
            "numpy": np.arange(4, dtype=np.int64),
            "dtype_obj": np.dtype("float64"),
        },
    }
    path = tmp_path / "fake_numpy_ckpt.pt"
    torch.save(blob, path)
    loaded = local_inference.torch_load_checkpoint(str(path))
    assert loaded["step"] == 1
    assert list(loaded["rng_state"]["numpy"]) == [0, 1, 2, 3]


def test_untrusted_pickle_is_refused(tmp_path):
    """A checkpoint whose globals fall outside the allowlist is refused."""
    path = tmp_path / "evil.pt"
    torch.save({"model": {}, "custom": _NonAllowlistedGlobal()}, path)
    with pytest.raises(RuntimeError) as excinfo:
        local_inference.torch_load_checkpoint(str(path))
    assert "Refusing" in str(excinfo.value)


def test_missing_checkpoint_file_raises(tmp_path):
    with pytest.raises(Exception):
        local_inference.torch_load_checkpoint(str(tmp_path / "nope.pt"))


@requires_checkpoint
def test_load_model_reports_architecture():
    model, cfg = local_inference.load_model(str(CKPT), "cpu")
    assert cfg["vocab_size"] == 5822
    assert cfg["d_model"] == 384
    assert cfg["n_layers"] == 12
    assert cfg["max_seq_len"] == 2048
    assert model.num_params() == 28_787_328


# --------------------------------------------------------------------------- #
# 2. Tokenizer compatibility / tokenization / decoding
# --------------------------------------------------------------------------- #
def test_tokenizer_file_loads_and_matches_contract():
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(TOK))
    assert tok.get_vocab_size() == 5822
    assert tok.token_to_id("<bos>") == 1
    assert tok.token_to_id("<eos>") == 2
    from training.tokenizer_contract import CONTRACT_V1, detect_contract

    assert detect_contract(tok) == CONTRACT_V1


def test_encode_decode_roundtrip():
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(TOK))
    text = "def is_even(n):\n    return n % 2 == 0"
    ids = tok.encode(text).ids
    assert ids, "encoding produced no ids"
    assert tok.decode(ids) == text


# --------------------------------------------------------------------------- #
# 3. Deterministic generation smoke test (greedy, tiny budget)
# --------------------------------------------------------------------------- #
@requires_checkpoint
def test_greedy_generation_is_deterministic_and_nonempty():
    tokenizers = pytest.importorskip("tokenizers")
    model, _ = local_inference.load_model(str(CKPT), "cpu")
    tok = tokenizers.Tokenizer.from_file(str(TOK))
    prompt = la.build_prompt(tok, "Write a Python function is_even(n).")
    out1 = local_inference.generate_ids(
        model, tok, prompt, max_new_tokens=8, temperature=0.0, device="cpu"
    )
    out2 = local_inference.generate_ids(
        model, tok, prompt, max_new_tokens=8, temperature=0.0, device="cpu"
    )
    assert out1["text"] == out2["text"], "greedy decoding must be deterministic"
    assert out1["completion_tokens"] >= 1
    assert out1["finish_reason"] in ("stop", "length")


# --------------------------------------------------------------------------- #
# 4/6. Local application startup + localhost-only binding
# --------------------------------------------------------------------------- #
def test_validate_host_accepts_loopback():
    for host in ("127.0.0.1", "localhost", "::1"):
        assert la.validate_host(host) == host.lower()


def test_validate_host_rejects_non_loopback():
    for host in ("0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com", ""):
        with pytest.raises(ValueError):
            la.validate_host(host)


def test_server_defaults_are_loopback_and_local_only():
    assert la.SERVER_NAME == "127.0.0.1"
    assert la.validate_host(la.SERVER_NAME) == "127.0.0.1"


def test_main_refuses_non_loopback_host(capsys):
    rc = la.main(["--host", "0.0.0.0"])
    assert rc == 2
    assert "loopback" in capsys.readouterr().err.lower()


def test_launch_passes_validated_host_and_disables_share():
    source = inspect.getsource(la.main)
    assert "server_name=host" in source, "launch must pass the validated loopback host"
    assert "share=False" in inspect.getsource(la), "sharing must stay disabled"


# --------------------------------------------------------------------------- #
# 5. No external model-provider fallback
# --------------------------------------------------------------------------- #
def test_module_references_no_external_llm_endpoints():
    source = Path(la.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "api.openai.com",
        "anthropic.com",
        "generativelanguage.googleapis",
        "localhost:11434",
        "huggingface.co",
        "api.cohere.com",
        "openrouter.ai",
    ):
        assert forbidden not in source, f"external endpoint referenced: {forbidden}"


def test_generation_path_has_no_network_fallback():
    source = Path(la.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("requests.post", "urllib.request", "httpx.", "telemetry"):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# 6. Missing / incompatible checkpoint handling
# --------------------------------------------------------------------------- #
def test_load_runtime_missing_checkpoint(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        la.load_runtime(str(tmp_path / "missing.pt"), str(TOK))
    assert "Checkpoint not found" in str(excinfo.value)


def test_load_runtime_missing_tokenizer(tmp_path):
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"")  # tokenizer missing check fires first
    with pytest.raises(RuntimeError) as excinfo:
        la.load_runtime(str(fake_ckpt), str(tmp_path / "missing_tok.json"))
    assert "Tokenizer not found" in str(excinfo.value)


def test_load_runtime_rejects_checkpoint_without_model_config(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"step": 3, "model": {}}, bad)
    with pytest.raises(RuntimeError) as excinfo:
        la.load_runtime(str(bad), str(TOK))
    assert "Failed to load checkpoint" in str(excinfo.value)


def test_load_runtime_rejects_incompatible_tokenizer(tmp_path, monkeypatch):
    """A tokenizer whose vocabulary size does not match the checkpoint must be
    refused, not silently paired."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models

    small_tok_path = tmp_path / "tiny_tok.json"
    tiny = Tokenizer(models.BPE(vocab={"<bos>": 0, "<eos>": 1, "a": 2}, merges=[]))
    tiny.save(str(small_tok_path))

    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")  # never read: load_model is patched

    class _FakeModel:
        def num_params(self):
            return 1

    cfg = {"vocab_size": 5822, "d_model": 384, "n_layers": 12, "max_seq_len": 2048}
    monkeypatch.setattr(la, "load_model", lambda *a, **k: (_FakeModel(), cfg))

    with pytest.raises(RuntimeError) as excinfo:
        la.load_runtime(str(fake_ckpt), str(small_tok_path))
    assert "mismatch" in str(excinfo.value).lower()
