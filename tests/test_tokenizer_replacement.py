"""Tests for v1.5.0 tokenizer replacement.

Verifies that:
1. The new cleared tokenizer (taksh_code_bpe_v1.json) loads successfully.
2. Its vocab size matches the model embedding width (5,822).
3. Round-trip encoding/decoding works for code samples.
4. The default CLI tokenizer path points to the new cleared tokenizer.
5. Old uncleared tokenizer (qm_code_bpe_v0.json) is NOT the default anymore.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
NEW_TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
OLD_TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"


def test_new_tokenizer_loads():
    """The new cleared tokenizer file exists and loads."""
    assert NEW_TOKENIZER.is_file(), f"new tokenizer missing: {NEW_TOKENIZER}"
    tok = Tokenizer.from_file(str(NEW_TOKENIZER))
    assert tok is not None


def test_new_tokenizer_vocab_matches_model():
    """New tokenizer vocab size equals the model's expected vocab width."""
    tok = Tokenizer.from_file(str(NEW_TOKENIZER))
    assert tok.get_vocab_size() == 5822


def test_new_tokenizer_roundtrip():
    """Code samples round-trip through the new tokenizer without corruption."""
    tok = Tokenizer.from_file(str(NEW_TOKENIZER))
    samples = [
        "def add(a, b):\n    return a + b\n",
        "def multiply(x, y):\n    return x * y\n",
        "def is_even(n):\n    return n % 2 == 0\n",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
        "import os\nimport sys\nfrom pathlib import Path\n",
    ]
    for s in samples:
        ids = tok.encode(s).ids
        decoded = tok.decode(ids)
        assert decoded == s, f"roundtrip failed for {s[:40]!r}: {decoded!r}"


def test_new_tokenizer_has_special_tokens():
    """The new tokenizer registers the required special tokens."""
    tok = Tokenizer.from_file(str(NEW_TOKENIZER))
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<bos>") == 1
    assert tok.token_to_id("<eos>") == 2


def test_default_cli_tokenizer_is_cleared():
    """serving/server_security.py defaults to the new cleared tokenizer."""
    src = Path("serving/server_security.py").read_text(encoding="utf-8")
    assert 'default="tokenizer/taksh_code_bpe_v1.json"' in src


def test_default_local_inference_tokenizer_is_cleared():
    """serving/local_inference.py defaults to the new cleared tokenizer."""
    src = Path("serving/local_inference.py").read_text(encoding="utf-8")
    assert 'default="tokenizer/taksh_code_bpe_v1.json"' in src


def test_old_uncleared_tokenizer_not_default():
    """Neither inference entry point defaults to the uncleared qm_code_bpe_v0.json."""
    for f in ["serving/server_security.py", "serving/local_inference.py"]:
        src = Path(f).read_text(encoding="utf-8")
        assert 'default="tokenizer/qm_code_bpe_v0.json"' not in src, (
            f"{f} still defaults to uncleared tokenizer"
        )


def test_corpus_manifest_exists():
    """The tokenizer corpus manifest documents provenance."""
    manifest = REPO / "data" / "tokenizer_corpus" / "taksh_code_v1.manifest.json"
    assert manifest.is_file(), f"corpus manifest missing: {manifest}"
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["file_count"] > 0
    assert data["total_bytes"] > 0
    assert "corpus_sha256" in data
