"""Tokenizer fingerprinting (L1.1.1).

Two independent fingerprints for the same artifact:

- ``file_fingerprint``: SHA-256 of the raw bytes — detects any byte-level change
  (whitespace, key order, metadata).
- ``vocabulary_digest``: SHA-256 of the canonical serialization of the
  ``{token: id}`` mapping — detects any *vocabulary* change while remaining
  invariant to whitespace-only reformatting.

Neither reads the model. Neither rewrites, regenerates or normalizes the
artifact. Mismatch paths raise :class:`TokenizerFingerprintMismatch`; no path
silently accepts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class TokenizerFingerprintMismatch(RuntimeError):
    """The tokenizer does not match the expected fingerprint."""


def file_fingerprint(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vocabulary_digest(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    vocab = doc.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(
            f"{path} has no `model.vocab` mapping — cannot fingerprint it; "
            f"this is not a BPE tokenizer JSON")
    canonical = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SPECIAL_TOKEN_IDS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}


def assert_special_token_ids(path: str | Path) -> dict[str, int]:
    """Verify the <pad>/<bos>/<eos>/<unk> ids are exactly the pinned values."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    vocab = doc.get("model", {}).get("vocab", {})
    resolved = {}
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = vocab.get(token)
        if actual != expected:
            raise TokenizerFingerprintMismatch(
                f"special token {token!r} has id {actual!r}, expected {expected} "
                f"(tokenizer contract violation)")
        resolved[token] = int(expected)
    return resolved


def verify_against(path: str | Path, *, expected_file_fingerprint: str | None = None,
                   expected_vocabulary_digest: str | None = None) -> dict[str, str | dict]:
    """Verify a tokenizer against optional expected fingerprints.

    Returns the measured fingerprints if they match; raises otherwise. Passing
    ``None`` for both expected values measures and returns without comparing.
    """
    file_hash = file_fingerprint(path)
    vocab_hash = vocabulary_digest(path)
    if expected_file_fingerprint is not None and file_hash != expected_file_fingerprint:
        raise TokenizerFingerprintMismatch(
            f"tokenizer {path} file fingerprint mismatch: {file_hash} "
            f"!= {expected_file_fingerprint}")
    if expected_vocabulary_digest is not None and vocab_hash != expected_vocabulary_digest:
        raise TokenizerFingerprintMismatch(
            f"tokenizer {path} vocabulary digest mismatch: {vocab_hash} "
            f"!= {expected_vocabulary_digest}")
    return {"file_sha256": file_hash, "vocabulary_digest": vocab_hash,
            "special_token_ids": assert_special_token_ids(path)}
