"""Tokenizer control-token contract for ``qm.chat_instruction/2`` (v1.1.0).

The v1.0.0 tokenizers register only ``<bos>`` and ``<eos>``. A chat contract built
on those two markers cannot say *who* is speaking, so the model is asked to infer
turn structure from prose. Worse, any renderer that writes ``<|system|>`` into the
text gets byte fragments back, because the marker is not a token.

This module makes the reserved range explicit and checkable:

* :data:`CONTROL_TOKENS` is the reserved set (system, user, assistant,
  end-of-turn, tool, fill-in-the-middle). Each must encode to exactly one id.
* :func:`build_contract_tokenizer` derives a v2 tokenizer from an existing
  artifact by **appending** the control tokens. No BPE merges are retrained and
  no existing id changes, so the derivation is deterministic and auditable.
* :func:`detect_contract` reports which contract a tokenizer can serve. It never
  guesses: an unregistered marker means the tokenizer cannot serve v2.
* :func:`tokenizer_fingerprint` hashes the serialized tokenizer, so a checkpoint
  can bind the exact tokenizer it was trained with.

Scope note: v1.1.0 reserves and validates these tokens; shipping a v2 tokenizer
artifact and resizing embeddings for the S2 rung belongs to v1.2.0/v1.3.0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CONTRACT_V1 = "qm.plain_instruction/1"
CONTRACT_V2 = "qm.chat_instruction/2"

#: Reserved single-token markers, in a fixed order. Appending them in this order
#: is what makes derived tokenizers byte-reproducible.
CONTROL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|tool|>",
    "<|fim|>",
)

BOS = "<bos>"
EOS = "<eos>"


class TokenizerContractError(RuntimeError):
    """The tokenizer cannot represent the requested prompt contract."""


def contract_ids(tokenizer) -> dict[str, int] | None:
    """Return ``{marker: id}`` if every control token is a single registered token.

    ``None`` means this tokenizer does not support contract v2. Partial
    registration is treated as unsupported: a half-marked vocabulary is more
    dangerous than an obviously legacy one.
    """
    resolved: dict[str, int] = {}
    for marker in CONTROL_TOKENS:
        token_id = tokenizer.token_to_id(marker)
        if token_id is None:
            return None
        # A registered token must decode back to itself; otherwise it is an alias
        # that would render as different text than it encodes.
        if tokenizer.id_to_token(int(token_id)) != marker:
            return None
        resolved[marker] = int(token_id)
    return resolved


def detect_contract(tokenizer) -> str:
    """Return the strongest contract this tokenizer can serve, or raise."""
    if contract_ids(tokenizer) is not None:
        return CONTRACT_V2
    if tokenizer.token_to_id(BOS) is not None and tokenizer.token_to_id(EOS) is not None:
        return CONTRACT_V1
    raise TokenizerContractError(
        "tokenizer registers neither the v2 control tokens nor the legacy "
        "<bos>/<eos> pair; no prompt contract can be rendered without shredding "
        "markers into byte fragments"
    )


def require_contract(tokenizer, contract_id: str) -> dict[str, int] | None:
    """Verify that ``tokenizer`` can serve ``contract_id``; return its marker ids."""
    available = detect_contract(tokenizer)
    if contract_id == CONTRACT_V2:
        ids = contract_ids(tokenizer)
        if ids is None:
            raise TokenizerContractError(
                f"contract {contract_id} requires single-token control markers "
                f"{list(CONTROL_TOKENS)}; the tokenizer is only {available}"
            )
        return ids
    if contract_id == CONTRACT_V1:
        if tokenizer.token_to_id(BOS) is None or tokenizer.token_to_id(EOS) is None:
            raise TokenizerContractError(
                f"contract {contract_id} requires <bos>/<eos> single tokens"
            )
        return None
    raise TokenizerContractError(f"unknown prompt contract: {contract_id!r}")


def build_contract_tokenizer(source: str | Path, *, verify: bool = True):
    """Derive a contract-v2 tokenizer from ``source`` without retraining.

    The source artifact is copied, then the control tokens are appended as
    special tokens. Existing ids are untouched. The result is deterministic:
    the same input artifact and tokenizer library version produce the same
    serialized bytes and therefore the same fingerprint.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(source))
    if contract_ids(tok) is None:
        tok.add_special_tokens(list(CONTROL_TOKENS))
    if verify:
        ids = contract_ids(tok)
        if ids is None:
            raise TokenizerContractError(
                "derived tokenizer still lacks single-token control markers"
            )
        for marker in CONTROL_TOKENS:
            enc = tok.encode(marker, add_special_tokens=False)
            if enc.ids != [ids[marker]]:
                raise TokenizerContractError(
                    f"{marker} does not encode to exactly one id: {enc.ids}"
                )
    return tok


def tokenizer_fingerprint(tokenizer) -> str:
    """SHA-256 of the serialized tokenizer (the artifact serving must match)."""
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def tokenizer_file_fingerprint(path: str | Path) -> str:
    """SHA-256 of a tokenizer file on disk, byte for byte."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
