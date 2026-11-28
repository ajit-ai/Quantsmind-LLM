"""Fill-in-the-middle (FIM) text protocol for Taksh completions.

What this is
------------
A deterministic way to turn ``(prefix, suffix)`` into one prompt and to report
which part of the completion is the middle. It is shared by every serving
surface so a client sees one convention, not one per endpoint.

What this is **not**
--------------------
The shipped tokenizers (``qm_code_bpe_v0``, ``taksh_code_bpe_v1``) register
only ``<pad>``, ``<bos>``, ``<eos>`` and ``<unk>``. The sentinels below are
therefore **literal ASCII text**, not vocabulary tokens: the tokenizer encodes
them as ordinary byte fragments. That is stated rather than hidden because the
repository previously shipped a template that *pretended* markers were special
tokens (see ``docs/OPEN_ISSUES.md`` C3).

Promoting these sentinels to real control tokens is a vocabulary change:
new tokenizer, resized embeddings, retrained checkpoint. It is not a serving
flag, and this module does not do it.

The current checkpoint has never been trained on this protocol, so FIM output
quality is **not claimed**. The endpoint exists so an editor can speak a stable
contract to the local model and so future training can target the same bytes.
"""

from __future__ import annotations

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

FIM_PROTOCOL_ID = "taksh.fim_text/1"


def render_fim(prefix: str, suffix: str = "") -> str:
    """Render a FIM prompt. Without a suffix this is ordinary completion."""
    if suffix:
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    return prefix


def protocol_note() -> str:
    """One-line honesty note for API metadata."""
    return (
        f"{FIM_PROTOCOL_ID}: sentinels are literal text, not tokenizer special "
        "tokens; the checkpoint is not FIM-trained and no quality is claimed"
    )
