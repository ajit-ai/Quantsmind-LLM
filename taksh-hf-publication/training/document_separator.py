"""Document boundary semantics for packed token streams (A3 / B6).

``data/encode_dataset.py`` used to concatenate every corpus record with no
separator at all, while ``TokenDataset.batch`` samples random offsets across the
whole stream. A training window therefore routinely contained the tail of one
unrelated file followed by the head of another, with nothing marking the join.
The model was trained to believe any file continues into any other.

Separator choice
----------------
The shipped code tokenizer (``tokenizer/qm_code_bpe_v0.json``) defines exactly
four specials: ``<pad> <bos> <eos> <unk>``. There is no ``<sep>``, and inventing
one would require adding vocabulary and retraining the tokenizer, so this module
uses **``<eos>``** as the document separator:

* it already exists in the vocabulary, so no token id can be out of range;
* it already carries "this sequence ends here" meaning, and the serving stack
  already stops generation on it, so preprocessing, training and inference agree;
* using it means the model learns to emit it at document boundaries, which is
  exactly what inference needs.

A blank line or ``\\n\\n`` was rejected because the BPE vocabulary contains no
token for it (``token_to_id("\\n\\n")`` is ``None``); it would be shredded into
ordinary byte tokens and would not be distinguishable from content.

Oversized documents
-------------------
A record longer than ``seq_len`` is **not** truncated and **not** split. It is
written whole into the stream and simply spans consecutive training windows. This
is the documented behaviour: the encoder never destroys content, and the reader
windows across boundaries only where a document genuinely continues.
"""

from __future__ import annotations

DOC_SEPARATOR_NAME = "<eos>"
DOC_SEPARATOR_ID = 2  # <eos> in the shipped code tokenizer; always re-resolved


class SeparatorUnavailable(RuntimeError):
    """The tokenizer does not define the document separator token."""


def resolve_separator_id(tokenizer) -> int:
    """Return the token id used to mark a document boundary.

    Raises ``SeparatorUnavailable`` with an actionable message rather than
    silently falling back to a byte sequence that the vocabulary cannot represent.
    """
    token_id = tokenizer.token_to_id(DOC_SEPARATOR_NAME)
    if token_id is None:
        raise SeparatorUnavailable(
            f"tokenizer has no {DOC_SEPARATOR_NAME!r} token, so document "
            "boundaries cannot be marked. Refusing to write a stream whose "
            "documents run into each other. Either use a tokenizer that defines "
            f"{DOC_SEPARATOR_NAME!r} or add it via tokenizer/train_tokenizer.py."
        )
    return int(token_id)


def encode_records(
    tokenizer,
    records,
    max_tokens: int | None = None,
    skip_empty: bool = True,
) -> tuple[list[int], int]:
    """Encode records into one flat id stream, separated by the separator token.

    Returns ``(ids, records_emitted)``.

    ``skip_empty`` drops records that encode to nothing. A dropped record does
    *not* contribute a separator either, so the stream never contains two
    separators in a row and never begins or ends with one.

    ``max_tokens`` truncates **at document granularity**. A document is never cut
    in half: if the next document would exceed the budget, encoding stops at the
    last whole document. The single exception is a document that is larger than
    the entire budget on its own -- there is nothing whole to fall back to, so it
    is truncated and reported via the returned count. Cutting mid-document would
    leave a window that opens with the tail of a document and closes with nothing,
    which is the exact corruption the separator exists to prevent.
    """
    sep = resolve_separator_id(tokenizer)
    ids: list[int] = []
    emitted = 0
    truncated = False

    for record in records:
        chunk = tokenizer.encode(record).ids
        if skip_empty and not chunk:
            continue

        if max_tokens is not None:
            needed = len(chunk) + (1 if ids else 0)
            if len(ids) + needed > max_tokens:
                if not ids:
                    # Nothing emitted yet and this document cannot fit: emit as
                    # much as the budget allows rather than returning nothing.
                    ids.extend(chunk[:max_tokens])
                    emitted += 1
                    truncated = True
                break

        if ids:
            ids.append(sep)
        ids.extend(chunk)
        emitted += 1

    return ids, emitted