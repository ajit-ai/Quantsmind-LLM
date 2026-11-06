"""Tokenization and sequence construction for Taksh v1.3.0.

Two policies are implemented and both are explicit.

``document`` (default)
    Every document is tokenized on its own, optionally terminated with an
    explicit end-of-document token, and split into windows of at most
    ``context_length`` tokens. A window therefore never contains information
    from two documents, and the attention pattern inside a window cannot leak
    across a document boundary.

``packed``
    Documents are concatenated (each terminated) into a fixed-length stream.
    This is more token-efficient, but a window then attends across document
    boundaries. Packing therefore has to be *asked for* explicitly
    (``allow_cross_document=True``); the default is the document-preserving
    policy, and the choice is recorded in the run manifest.

Reserved-token safety is enforced at this boundary, not only in the data
pipeline:

* source text containing a reserved spelling is refused
  (``reserved_literal_in_source``) -- the file is never rewritten;
* a produced id in the control range is refused unless it is explicitly
  authorised (``unauthorized_control_token``);
* an id beyond the model's embedding width is refused
  (``token_id_beyond_model_vocab``).

Padding is suffix-only. Under causal masking a suffix cannot influence the
positions before it, so a padded tail is safe even though this model has no
padding attention mask; the invariant is asserted by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from training.objective import IGNORE_INDEX

RESERVED_LITERALS = (
    "<bos>",
    "<eos>",
    "<unk>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|tool|>",
    "<|fim|>",
)


class SequenceError(RuntimeError):
    """A sequence could not be built safely."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class TokenizedSample:
    """One tokenized document."""

    source_id: str
    file_path: str
    token_ids: list = field(default_factory=list)


@dataclass
class Batch:
    """A padded microbatch with its labels and attention mask."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    supervised_tokens: int
    document_count: int
    cross_document: bool = False


class SequenceBuilder:
    """Turns raw documents into training windows.

    The builder is the only place where tokenization happens for pretraining,
    so the reserved-token policy, the model vocabulary bound and the boundary
    policy are enforced once and are testable in isolation.
    """

    def __init__(
        self,
        tokenizer,
        *,
        context_length: int,
        model_vocab: int | None = None,
        policy: str = "document",
        append_eos: bool = True,
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        control_token_floor: int | None = None,
        authorized_control_ids=(),
        reserved_literals=RESERVED_LITERALS,
        allow_cross_document: bool = False,
    ):
        if context_length < 2:
            raise SequenceError(
                "invalid_context_length", f"context_length={context_length}"
            )
        if policy not in ("document", "packed"):
            raise SequenceError("unknown_policy", f"policy={policy!r}")
        if policy == "packed" and not allow_cross_document:
            raise SequenceError(
                "cross_document_packing_not_authorised",
                "packed sequences let a window attend across document boundaries; "
                "pass allow_cross_document=True to accept that explicitly",
            )
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.model_vocab = model_vocab
        self.policy = policy
        self.append_eos = bool(append_eos)
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.control_token_floor = control_token_floor
        self.authorized_control_ids = set(int(i) for i in authorized_control_ids)
        self.reserved_literals = tuple(reserved_literals)
        self.allow_cross_document = bool(allow_cross_document)
        self.eos_id = tokenizer.token_to_id(eos_token) if eos_token else None
        self.pad_id = tokenizer.token_to_id(pad_token) if pad_token else None
        if self.pad_id is None:
            raise SequenceError("pad_token_missing", f"token {pad_token!r} not found")
        self.token_count = 0
        self.document_count = 0

    # ------------------------------------------------------------------ safety
    def _check_source_text(self, text: str) -> None:
        for literal in self.reserved_literals:
            if literal in text:
                raise SequenceError(
                    "reserved_literal_in_source",
                    f"source text contains the reserved spelling {literal!r}; "
                    "such a sample would inject a control id into the token "
                    "stream and is refused rather than rewritten",
                )

    def _check_ids(self, ids) -> None:
        for token_id in ids:
            if self.model_vocab is not None and token_id >= self.model_vocab:
                raise SequenceError(
                    "token_id_beyond_model_vocab",
                    f"id {token_id} >= model vocab {self.model_vocab}",
                )
            if (
                self.control_token_floor is not None
                and token_id >= self.control_token_floor
                and token_id not in self.authorized_control_ids
            ):
                raise SequenceError(
                    "unauthorized_control_token",
                    f"id {token_id} is in the reserved control range "
                    f"(>= {self.control_token_floor}) and is not authorised",
                )

    # -------------------------------------------------------------- tokenizing
    def encode(self, text: str) -> list:
        """Tokenize one document, refusing reserved spellings and bad ids."""
        self._check_source_text(text)
        ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
        self._check_ids(ids)
        self.token_count += len(ids)
        self.document_count += 1
        return ids

    def encode_document(self, text: str) -> list:
        """Tokenize a document and append the end-of-document token.

        The terminator is inserted by the builder, not present in the source,
        which is why the reserved-literal check does not reject it.
        """
        ids = self.encode(text)
        if self.append_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    # -------------------------------------------------------------- sequences
    def build_windows(self, documents) -> list:
        """Build training windows from ``(source_id, file_path, text)`` tuples.

        Returns a list of ``(token_ids, document_count, cross_document)``.
        """
        if self.policy == "document":
            return self._build_document_windows(documents)
        return self._build_packed_windows(documents)

    def _build_document_windows(self, documents) -> list:
        windows = []
        for source_id, file_path, text in documents:
            ids = self.encode_document(text)
            for start in range(0, len(ids), self.context_length):
                chunk = ids[start : start + self.context_length]
                if len(chunk) < 2:
                    # A window shorter than two tokens cannot produce a single
                    # next-token prediction, so it carries no training signal.
                    continue
                windows.append((chunk, 1, False))
        return windows

    def _build_packed_windows(self, documents) -> list:
        stream = []
        for source_id, file_path, text in documents:
            stream.extend(self.encode_document(text))
        windows = []
        for start in range(0, len(stream), self.context_length):
            chunk = stream[start : start + self.context_length]
            if len(chunk) < 2:
                continue
            windows.append((chunk, 1, True))
        return windows

    # ------------------------------------------------------------------ batches
    def to_batch(self, windows) -> Batch:
        """Pad a group of windows into a microbatch.

        Labels are the next-token targets of each window; padding positions
        are masked with ``IGNORE_INDEX`` and never predicted. The attention
        mask marks the valid prefix of every row.
        """
        if not windows:
            raise SequenceError("empty_batch", "no windows to batch")
        from training.objective import shift_for_causal

        rows = []
        labels_rows = []
        masks = []
        supervised = 0
        cross = False
        for token_ids, document_count, is_cross in windows:
            tokens = torch.tensor(token_ids, dtype=torch.long)
            inputs, labels = shift_for_causal(tokens, self.pad_id)
            rows.append(inputs)
            labels_rows.append(labels)
            masks.append(torch.ones_like(inputs))
            supervised += int((labels != IGNORE_INDEX).sum().item())
            cross = cross or bool(is_cross)
        length = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), length), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for index, (row, label_row, mask) in enumerate(zip(rows, labels_rows, masks)):
            size = row.numel()
            input_ids[index, :size] = row
            labels[index, :size] = label_row
            attention_mask[index, :size] = mask
        return Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            supervised_tokens=supervised,
            document_count=len(windows),
            cross_document=cross,
        )

    def statistics(self) -> dict:
        """Token accounting for the run manifest."""
        return {
            "tokens_encoded": self.token_count,
            "documents_encoded": self.document_count,
            "policy": self.policy,
            "context_length": self.context_length,
            "append_eos": self.append_eos,
            "cross_document_allowed": self.allow_cross_document,
        }
