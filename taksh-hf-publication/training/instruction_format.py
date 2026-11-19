"""Instruction formatting and assistant-only loss masking (M4).

The format is plain text with ``<bos>``/``<eos>`` boundaries -- the two tokens
``tokenizer/qm_code_bpe_v0.json`` actually registers (ids 1 and 2). **No special
tokens are invented**, and no chat template is used, because that tokenizer has
no role markers: rendering ``<|user|>`` would shred it into byte fragments and
the model would never learn a real turn delimiter.

::

    <bos>### Instruction
    {instruction}

    ### Input
    {context}

    ### Response
    {response}<eos>

``### Input`` is omitted entirely when the example has no context field.

Why the prompt and completion are tokenized *separately*
---------------------------------------------------------

The obvious implementation encodes the whole rendered string, then re-renders
and encodes the prompt alone to work out where supervision starts. That is what
``training/sft.py`` does, and it is fragile: tokenization is not guaranteed to
be prefix-consistent at the boundary, so the mask can land one token early or
late, and a token at the boundary is silently supervised or silently ignored.

Here the prompt ids and the completion ids are produced independently and
concatenated. The supervision boundary is then exact *by construction* -- it is
the length of the prompt id list, not a guess. The tests assert this.

Truncation policy
-----------------

An example that does not fit is truncated from the **left of the prompt** (its
tail is the most recent context), never from the right. If the completion alone
still does not fit, the example is rejected: silently truncating a target
response leaves nothing to learn from while looking like a successful dataset.
"""

from __future__ import annotations

IGNORE_INDEX = -100

PROMPT_HEADER = "### Instruction"
INPUT_HEADER = "### Input"
RESPONSE_HEADER = "### Response"


class FormattingError(ValueError):
    """An example cannot be rendered or cannot fit; it must not be trained on."""


def render_prompt(instruction: str, context: str = "") -> str:
    """The prompt half of an example, without boundary tokens."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise FormattingError("instruction is empty; nothing to prompt with")
    parts = [PROMPT_HEADER, instruction.strip()]
    if isinstance(context, str) and context.strip():
        parts += ["", INPUT_HEADER, context.strip()]
    return "\n".join(parts) + "\n\n" + RESPONSE_HEADER + "\n"


def render_completion(response: str) -> str:
    if not isinstance(response, str) or not response.strip():
        raise FormattingError(
            "response is empty; an example with no target cannot be supervised"
        )
    return response


def encode_example(tokenizer, instruction: str, response: str, context: str = "",
                   max_length: int = 512, bos: str = "<bos>", eos: str = "<eos>",
                   supervise_prompt: bool = False):
    """Tokenize one example and build its loss mask.

    Returns a dict with ``input_ids``, ``labels``, ``prompt_tokens``,
    ``supervised_tokens``, and a ``truncated``/``rejected`` record explaining
    what happened. Raises :class:`FormattingError` when the example cannot be
    used at all.

    ``supervise_prompt=True`` trains on the whole sequence (used for
    continued pretraining-style tuning), but the default is assistant-only.
    """
    bos_id = tokenizer.token_to_id(bos)
    eos_id = tokenizer.token_to_id(eos)
    if bos_id is None or eos_id is None:
        raise FormattingError(
            f"tokenizer does not register both {bos!r} and {eos!r}; refusing to "
            "fabricate boundary tokens"
        )

    prompt_text = render_prompt(instruction, context)
    completion_text = render_completion(response)

    prompt_ids = tokenizer.encode(prompt_text).ids
    completion_ids = tokenizer.encode(completion_text).ids

    ids = [bos_id] + prompt_ids + completion_ids + [eos_id]
    prompt_len = 1 + len(prompt_ids)

    notes = {"truncated": False, "truncated_prompt_tokens": 0, "rejected": None}

    if len(ids) > max_length:
        # Keep the completion whole; trim the prompt's front.
        budget = max_length - len(completion_ids) - 1  # 1 = trailing eos
        if budget < 1:
            notes["rejected"] = (
                f"completion alone is {len(completion_ids) + 1} tokens but "
                f"max_length is {max_length}; refusing to truncate the target"
            )
            raise FormattingError(notes["rejected"])
        keep = max(1, budget - 1)  # 1 = leading bos
        dropped = max(0, len(prompt_ids) - keep)
        ids = [bos_id] + prompt_ids[-keep:] + completion_ids + [eos_id]
        prompt_len = 1 + keep
        notes["truncated"] = True
        notes["truncated_prompt_tokens"] = dropped

    if len(ids) < 2:
        raise FormattingError("example encodes to fewer than 2 tokens")

    input_ids = ids[:-1]
    labels = list(ids[1:])

    if not supervise_prompt:
        masked = min(prompt_len, len(labels))
        for position in range(masked):
            labels[position] = IGNORE_INDEX

    supervised = sum(1 for label in labels if label != IGNORE_INDEX)
    if supervised == 0:
        raise FormattingError(
            "masking left zero supervised tokens; the example would contribute "
            "nothing to the loss"
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_tokens": prompt_len,
        "supervised_tokens": supervised,
        "total_tokens": len(input_ids),
        "bos_id": bos_id,
        "eos_id": eos_id,
        **notes,
    }


def collate(batch, pad_id=0):
    """Right-pad a batch. Padding is masked, and causal attention makes
    right-padding safe: a real position never attends to a later pad token."""
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[index, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention[index, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention,
    }


def describe_mask(example, tokenizer=None) -> dict:
    """Human-readable view of where supervision starts, for reports and tests."""
    labels = example["labels"]
    total = len(labels)
    masked = sum(1 for label in labels if label == IGNORE_INDEX)
    described = {
        "total_positions": total,
        "masked_positions": masked,
        "supervised_positions": total - masked,
        "supervision_starts_at": total - (total - masked),
        "truncated": example.get("truncated", False),
    }
    if tokenizer is not None:
        first = total - (total - masked)
        described["first_supervised_text"] = tokenizer.decode(
            labels[first:first + 24]
        ).replace("\n", "\\n")
    return described