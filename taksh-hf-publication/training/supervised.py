"""Instruction-to-code supervision with response-only loss masking (v2.1.0).

The v2.0.0 finding was that the *training text did not contain the
instruction*: the curriculum trained on bare reference code while the
evaluator prompted with an instruction plus a signature. A causal model can
only condition on tokens it has seen, so that setup could never teach
instruction-to-operation mapping, whatever the learning rate.

This module fixes the supervision path for the synthetic training only:

* the **training text** is ``instruction + signature + body`` — the model sees
  the instruction and then continues with the requested implementation;
* the **loss** is masked to the response (body plus the end-of-document
  token), so the objective is "given this instruction and signature, produce
  this body", and prompt tokens are never a learning target;
* the **evaluation prompt** is exactly the prefix the model was trained on,
  so training and inference agree.

Nothing here touches production serving, the tokenizer, or the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from training.objective import IGNORE_INDEX

PROMPT_SEPARATOR = "\n"


class SupervisionError(ValueError):
    """A supervised example is not safe to train on."""


@dataclass
class SupervisedExample:
    """One instruction-to-code example with its supervision metadata."""

    example_id: str
    instruction: str
    signature: str
    body: str
    operation: str
    family: str
    split: str
    seed: int
    content_sha256: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return f"{self.instruction}{PROMPT_SEPARATOR}{self.signature}{PROMPT_SEPARATOR}"

    @property
    def target_text(self) -> str:
        return f"{self.body}{PROMPT_SEPARATOR}"

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.target_text

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "instruction": self.instruction,
            "signature": self.signature,
            "body": self.body,
            "operation": self.operation,
            "family": self.family,
            "split": self.split,
            "seed": self.seed,
            "content_sha256": self.content_sha256,
            "metadata": dict(self.metadata),
        }


def build_example(
    *,
    example_id: str,
    instruction: str,
    signature: str,
    body: str,
    operation: str,
    family: str,
    split: str,
    seed: int,
    metadata: dict | None = None,
) -> SupervisedExample:
    """Build one example and hash its training text."""
    if not instruction.strip():
        raise SupervisionError(f"{example_id}: the instruction is empty")
    if not body.strip():
        raise SupervisionError(f"{example_id}: the target body is empty")
    example = SupervisedExample(
        example_id=example_id,
        instruction=instruction,
        signature=signature,
        body=body,
        operation=operation,
        family=family,
        split=split,
        seed=seed,
        metadata=dict(metadata or {}),
    )
    example.content_sha256 = hashlib.sha256(
        example.full_text.encode("utf-8")
    ).hexdigest()
    return example


def supervise(
    tokenizer,
    example: SupervisedExample,
    *,
    max_length: int = 256,
    eos_token: str = "<eos>",
    required_markers=(),
) -> dict:
    """Tokenise one example and mask the loss to the response only.

    Returns the token ids, the labels, the prompt and target boundaries, the
    supervised and masked positions, the end-of-document position, whether the
    example was truncated, and whether the required operation survived
    truncation.
    """
    prompt_ids = tokenizer.encode(example.prompt_text, add_special_tokens=False).ids
    target_ids = tokenizer.encode(example.target_text, add_special_tokens=False).ids
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise SupervisionError(f"the tokenizer has no {eos_token!r} token")
    target_ids = list(target_ids) + [eos_id]

    ids = list(prompt_ids) + list(target_ids)
    truncated = len(ids) > max_length
    if truncated:
        ids = ids[:max_length]
        target_ids = target_ids[: max(0, max_length - len(prompt_ids))]

    prompt_len = min(len(prompt_ids), len(ids))
    labels = list(ids[1:])
    # ``labels[i]`` predicts ``ids[i + 1]``, so the label that predicts the
    # *first target token* sits at index ``prompt_len - 1``. Only the labels
    # that predict prompt tokens (indices 0 .. prompt_len - 2) are masked.
    # Masking one label too many would hide the first target token, and the
    # model would learn to emit the terminator instead of the body.
    for index in range(max(0, min(prompt_len - 1, len(labels)))):
        labels[index] = IGNORE_INDEX
    supervised = [
        index for index, label in enumerate(labels) if label != IGNORE_INDEX
    ]
    masked = [index for index, label in enumerate(labels) if label == IGNORE_INDEX]
    kept_text = tokenizer.decode(ids[prompt_len:]).strip()
    return {
        "ids": ids,
        "labels": labels,
        "prompt_len": prompt_len,
        "target_len": len(ids) - prompt_len,
        "supervised_positions": supervised,
        "masked_positions": masked,
        "eos_position": len(ids) - 1 if eos_id in ids else None,
        "truncated": truncated,
        "operation_preserved": all(marker in kept_text for marker in required_markers),
        "kept_target_text": kept_text,
    }


def verify_supervision(tokenizer, example: SupervisedExample, **kwargs) -> list:
    """Return every supervision defect found for one example.

    An empty list means the example is safe to train on: the instruction
    precedes the target, the response boundary is explicit, the target tokens
    are supervised, the prompt tokens are masked, the end-of-document token is
    supervised, and truncation did not remove the required operation.
    """
    problems: list = []
    supervision = supervise(tokenizer, example, **kwargs)
    ids = supervision["ids"]
    labels = supervision["labels"]
    if not supervision["supervised_positions"]:
        problems.append("no target token contributes to the loss")
    if supervision["prompt_len"] == 0:
        problems.append("the prompt is empty")
    if supervision["target_len"] == 0:
        problems.append("the target is empty")
    if supervision["eos_position"] is None:
        problems.append("the end-of-document token was not produced")
    if supervision["truncated"] and not supervision["operation_preserved"]:
        problems.append("truncation removed the required operation")
    for index in supervision["masked_positions"]:
        if index >= supervision["prompt_len"]:
            problems.append("a prompt token index is inside the target region")
    if supervision["prompt_len"] and len(supervision["masked_positions"]) != (
        supervision["prompt_len"] - 1
    ):
        problems.append(
            "the number of masked labels is not prompt_len - 1: the first "
            "target token would be hidden from the loss"
        )
    if len(ids) != len(labels) + 1:
        problems.append("the label count is not the token count minus one")
    return problems
