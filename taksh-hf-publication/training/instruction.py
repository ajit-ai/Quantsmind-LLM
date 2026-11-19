"""Instruction-learning contract and supervised sequence construction (v1.5.0).

This is the *engineering foundation* for supervised instruction tuning. It
does not assume that an approved instruction dataset exists, and it cannot
mark a checkpoint instruction-qualified: that label is only ever set by an
authorized training and evaluation process, never by the fact that this
module runs.

The contract is versioned and every example must carry provenance and an
authorization state. In production mode an example without provenance, with an
unresolved licence decision, or with a contamination class that suggests it
was seen in training is rejected.

Sequence construction reuses the v1.1.0 prompt contract and the v1.3.0
objective: the prompt half is masked out of the loss, the response is
supervised, and the terminator is the tokenizer's own end-of-document token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INSTRUCTION_SCHEMA_VERSION = "quantsmind.instruction_example/1"

#: Authorization states an example may declare.
AUTHORIZATION_STATES = ("approved", "pending", "rejected", "unresolved")

#: Contamination classes, mirroring the evaluation vocabulary.
CONTAMINATION_CLASSES = ("authored", "published_suite", "seen_in_training", "unknown")

REQUIRED_FIELDS = (
    "example_id",
    "language",
    "instruction",
    "response",
    "provenance",
    "authorization",
    "contamination",
    "prompt_contract",
    "token_budget",
    "split",
)


class InstructionContractError(ValueError):
    """An instruction example does not satisfy the contract."""


@dataclass
class InstructionExample:
    """One supervised instruction example."""

    example_id: str
    language: str
    instruction: str
    response: str
    context: str = ""
    provenance: dict = field(default_factory=dict)
    authorization: str = "pending"
    contamination: str = "unknown"
    prompt_contract: str = "qm.chat_instruction/2"
    token_budget: int = 512
    split: str = "train"

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "language": self.language,
            "instruction": self.instruction,
            "response": self.response,
            "context": self.context,
            "provenance": dict(self.provenance),
            "authorization": self.authorization,
            "contamination": self.contamination,
            "prompt_contract": self.prompt_contract,
            "token_budget": self.token_budget,
            "split": self.split,
        }


def validate_example(example: dict, *, production: bool = False) -> list:
    """Return every problem with an example.

    ``production=True`` additionally requires approved provenance, an approved
    authorization state and a non-training contamination class.
    """
    problems: list = []
    if not isinstance(example, dict):
        return ["example must be a mapping"]
    for field in REQUIRED_FIELDS:
        if field not in example:
            problems.append(f"missing '{field}'")
    if not isinstance(example.get("instruction"), str) or not example.get("instruction", "").strip():
        problems.append("instruction must be a non-empty string")
    if not isinstance(example.get("response"), str) or not example.get("response", "").strip():
        problems.append("response must be a non-empty string")
    if example.get("authorization") not in AUTHORIZATION_STATES:
        problems.append(
            f"authorization must be one of {list(AUTHORIZATION_STATES)}"
        )
    if example.get("contamination") not in CONTAMINATION_CLASSES:
        problems.append(f"contamination must be one of {list(CONTAMINATION_CLASSES)}")
    provenance = example.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source"):
        problems.append("provenance.source is required")
    if int(example.get("token_budget", 0) or 0) <= 0:
        problems.append("token_budget must be positive")
    if example.get("split") not in ("train", "val", "test"):
        problems.append("split must be train, val or test")
    if production:
        if example.get("authorization") != "approved":
            problems.append(
                f"production mode requires an approved authorization state, got "
                f"{example.get('authorization')!r}"
            )
        if example.get("contamination") == "seen_in_training":
            problems.append("production mode rejects examples seen in training")
        if not isinstance(provenance, dict) or not provenance.get("license"):
            problems.append("production mode requires provenance.license")
    return problems


def load_examples(path, *, production: bool = False) -> list:
    """Load and validate a JSONL instruction file."""
    import json
    from pathlib import Path

    examples = []
    problems = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        example = json.loads(line)
        issues = validate_example(example, production=production)
        if issues:
            problems.extend(f"line {number}: {issue}" for issue in issues)
        examples.append(example)
    if problems:
        raise InstructionContractError("; ".join(problems))
    return examples


def build_supervised_sequence(tokenizer, example: dict, *, max_length: int = 512) -> dict:
    """Build one supervised sequence with response-only loss masking.

    The prompt half is rendered through the declared chat contract and
    masked out of the loss; the response and the terminator are supervised.
    The result records exactly which token positions are supervised, so a
    reviewer can check the mask without re-deriving it.
    """
    from training.objective import IGNORE_INDEX

    contract = example.get("prompt_contract", "qm.chat_instruction/2")
    if contract != "qm.chat_instruction/2":
        raise InstructionContractError(
            f"unsupported prompt contract {contract!r}; this pipeline supports "
            "'qm.chat_instruction/2'"
        )
    from training.chat_template import render_chat

    user_content = example["instruction"]
    if example.get("context"):
        user_content = f"{example['context']}\n\n{example['instruction']}"
    prompt = render_chat(
        tokenizer, [{"role": "user", "content": user_content}], default_system=""
    )
    response = example["response"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    response_ids = tokenizer.encode(response, add_special_tokens=False).ids
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    budget = min(int(example.get("token_budget", max_length)), int(max_length))
    ids = (prompt_ids + response_ids)[:budget]
    prompt_len = min(len(prompt_ids), len(ids))
    labels = list(ids[1:])
    for index in range(min(prompt_len, len(labels))):
        labels[index] = IGNORE_INDEX
    return {
        "ids": ids,
        "labels": labels,
        "prompt_tokens": prompt_len,
        "response_tokens": max(0, len(ids) - prompt_len),
        "supervised_tokens": sum(1 for label in labels if label != IGNORE_INDEX),
        "truncated": len(prompt_ids) + len(response_ids) > budget,
        "prompt_contract": contract,
    }
