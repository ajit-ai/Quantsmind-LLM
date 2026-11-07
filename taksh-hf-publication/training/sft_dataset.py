"""Instruction-tuning dataset for M4 (M4 training engine input).

Thin wrapper around :func:`training.instruction_format.encode_example` that adds
shuffled epoch cycling, coverage accounting, and an explicit record of every
example that could not be used and why.

Reused from M3 rather than duplicated: ``training/checkpoint.py`` for weight
compatibility, ``training/validation.py`` for validation passes,
``training/manifest.py`` for the run record.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.instruction_format import FormattingError, encode_example


class InstructionDataset:
    """Instruction examples encoded with an exact assistant-only loss mask."""

    def __init__(self, jsonl_path, tokenizer, max_length=512,
                 supervise_prompt=False, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.supervise_prompt = supervise_prompt
        self.seed = seed
        self.path = Path(jsonl_path)

        self.records = []
        malformed = []
        text = self.path.read_text(encoding="utf-8-sig")
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                self.records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                malformed.append({"line": number, "reason": str(exc)})

        self.rejected = []
        self.encoded = []
        for record in self.records:
            try:
                encoded = encode_example(
                    tokenizer,
                    record.get("instruction", ""),
                    record.get("response", ""),
                    record.get("input", "") or "",
                    max_length=max_length,
                    supervise_prompt=supervise_prompt,
                )
            except FormattingError as exc:
                self.rejected.append({
                    "id": record.get("id"), "reason": str(exc),
                })
                continue
            encoded["id"] = record.get("id")
            encoded["task"] = record.get("task")
            self.encoded.append(encoded)

        if not self.encoded:
            raise ValueError(
                f"{self.path}: no example could be encoded. "
                f"{len(self.rejected)} rejected; first: "
                + (self.rejected[0]["reason"] if self.rejected else "none")
            )

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]

    def report(self) -> dict:
        supervised = sum(item["supervised_tokens"] for item in self.encoded)
        total = sum(item["total_tokens"] for item in self.encoded)
        truncated = sum(1 for item in self.encoded if item.get("truncated"))
        return {
            "path": str(self.path),
            "records_read": len(self.records),
            "records_encoded": len(self.encoded),
            "records_rejected": len(self.rejected),
            "rejected_examples": self.rejected[:10],
            "truncated": truncated,
            "total_tokens": total,
            "supervised_tokens": supervised,
            "masked_tokens": total - supervised,
            "supervised_fraction": (supervised / total) if total else 0.0,
            "mean_tokens_per_example": (total / len(self.encoded))
            if self.encoded else 0.0,
            "max_length": self.max_length,
            "supervise_prompt": self.supervise_prompt,
        }


class EpochShuffler:
    """Shuffled, epoch-cycling index stream with coverage accounting.

    The earlier SFT loop indexed ``dataset[i % len(dataset)]`` for
    ``i`` in ``range(micro_batch_size)``, so every step saw the same first batch
    and most of the dataset was never trained on. This class fixes that and
    reports coverage explicitly.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int = 42):
        if n_items <= 0:
            raise ValueError("cannot build batches from an empty dataset")
        self.n_items = n_items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._order: list[int] = []
        self._pos = 0
        self.consumed = 0
        self.seen: set[int] = set()
        self.epochs_completed = 0
        self.refills = 0

    def _refill(self) -> None:
        self._order = list(range(self.n_items))
        self.rng.shuffle(self._order)
        if self._pos >= self.n_items:
            self.epochs_completed += 1
        self._pos = 0
        self.refills += 1

    def next_indices(self) -> list[int]:
        if not self._order:
            self._refill()
        out: list[int] = []
        while len(out) < self.batch_size:
            if self._pos >= self.n_items:
                self._refill()
            out.append(self._order[self._pos])
            self._pos += 1
        self.consumed += len(out)
        self.seen.update(out)
        return out

    @property
    def unique_coverage(self) -> float:
        return len(self.seen) / self.n_items if self.n_items else 0.0

    def coverage_report(self) -> dict:
        return {
            "dataset_size": self.n_items,
            "consumed_examples": self.consumed,
            "unique_examples_seen": len(self.seen),
            "unique_coverage": round(self.unique_coverage, 6),
            "epochs_completed": self.epochs_completed,
            "shuffle_refills": self.refills,
        }