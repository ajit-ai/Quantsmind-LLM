"""Validated read interface between data preparation and training (v1.2.0).

The v1.3.0 training pipeline must not be able to train on an unprepared
corpus, and it must not be able to train on a corpus that was prepared with a
different tokenizer. This module is the boundary that enforces both.

* The manifest schema must be the one this version writes.
* The artifact's checksum and record count must match the manifest.
* Every record's content hash is re-checked while streaming.
* When a tokenizer is supplied, its file hash must equal the one recorded in
  the manifest, and the prompt contract recorded in the manifest must be the
  one the tokenizer actually serves.
* A split that the manifest reports as not ready is refused, unless the
  caller explicitly asks for it and says so.

There is no silent fallback: a missing manifest, a missing split, a checksum
mismatch or an unapproved corpus are errors, not warnings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from data.python_v1 import MANIFEST_SCHEMA, RECORD_SCHEMA
from data.python_v1.registry import file_sha256

SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}


class DatasetInterfaceError(RuntimeError):
    """The prepared dataset cannot be consumed as presented."""


@dataclass(frozen=True)
class Sample:
    """One training-ready record."""

    text: str
    source_id: str
    file_path: str
    split: str
    content_sha256: str
    component_id: str


class PythonDataset:
    """A validated, ordered view of one prepared split."""

    def __init__(
        self,
        manifest_path,
        split: str = "train",
        *,
        tokenizer_path=None,
        verify: bool = True,
        require_ready: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise DatasetInterfaceError(
                f"dataset manifest not found: {self.manifest_path}. Refusing to "
                "fall back to an unapproved or unverified dataset."
            )
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetInterfaceError(f"manifest is not valid JSON: {exc}") from exc
        if self.manifest.get("schema") != MANIFEST_SCHEMA:
            raise DatasetInterfaceError(
                f"manifest schema is {self.manifest.get('schema')!r}, expected "
                f"{MANIFEST_SCHEMA!r}"
            )
        if split not in SPLIT_FILES:
            raise DatasetInterfaceError(
                f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}"
            )
        self.split = split

        splits = self.manifest.get("splits", {})
        if require_ready and not splits.get("ready", False):
            raise DatasetInterfaceError(
                "the prepared dataset is not ready for evaluation: "
                + "; ".join(splits.get("insufficient", ["no reason recorded"]))
                + ". Pass require_ready=False to inspect it anyway."
            )

        artifacts = {
            a.get("name"): a for a in self.manifest.get("artifacts", [])
        }
        name = SPLIT_FILES[split]
        entry = artifacts.get(name)
        if entry is None:
            raise DatasetInterfaceError(
                f"manifest has no artifact named {name!r}; available: "
                f"{sorted(artifacts)}"
            )
        self.artifact = entry
        self.path = self.manifest_path.parent / name
        if not self.path.is_file():
            raise DatasetInterfaceError(f"artifact file is missing: {self.path}")
        if verify:
            actual = file_sha256(self.path)
            if actual != entry.get("sha256"):
                raise DatasetInterfaceError(
                    f"{name} checksum mismatch: manifest says "
                    f"{entry.get('sha256')}, file is {actual}"
                )
        self.tokenizer_path = tokenizer_path
        if tokenizer_path is not None:
            self._check_tokenizer(tokenizer_path)

    def _check_tokenizer(self, tokenizer_path) -> None:
        recorded = self.manifest.get("tokenizer", {}) or {}
        declared = recorded.get("sha256")
        path = Path(tokenizer_path)
        if not path.is_file():
            raise DatasetInterfaceError(f"tokenizer not found: {path}")
        actual = file_sha256(path)
        if declared and actual != declared:
            raise DatasetInterfaceError(
                "tokenizer does not match the one the dataset was prepared "
                f"with: manifest {declared}, file {actual}"
            )
        contract = recorded.get("contract_id")
        if contract:
            from training.tokenizer_contract import detect_contract

            try:
                served = detect_contract(_load_tokenizer(path))
            except Exception as exc:  # noqa: BLE001 - reported as a refusal
                raise DatasetInterfaceError(
                    f"tokenizer cannot serve a prompt contract: {exc}"
                ) from exc
            if served != contract:
                raise DatasetInterfaceError(
                    f"tokenizer serves contract {served!r} but the dataset was "
                    f"prepared under {contract!r}"
                )

    def __len__(self) -> int:
        return int(self.artifact.get("records", 0))

    def __iter__(self):
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetInterfaceError(
                        f"{self.path.name}:{number} is not valid JSON: {exc}"
                    ) from exc
                if row.get("record_schema") != RECORD_SCHEMA:
                    raise DatasetInterfaceError(
                        f"{self.path.name}:{number} has record schema "
                        f"{row.get('record_schema')!r}, expected {RECORD_SCHEMA!r}"
                    )
                digest = hashlib.sha256(row.get("text", "").encode("utf-8")).hexdigest()
                if digest != row.get("content_sha256"):
                    raise DatasetInterfaceError(
                        f"{self.path.name}:{number} content hash does not match "
                        "its text"
                    )
                count += 1
                yield Sample(
                    text=row["text"],
                    source_id=row.get("source_id", ""),
                    file_path=row.get("file_path", ""),
                    split=row.get("split", self.split),
                    content_sha256=row.get("content_sha256", ""),
                    component_id=row.get("component_id", ""),
                )
        if count != len(self):
            raise DatasetInterfaceError(
                f"{self.path.name} holds {count} records but the manifest "
                f"declares {len(self)}"
            )


def _load_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
