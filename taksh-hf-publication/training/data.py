import json
from pathlib import Path

import numpy as np
import torch

_DTYPE_MAP = {"uint16": np.uint16, "uint32": np.uint32}


class TokenArtifactIncompatible(RuntimeError):
    """The token artifact does not match the model it would train."""


class TokenDataset:
    def __init__(self, tokens_path: str, meta_path: str | None = None, seed: int = 42):
        p = Path(tokens_path)
        meta_file = Path(meta_path) if meta_path else p.with_suffix(".meta.json")
        self.vocab_size = None
        self.meta = {}
        self.tokens_path = p
        if meta_file.exists():
            self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if self.meta.get("dtype") not in _DTYPE_MAP:
                raise TokenArtifactIncompatible(
                    f"{meta_file}: unknown dtype {self.meta.get('dtype')!r}. "
                    "Refusing to guess how to read this artifact."
                )
            dtype = _DTYPE_MAP[self.meta["dtype"]]
            self.vocab_size = int(self.meta.get("vocab_size", 0)) or None
        else:
            dtype = np.uint16
        self.data = np.memmap(p, dtype=dtype, mode="r")
        self.rng = np.random.default_rng(seed)
        if len(self.data) < 2:
            raise ValueError(f"token file too small: {p} ({len(self.data)} tokens)")

    def validate_for_model(self, model_vocab: int, tokenizer_vocab: int | None = None):
        """Check the artifact against the model/tokenizer it will be used with.

        Returns a list of human-readable problems; empty means compatible.
        Artifacts predating ``schema: quantsmind.token_stream/1`` carry no
        provenance, so only the vocabulary cross-check is possible for them and
        that limitation is reported rather than hidden.
        """
        problems = []
        if not self.meta:
            return [
                "token artifact has no metadata file; provenance (tokenizer, source "
                "corpus, separator policy) cannot be verified"
            ]
        if self.meta.get("schema") != "quantsmind.token_stream/1":
            problems.append(
                f"token artifact schema is {self.meta.get('schema')!r}, not "
                "'quantsmind.token_stream/1'; it carries no tokenizer or "
                "separator provenance and cannot be verified"
            )
            problems.append(
                "LEGACY ARTIFACT: produced before document separators existed, so "
                "its documents run together with no boundary (OPEN_ISSUES B6). "
                "Regenerate with data.encode_dataset to restore boundaries before "
                "training on it."
            )
        artifact_vocab = self.vocab_size
        if artifact_vocab and artifact_vocab > model_vocab:
            problems.append(
                f"artifact vocab_size {artifact_vocab} exceeds model vocab_size "
                f"{model_vocab}; ids would index outside the embedding matrix"
            )
        if tokenizer_vocab and artifact_vocab and artifact_vocab != tokenizer_vocab:
            problems.append(
                f"artifact vocab_size {artifact_vocab} does not match the "
                f"tokenizer's {tokenizer_vocab}; the artifact was produced by a "
                "different tokenizer"
            )
        return problems

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - seq_len - 1
        if hi < 0:
            raise ValueError(
                f"cannot draw a window of {seq_len} tokens from "
                f"{self.tokens_path}: the stream holds {len(self.data)} tokens, "
                f"so at least {seq_len + 1} are required. Either lower "
                "data.seq_len or encode a larger corpus."
            )
        offsets = self.rng.integers(0, hi + 1, size=batch_size)
        x = np.stack([self.data[o : o + seq_len] for o in offsets])
        y = np.stack([self.data[o + 1 : o + 1 + seq_len] for o in offsets])
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)),
        )
