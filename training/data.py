import json
from pathlib import Path

import numpy as np
import torch

_DTYPE_MAP = {"uint16": np.uint16, "uint32": np.uint32}


class TokenDataset:
    def __init__(self, tokens_path: str, meta_path: str | None = None, seed: int = 42):
        p = Path(tokens_path)
        meta_file = Path(meta_path) if meta_path else p.with_suffix(".meta.json")
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            dtype = _DTYPE_MAP[meta["dtype"]]
        else:
            dtype = np.uint16
        self.data = np.memmap(p, dtype=dtype, mode="r")
        self.rng = np.random.default_rng(seed)
        if len(self.data) < 2:
            raise ValueError(f"token file too small: {p} ({len(self.data)} tokens)")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - seq_len - 1
        offsets = self.rng.integers(0, hi + 1, size=batch_size)
        x = np.stack([self.data[o : o + seq_len] for o in offsets])
        y = np.stack([self.data[o + 1 : o + 1 + seq_len] for o in offsets])
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)),
        )
