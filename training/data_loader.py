"""Deterministic dataset streaming for Taksh v1.3.0.

The reader is the v1.2.0 interface (``data.python_v1.interface``): artifacts
are read in manifest order, their checksums are verified, and every record's
content hash is re-checked while streaming. No other corpus path exists, so
a run cannot silently fall back to a historical token stream.

Sampling is deterministic and resume-aware. For a given ``(seed, epoch)`` the
window order is a pure function, produced by a seeded ``torch.Generator``;
the sampler state records the epoch and the position inside it, so a resumed
run replays exactly the order the interrupted run would have used.
"""

from __future__ import annotations

import torch

from data.python_v1.interface import PythonDataset


class ManifestDataset:
    """Ordered view of one prepared split."""

    def __init__(self, manifest_path, split: str = "train", *, require_ready: bool = True):
        self.split = split
        self._dataset = PythonDataset(
            manifest_path, split, verify=True, require_ready=require_ready
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for sample in self._dataset:
            yield {
                "source_id": sample.source_id,
                "file_path": sample.file_path,
                "text": sample.text,
                "content_sha256": sample.content_sha256,
            }

    @property
    def manifest(self) -> dict:
        return self._dataset.manifest


class DeterministicSampler:
    """Seeded permutation of window indices, stable per epoch."""

    def __init__(self, count: int, seed: int, shuffle: bool = True):
        self.count = int(count)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    def order(self, epoch: int) -> list:
        """Window indices for one epoch. Pure function of (seed, epoch)."""
        if self.count == 0:
            return []
        if not self.shuffle:
            return list(range(self.count))
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + int(epoch))
        return torch.randperm(self.count, generator=generator).tolist()

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "shuffle": self.shuffle,
            "count": self.count,
            "type": "deterministic_epoch_permutation",
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        if state.get("type") not in (None, "deterministic_epoch_permutation"):
            raise ValueError(f"unsupported sampler type: {state.get('type')!r}")
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.count = int(state.get("count", self.count))


class WindowStream:
    """Turns windows into microbatches and accumulation windows."""

    def __init__(
        self,
        windows,
        *,
        sampler: DeterministicSampler,
        batch_builder,
        micro_batch_size: int = 1,
        grad_accum_steps: int = 1,
        drop_last: bool = True,
    ):
        if micro_batch_size < 1 or grad_accum_steps < 1:
            raise ValueError("micro_batch_size and grad_accum_steps must be >= 1")
        self.windows = list(windows)
        self.sampler = sampler
        self.batch_builder = batch_builder
        self.micro_batch_size = int(micro_batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        return len(self.windows)

    def microbatches(self, epoch: int, start_position: int = 0):
        """Yield ``(position, batch)`` pairs for one epoch.

        ``position`` is the index of the microbatch inside the epoch, which
        is what a resumed run needs in order to continue exactly where it
        stopped.
        """
        order = self.sampler.order(epoch)
        position = 0
        for start in range(0, len(order), self.micro_batch_size):
            chunk = order[start : start + self.micro_batch_size]
            if self.drop_last and len(chunk) < self.micro_batch_size:
                break
            if position < start_position:
                position += 1
                continue
            batch = self.batch_builder([self.windows[index] for index in chunk])
            yield position, batch
            position += 1

    def accumulation_windows(self, epoch: int, start_position: int = 0):
        """Yield lists of microbatches that form one optimizer step."""
        window: list = []
        for position, batch in self.microbatches(epoch, start_position):
            window.append(batch)
            if len(window) == self.grad_accum_steps:
                yield window
                window = []
        if window:
            yield window

    def state_dict(self, epoch: int, position: int) -> dict:
        state = self.sampler.state_dict()
        state.update({"epoch": int(epoch), "position": int(position)})
        return state
