from pathlib import Path

import numpy as np
import torch

TEXT_VOCAB_LIMIT = 49152  # official 49152-token BPE vocab; specials start above it


class ImageTextDataset:
    """Minimal beamless image/caption dataset for the M1.1 fit-check.

    Loads an .npz produced by scripts/synth_images.py containing uint8 (N,H,W,3)
    images and a str caption per image. Captions are encoded with the committed
    BPE tokenizer (no training). Only synthetic, hermetic data.
    """

    def __init__(self, data_path: str, tokenizer_path: str = "tokenizer/qm_bpe_v0.json", seed: int = 42):
        d = np.load(data_path, allow_pickle=True)
        images = d["images"]
        captions = list(map(str, d["captions"]))
        assert images.ndim == 4 and images.shape[3] == 3, "expected (N,H,W,3) uint8 images"
        self.images = images
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(tokenizer_path)
        self.ids = [self.tok.encode(c).ids for c in captions]
        for ids in self.ids:
            assert all(i < TEXT_VOCAB_LIMIT for i in ids), "caption encodes to special/reserved id"
        self.max_caption_len = max((len(x) for x in self.ids), default=0)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def sample(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single sample: (images (1,3,H,W) float32, caption_ids (1,T) long, lengths (1))."""
        return self.batch_of([i])

    def batch_of(self, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = self.images[idxs]
        imgs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0
        cap = np.zeros((len(idxs), self.max_caption_len), dtype=np.int64)
        lengths = np.zeros(len(idxs), dtype=np.int64)
        for k, i in enumerate(idxs):
            ids = self.ids[int(i)]
            cap[k, : len(ids)] = ids
            lengths[k] = len(ids)
        return imgs, torch.from_numpy(cap).long(), torch.from_numpy(lengths).long()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (images (B,3,H,W) float32 in [0,1], caption_ids (B,T) long, lengths (B))."""
        idx = self.rng.integers(0, len(self.images), size=batch_size)
        return self.batch_of([int(i) for i in idx])


def resolve_data_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MM data not found: {path} (run scripts/synth_images.py first)")
    return p