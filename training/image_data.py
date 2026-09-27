from pathlib import Path

import numpy as np
import torch

from training.mm_manifest import SampleRecord, load_manifest

TEXT_VOCAB_LIMIT = 49152  # official 49152-token BPE vocab; specials start above it


def preprocess_image(img, size: int = 112, normalize: dict | None = None) -> np.ndarray:
    """Deterministic M1 preprocessing for a real image.

    Steps: RGB conversion -> cover-resize (LANCZOS) -> center crop to ``size`` ->
    float32 (size, size, 3) tensor in [0, 1]. ``normalize`` optionally applies
    per-channel (mean, std) in [0,1] space. No augmentation, no randomness.

    EXIF orientation is corrected first (``ImageOps.exif_transpose``), so images
    carrying an orientation flag (e.g. portrait phone photos) are processed
    upright, matching what the validator decodes and what authors intended.
    Images without EXIF orientation pass through pixel-identical.
    """
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img = img.resize((nw, nh), resample)
    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (size, size, 3)
    if normalize:
        mean = np.asarray(normalize["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(normalize["std"], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
    return arr


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


class RealImageDataset:
    """Real image/caption dataset driven by a provenance-aware MANIFEST.json.

    1) MANIFEST.json -> 2) image file -> 3) RGB conversion -> 4) deterministic
    resize/crop -> 5) M1 image tensor -> 6) caption/token sequence.

    Exposes the same batch/sample contract as ImageTextDataset (the synthetic
    path is unchanged). Requires Pillow for decoding image files.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        tokenizer_path: str = "tokenizer/qm_bpe_v0.json",
        image_size: int = 112,
        normalize: dict | None = None,
        seed: int = 42,
        max_caption_len: int | None = None,
    ):
        self.manifest = load_manifest(manifest_path)
        self.root = self.manifest.path.parent
        self.image_size = image_size
        self.normalize = normalize
        self.split = split
        self.samples = self.manifest.for_split(split)
        if not self.samples:
            raise ValueError(f"split {split!r} has no samples in {manifest_path}")
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(tokenizer_path)
        self.ids = [self.tok.encode(s.caption).ids for s in self.samples]
        for record, ids in zip(self.samples, self.ids):
            assert all(i < TEXT_VOCAB_LIMIT for i in ids), (
                f"caption of sample {record.sample_id} encodes to a special/reserved id"
            )
        if max_caption_len is not None:
            if not isinstance(max_caption_len, int) or max_caption_len <= 0:
                raise ValueError(
                    f"max_caption_len must be a positive int, got {max_caption_len!r}"
                )
            self.ids = [ids[:max_caption_len] for ids in self.ids]
            self.max_caption_len = max_caption_len
        else:
            self.max_caption_len = max((len(x) for x in self.ids), default=0)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, record: SampleRecord) -> np.ndarray:
        from PIL import Image

        path = record.resolved_path(self.root)
        if not path.is_file():
            raise FileNotFoundError(f"missing image for sample {record.sample_id}: {path}")
        with Image.open(path) as img:
            return preprocess_image(img, self.image_size, self.normalize)

    def sample(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.batch_of([i])

    def batch_of(self, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arrs = [self._load_image(self.samples[int(i)]) for i in idxs]
        imgs = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2)  # (B,3,H,W)
        cap = np.zeros((len(idxs), self.max_caption_len), dtype=np.int64)
        lengths = np.zeros(len(idxs), dtype=np.int64)
        for k, i in enumerate(idxs):
            ids = self.ids[int(i)]
            cap[k, : len(ids)] = ids
            lengths[k] = len(ids)
        return imgs, torch.from_numpy(cap).long(), torch.from_numpy(lengths).long()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, len(self.samples), size=batch_size)
        return self.batch_of([int(i) for i in idx])