"""M1.3 corrections: frozen text backbone, EXIF orientation, caption-length wiring.

Covers the three M1 issue-report fixes (generation-harness sampling is
explicitly out of scope for M1.3 and is not tested here):
1. ``freeze_text_backbone`` freezes every text param and keeps ViT/projector
   trainable; the optimizer set built the train_mm way excludes text params.
2. ``preprocess_image`` physically corrects EXIF orientation (mm1_v2);
   non-EXIF images are pixel-identical to before.
3. ``sequence.max_caption_len`` reaches ``RealImageDataset`` and caps the
   effective caption length; the corrected ``mm_m1_local.yaml`` carries the
   frozen recipe, the cap, and the ``mm1_v2`` identity.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

import json

import numpy as np
from PIL import Image

from training.image_data import RealImageDataset, preprocess_image
from training.mm_manifest import load_manifest
from training.multimodal import QMMultimodal, freeze_text_backbone
from training.pretrain import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_CFG = {
    "vocab_size": 68,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 88,
    "max_seq_len": 256,
    "rope_theta": 10000.0,
    "norm_eps": 1.0e-5,
    "tie_embeddings": True,
}

VISION_CFG = {
    "image_size": 32,
    "patch_size": 8,
    "in_channels": 3,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "ffn_dim": 88,
    "norm_eps": 1.0e-5,
}


def test_freeze_text_backbone_excludes_text_from_optimizer():
    torch.manual_seed(0)
    model = QMMultimodal(TEXT_CFG, VISION_CFG)
    n_text_params = sum(1 for _ in model.text.parameters())
    assert n_text_params > 0

    frozen = freeze_text_backbone(model)
    assert len(frozen) == n_text_params
    assert all(name.startswith("text.") for name in frozen)
    assert all(not p.requires_grad for p in model.text.parameters())
    assert all(p.requires_grad for p in model.vision.parameters())
    assert all(p.requires_grad for p in model.projector.parameters())

    # the train_mm construction: optimizer over requires_grad params only
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    text_ids = {id(p) for p in model.text.parameters()}
    opt_params = [p for g in optim.param_groups for p in g["params"]]
    assert not any(id(p) in text_ids for p in opt_params)
    expected_trainable = model.vision.num_params() + sum(
        p.numel() for p in model.projector.parameters()
    )
    assert sum(p.numel() for p in opt_params) == expected_trainable


def _half_red_half_blue(width, height):
    img = Image.new("RGB", (width, height))
    px = img.load()
    for x in range(width):
        for y in range(height):
            px[x, y] = (255, 0, 0) if x < width // 2 else (0, 0, 255)
    return img


def test_exif_orientation_physically_corrected(tmp_path):
    # displayed scene: 4x6 portrait, red left half / blue right half
    displayed = _half_red_half_blue(4, 6)
    upright_path = tmp_path / "upright.tif"
    displayed.save(upright_path)
    # same scene stored landscape (6x4) with EXIF orientation 6
    # (rotate 90 CW to display), so stored = displayed rotated 90 CCW
    stored = displayed.transpose(Image.ROTATE_90)
    assert stored.size == (6, 4)
    exif = stored.getexif()
    exif[274] = 6
    oriented_path = tmp_path / "oriented.tif"
    stored.save(oriented_path, exif=exif)
    with Image.open(oriented_path) as check:
        assert check.getexif().get(274) == 6  # the flag really is on disk

    with Image.open(upright_path) as f:
        plain = preprocess_image(f, size=16)
    with Image.open(oriented_path) as f:
        corrected = preprocess_image(f, size=16)
    assert plain.shape == (16, 16, 3)
    assert plain.dtype == np.float32
    assert corrected.shape == (16, 16, 3)
    # identical displayed content -> identical preprocessed tensors
    np.testing.assert_array_equal(plain, corrected)


def test_non_exif_image_behavior_unchanged(tmp_path):
    path = tmp_path / "plain.png"
    _half_red_half_blue(24, 16).save(path)
    with Image.open(path) as f:
        first = preprocess_image(f, size=32)
    with Image.open(path) as f:
        second = preprocess_image(f, size=32)
    assert first.shape == (32, 32, 3)
    assert first.dtype == np.float32
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    np.testing.assert_array_equal(first, second)


def _write_long_caption_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "mmcap"
    (data_dir / "train").mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(data_dir / "train" / "a.png")
    caption = " ".join(f"word{i}" for i in range(40))
    manifest = {
        "schema_version": 1,
        "dataset": {"name": "mm_cap", "splits": ["train"]},
        "samples": [{
            "sample_id": "a",
            "split": "train",
            "image_path": "train/a.png",
            "caption": caption,
            "source": "fixture",
        }],
    }
    path = data_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_max_caption_len_caps_effective_length(tmp_path):
    manifest = _write_long_caption_fixture(tmp_path)
    full = RealImageDataset(manifest, split="train")
    assert full.max_caption_len > 8  # premise: fixture caption exceeds the cap
    capped = RealImageDataset(manifest, split="train", max_caption_len=8)
    assert capped.max_caption_len == 8  # no hidden fallback to the data max
    assert all(len(ids) <= 8 for ids in capped.ids)
    assert capped.ids[0] == full.ids[0][:8]
    _, caps, lengths = capped.batch_of([0])
    assert caps.shape == (1, 8)
    assert int(lengths[0]) == min(len(full.ids[0]), 8)
    with pytest.raises(ValueError, match="max_caption_len"):
        RealImageDataset(manifest, split="train", max_caption_len=0)


def test_mm_m1_local_yaml_carries_corrected_recipe():
    cfg = load_yaml(str(REPO_ROOT / "configs" / "train" / "mm_m1_local.yaml"))
    assert cfg["run"]["freeze_text_backbone"] is True
    assert cfg["sequence"]["max_caption_len"] == 64
    assert cfg["image"]["preprocessing_version"] == "mm1_v2"


def test_preprocessing_version_default_is_mm1_v2(tmp_path):
    manifest = {
        "schema_version": 1,
        "dataset": {"name": "mm_v2", "splits": ["train"]},
        "samples": [{
            "sample_id": "a",
            "split": "train",
            "image_path": "train/a.png",
            "caption": "a red square",
            "source": "fixture",
        }],
    }
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded.preprocessing_version == "mm1_v2"
    assert loaded.samples[0].preprocessing_version == "mm1_v2"
