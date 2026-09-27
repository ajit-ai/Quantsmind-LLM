import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")

import numpy as np
import torch

from PIL import Image

from training.image_data import RealImageDataset, preprocess_image

M1_SIZE = 112


@pytest.fixture
def fixtures(tmp_path: Path):
    """Tiny hermetic image set: train/val/test splits each with a PNG file."""
    data_dir = tmp_path / "mm1"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "val").mkdir()
    (data_dir / "test").mkdir()

    # non-square RGB image (exercises cover-resize + center crop)
    Image.new("RGB", (224, 96), (255, 0, 0)).save(data_dir / "train" / "a.png")
    # grayscale (exercises RGB conversion)
    Image.new("L", (64, 48), 200).save(data_dir / "val" / "b.png")
    Image.new("RGB", (112, 112), (0, 255, 0)).save(data_dir / "test" / "c.png")

    samples = [
        {"sample_id": "a", "split": "train", "image_path": "train/a.png",
         "caption": "a red rectangle", "source": "fixture", "license": "CC0"},
        {"sample_id": "b", "split": "val", "image_path": "val/b.png",
         "caption": "a grey square", "source": "fixture", "license": "CC0"},
        {"sample_id": "c", "split": "test", "image_path": "test/c.png",
         "caption": "a green circle", "source": "fixture", "license": "CC0"},
    ]
    manifest = {
        "schema_version": 1,
        "preprocessing_version": "mm1_v1",
        "dataset": {"name": "mm_fixture", "splits": ["train", "val", "test"]},
        "samples": samples,
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir


def test_image_loading_shape(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="train")
    imgs, caps, lens = ds.sample(0)
    assert imgs.shape == (1, 3, M1_SIZE, M1_SIZE)
    assert imgs.dtype == torch.float32
    assert 0.0 <= imgs.min().item() and imgs.max().item() <= 1.0


def test_rgb_conversion(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="val")
    imgs, _, _ = ds.sample(0)  # grayscale source -> RGB
    assert imgs.shape == (1, 3, M1_SIZE, M1_SIZE)
    # 200/255 linear grey, equal across all three channels
    assert torch.allclose(imgs[0, 0], imgs[0, 1], atol=1e-6)
    assert torch.allclose(imgs[0, 1], imgs[0, 2], atol=1e-6)


def test_expected_image_dimensions(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="test")
    imgs, _, _ = ds.sample(0)
    assert imgs.shape == (1, 3, 112, 112)
    assert imgs.shape[2] == M1_SIZE and imgs.shape[3] == M1_SIZE


def test_caption_handling(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="train")
    imgs, caps, lens = ds.sample(0)
    assert caps.ndim == 2 and caps.shape[1] == ds.max_caption_len
    assert caps[0, : lens[0]].lt(49152).all()  # BPE ids, never special/reserved
    assert lens[0].item() == len(ds.tok.encode("a red rectangle").ids)


def test_deterministic_preprocessing(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="train")
    imgs1, _, _ = ds.sample(0)
    imgs2, _, _ = ds.sample(0)
    assert torch.equal(imgs1, imgs2)


def test_preprocess_function_deterministic(fixtures):
    with Image.open(fixtures / "train" / "a.png") as img:
        a = preprocess_image(img, M1_SIZE)
    with Image.open(fixtures / "train" / "a.png") as img:
        b = preprocess_image(img, M1_SIZE)
    assert np.array_equal(a, b)
    assert a.shape == (M1_SIZE, M1_SIZE, 3)


def test_missing_image_failure(fixtures):
    data_dir = fixtures
    manifest_path = data_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"].append(
        {"sample_id": "ghost", "split": "train", "image_path": "train/ghost.png",
         "caption": "a violet sphere", "source": "fixture", "license": "CC0"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ds = RealImageDataset(manifest_path, split="train")
    with pytest.raises(FileNotFoundError, match="ghost"):
        ds.batch_of([1])


def test_malformed_manifest_failure(fixtures):
    manifest_path = fixtures / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["samples"][0]["source"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        RealImageDataset(manifest_path, split="train")


def test_batch_api_matches_synthetic_contract(fixtures):
    ds = RealImageDataset(fixtures / "MANIFEST.json", split="train")
    imgs, caps, lens = ds.batch(2)
    assert imgs.shape[0] == 2
    assert caps.shape[0] == 2 and lens.shape[0] == 2


def test_split_with_no_samples(fixtures):
    manifest_path = fixtures / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = [s for s in manifest["samples"] if s["split"] == "test"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # "train" is a valid split name but now has zero samples
    with pytest.raises(ValueError, match="no samples"):
        RealImageDataset(manifest_path, split="train")