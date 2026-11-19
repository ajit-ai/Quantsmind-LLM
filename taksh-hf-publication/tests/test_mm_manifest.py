import json
from pathlib import Path

import pytest

from training.mm_manifest import SPLITS, SampleRecord, load_manifest


def write_manifest(tmp_path: Path, samples: list[dict], extra: dict | None = None) -> Path:
    path = tmp_path / "MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "preprocessing_version": "mm1_v1",
        "dataset": {"name": "mm_test", "splits": list(SPLITS), "image_resolution": [112, 112]},
        "samples": samples,
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def valid_sample(**overrides) -> dict:
    s = {
        "sample_id": "mm1-000001",
        "split": "train",
        "image_path": "train/mm1-000001.jpg",
        "caption": "a small red square is on the left",
        "source": "fixture dataset",
        "license": "CC0",
    }
    s.update(overrides)
    return s


def test_manifest_parses_valid(tmp_path):
    p = write_manifest(tmp_path, [valid_sample()])
    m = load_manifest(p)
    assert m.schema_version == 1
    assert len(m.samples) == 1
    s = m.samples[0]
    assert isinstance(s, SampleRecord)
    assert s.sample_id == "mm1-000001"
    assert s.license == "CC0"


def test_split_filtering(tmp_path):
    samples = [
        valid_sample(sample_id="a", split="train"),
        valid_sample(sample_id="b", split="val"),
        valid_sample(sample_id="c", split="test"),
    ]
    p = write_manifest(tmp_path, samples)
    m = load_manifest(p)
    assert [s.sample_id for s in m.for_split("train")] == ["a"]
    assert [s.sample_id for s in m.for_split("val")] == ["b"]
    assert [s.sample_id for s in m.for_split("test")] == ["c"]
    with pytest.raises(ValueError):
        m.for_split("nope")


def test_unknown_provenance_stays_none(tmp_path):
    s = valid_sample()
    del s["license"]
    p = write_manifest(tmp_path, [s])
    m = load_manifest(p)
    assert m.samples[0].license is None
    assert m.samples[0].source_url is None
    assert m.samples[0].attribution is None


def test_malformed_missing_required_field(tmp_path):
    s = valid_sample()
    del s["source"]
    p = write_manifest(tmp_path, [s])
    with pytest.raises(ValueError, match="source"):
        load_manifest(p)


def test_malformed_invalid_split(tmp_path):
    p = write_manifest(tmp_path, [valid_sample(split="banana")])
    with pytest.raises(ValueError, match="split"):
        load_manifest(p)


def test_malformed_missing_samples(tmp_path):
    p = write_manifest(tmp_path, [valid_sample()], extra={"samples": "not-a-list"}) 
    with pytest.raises(ValueError, match="samples"):
        load_manifest(p)


def test_malformed_bad_json(tmp_path):
    p = tmp_path / "MANIFEST.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_manifest(p)


def test_missing_manifest_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_invalid_dimension_fields(tmp_path):
    p = write_manifest(tmp_path, [valid_sample(width="huge")])
    with pytest.raises(ValueError, match="width"):
        load_manifest(p)


def test_empty_manifest(tmp_path):
    p = write_manifest(tmp_path, [])
    m = load_manifest(p)
    assert m.samples == []
    assert m.for_split("train") == []