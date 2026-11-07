"""Hermetic tests for the real-data manifest validator.

Uses tiny tmp_path fixtures only (no network, no real licensed data).  Verifies
the schema/provenance/file/checksum/dedup/hash behaviour of
``training.validate_manifest``.
"""
import hashlib
import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")

from PIL import Image

from training.mm_manifest import valid_iso_date, valid_sha256_hex
from training.validate_manifest import _canonical_digest, validate_manifest


def png_bytes(size=(32, 32), color=(10, 200, 30)) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny manifest with one complete-provenance sample and a real PNG."""
    (tmp_path / "train").mkdir()
    img = png_bytes()
    (tmp_path / "train" / "a.jpg").write_bytes(img)
    samples = [
        {
            "sample_id": "mm1-000001",
            "split": "train",
            "image_path": "train/a.jpg",
            "caption": "a green rectangle",
            "source": "fixture-dataset",
            "license": "CC0",
            "license_ref": "https://example.in/cc0",
            "acquisition_date": "2026-09-20",
            "checksum": sha256_hex(img),
            "width": 32,
            "height": 32,
        }
    ]
    manifest = {
        "schema_version": 1,
        "preprocessing_version": "mm1_v1",
        "dataset": {"name": "mm_fixture", "splits": ["train", "val", "test"]},
        "samples": samples,
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def rewrite(repo: Path, mutate):
    manifest = json.loads((repo / "MANIFEST.json").read_text(encoding="utf-8"))
    mutate(manifest)
    (repo / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------- schema


def test_valid_complete_manifest(repo):
    rep = validate_manifest(repo / "MANIFEST.json", require_complete_provenance=True)
    assert rep["valid"] is True
    assert rep["errors"] == []
    assert rep["samples"] == 1
    assert rep["counts"]["train"] == 1
    assert rep["schema_version"] == 1
    assert rep["sha256_stable"] is True


def test_empty_manifest_valid_with_warning(tmp_path):
    manifest = {
        "schema_version": 1,
        "preprocessing_version": "mm1_v1",
        "dataset": {"name": "empty", "splits": ["train", "val", "test"]},
        "samples": [],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    rep = validate_manifest(tmp_path / "MANIFEST.json")
    assert rep["valid"] is True
    assert any("no samples" in w for w in rep["warnings"])


def test_missing_manifest_file(tmp_path):
    rep = validate_manifest(tmp_path / "nope.json")
    assert rep["valid"] is False
    assert rep["errors"]


def test_malformed_json_manifest(tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{not json", encoding="utf-8")
    rep = validate_manifest(tmp_path / "MANIFEST.json")
    assert rep["valid"] is False
    assert any("parse" in e for e in rep["errors"])


def test_missing_required_source_field(repo):
    rewrite(repo, lambda m: m["samples"][0].pop("source"))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert any("source" in e for e in rep["errors"])


def test_invalid_checksum_format(repo):
    rewrite(repo, lambda m: m["samples"][0].update(checksum="not-a-checksum"))
    rep = validate_manifest(repo / "MANIFEST.json", check_images=False)
    assert rep["valid"] is False
    assert any("checksum" in e.lower() for e in rep["errors"])


def test_invalid_acquisition_date(repo):
    rewrite(repo, lambda m: m["samples"][0].update(acquisition_date="20-20-2026"))
    rep = validate_manifest(repo / "MANIFEST.json", check_images=False)
    assert rep["valid"] is False
    assert any("acquisition_date" in e for e in rep["errors"])


# ---------------------------------------------------------------- provenance


def test_strict_mode_rejects_incomplete_provenance(repo):
    rewrite(repo, lambda m: m["samples"][0].pop("license"))
    lax = validate_manifest(repo / "MANIFEST.json")
    assert lax["valid"] is True  # schema allows unknown license
    assert any("license unknown" in n for n in lax["notes"])
    strict = validate_manifest(repo / "MANIFEST.json", require_complete_provenance=True)
    assert strict["valid"] is False
    assert any("license" in e for e in strict["errors"])


def test_strict_mode_rejects_missing_checksum(repo):
    rewrite(repo, lambda m: m["samples"][0].pop("checksum"))
    strict = validate_manifest(repo / "MANIFEST.json", require_complete_provenance=True)
    assert strict["valid"] is False
    assert any("checksum" in e for e in strict["errors"])


def test_strict_mode_rejects_unauthenticated_fixture(tmp_path):
    """A sample with no license/date/checksum (like a unit-test fixture)
    must fail the real-data admission gate."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "x.jpg").write_bytes(png_bytes())
    samples = [{
        "sample_id": "f1", "split": "train", "image_path": "train/x.jpg",
        "caption": "a test fixture", "source": "unit-test",
    }]
    manifest = {"schema_version": 1, "dataset": {"name": "d"},
                "samples": samples}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    rep = validate_manifest(tmp_path / "MANIFEST.json",
                            require_complete_provenance=True)
    assert rep["valid"] is False
    assert rep["summary"]["provenance_incomplete_count"] == 1


# ---------------------------------------------------------------- image checks


def test_missing_image_file(repo):
    (repo / "train" / "a.jpg").unlink()
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert rep["summary"]["images_missing"] == 1
    assert any("image file not found" in e for e in rep["errors"])


def test_corrupt_image_file(repo):
    (repo / "train" / "a.jpg").write_bytes(b"this is not an image at all")
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert rep["summary"]["images_unreadable"] == 1


def test_checksum_mismatch(repo):
    rewrite(repo, lambda m: m["samples"][0].update(checksum="0" * 64))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert rep["summary"]["checksum_mismatches"] == 1
    assert any("checksum mismatch" in e for e in rep["errors"])


def test_declared_dimension_mismatch_is_warning(repo):
    rewrite(repo, lambda m: m["samples"][0].update(width=640, height=480))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is True  # schema ok, image ok -> warning only
    assert rep["summary"]["dimension_mismatches"] == 1
    assert any("width 640" in w for w in rep["warnings"])


def test_no_image_checks_skips_file_access(repo):
    (repo / "train" / "a.jpg").unlink()
    rep = validate_manifest(repo / "MANIFEST.json", check_images=False)
    assert rep["valid"] is True


# ---------------------------------------------------------------- duplicates


def test_duplicate_sample_id(repo):
    img2 = png_bytes(color=(1, 2, 3))
    (repo / "train" / "b.jpg").write_bytes(img2)
    rewrite(repo, lambda m: m["samples"].append(dict(
        m["samples"][0], sample_id="mm1-000001", image_path="train/b.jpg",
        checksum=sha256_hex(img2))))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert rep["summary"]["duplicate_sample_ids"] == 1


def test_duplicate_image_path(repo):
    rewrite(repo, lambda m: m["samples"].append(dict(
        m["samples"][0], sample_id="mm1-000002")))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is False
    assert rep["summary"]["duplicate_image_paths"] == 1


def test_duplicate_content_detected(repo):
    data = (repo / "train" / "a.jpg").read_bytes()
    (repo / "train" / "b.jpg").write_bytes(data)  # same bytes, different name
    rewrite(repo, lambda m: m["samples"].append(dict(
        m["samples"][0], sample_id="mm1-000002", image_path="train/b.jpg")))
    rep = validate_manifest(repo / "MANIFEST.json")
    assert rep["valid"] is True  # content duplication is a warning, not an error
    assert rep["summary"]["duplicate_content"] == 2
    assert any("duplicate image content" in w for w in rep["warnings"])


# ---------------------------------------------------------------- hashing


def test_deterministic_sha256_stable(repo):
    rep = validate_manifest(repo / "MANIFEST.json")
    raw = (repo / "MANIFEST.json").read_bytes()
    assert rep["sha256"] == sha256_hex(raw)
    assert rep["sha256_stable"] is True


def test_canonical_digest_formatting_independent(tmp_path):
    samples = [
        {"sample_id": "b", "caption": "x", "license": "CC0"},
        {"sample_id": "a", "caption": "y", "license": "CC0"},
    ]
    blob1 = json.dumps(samples, sort_keys=False)
    blob2 = json.dumps([dict(reversed(list(s.items()))) for s in samples[::-1]],
                       sort_keys=True, indent=2)
    # order/formatting must not change the canonical content digest
    assert _canonical_digest(samples) == _canonical_digest(json.loads(blob2))
    assert _canonical_digest(samples) == _canonical_digest(json.loads(blob1))


# ---------------------------------------------------------------- helpers


def test_valid_sha256_hex():
    assert valid_sha256_hex("0" * 64)
    assert not valid_sha256_hex("0" * 63)
    assert not valid_sha256_hex("Z" * 64)


def test_valid_iso_date():
    assert valid_iso_date("2026-09-20")
    assert not valid_iso_date("2026-13-01")
    assert not valid_iso_date("2026-2-1")
    assert not valid_iso_date("not-a-date")


def test_report_shape(repo):
    rep = validate_manifest(repo / "MANIFEST.json")
    assert {"valid", "manifest", "sha256", "canonical_digest", "samples",
            "counts", "errors", "warnings", "notes", "per_sample", "summary"} <= set(rep)
    assert rep["per_sample"][0]["sample_id"] == "mm1-000001"