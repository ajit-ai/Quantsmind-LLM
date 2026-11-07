"""Manifest validation for the M1 real-data run.

Reuses ``training/mm_manifest.load_manifest`` for schema/provenance parsing
(never duplicates its logic) and adds file-level checks: referenced image
existence and readability, split/sample bookkeeping, duplicate detection,
image-checksum consistency, caption presence, and deterministic hashing.

This command never modifies a manifest and never invents data.  By default it
*reports* provenance gaps (a record whose license is unknown is still allowed
by the schema); pass ``--require-complete-provenance`` to REJECT records whose
provenance is incomplete, which is the gate real licensed training data must
pass before being used for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from training.mm_manifest import SPLITS, load_manifest

TEXT_VOCAB_LIMIT = 49152  # BPE vocab boundary; special ids start above it


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(samples: list[dict]) -> str:
    """Stable digest of the sample content, independent of key/formatting order.

    ``Manifest.sha256()`` hashes the raw file bytes (order-sensitive); this
    canonical digest lets two differently-formatted manifests with the same
    logical samples compare equal.
    """
    canonical = sorted(samples, key=lambda s: str(s.get("sample_id", "")))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_hex(blob.encode("ascii"))


def _image_report(path: Path):
    """(sha256_hex, width, height) after a forced decode; raises on corrupt files.

    Dimensions are measured after EXIF orientation correction, matching what
    ``preprocess_image`` (mm1_v2) feeds training. The checksum always covers
    the raw file bytes.
    """
    from PIL import Image, ImageOps

    data = path.read_bytes()
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img.load()  # force decode: raises on truncated/corrupt images
        w, h = img.size
    return _sha256_hex(data), w, h


def validate_manifest(
        manifest_path: str | Path,
        require_complete_provenance: bool = False,
        check_images: bool = True,
        tokenizer_path: str | None = None,
    ) -> dict:
    """Validate a MANIFEST.json and the images it references.

    Parameters
    ----------
    manifest_path : str | Path
        Path to a schema-version-1 ``MANIFEST.json``.
    require_complete_provenance : bool
        When True, every sample must carry a license, an acquisition date, and
        an image checksum; missing/unknown provenance becomes an error rather
        than a note.  Real licensed training data is admitted only in this mode.
    check_images : bool
        When True (default), verify referenced image files exist, decode, and
        (when declared) match the recorded checksum/dimensions.
    tokenizer_path : str | None
        When set, captions are tokenized and checked to stay below the BPE
        vocab boundary (mirrors ``RealImageDataset``).
    image_size : int
        Unused explicitly; kept for API symmetry with image preprocessing.

    Returns
    -------
    dict
        A review report: ``valid`` (bool), ``errors``/``notes``/``warnings``,
        per-sample issues, split counts, deterministic hashes, summary.
    """
    path = Path(manifest_path)
    manifest_dir = path.parent
    errors, warnings, notes = [], [], []

    if not path.is_file():
        return {
            "valid": False,
            "manifest": str(path),
            "errors": [f"manifest file not found: {path}"],
            "schema_version": None,
            "sha256": None,
            "canonical_digest": None,
            "samples": 0,
            "counts": {},
            "summary": {},
        }

    raw = path.read_bytes()
    file_sha256 = _sha256_hex(raw)

    try:
        manifest = load_manifest(path)
    except FileNotFoundError as exc:
        return {
            "valid": False,
            "manifest": str(path),
            "errors": [str(exc)],
            "schema_version": None,
            "sha256": file_sha256,
            "canonical_digest": None,
            "samples": 0,
            "counts": {},
            "summary": {},
        }
    except ValueError as exc:
        return {
            "valid": False,
            "manifest": str(path),
            "errors": [f"schema/parse error: {exc}"],
            "schema_version": None,
            "sha256": file_sha256,
            "canonical_digest": None,
            "samples": 0,
            "counts": {},
            "summary": {},
        }

    samples = manifest.samples
    if not samples:
        warnings.append("manifest has no samples (empty template); real training data is absent")

    counts = {split: 0 for split in SPLITS}
    for s in samples:
        counts.setdefault(s.split, 0)
        counts[s.split] += 1

    seen_ids = {}
    seen_paths = {}
    checksum_to_ids = {}
    per_sample = []
    duplicate_sample_ids, duplicate_paths, duplicate_content = [], [], []
    checksum_mismatches, images_missing, images_unreadable = [], [], []
    dimension_mismatches, provenance_gaps = [], []

    for s in samples:
        sample_errors, sample_warnings, sample_notes = [], [], []

        if s.sample_id in seen_ids:
            sample_errors.append(f"duplicate sample_id (also seen at {seen_ids[s.sample_id]})")
            duplicate_sample_ids.append(s.sample_id)
        else:
            seen_ids[s.sample_id] = s.sample_id

        img_path = s.resolved_path(manifest_dir)
        path_key = str(img_path)
        if path_key in seen_paths:
            sample_errors.append(
                f"duplicate image file reference (also sample {seen_paths[path_key]})"
            )
            duplicate_paths.append(s.sample_id)
        else:
            seen_paths[path_key] = s.sample_id

        if s.license in (None, ""):
            msg = "license unknown/absent"
            if require_complete_provenance:
                sample_errors.append(msg)
            else:
                sample_notes.append(msg)
            provenance_gaps.append((s.sample_id, "license"))
        if s.acquisition_date in (None, ""):
            msg = "acquisition_date absent"
            if require_complete_provenance:
                sample_errors.append(msg)
            else:
                sample_notes.append(msg)
            provenance_gaps.append((s.sample_id, "acquisition_date"))
        if s.checksum in (None, ""):
            msg = "image checksum absent"
            if require_complete_provenance:
                sample_errors.append(msg)
            else:
                sample_notes.append(msg)
            provenance_gaps.append((s.sample_id, "checksum"))

        file_checksum = dimension_w = dimension_h = None
        if check_images:
            if not img_path.is_file():
                msg = f"image file not found: {img_path}"
                sample_errors.append(msg)
                images_missing.append(s.sample_id)
            else:
                try:
                    file_checksum, dimension_w, dimension_h = _image_report(img_path)
                except Exception as exc:  # noqa: BLE001 - corrupt/undecodable file
                    msg = f"image unreadable: {exc}"
                    sample_errors.append(msg)
                    images_unreadable.append(s.sample_id)
                else:
                    checksum_to_ids.setdefault(file_checksum, []).append(s.sample_id)
                    if s.checksum and s.checksum != file_checksum:
                        sample_errors.append(
                            f"checksum mismatch: declared {s.checksum} != actual {file_checksum}"
                        )
                        checksum_mismatches.append(s.sample_id)
                    if s.width is not None and s.width != dimension_w:
                        sample_warnings.append(
                            f"declared width {s.width} != actual {dimension_w}"
                        )
                        dimension_mismatches.append(s.sample_id)
                    if s.height is not None and s.height != dimension_h:
                        sample_warnings.append(
                            f"declared height {s.height} != actual {dimension_h}"
                        )
                        dimension_mismatches.append(s.sample_id)

        if tokenizer_path and s.caption:
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(tokenizer_path)
                ids = tok.encode(s.caption).ids
                bad = [i for i in ids if i >= TEXT_VOCAB_LIMIT]
                if bad:
                    sample_errors.append(
                        f"caption encodes to reserved/special ids: {sorted(set(bad))[:5]}"
                    )
            except Exception as exc:  # noqa: BLE001 - tokenizer failure is a validation error
                sample_errors.append(f"caption not tokenizable: {exc}")

        per_sample.append({
            "sample_id": s.sample_id,
            "split": s.split,
            "errors": sample_errors,
            "warnings": sample_warnings,
            "notes": sample_notes,
        })
        errors.extend(f"{s.sample_id}: {e}" for e in sample_errors)
        warnings.extend(f"{s.sample_id}: {e}" for e in sample_warnings)
        notes.extend(f"{s.sample_id}: {e}" for e in sample_notes)

    # content duplication: distinct paths/videos sharing a checksum
    for checksum, ids in sorted(checksum_to_ids.items()):
        if len(ids) > 1:
            warnings.append(
                f"duplicate image content (same sha256 {checksum}): {ids}"
            )
            duplicate_content.extend(ids)

    # deterministic hashing checks (computed twice on the same bytes)
    sha256_stable = (_sha256_hex(raw) == file_sha256)
    canonical_digest = _canonical_digest([asdict(s) for s in samples])

    summary = {
        "images_missing": len(images_missing),
        "images_unreadable": len(images_unreadable),
        "duplicate_sample_ids": len(set(duplicate_sample_ids)),
        "duplicate_image_paths": len(set(duplicate_paths)),
        "duplicate_content": len(set(duplicate_content)),
        "checksum_mismatches": len(checksum_mismatches),
        "dimension_mismatches": len(set(dimension_mismatches)),
        "provenance_gaps": len(provenance_gaps),
        "provenance_incomplete_count": len({sid for sid, _ in provenance_gaps}),
    }

    return {
        "valid": not errors,
        "manifest": str(path),
        "schema_version": manifest.schema_version,
        "preprocessing_version": manifest.preprocessing_version,
        "dataset_name": manifest.dataset_name,
        "sha256": file_sha256,
        "sha256_stable": sha256_stable,
        "canonical_digest": canonical_digest,
        "samples": len(samples),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "per_sample": per_sample,
        "summary": summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.validate_manifest",
        description="Validate an M1 real-data MANIFEST.json (schema, provenance, images).",
    )
    parser.add_argument("--manifest", required=True, help="path to MANIFEST.json")
    parser.add_argument(
        "--require-complete-provenance",
        action="store_true",
        help="reject samples without license + acquisition_date + image checksum "
             "(gate for real licensed training data)",
    )
    parser.add_argument(
        "--no-image-checks", action="store_true",
        help="skip image existence/readability/checksum checks",
    )
    parser.add_argument("--tokenizer", default=None,
                        help="BPE tokenizer; when set, captions are checked for reserved ids")
    parser.add_argument("--out", default=None, help="write the validation report as JSON")
    args = parser.parse_args(argv)

    report = validate_manifest(
        args.manifest,
        require_complete_provenance=args.require_complete_provenance,
        check_images=not args.no_image_checks,
        tokenizer_path=args.tokenizer,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "VALID" if report["valid"] else "INVALID"
    print(
        f"manifest={report['manifest']} status={status} samples={report['samples']} "
        f"sha256={report['sha256']}"
    )
    print(
        f"split counts: " + ", ".join(f"{k}={v}" for k, v in sorted(report["counts"].items()))
    )
    for w in report["warnings"]:
        print(f"warning: {w}")
    for e in report["errors"]:
        print(f"error: {e}")
    for n in report["notes"]:
        print(f"note: {n}")
    print(f"summary: {json.dumps(report['summary'])}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())