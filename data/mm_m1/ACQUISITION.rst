Acquisition & provenance procedure (M1 real-data run)
=====================================================

This document is the safe procedure for populating ``data/mm_m1/`` with the
first real, licensed image/caption data for the M1 evaluation run.  It is a
*procedure*, not a dataset: it contains no records, no licenses, no URLs.

Nothing in this document authorizes downloading arbitrary web data.  Real data
is added only by a human, dataset by dataset, with a written acquisition log
(see `The acquisition log`_) and after a legal/provenance review.  Until then,
``data/mm_m1/MANIFEST.json`` stays an empty template.

Hard rule 0 — synthetic material never becomes "real" data
----------------------------------------------------------

* Synthetic hermetic pairs from ``scripts/synth_images.py`` /
  ``ImageTextDataset`` are stored as ``.npz`` for the synthetic fit-check path
  only.  They are NEVER written into ``data/mm_m1/`` and NEVER referenced by a
  ``MANIFEST.json``.
* A record is real licensed training data ONLY if it passes
  ``python -m training.validate_manifest --manifest <candidate> --require-complete-provenance``
  with its referenced image files present and readable.  Anything that fails
  that gate is excluded from training, full stop.
* Unit-test fixtures (like the ones under ``tests/``) live in ``tmp_path`` at
  test time and are never committed into ``data/mm_m1/``.
* Any manifest that survives these rules is, by construction, unable to
  silently carry a synthetic or test fixture as a licensed real sample: those
  carry no license, no acquisition date, and no image checksum, and strict
  validation rejects them.

Record contract (schema version 1)
----------------------------------

Every admitted sample must be a JSON object inside ``samples`` with exactly
these semantics:

``sample_id`` (required)
    Stable, unique ASCII id, e.g. ``mm1-000001``.  Never reused across
    re-acquisitions; a retracted sample id stays retracted.

``split`` (required)
    One of ``{train, val, test}``.  The split is decided and recorded here; it
    is never inferred from directory placement.  Real-data exclusions (dedup,
    tests) happen before the split is assigned.

``image_path`` (required)
    Path to the image file, relative to the manifest directory
    (``data/mm_m1/``).  See `Image-file naming rules`_.

``caption`` (required)
    A non-empty, human-written natural-language description of the image in the
    source language.  Captions must encode to BPE ids below ``49152`` (no
    special/reserved ids) — the validator rejects them otherwise.

``source`` (required)
    The identity of the original dataset/provider, e.g. ``OAI/CarDD v2 research
    release``.  This is what the data card and training metadata record.  It
    must name something real and documented in the acquisition log.

``source_url`` (optional)
    The original source URL or acquisition reference.  When the data was
    acquired by other documented means (physical media, direct handoff), record
    the acquisition-log reference instead.  ``null`` means genuinely unknown.

``license`` (optional)
    The actual license of the original material when known; ``null`` when
    unknown.  Complete-provenance validation REJECTS unknown licenses for real
    training data.  Recognized license identifiers (SPDX or well-known dataset
    license) are strongly preferred over prose.

``license_ref`` (optional)
    Evidence for ``license``: the license URL, a citation, or a registry id
    (e.g. ``https://spdx.org/licenses/CC-BY-4.0``).  Required for strict
    admission of real data.

``acquisition_date`` (optional)
    ISO-8601 date (``YYYY-MM-DD``) on which QuantsMind acquired the original
    material.  Required for strict admission of real data.

``checksum`` (optional)
    64-char lowercase hex sha256 of the image file *as recorded* (raw bytes).
    Required for strict admission of real data; the validator compares it
    against the on-disk file and fails on mismatch.

``attribution`` (optional)
    Creator/owner name when required or useful; otherwise ``null``.

``width`` / ``height`` (optional)
    Pixel dimensions of the *original* image.  When recorded they must match
    the file; the validator raises a warning on mismatch (a resize is
    suspicious and must be logged as a transformation).

``preprocessing_version`` (optional)
    Defaults to ``mm1_v2``; new acquisitions must record ``mm1_v2``.
    ``mm1_v1`` is frozen as historical evidence (pre-EXIF-correction pipeline)
    and must not be relabeled. Changing or introducing a preprocessing step is
    a transformation (below) and must be versioned and documented — the loader
    records it in checkpoint metadata.

Dataset identity
----------------

``dataset.name`` identifies the shipped set (e.g. ``mm_m1``); it must not
contain provenance.  Per-acquisition identity, licensing and URLs belong in the
acquisition log and in each sample's ``source``/``source_url``/``license``/
``license_ref`` — not in the dataset name.

Image-file naming rules
-----------------------

* Files live under ``data/mm_m1/train/``, ``val/``, or ``test/`` matching the
  sample's split.
* Names: lowercase ASCII, ``[a-z0-9._-]`` only; no spaces, quotes, non-ASCII, or
  path tricks.  Recommended ``<sourceid>-<zero-padded-index>.jpg``.
* The recorded ``image_path`` must resolve strictly inside the split directory;
  resolution is relative to the manifest directory and never escapes it.
* Never overwrite an existing file with different content: a changed file must
  get a new name and a new ``checksum``; the old id stays retired.

Duplicate handling
------------------

* ``sample_id`` must be unique; duplicate ids are a validation error.
* Two samples must not reference the same image file; identical resolved paths
  are a validation error.
* Identical image *content* (same sha256) across distinct samples is a
  ``duplicate_content`` warning: a human must dedupe or explicitly retain both
  (e.g. different captions) and log the decision.  Copyright dedup (not just
  exact bytes) is also required before training, following the project's
  provenance-ledger principle.
* ``checksum`` values must be mutually consistent: equal checksum values imply
  equal bytes, verified against the file.

Recorded transformations
------------------------

The M1 pipeline applies a fixed deterministic transform
(``preprocess_image``: RGB conversion, cover-resize, center crop to
``112 x 112``, float32 ``[0, 1]``).  Per-sample cropping/resizing decisions are
NOT encoded in the manifest; they are covered by ``preprocessing_version`` and
by the transform itself.  Version history:

* ``mm1_v1`` — RGB conversion, cover-resize, center crop to ``112 x 112``;
  EXIF orientation flags ignored (historical; frozen, do not relabel).
* ``mm1_v2`` — ``mm1_v1`` plus EXIF orientation correction
  (``ImageOps.exif_transpose`` applied on load, before conversion/resize).
  Recorded width/height are the post-correction pixel dimensions, matching
  what the validator decodes and what training sees.

Any additional transformation (explicit
``normalize``, caption rewriting, format conversion, lossy re-encode) must be:

1. recorded as a note on the affected samples or in the acquisition log;
2. covered by a new ``preprocessing_version`` when it changes the bytes seen
   by training.

Rejecting records with incomplete provenance
--------------------------------------------

Run, for every candidate manifest:

``python -m training.validate_manifest --manifest <candidate> --require-complete-provenance --tokenizer tokenizer/qm_bpe_v0.json``

Incomplete records (missing license / acquisition date / checksum, unreadable
or missing images, checksum mismatch, reserved-id captions, malformed fields)
FAIL the validation with exit code 1 and are removed from the manifest before
it enters training.  The strict gate is the machine-readable guarantee that no
under-provenanced sample is ever presented as licensed real training data.

The acquisition log
-------------------

Each dataset addition is recorded in a human-written, dated log entry (outside
the repo, or in a future standalone file) stating: dataset/provider identity,
acquisition method and date, exact source URL or reference, license text or id,
license evidence URL, number of items, duplicate/exclusion report, and the
person who performed the legal/provenance review.  Manifest field values
reference this log; they never substitute for it.

Checksums & reproducibility
---------------------------

* ``MANIFEST.json`` is hashed as raw bytes via ``Manifest.sha256()`` (this is
  the identity recorded in training checkpoint metadata).
* The validator reports both that raw ``sha256`` and a canonical content digest
  (sorted samples, sorted keys, compact JSON) so formatting changes don't mask
  a logical identity change.
* Per-image ``checksum`` values act as the byte-level integrity record; a file
  that changes without a manifest update is caught as a checksum mismatch.

What stays out of scope
-----------------------

This procedure does not add images, does not invent licenses/URLs/checksums/
dataset names, does not populate ``MANIFEST.json``, does not alter the M1
promotion gate, and does not train the model.  When real data is eventually
admitted it must reproduce exactly per this document and the validator.