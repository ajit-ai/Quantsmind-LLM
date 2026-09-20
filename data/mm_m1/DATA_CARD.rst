MANIFEST.json
Data card and directory structure for the M1.2 real-image dataset.

Purpose
=======
Real image/caption pairs for the M1 architecture milestone. No real dataset is
included in this repository: images live outside the repo and are referenced
through a provenance-aware ``MANIFEST.json``. The repository only ships the
manifest *schema*, this data card, and the loader/parser code.

Directory layout
================

.. code-block:: text

   data/mm_m1/
       MANIFEST.json      # provenance-aware sample index (schema version 1)
       DATA_CARD.rst      # this file
       train/             # split directories (image files; never committed)
       val/
       test/

Split policy
============
Every sample declares exactly one split in ``{train, val, test}``. Splits are
recorded in the manifest, not inferred from directory placement.

Provenance rules
================
* ``source`` is required for every sample. It names the original dataset /
  provider (e.g. ``X-CC0-localized``).
* ``license`` must be the actual license of the *original* material when known.
  When unknown, it must be ``null`` — never an invented value.
* ``source_url`` and ``attribution`` are recorded when known; otherwise ``null``
  or omitted.
* The QuantsMind project does not fabricate provenance or licensing information
  for data it did not itself collect. Each addition must be reviewed against a
  documented acquisition log.
* Derived images we generate ourselves are described by ``source`` plus a local
  generation note, never attributed to a third party.

Sample schema (schema_version 1)
================================

.. code-block:: json

   {
     "sample_id": "mm1-000001",
     "split": "train",
     "image_path": "train/mm1-000001.jpg",
     "caption": "a red hatchback parked beside a parked bus",
     "source": "OAI/CarDD v2 research release",
     "source_url": "https://example.in/dataset",   // optional
     "license": "CC BY 4.0",                        // null when unknown
     "attribution": null,                           // optional
     "width": 1920,                                 // optional
     "height": 1080,                                // optional
     "preprocessing_version": "mm1_v1"
   }

Paths are relative to ``data/mm_m1/`` (the directory containing the manifest).

Preprocessing
=============
* RGB conversion, deterministic cover-resize, center crop to ``112 x 112``
  (the M1.1 image resolution), conversion to a float32 [0, 1] tensor.
* No augmentation in M1.2.

Status
======
M1 is not promoted. M1.2 establishes real-data *infrastructure* only.