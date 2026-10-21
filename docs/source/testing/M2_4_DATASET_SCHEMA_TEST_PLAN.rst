M2.4 — Unified Dataset Schema: Test Plan
=========================================

.. contents:: On this page
   :local:

Scope
-----

This plan covers **M2.4 Unified Dataset Schema** only. It is not an M1, M2.3,
M2.5+ or training-execution plan.

Pre-conditions
--------------

* ``data/schema.py`` defines per-kind contracts for ``corpus`` / ``sft`` / ``vqa``.
* Two levels are distinguished: ``encoding`` (tokenizer boundary) and ``release``
  (distributable corpus).
* ``data/encode_dataset.py`` validates at the encoding boundary.

Layers exercised
----------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Layer
     - Activity
     - Expected
   * - A
     - Automated: ``tests/test_dataset_schema.py``
     - Schema definition, validation, audit and encoder-boundary behaviour
   * - A
     - Automated: related suites
     - Encoder/token-contract and end-to-end smoke suites still pass
   * - A
     - Automated: full pytest
     - Whole repository green
   * - B
     - Manual: schema audit CLI over the real datasets
     - Conformance reported per file, read-only, with deviations explained
   * - B
     - Manual: encoding a conforming corpus
     - Artifact produced, metadata carries the schema version

Not covered
-----------

* No claim that any dataset was *rewritten* to conform — legacy shards are only
  reported.
* No performance or scale testing.
* No live corpus acquisition.

Pass criteria
-------------

#. Every test in ``tests/test_dataset_schema.py`` passes.
#. A malformed record is rejected at ``--strict`` and warned about by default.
#. ``audit_dataset`` never modifies the file it inspects.
#. Related and full suites pass with no regression.