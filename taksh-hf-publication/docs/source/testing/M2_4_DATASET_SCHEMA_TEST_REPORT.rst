M2.4 — Unified Dataset Schema: Test Report
===========================================

.. contents:: On this page
   :local:
   :depth: 2

Date
----

2026-10-18 · Branch ``develop`` · HEAD ``4a5a144`` (unchanged; no commits made)

Automated evidence (Layer A)
----------------------------

Executed from the repository root:

.. code-block:: text

    python -m pytest tests/test_dataset_schema.py -q
    -> 37 passed

    python -m pytest tests/test_token_contract_and_chat.py -q
    python -m pytest tests/test_training_smoke_e2e.py -q
    -> 34 passed (combined)

    python -m pytest tests -q
    -> 523 passed

Full-suite baseline before this slice was 486 passed. M2.4 adds 37 tests; no
existing test was weakened or removed.

Defects found and fixed
-----------------------

Two defects were found by the new tests and fixed:

1. **Production code.** ``data/schema.py::audit_dataset`` returned early for a
   missing file **without** a ``conforms`` key, so a caller reading the summary
   raised ``KeyError: 'conforms'``. The key is now initialised before the early
   return. Covered by ``test_audit_handles_a_missing_file``.
2. **Test fixture.** ``tests/test_dataset_schema.py`` used
   ``np.asarray(..., file=...)``, which numpy does not accept; corrected to
   ``np.asarray(...).tofile(...)``.

No other defect was found or fixed in this slice.

Dataset conformance (measured, read-only)
-----------------------------------------

``python -m data.schema <files> --level release``

.. list-table::
   :header-rows: 1
   :widths: 34 20 16 30

   * - Dataset
     - Result
     - Kind
     - Note
   * - ``data/code_corpus_v2.jsonl``
     - **437/437 CONFORMS**
     - corpus
     - Licence-gated corpus with full provenance
   * - ``data/code_corpus_v1.jsonl``
     - **447/447 CONFORMS**
     - corpus
     - Licence-gated corpus with full provenance
   * - ``data/code_sft_v0.jsonl``
     - **442/442 CONFORMS**
     - sft
     - Conforms to the ``sft`` contract (corpus quarantined separately)
   * - ``data/corpus_v0.jsonl``
     - 0/9,308 DEVIATES
     - corpus
     - Legacy prose corpus; lacks provenance fields
   * - ``data/code_corpus_v0.jsonl``
     - 0/77 DEVIATES
     - corpus
     - Legacy self-sourced shard
   * - ``data/internal_corpus_v0.jsonl``
     - 0/5 DEVIATES
     - corpus
     - Legacy internal demo corpus

Deviating shards are **reported, not rewritten**. A test asserts the size and
mtime of ``data/corpus_v0.jsonl`` are unchanged by auditing.

Manual / operator evidence (Layer B)
------------------------------------

Partial only. The schema audit CLI was executed over the real datasets above and
its output recorded. A **full** Layer-B workflow walkthrough (acquisition →
validation → audit → processing → tokenization → artifact → validation → training
preparation) has **not** been performed as a single recorded operator run.

Encoder boundary behaviour (observed)
-------------------------------------

.. code-block:: text

    default          warn, continue
    --strict         refuse; no artifact written; SystemExit names the failures
    --no-validate    bypass; metadata records dataset_schema_validated: false

Open policy issue — **not changed in this slice**
-------------------------------------------------

Should production dataset encoding **fail closed by default**? Current behaviour
warns by default. ``--strict`` refuses; ``--no-validate`` bypasses. Recorded as a
policy decision for the user; the implementation was deliberately left unchanged.

Acceptance state
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - State
     - Status
     - Basis
   * - IMPLEMENTED
     - **yes**
     - ``data/schema.py``, encoder integration
   * - AUTOMATED TESTED
     - **yes**
     - 37 + 34 + 523 tests recorded above
   * - OPENCODE MANUAL TESTED
     - **partial**
     - Audit CLI run; full workflow walkthrough not performed
   * - USER ACCEPTED
     - **no**
     - Not yet performed
   * - TECHNICAL REVIEW
     - **no**
     - Not yet performed
   * - FORMALLY CLOSED
     - **no**
     - Blocked on the states above

**M2.4 is implemented and automatically tested. It is not formally closed.**