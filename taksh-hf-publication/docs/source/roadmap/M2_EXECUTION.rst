M2 — Dataset Expansion and Training Readiness
==============================================

.. contents:: On this page
   :local:
   :depth: 2

Authority
---------

``docs/ROADMAP.md`` is the single authoritative execution roadmap. This document is
its detailed record for **M2** and does not define a competing roadmap.

M2 is a **training-readiness** milestone. It is *not* authorisation to train, and
no base-model pretraining has been launched.

Slice status
------------

Statuses are evidence-based. Nothing below is upgraded without a repository
artifact, a recorded test result, or a recorded human decision.

.. list-table:: M2 slice status
   :header-rows: 1
   :widths: 8 26 22 44

   * - Slice
     - Name
     - Status
     - Evidence basis
   * - M2.1
     - Architecture and configuration
     - Complete
     - 7 train configs validate; ``validate_train_config()`` / ``unused_config_keys()``;
       model configs defined under ``configs/model/``
   * - M2.2
     - Resumable data acquisition
     - Complete
     - ``--resume`` in the corpus builder; interrupted-run and partial-line recovery
       tests
   * - M2.3
     - Corpus expansion
     - **Blocked / incomplete**
     - ~2.24M tokens against the 100M target. Requires allowlist expansion
       (human authorisation) and two unresolved licensing decisions
   * - M2.4
     - Unified dataset schema
     - Implemented / automated-tested / manual-tested
     - ``data/schema.py``; conformance measured on the real shards. Manual
       verification found four diagnostic defects (a missing corpus or tokenizer
       raised a raw traceback, a missing file was reported as "0 of 0 records
       fail", and empty versus nonexistent was indistinguishable). All four are
       fixed and regression-tested
   * - M2.5
     - Deterministic dataset processing
     - Implemented / automated-tested
     - Content-hash splits, order-independent source-group assignment,
       byte-identical re-encoding; 14 determinism/reproducibility tests
   * - M2.6
     - Deduplication and corpus composition
     - Implemented / automated-tested
     - Exact + MinHash dedup applied before splitting; cross-split leakage checks
       asserted in the manifest; 11 dedup/leakage tests
   * - M2.7
     - Tokenization and token budget
     - Implemented / automated-tested / manual-tested
     - ``<eos>`` document separators, document-granular token budget, artifact
       provenance and compatibility validation; regenerated artifact verified.
       A code token artifact (2,239,771 tokens) is now built from the released
       corpus and shipped inside the release set
   * - M2.8
     - Dataset versioning and release artifacts
     - Implemented / automated-tested / manual-tested
     - ``data/release.py``: a release is a named set (shard, provenance manifest,
       token artifact, token metadata) with ``RELEASE.json``,
       ``CHECKSUMS.sha256`` and a verify command that re-hashes every file. The
       token artifact's recorded ``corpus_sha256`` is checked against the
       released shard, so a token file from a different corpus cannot pass as
       part of the release
   * - M2.9
     - Training preparation
     - Implemented / automated-tested / manual-tested
     - Config validation, strict checkpoint compatibility, append-only loss log,
       SFT coverage accounting, end-to-end training smoke test. **No real training
       run has been launched.** ``configs/train/pretrain_code_v2.yaml`` points at
       the released code token artifact with a vocab-matched model
   * - M2.10
     - Evaluation preparation
     - Implemented / automated-tested / manual-tested
     - Held-out set materialised from the source-group-isolated test split with
       separation re-verified (``evaluation/holdout.py``); reproducibility
       records (``evaluation/run_manifest.py``); human-eval rubric, blinding
       protocol and reliability gate (``evaluation/human_eval.py``). **No human
       judgements exist yet, so no human score is releasable**
   * - M2.11
     - Documentation
     - Updated in this pass
     - This record, the release/gate page, and the consolidated acceptance
       checklist. Built clean by Sphinx
   * - M2.12
     - M2 release / training-readiness gate
     - **Implemented; verdict BLOCKED**
     - ``data/m2_gate.py`` reads real artifacts and reports evidence for ten
       checks. Verdict: ``M2 BLOCKED`` — 1 technical failure, 3 blocked on human
       decisions. See :doc:`M2_RELEASE_AND_GATE`. The gate is proven capable of
       returning ``M2 READY`` (``test_gate_reports_ready_when_everything_is_satisfied``)

Acceptance states
-----------------

These are **distinct** states. They must not be collapsed into a single "PASS".

.. code-block:: text

    IMPLEMENTED
        ↓
    AUTOMATED TESTED
        ↓
    OPENCODE MANUAL TESTED
        ↓
    USER ACCEPTED
        ↓
    TECHNICAL REVIEW
        ↓
    FORMALLY CLOSED

.. list-table:: Current acceptance state per slice
   :header-rows: 1
   :widths: 10 16 18 18 16 16 18

   * - Slice
     - Implemented
     - Automated tested
     - Manual tested
     - User accepted
     - Technical review
     - Formally closed
   * - M2.1
     - yes
     - yes
     - partial
     - no
     - no
     - no
   * - M2.2
     - yes
     - yes
     - partial
     - no
     - no
     - no
   * - M2.3
     - no
     - n/a
     - n/a
     - no
     - no
     - no
   * - M2.4
     - yes
     - yes
     - partial
     - no
     - no
     - no
   * - M2.5
     - yes
     - yes
     - partial
     - no
     - no
     - no
   * - M2.6
     - yes
     - yes
     - partial
     - no
     - no
     - no
   * - M2.7
     - yes
     - yes
     - yes
     - no
     - no
     - no
   * - M2.8
     - partial
     - partial
     - no
     - no
     - no
     - no
   * - M2.9
     - yes
     - yes
     - yes
     - no
     - no
     - no
   * - M2.10
     - partial
     - partial
     - no
     - no
     - no
     - no
   * - M2.11
     - partial
     - partial
     - no
     - no
     - no
     - no
   * - M2.12
     - no
     - n/a
     - n/a
     - no
     - no
     - no

"partial" manual testing means an operator command was executed and its output
recorded during implementation work — not a full Layer-B workflow walkthrough of
the slice. **No slice is USER ACCEPTED or TECHNICALLY REVIEWED.**

Validation layers
-----------------

.. list-table::
   :header-rows: 1
   :widths: 10 26 64

   * - Layer
     - Owner
     - What it establishes
   * - **A**
     - Automated testing (OpenCode/Space Bunny)
     - Unit, integration, regression, CLI, dataset-validation, deterministic
       processing, artifact validation and training-data smoke tests; full pytest;
       reproducibility checks. **Passing automated tests alone does NOT accept M2.**
   * - **B**
     - OpenCode manual / agent-driven testing
     - Important workflows exercised as a real operator, end to end: acquisition →
       validation → schema audit → processing → tokenization → artifact generation →
       artifact validation → training preparation. Each records command, inputs,
       actual output, warnings/errors, generated artifacts, logs, failure and
       recovery behaviour. Outputs must not be fabricated.
   * - **C**
     - Independent user manual testing
     - The user independently exercises the workflow as a separate gate. For model
       functionality this eventually includes prompt/response, code generation,
       completion, explanation, correction, conversational behaviour, context
       handling, checkpoint loading and failure behaviour. Must not duplicate the
       automated suite.
   * - **D**
     - Technical review (ChatGPT/technical reviewer)
     - After A–C, assess supplied evidence for architectural correctness, scope
       correctness, test adequacy, reproducibility, release readiness, known
       limitations, unsupported claims and documentation consistency.

M2 completion gate
------------------

M2 is **NOT COMPLETE** merely because individual implementation slices are done.
All of the following must hold:

#. Required corpus expansion achieved, or formally waived by the user.
#. Licensing decisions resolved (``psf/black``, ``pypa/setuptools``).
#. Dataset schema established (M2.4 formally closed).
#. Dataset processing shown deterministic.
#. Deduplication and corpus composition established.
#. Tokenization and token budget validated.
#. Dataset versioning and release artifacts established (M2.8).
#. Training preparation reproducible (M2.9).
#. Evaluation preparation established (M2.10).
#. Documentation complete (M2.11).
#. Layer A — automated tests pass.
#. Layer B — OpenCode manual tests recorded.
#. Layer C — user manual acceptance recorded.
#. Layer D — technical review completed.
#. Remaining limitations explicitly documented.

No base-model training is an M2 requirement, and none has occurred.

Open human decisions
--------------------

1. **Licensing** — ``psf/black`` (GPL-2.0 subtree) and ``pypa/setuptools``
   (AGPL-3.0 text). Excluded pending explicit human decision. Not resolved by
   assumption.
2. **Corpus scale** — M2.3 requires allowlist expansion; not authorised.
3. **Schema validation policy** — should production dataset encoding fail closed
   by default? Current behaviour: warns by default, refuses under ``--strict``,
   bypassable with ``--no-validate``. **Unchanged in this slice.**
4. **Multimodal track** — deferred; reactivation requires an explicit decision.
5. **Roadmap reconciliation** — resolved 2026-10-18; ``docs/ROADMAP.md`` is
   authoritative and multimodal is deferred.

Known limitations carried forward
----------------------------------

- Code corpus is ~2.24M tokens against a 100M target.
- No role tokens in the tokenizer vocabulary; conversational SFT compatibility is
  not claimed.
- The pilot token artifact is general prose, not the code corpus.
- Passing tests and valid tokenization do not establish conversational or coding
  capability.