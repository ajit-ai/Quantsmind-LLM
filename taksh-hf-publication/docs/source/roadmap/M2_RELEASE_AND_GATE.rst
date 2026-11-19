M2 Release Artifacts, Evaluation Preparation and Readiness Gate
===============================================================

.. contents:: On this page
   :local:
   :depth: 2

Purpose
-------

This page records the M2 work that turns a corpus into a *release*, an
evaluation set, and a *verdict*. It is the evidence behind the slice statuses in
:doc:`M2_EXECUTION`.

The central claim of this page is narrow and checkable: **M2 is not complete,
and the reason is enumerated rather than softened.**

Release artifacts (M2.8)
------------------------

A corpus is not released because a JSONL file exists. ``data/release.py``
defines a release as a named, verifiable set:

.. code-block:: text

    data/releases/code_corpus-v2/
        RELEASE.json                     identity, contents, checksums, provenance
        CHECKSUMS.sha256                 sha256sum -c compatible
        code_corpus_v2.jsonl             437 records, licence-gated
        code_corpus_v2.manifest.json     provenance, splits, licence breakdown
        code_corpus_v2.tokens.bin        2,239,771 uint16 token ids
        code_corpus_v2.tokens.meta.json  encoding provenance

Build and verify::

    python -m data.release build \
        --corpus data/code_corpus_v2.jsonl \
        --manifest data/code_corpus_v2.manifest.json \
        --extra data/code_corpus_v2.tokens.bin \
        --extra data/code_corpus_v2.tokens.meta.json \
        --out data/releases/code_corpus-v2 \
        --dataset-id code_corpus-v2

    python -m data.release verify --release data/releases/code_corpus-v2

Nine checks run on verify; all nine currently pass:

.. list-table:: Release verification checks
   :header-rows: 1
   :widths: 34 66

   * - Check
     - What a failure means
   * - ``schema_ok``
     - The record was written by a different tool version
   * - ``files_present``
     - A released file was deleted
   * - ``checksums_match``
     - A released file was modified after the release was built
   * - ``checksum_file_agrees``
     - ``CHECKSUMS.sha256`` and ``RELEASE.json`` disagree with each other
   * - ``record_count_matches``
     - The shard no longer holds the number of records claimed
   * - ``corpus_status_complete``
     - The underlying corpus manifest is not ``complete``
   * - ``no_integrity_failures``
     - The corpus manifest records integrity failures
   * - ``token_chain_matches_shard``
     - The token artifact was encoded from a *different* corpus
   * - ``token_artifact_schema_validated``
     - The token artifact was encoded with schema validation switched off

The ``token_chain_matches_shard`` check is the one worth understanding. The
token artifact's own metadata records the SHA-256 of the corpus it was encoded
from. If that hash is not among the released files, the token file was produced
from something else, and the release says so rather than shipping a mismatched
pair.

.. note::

   ``RELEASE.json`` carries ``"released": false``. A verified local set is not a
   publication. Nothing in this module uploads, tags, or publishes anything;
   those remain separate, explicitly authorised operations.

Evaluation preparation (M2.10)
------------------------------

Held-out data
~~~~~~~~~~~~~

``evaluation/holdout.py`` materialises the held-out split and proves it is held
out, by intersecting content SHA-1 sets *and* repository sets against the
training split. Both must be empty; one clean intersection is not enough.

.. code-block:: text

    evaluation/code_holdout_v2.jsonl         46 items
    evaluation/code_holdout_v2.manifest.json

These items are real source files from 2 repositories that appear in no training
split. The manifest records the separation method, the corpus checksum, and the
protocol.

What this set cannot do
~~~~~~~~~~~~~~~~~~~~~~~

It is **not** an executable benchmark, and the manifest says so explicitly:

* It supports per-token cross-entropy loss and nothing more. It cannot report
  pass@1, because a source file is not a function specification with a test
  harness.
* HumanEval (164 problems) and MBPP (500 problems) are **not present** in this
  repository. The 5-problem subsets in ``evaluation/eval_code.py`` exist only to
  prove the generation-and-execute harness runs end to end, and any pass rate
  measured over 5 problems is stamped ``unmeasured``.
* The held-out set spans 2 repositories. It measures generalisation to unseen
  code, not breadth of capability.

Baseline
~~~~~~~~

The manifest records the uniform-random loss floor (``ln(vocab_size)``, 8.67 for
the 5822-token code vocabulary). A held-out loss at or above that number means
the model has learned nothing, and should be reported as such.

Reproducibility
~~~~~~~~~~~~~~~

``evaluation/run_manifest.py`` defines what an evaluation run must record:
checkpoint, seed, dataset release, held-out set, tokenizer, config, task set and
task-set version -- each with a checksum. A run missing any of them is reported
``certified: false`` with the specific gaps named. There is no "mostly
reproducible".

It also checks for drift: if the held-out manifest's recorded corpus checksum is
not among the released files, the release and the evaluation set have diverged.

Human evaluation
~~~~~~~~~~~~~~~~

``evaluation/human_eval.py`` defines a versioned rubric, a blind pairwise
protocol, and a reliability gate. Two properties matter:

* A single annotator cannot produce a released score. With no item judged by two
  or more annotators, kappa is undefined and the report says human scores must
  not be released.
* Kappa is paired Cohen's kappa over annotator pairs. A marginal-based chance
  term would score systematically anti-correlated annotators as merely mediocre;
  the paired form correctly reports them as worse than chance.

Two independent routes to releasability exist, because kappa's variance
correction is undefined when every item is unanimous under a single label: full
unanimity across at least two annotators, or kappa at or above 0.6. A single
disputed item fails the whole batch.

.. warning::

   **No human judgements have been collected.** The rubric weights are
   provisional and uncalibrated. No human score for this model exists.

The readiness gate (M2.12)
--------------------------

``data/m2_gate.py`` reads real artifacts and reports the evidence for each check.
There are no hardcoded results: a missing artifact fails, and a fact that only a
human can supply is reported ``BLOCKED`` rather than guessed.

.. code-block:: text

    PASS      the requirement is met, and the evidence is named
    FAIL      a technical deficiency; someone could fix it by writing code
    BLOCKED   a decision only a human can make

Current verdict: ``M2 BLOCKED`` -- 1 technical failure, 3 blocked on human
decisions.

.. list-table:: Gate checks
   :header-rows: 1
   :widths: 16 44 40

   * - Check
     - Requirement
     - Current status
   * - ``M2.4-schema``
     - Records satisfy the release schema
     - PASS (437/437)
   * - ``M2.3-licence``
     - Every source has an approved licence
     - **BLOCKED** -- 2 sources need a human licence decision
   * - ``M2.3-scale``
     - Corpus draws from at least 20 sources
     - **BLOCKED** -- 9 repositories; allowlist expansion needs authorisation
   * - ``M2.3-tokens``
     - Token corpus reaches 100,000,000 tokens
     - FAIL -- 2,239,771 (2.24% of target); blocked upstream by source scale
   * - ``M2.6-splits``
     - Splits do not share repositories
     - PASS (46 test vs 193 train, no shared repos)
   * - ``M2.8-release``
     - Released artifacts verify
     - PASS (9 of 9 checks)
   * - ``M2.9-training``
     - Config resolves to a real, vocab-matched artifact
     - PASS (model vocab 5822 = tokenizer vocab 5822)
   * - ``M2.10-holdout``
     - Held-out data exists and is separated
     - PASS (46 items, 2 repositories)
   * - ``M2.10-benchmark``
     - Evaluation capability is not overstated
     - PASS (declared non-executable)
   * - ``M2.11-human-eval``
     - Human evaluation defined, calibrated, releasable
     - **BLOCKED** -- no annotators, no judgements, no agreement figure

Reproduce with::

    python -m data.m2_gate

The gate is not rigged to fail. A test builds a complete, self-consistent world
in which every artifact-backed check passes and asserts the verdict is
``M2 READY``
(``tests/test_m2_gate.py::test_gate_reports_ready_when_everything_is_satisfied``).

Consolidated acceptance checklist
---------------------------------

These are **distinct** states and must not be collapsed into a single "PASS". No
slice has reached user acceptance or technical review, and none has been
formally closed. Those require a human, not a test run.

.. list-table:: M2 acceptance state
   :header-rows: 1
   :widths: 10 15 15 15 15 15 15

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
     - yes
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
     - yes
     - yes
     - yes
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
     - yes
     - yes
     - yes
     - no
     - no
     - no
   * - M2.11
     - yes
     - yes
     - yes
     - no
     - no
     - no
   * - M2.12
     - yes
     - yes
     - yes
     - no
     - no
     - no

What closes M2
---------------

Three items are outside engineering's reach and are stated here so they are not
quietly forgotten:

1. **Licence decisions.** Two sources (``psf-black``, ``pypa-setuptools``)
   need a human judgement on whether their licences are acceptable as training
   data. This is a legal question, not a code change.
2. **Source authorisation.** Expanding the allowlist from 9 repositories toward
   the 20-source target requires explicit human authorisation before anything is
   fetched. Until then the token shortfall (2.24% of target) cannot be closed,
   because the shortfall is caused by the source restriction.
3. **Human evaluation.** The rubric and protocol are defined, but no annotators
   have been recruited and no judgements collected, so no agreement figure exists
   and no human score may be released.

Additionally, **user acceptance (Layer C)** and **technical review (Layer D)** for
every slice remain outstanding. They are human steps. No test suite can close
them, and this document does not claim otherwise.