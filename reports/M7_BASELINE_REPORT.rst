M7 Baseline Inspection Report
=============================

Target version: v0.7.0 (historical baseline record -- never released)

Current release target: v1.0.0 (Apache-2.0 source release; publication tracked separately).
See ``docs/source/releases/M7_RELEASE_READINESS.rst``.

Report date: 2026-10-20

.. contents:: On this page
   :local:
   :depth: 2

Purpose
-------

This report records the repository, evidence, documentation and serving state
observed **before** the formal M7 documentation, versioning and compatibility
changes were made. It is an inspection record, not a capability claim.

One sequencing fact is stated plainly: M7 dataset, training and evaluation
artifacts already existed as uncommitted work when the formal M7 planning
documents were requested. They were produced in this working session and are
preserved. This report does not pretend they did not exist at baseline, and it
does not use hindsight to rewrite their numbers.

Baseline identity
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Observation
   * - Repository
     - ``F:\\Codes\\Git\\Quantsmind-LLM``
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``
   * - Tags / releases
     - 0 tags; no product release
   * - Working tree
     - **177 dirty entries**: 37 modified, 140 untracked; all preserved
   * - Product name in repository
     - ``Quantsmind-LLM`` and ``Quantsmind Coder``
   * - Requested product name
     - **Taksh** (product-facing rename only; repository name and URLs unchanged)
   * - Requested target version
     - **v0.7.0**
   * - Documentation version at baseline
     - ``0.1.0`` in ``docs/source/conf.py``
   * - Test baseline
     - 892 passed before the final M7 import cleanup; 42 relevant M7 tests
       passed after it
   * - Sphinx baseline
     - Sphinx 7.4.7 available; existing HTML build loaded with no source changes

M2--M6 evidence at baseline
---------------------------

The existing milestone record is negative or partial. M7 must not silently
upgrade any of it.

.. list-table:: Prior evidence
   :header-rows: 1
   :widths: 12 30 22 36

   * - Milestone
     - Recorded scope
     - Verdict
     - Evidence basis
   * - M2
     - Dataset expansion and training readiness
     - ``M2 BLOCKED``
     - 2.24M of 100M tokens; two licence decisions and human authorisations
       outstanding
   * - M3
     - Base pretraining from random initialisation
     - Implemented, demonstrated, not accepted
     - 2,853,280 params, 400 steps, 1.17M tokens; generated code does not parse
   * - M4
     - Instruction tuning
     - Implemented, negative capability result
     - 2,853,280 params; no usable functional evidence
   * - M5
     - Evaluation and strategy selection
     - ``INSUFFICIENT_EVIDENCE``
     - 1/20 syntax-valid for M3, 0/20 for M4; no executable submissions
   * - M6
     - Documented as ``Local Capability Validation``, not serving
     - ``INSUFFICIENT_EVIDENCE``
     - Three local candidates, all 0 executable; no coding-capable local
       candidate acquired

**Roadmap conflict.** ``docs/ROADMAP.md`` calls M6 "Serving and Product
Integration" and M7 "Release / Production Readiness". The published
``docs/source/roadmap/M6_EXECUTION.rst`` and ``reports/M6_CAPABILITY_REPORT.rst``
instead record **M6 Local Capability Validation**. The new M7 request uses
"Sovereign Capability Experiment". These historical labels conflict and must not
be silently renumbered. The M7 documents record the actual evidence and keep
the historical artifacts as written.

M7 data readiness
-----------------

.. list-table:: Dataset partitions
   :header-rows: 1
   :widths: 22 14 64

   * - Partition
     - Records
     - Notes
   * - Train
     - 507
     - 200 generation, 100 documentation, 100 explanation, 100 completion,
       7 fixtures; 58,123 supervised tokens
   * - Validation
     - 58
     - aiohttp, disjoint file buckets from development
   * - Development
     - 42
     - aiohttp plus two fixtures; not used for training or selection
   * - Locked
     - 159
     - 40 generation, 54 documentation, 48 explanation, 14 completion,
       1 SQL, 1 config, 1 format

.. list-table:: Licence and isolation
   :header-rows: 1
   :widths: 38 62

   * - Check
     - Observation
   * - Locked source
     - ``psf-requests``, 156 records, Apache-2.0, plus 3 CC0 fixtures
   * - Verified sources
     - 7 sources; 162 records withheld from 2 unverified sources
   * - Provenance
     - every repository-derived record carries source, licence, source sha1 and
       source-file sha256
   * - Split isolation
     - 0 shared ids, examples, responses or source files; train and locked do
       not share a source
   * - M5 benchmark
     - unchanged at manifest hash ``52ea2ed5db839b22...``; no exact prompt or
       reference overlap with M7 training
   * - Functional oracles
     - 8 reference functions, 29 cases, expected values produced by execution
       in a subprocess

The data is ready for the **bounded pilot** that was run. It is not a
production instruction corpus.

M7 resource readiness at baseline
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Item
     - Observation
   * - Preflight verdict
     - ``FEASIBLE``
   * - Model
     - 9,359,104 params, vocab 5,822, context 512
   * - Measured peak requirement
     - 309,600,256 bytes over 3 real optimizer steps
   * - Current-availability budget
     - 315,033,600 bytes
   * - Total-RAM budget
     - 2,939,846,246 bytes
   * - Projected pretraining
     - 40.8 minutes for 300 steps at 4,096 tokens/step
   * - Actual pretraining
     - 300 steps, 1,228,800 tokens, 27.2 minutes, peak RSS 700,182,528 bytes
   * - Actual SFT
     - 120 steps, 491,520 tokens, 9.8 minutes, peak RSS 907,993,088 bytes
   * - Device
     - CPU only; CUDA absent; 8 logical CPUs, 4 torch threads
   * - Disk free
     - 190.9 GiB at preflight

The run completed inside the resource envelope without a resource-gate failure.
That is an engineering result, not evidence of capability.

M7 artifact state at baseline
-----------------------------

.. list-table:: Existing M7 artifacts
   :header-rows: 1
   :widths: 48 52

   * - Path
     - State
   * - ``data/m7/``
     - Dataset partitions, manifest, pretraining token artifacts and oracles
   * - ``configs/eval/m7_protocol.yaml``
     - Pre-registered protocol, gates, thresholds and limitations
   * - ``configs/train/pretrain_m7.yaml``, ``sft_m7.yaml``
     - Training configurations
   * - ``runs/m7_pretrain``, ``runs/m7_sft``
     - Completed runs with checkpoints and manifests
   * - ``runs/m7_untrained``
     - Step-0 control checkpoint
   * - ``runs/m7_eval``, ``runs/m7_eval_secondary``
     - Primary and secondary locked evaluations
   * - ``reports/M7_GATES.json``, ``M7_OUTCOME.json``, ``M7_PREFLIGHT.json``,
       ``M7_AUTHORIZATION.json``
     - Machine-readable gate, outcome, resource and authorization records
   * - ``reports/M7_CANDIDATE_REPORT.rst``
     - Candidate training and evaluation report
   * - ``tests/test_m7_evaluation.py``
     - Task-aware syntax-scoring tests only

Measured M7 result at baseline
------------------------------

.. list-table:: Locked code generation, 40 tasks
   :header-rows: 1
   :widths: 34 14 14 14 24

   * - Model
     - Params
     - Syntax
     - Functional
     - Common validation loss
   * - Untrained control
     - 9.36M
     - 0/40
     - 0/8
     - 8.7339
   * - M3 base
     - 2.85M
     - 0/40
     - 0/8
     - 4.9685
   * - M4 tuned
     - 2.85M
     - 0/40
     - 0/8
     - 4.9220
   * - M7 candidate
     - 9.36M
     - 0/40
     - 0/8
     - 5.9473

The candidate reached evidence level **0**. The machine-readable verdict is
``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED``. Gate G failed; integrity
gate F and independent-evaluation gate H passed.

Secondary locked evaluation also failed: documentation 0/54, completion 0/14,
SQL 0/1, config 0/1, format 0/1. Explanation format passed 46/48, but semantic
correctness has no automatic oracle and is not a capability claim. The secondary
run used 128 new tokens versus the protocol's 192, and 6 completion prompts
exceeded the 512-token context. Both limitations must remain attached to those
numbers.

M6 serving and integration state at baseline
--------------------------------------------

The serving layer exists in ``serving/`` and predates M7. The OpenAI-compatible
server provides:

* ``GET /health``, ``GET /v1/models`` and non-streaming
  ``POST /v1/chat/completions``;
* loopback binding by default, explicit ``--allow-remote`` opt-in and optional
  ``QM_API_KEY`` bearer authentication;
* startup-only checkpoint loading; no HTTP endpoint accepts a checkpoint path;
* real-checkpoint tests against ``runs/code_pretrain_tiny``;
* explicit rejection of unsupported OpenAI parameters.

**Baseline compatibility failure.** Loading the M7 SFT checkpoint through
``serving.openai_server.load_state`` failed with ``KeyError: 'model_config'``.
The M7 fine-tuning checkpoint records its architecture under
``config.run.base_model_config``, while ``serving.local_inference.load_model``
only reads ``config.run.model_config``. The general loader
``training.generate.load_checkpoint_for_generation`` already handles both
fields, but serving does not use it.

This is a real integration gap, not a cosmetic one. At baseline, M7 serving
compatibility is **BLOCKED**. It must be fixed and covered by a real-checkpoint
test before the final report can claim compatibility. Loading the checkpoint
through the training generation path is known to pass; that does not repair the
serving path.

Documentation state at baseline
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Check
     - Observation
   * - Sphinx source
     - ``docs/source`` with RST; Sphinx 7.4.7; ``sphinx-rtd-theme``
   * - Existing M7 documentation
     - None
   * - Existing serving documentation
     - Scattered across README and ``docs/OPEN_ISSUES.md``; no M7 serving
       compatibility record
   * - Index title
     - ``QuantsMind-LLM``
   * - Index toctree
     - M2 through M6 only
   * - Version
     - ``0.1.0``
   * - Required M7 RST files
     - Not present:
       ``M7_SOVEREIGN_CAPABILITY_EXPERIMENT.rst``,
       ``M7_EXPERIMENT_PLAN.rst``, ``M7_EXPERIMENT_REPORT.rst``,
       ``M7_REPRODUCIBILITY.rst``, ``M7_MANUAL_VERIFICATION.rst``
   * - Existing build
     - No-op Sphinx build succeeded; a forced rebuild of new documents is still
       required
   * - Encoding
     - Several pre-existing Markdown and RST files contain mojibake; broad
       rewriting is out of scope and risks losing history

M7 test coverage at baseline
----------------------------

``tests/test_m7_evaluation.py`` tests only task-aware syntax scoring. It does
not test:

* rejection of an artificially green result;
* a missing dataset, protocol, checkpoint or evaluation summary;
* failed or incomplete training manifests;
* resource-gate rejection;
* preservation of the M5 benchmark hash;
* serving compatibility with the M7 checkpoint.

Those are required M7 tests. The baseline test suite passing does not close the
documentation or serving gaps above.

Experiment justification at baseline
------------------------------------

The bounded experiment was justified as the smallest test that could distinguish
"more parameters and more data" from "no capability":

* 9.36M parameters, 300 pretraining steps and 120 SFT steps on 1.66M total
  training tokens, all CPU-feasible;
* an untrained same-architecture control and the existing M3/M4 checkpoints;
* a source-disjoint locked partition from ``psf-requests``;
* functional oracles executed from reference implementations;
* pre-registered thresholds fixed before evaluation;
* no hosted API and no external model.

The measured answer is negative. That does not retroactively invalidate the
experiment; it means the experiment must be reported as negative and must not be
extended by tuning against the locked set.

Known limitations and conflicts
-------------------------------

1. **Naming conflict.** Roadmap M6/M7 labels disagree with the published M6
   capability record and the new M7 request. Historical labels are preserved;
   the new M7 scope is "Sovereign Capability Experiment".
2. **Product naming.** The requested product name is **Taksh**; the repository,
   schema identifiers and historical reports keep ``Quantsmind``. A broad
   textual rewrite would break schemas and provenance. Only product-facing
   documentation and application titles should change.
3. **Gate naming collision.** The request defines readiness Gates A, B and C
   (data, resources, justification). The existing protocol defines engineering
   Gates A--H (licence, dataset, isolation, resources, authorization,
   integrity, capability, independent evaluation). Both must be documented with
   an explicit mapping, not conflated.
4. **Pre-existing work.** M7 code and results were already uncommitted at
   baseline. The formal plan cannot claim they were produced after approval.
5. **Small sample.** 507 training records, 40 locked generation tasks and 8
   functional probes are pilot evidence. Levels 4 and 5 were never available.
6. **Secondary evaluation mismatch.** Secondary generation used 128 new tokens,
   not the protocol's 192. Six completion prompts did not fit the 512-token
   context. These are recorded failures/limitations, not zeroes.
7. **No human evaluation.** The protocol requires independent human review for
   level 5. None occurred.
8. **No release.** ``v0.7.0`` is a target version. No commit, tag, push, merge,
   package publication or release was performed.

Baseline decision
-----------------

Proceed with documentation, versioning, serving compatibility repair and
negative-path tests. Do **not** retrain, tune against the locked partition,
alter the pre-registered thresholds, overwrite M5 evidence, or describe the
negative result as a capability success.
