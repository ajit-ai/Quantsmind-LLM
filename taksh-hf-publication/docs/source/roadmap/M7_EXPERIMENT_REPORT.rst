M7 Sovereign Capability Experiment -- Report
============================================

Release target: v1.0.0 (Apache-2.0 source release; publication tracked separately)

.. contents:: On this page
   :local:
   :depth: 2

Verdict
-------

**COMPLETE -- NEGATIVE CAPABILITY RESULT, REPORTED AS MEASURED.**

M7 was justified, resourced, run and evaluated. The integrity gates passed; the
capability gate failed:

* common-validation loss **5.9473**, worse than M3 (4.9685) and M4 (4.9220);
* locked code generation **0/40** syntax-valid and **0/8** functional;
* **0/29** functional cases passed;
* evidence level **0**;
* outcome ``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED``.

No model or weight release was cut by this experiment. The result does not
support a claim that the model can write code, follow coding instructions
reliably, or meet a production standard.

Baseline and scope
------------------

The pre-documentation state is in ``reports/M7_BASELINE_REPORT.rst``. The
starting point was branch ``develop`` at ``4a5a144``, with no tags and no
release at baseline. M2 was ``BLOCKED``; M3/M4 produced no parseable code; M5
and M6 were ``INSUFFICIENT_EVIDENCE``.

M7 asked whether a 9.36M-parameter model trained from random initialisation on
licence-verified code could demonstrate capability under local CPU and licence
constraints. It used no hosted API and no pretrained weights.

What was run
------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Component
     - Configuration
   * - Candidate
     - 9,359,104 parameters, vocab 5,822, context 512
   * - Pre-training
     - 300/300 steps, 1,228,800 tokens, 27.2 minutes, seed 42
   * - Fine-tuning
     - 120/120 steps, 491,520 tokens, 9.8 minutes, full fine-tuning
   * - Training data
     - 507 supervised records, 58,123 supervised tokens; 1,165,217 pretraining
       tokens
   * - Controls
     - Untrained 9.36M step-0 model; M3 base; M4 tuned
   * - Primary evaluation
     - 40 locked generation tasks, greedy, 192 new tokens
   * - Secondary evaluation
     - 119 locked non-generation records, 128 new tokens (deviation)
   * - Locked source
     - ``psf-requests`` (Apache-2.0) plus three CC0 fixtures
   * - Human review
     - None; level 5 unavailable
   * - Hosted services
     - None; CPU only

Training integrity
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Run
     - Result
     - Evidence
   * - Pre-training
     - 300/300
     - Final train loss 4.0803; best validation 5.8321; peak RSS 700,182,528
       bytes; 75 tensors reloaded in a fresh process
   * - Fine-tuning
     - 120/120
     - Final train loss 3.6300; best validation 4.6752; peak RSS 907,993,088
       bytes; 75 tensors reloaded in a fresh process
   * - Checkpoints
     - Reloadable
     - ``runs/m7_pretrain/checkpoint_best.pt`` and
       ``runs/m7_sft/checkpoint_best.pt``
   * - Historical controls
     - Unchanged
     - M3 and M4 hashes matched the pre-registered values
   * - Dataset provenance
     - Verified
     - Run manifests match dataset manifest and artifact hashes

The pre-training run exhausted its fixed 300-step budget below its soft target
of final train loss 3.0. The run is complete, not truncated. The target was not
a hard gate.

Primary locked result
---------------------

.. list-table:: Code generation, 40 locked tasks
   :header-rows: 1
   :widths: 30 14 14 16 26

   * - Model
     - Params
     - Syntax
     - Functional
     - Common validation loss
   * - Untrained
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

All four models produced no parseable top-level function on any locked
generation task. The candidate's functional failures were all ``NO_CODE``. The
best baseline syntax rate was 0.0, so the protocol required at least 0.10; the
candidate reached 0.0.

The validation comparison uses the same 8 windows of 512 tokens from
``data/m7/pretrain.val.tokens.bin`` for every checkpoint. M3's recorded
validation loss came from a different subset and is not used here.

Secondary locked result
-----------------------

.. list-table:: Candidate only, 119 locked records
   :header-rows: 1
   :widths: 28 14 58

   * - Task
     - Result
     - Limitation
   * - Documentation
     - 0/54
     - no docstring-only response passed the format check
   * - Explanation
     - 46/48
     - format only; semantic correctness is not evaluated
   * - Completion
     - 0/14
     - 6 prompts were 741--1,197 tokens and exceeded the 512-token context
   * - SQL
     - 0/1
     - no ``SELECT`` statement
   * - Config output
     - 0/1
     - format check failed
   * - Format following
     - 0/1
     - instruction check failed

The 46/48 explanation result is **not** a capability claim. It scores shape, not
meaning, and many outputs are visibly repetitive. Secondary generation used 128
new tokens instead of the protocol's 192; this deviation limits comparability.

Gates
-----

.. list-table::
   :header-rows: 1
   :widths: 8 26 10 56

   * - Gate
     - Name
     - Result
     - Evidence
   * - A
     - Licence and provenance
     - PASS
     - 7 verified sources; 162 withheld; source provenance on records
   * - B
     - Dataset semantic quality
     - PASS
     - 507/58/42/159 partitions; oracles; encoding checks
   * - C
     - Split integrity and leakage
     - PASS
     - 0 shared ids/examples/responses/files; M5 hash unchanged
   * - D
     - Resource feasibility
     - PASS
     - FEASIBLE; 309.6 MB measured requirement; 40.8 min projected
   * - E
     - Training authorization
     - PASS
     - content-addressed token binds protocol, data and configs
   * - F
     - Training integrity
     - PASS
     - completed steps, manifests, fresh-process reloads and hashes
   * - G
     - Capability evidence
     - **FAIL**
     - level 0; 0/40 syntax; 0/8 functional; worse than controls
   * - H
     - Independent evaluation
     - PASS
     - protocol hash and checkpoint hashes recorded; M5 unchanged; no human
       review

The top-level gate report is ``ok: false`` because Gate G is the capability
threshold. Integrity is not capability.

Failure observations
--------------------

These are observations and hypotheses, not proven causes:

1. **Compute per parameter.** M7 has 3.3 times M3's parameters but received 300
   pre-training steps rather than 400. The larger model may be under-trained.
2. **Template-heavy supervision.** The model reproduced docstring fragments and
   ``:param`` patterns instead of emitting complete functions on demand.
3. **Out-of-distribution prompts.** Every model, including the untrained
   control, scored 0/40. The locked prompts ask for functions from held-out
   ``psf-requests`` modules; this is difficult at this scale.
4. **SFT traded language loss for format.** Fine-tuning lowered supervised loss
   from 4.3700 to 3.6300 but raised common validation loss from 5.8321 to
   5.9473.

None of these observations changes the measured outcome.

Supported and unsupported claims
--------------------------------

Supported:

* the dataset, licence gate, split isolation, oracles and provenance are
  implemented and verified for a bounded pilot;
* both training runs completed with reloadable checkpoints;
* the evaluation is protocol-bound, source-disjoint and reproducible;
* the candidate does not reach evidence level 2 or higher.

Not supported:

* useful code generation, functional coding, or general programming capability;
* production readiness, security readiness or release readiness;
* explanation or documentation semantic correctness;
* level 5 independent human evaluation;
* any comparison with external models.

Threats to validity and deviations
----------------------------------

* 507 training records and 40 locked generation tasks are small samples.
* 8 functional probes from one repository are pilot evidence, not broad coverage.
* Secondary generation used 128 new tokens, not 192.
* Six completion prompts did not fit the 512-token context and are generation
  errors, not fair model failures.
* One seed and CPU-only execution; no variance estimate is claimed.
* No human review occurred.
* M5 is unchanged and is not used to rescue M7 evidence.
* The soft pre-training loss target was missed.

Remaining work
--------------

The negative result does not justify tuning against the locked set. A future
experiment needs a new pre-registered protocol, more and better-structured
training data, a larger compute budget, and a fresh independent evaluation. The
current candidate remains an engineering artifact, not a product.

Artifacts
---------

* ``reports/M7_GATES.json`` and ``reports/M7_OUTCOME.json``;
* ``reports/M7_PREFLIGHT.json`` and ``reports/M7_AUTHORIZATION.json``;
* ``runs/m7_pretrain``, ``runs/m7_sft``, ``runs/m7_eval`` and
  ``runs/m7_eval_secondary``;
* ``reports/M7_CANDIDATE_REPORT.rst`` and ``reports/M7_BASELINE_REPORT.rst``.

Final status
------------

.. list-table::
   :header-rows: 1
   :widths: 8 60 32

   * - #
     - Item
     - Status
   * - 1
     - Repository and M2--M6 baseline inspection
     - COMPLETE
   * - 2
     - Readiness Gate A -- data
     - COMPLETE (bounded pilot)
   * - 3
     - Readiness Gate B -- resources
     - COMPLETE
   * - 4
     - Readiness Gate C -- experimental justification
     - COMPLETE
   * - 5
     - Dataset construction, provenance and split isolation
     - COMPLETE
   * - 6
     - Pre-training, fine-tuning and checkpoint integrity
     - COMPLETE
   * - 7
     - Locked evaluation and protocol gates
     - COMPLETE (negative capability result)
   * - 8
     - Five required M7 RST documents
     - COMPLETE
   * - 9
     - Forced Sphinx HTML build and page verification
     - COMPLETE
   * - 10
     - Negative-path, missing-artifact, failed-run, resource and M5 tests
     - COMPLETE
   * - 11
     - M6 serving compatibility for the M7 checkpoint
     - COMPLETE
   * - 12
     - Taksh rename, v1.0.0 release metadata and packaging
     - COMPLETE -- v1.0.0 Apache-2.0 source release prepared; publication tracked
       separately

The experiment was originally documented under the interim target ``v0.7.0``.
The authoritative M7 release target is ``v1.0.0``; neither label is a model
release. v1.0.0 is an Apache-2.0 source release, not a weight or dataset
release. Packaging and release evidence is in
:doc:`../releases/M7_RELEASE_READINESS`. No capability or production claim is
made.

See :doc:`M7_MANUAL_VERIFICATION` and :doc:`M7_REPRODUCIBILITY` for checks.
