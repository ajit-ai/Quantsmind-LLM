M7 Sovereign Capability Experiment
==================================

Release target: v1.0.0 (Apache-2.0 source release; publication tracked separately)

.. contents:: On this page
   :local:
   :depth: 2

Summary
-------

M7 tests whether a small coding model trained from scratch on licence-verified
code can demonstrate measurable capability under this project's CPU and licence
boundary. It is a bounded experiment, not a product release.

**Verdict: training completed, capability not demonstrated.** The candidate
reached evidence level 0. It scored 0/40 on locked syntax-valid code generation
and 0/8 on locked functional probes. Its common-validation loss was 5.9473,
worse than both the M3 (4.9685) and M4 (4.9220) controls.

The engineering work is complete and verifiable. The capability claim fails and
is reported as a failure.

Document map
------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Document
     - Contents
   * - ``reports/M7_BASELINE_REPORT.rst``
     - Pre-documentation repository, evidence, serving and documentation state
   * - :doc:`M7_EXPERIMENT_PLAN`
     - Hypotheses, design, Gates A/B/C, pre-registered thresholds and commands
   * - :doc:`M7_EXPERIMENT_REPORT`
     - Measured result, gate outcomes, failures and limitations
   * - :doc:`M7_REPRODUCIBILITY`
     - Environment, hashes, commands, runtime and determinism limits
   * - :doc:`M7_MANUAL_VERIFICATION`
     - Step-by-step checks and recorded outcomes

Architecture
------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Component
     - M7 configuration
   * - Model
     - Decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU,
       tied embeddings
   * - Parameters
     - 9,359,104
   * - Vocabulary
     - 5,822, matching ``tokenizer/qm_code_bpe_v0.json``
   * - Context
     - 512 tokens
   * - Objective
     - Next-token prediction, followed by supervised fine-tuning
   * - Initialisation
     - Random, standard deviation 0.02; no pretrained weights
   * - Device
     - CPU only, torch 2.13.0+cpu

Experiment boundary
-------------------

M7 is sovereign:

* no hosted API, no external model and no pretrained checkpoint;
* training and evaluation ran locally on CPU;
* all training data passed an explicit licence gate;
* the locked evaluation source is ``psf-requests`` under Apache-2.0;
* the M5 benchmark was not modified and was not used as M7 evidence.

M7 is bounded:

* 507 supervised training records and 58,123 supervised tokens;
* 300 pre-training steps and 120 fine-tuning steps;
* 40 locked generation tasks and 8 functional probes;
* one seed, one host, no variance estimate;
* no human review, so evidence level 5 is unavailable.

Readiness gates
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Gate
     - Question
     - Result
   * - A -- data
     - Is the data licensed, labelled, isolated and fit for the test?
     - PASS for a bounded pilot; 7 sources verified, 162 records withheld, no
       split leakage
   * - B -- resources
     - Does the experiment fit the measured CPU, RAM, disk and time budget?
     - PASS; preflight ``FEASIBLE``, 309.6 MB measured requirement, 40.8 min
       projected pre-training
   * - C -- justification
     - Is this the smallest falsifiable test that can change the decision?
     - PASS; 9.36M parameters for 300 + 120 steps, with untrained and M3/M4
       controls

The protocol also has engineering gates A--H. Readiness Gate A maps to protocol
Gates A--C, readiness Gate B maps to protocol Gate D, and readiness Gate C has
no single protocol gate. The full mapping is in :doc:`M7_EXPERIMENT_PLAN`.

Outcome
-------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Result
   * - Training
     - 300/300 pre-training steps; 120/120 fine-tuning steps
   * - Checkpoints
     - Reloadable in a fresh process; 75 tensors each
   * - Locked syntax
     - 0/40 (0.0)
   * - Locked functional
     - 0/8 (0.0); 0/29 cases
   * - Common validation
     - 5.9473 vs untrained 8.7339, M3 4.9685, M4 4.9220
   * - Evidence level
     - 0
   * - Gate G
     - FAIL
   * - Gates A--F and H
     - PASS
   * - Release
     - Publication tracked separately; v1.0.0 source release documented in
       :doc:`../releases/M7_RELEASE_READINESS`

M6 serving and integration
--------------------------

The M6 published record is *Local Capability Validation*, not an M6 serving
acceptance. Serving code exists from earlier work. During M7 baseline inspection,
the fine-tuned checkpoint failed to load through
``serving.openai_server.load_state`` because the SFT checkpoint records its
architecture under ``config.run.base_model_config`` while serving read only
``config.run.model_config``.

The serving loader was repaired to resolve either field and to refuse an
unusable configuration instead of guessing. A real-checkpoint test now verifies
the M7 model through the OpenAI-compatible server. This proves API-shaped
compatibility, not model quality.

Historical naming
-----------------

The roadmap previously labelled M6 "Serving and Product Integration" and M7
"Release / Production Readiness". The published M6 record and the M7 request use
different scopes. Historical artifacts retain their original names. M7 now
closes the completed **Sovereign Capability Experiment** together with its
v1.0.0 packaging and public-release decision; packaging is complete as an
Apache-2.0 source release, with publication authorized and tracked separately.
There is no M8 milestone. The experiment was originally documented under the
interim target ``v0.7.0``; that label is historical and was never released.

Product naming
--------------

The product name is **Taksh**. The repository name, schema identifiers and
historical reports retain ``Quantsmind`` for provenance. This is a product-facing
rename only.

What this milestone does not claim
----------------------------------

* capable code generation or functional programming;
* production, security or release readiness;
* semantic correctness of explanations or documentation;
* comparison against any external model;
* independent human acceptance;
* that a negative result can be converted into a positive one by changing the
  thresholds after the fact.

Next steps
----------

Do not tune against the locked partition. A future experiment requires a fresh
protocol and evaluation, more and better-structured data, and a larger compute
budget. The existing result must remain reproducible as a negative result.
