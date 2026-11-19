M7 Sovereign Capability Experiment -- Plan
==========================================

Release target: v1.0.0 (Apache-2.0 source release; publication tracked separately)

.. contents:: On this page
   :local:
   :depth: 2

Status and sequencing
---------------------

This document formalises the M7 experiment plan. The evaluation thresholds were
fixed in ``configs/eval/m7_protocol.yaml`` **before** training; this narrative
plan records both that pre-registration and the execution that followed.

The M7 implementation, training and evaluation were already present as
uncommitted work when this formal document set was requested. The baseline
inspection records that fact rather than pretending the work happened after a
written plan. The experiment is reported as run, including the negative result.

Authority
---------

``docs/ROADMAP.md`` remains the single authoritative execution roadmap. This
document is the detailed M7 record and does not create a competing roadmap.

The M7 request defines three readiness gates:

* **Gate A -- data readiness**;
* **Gate B -- resource readiness**;
* **Gate C -- experimental justification**.

The existing M7 protocol also defines engineering gates A--H. They are related
but not the same. The mapping is explicit in `Gate naming and mapping`_.

Scope
-----

M7 asks one bounded question:

    Can a 9.36M-parameter coding model, trained from random initialisation on
    licence-verified code under this project's CPU and licence constraints,
    demonstrate measurable coding capability against an untrained control and
    the existing M3/M4 checkpoints?

The experiment is sovereign: no hosted API, no external model, and no pretrained
weights. It is a capability experiment, not a release, production-readiness
claim, product acceptance, or proof of commercial usefulness.

Baseline
--------

The pre-documentation state is recorded in
``reports/M7_BASELINE_REPORT.rst``. Key facts:

* branch ``develop``, HEAD ``4a5a144``, no tags, no release;
* 177 dirty working-tree entries at baseline, preserved;
* M2 is ``BLOCKED``; M3/M4 are implemented but do not produce parseable code;
  M5 and M6 are ``INSUFFICIENT_EVIDENCE``;
* M6's published record is *Local Capability Validation*, despite the roadmap
  label "Serving and Product Integration";
* M7's fine-tuned checkpoint did not load through the serving path at baseline;
  that is an integration defect to repair, not a capability result.

Hypotheses and falsification
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 50 40

   * - ID
     - Hypothesis
     - Falsified when
   * - H1
     - Pre-training on the M7 token stream lowers held-out loss below the
       untrained, M3 and M4 controls on the same validation windows.
     - The candidate's common-validation loss is not below every usable control.
   * - H2
     - SFT produces syntax-valid Python on at least 50% of locked generation
       tasks and beats the best baseline by at least 0.10 absolute.
     - Syntax rate is below 0.50, fewer than 20 tasks are attempted, or the gain
       is below 0.10.
   * - H3
     - At least 50% of the 8 locked functional probes pass, with at least 8
       attempted, and the rate beats the best baseline by at least 0.10.
     - The functional threshold or gain condition is missed.
   * - H4
     - At least two task categories reach 50% on at least three attempted
       records.
     - Fewer than two categories qualify.
   * - H5
     - Independent human review supports a level-5 claim.
     - No human review artifact exists. M7 does not claim this level.

The experiment is designed to make a negative result publishable. A failure to
reach a threshold is an outcome, not a reason to change the threshold.

Experimental design
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Design
   * - Candidate
     - ``configs/model/code_m3_9m.yaml``: decoder-only transformer, 9,359,104
       parameters, vocab 5,822, context 512, random initialisation
   * - Pre-training
     - 300 steps, micro-batch 1, gradient accumulation 8, 4,096 tokens/step,
       1,228,800 tokens, seed 42
   * - Fine-tuning
     - 120 steps, micro-batch 4, gradient accumulation 2, 491,520 tokens, full
       fine-tuning, seed 42
   * - Training data
     - 507 supervised records; 58,123 supervised tokens; 1,165,217 pretraining
       tokens after licence gating
   * - Controls
     - Same-architecture untrained step-0 model; M3 base; M4 tuned
   * - Primary evaluation
     - Locked ``code_generation``: 40 tasks, 8 functional oracles, 29 cases,
       greedy decoding, 192 new tokens
   * - Secondary evaluation
     - 119 locked non-generation records, 128 new tokens (a recorded deviation
       from the protocol's 192)
   * - Independence
     - Locked source is ``psf-requests`` (Apache-2.0) plus three CC0 fixtures;
       no shared ids, examples, responses or files with training
   * - Human review
     - Not performed; level 5 unavailable
   * - Hosted services
     - None; CPU only

Readiness Gate A -- data readiness
----------------------------------

.. list-table:: Gate A evidence
   :header-rows: 1
   :widths: 34 66

   * - Check
     - Result
   * - Partition sizes
     - Train 507; validation 58; development 42; locked 159
   * - Task coverage
     - Generation, documentation, explanation, completion, JSON, config, SQL,
       format following and constrained output
   * - Licence gate
     - 7 verified sources; 162 records withheld from 2 unverified sources;
       unreviewed records are not allowed
   * - Provenance
     - source, licence, source sha1 and source-file sha256 on every
       repository-derived record
   * - Split isolation
     - 0 shared ids, examples, responses or source files; train and locked do
       not share a source
   * - Functional oracles
     - 8 reference functions and 29 cases; expectations produced by subprocess
       execution, not hand-written guesses
   * - Encoding
     - 0 rejected records in the gate run; 6 completion prompts later exceeded
       the 512-token generation context and are disclosed
   * - M5 preservation
     - ``evaluation/m5_benchmark.jsonl`` hash unchanged at
       ``52ea2ed5db839b22...``; no exact M5 prompt/reference overlap with M7
       training

**Verdict: PASS for the bounded pilot.** The data supports the experiment but
is not a production corpus. The locked partition is source-disjoint, but the
sample is small and the 8 functional probes are pilot evidence.

Readiness Gate B -- resource readiness
--------------------------------------

.. list-table:: Gate B evidence
   :header-rows: 1
   :widths: 40 60

   * - Check
     - Result
   * - Preflight verdict
     - ``FEASIBLE``
   * - Measured peak requirement
     - 309,600,256 bytes over 3 real optimizer steps
   * - Current-availability budget
     - 315,033,600 bytes; no pressure warning
   * - Total-RAM budget
     - 2,939,846,246 bytes
   * - Projected pre-training
     - 40.8 minutes for 300 steps
   * - Actual pre-training
     - 27.2 minutes; peak RSS 700,182,528 bytes
   * - Actual fine-tuning
     - 9.8 minutes; peak RSS 907,993,088 bytes
   * - Disk
     - 190.9 GiB free at preflight
   * - Device
     - CPU only; CUDA absent; 8 logical CPUs, 4 torch threads

**Verdict: PASS.** The run fitted the machine with the measured safety margin.
Meeting the resource envelope says nothing about model quality.

Readiness Gate C -- experimental justification
----------------------------------------------

The experiment was the smallest test that could distinguish "more parameters and
more data" from "still no capability". The alternatives were:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Alternative
     - Why not chosen
     - Consequence
   * - Continue M3/M4 at 2.85M parameters
     - Already measured at 0/20 syntax; repeating it cannot test scale.
     - No new information.
   * - Train a 16.3M or larger model for 10,000 steps
     - Far beyond the CPU budget and would not fit the measured RAM envelope.
     - Blocked by resources, not by science.
   * - Use a hosted or downloaded pretrained model
     - Violates the sovereign boundary and cannot answer the from-scratch
       question.
     - Outside the licence and research scope.
   * - Tune against the locked set
     - Destroys the only independent evidence and turns measurement into
       rationalisation.
     - Prohibited.
   * - M7 9.36M, 300 + 120 steps
     - Fits the measured CPU and RAM budget; has an untrained control and
       source-disjoint locked evaluation.
     - Chosen. Bounded, falsifiable and affordable.

The experiment had a clear negative outcome available. It did not require a
positive result to justify running. The cost was approximately 37 minutes of
training plus evaluation on the local CPU, with no paid compute or hosted API.

**Verdict: PASS.** The experiment was justified, bounded and falsifiable.

Gate naming and mapping
-----------------------

Do not conflate the readiness gates with the protocol gates.

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Requested gate
     - Protocol gate
     - Meaning
   * - Gate A -- data readiness
     - Protocol Gate A + B + C
     - Licence/provenance, dataset semantic quality, split isolation and leakage
   * - Gate B -- resource readiness
     - Protocol Gate D
     - Measured CPU/RAM/disk preflight
   * - Gate C -- experimental justification
     - No single protocol gate
     - Narrative justification, alternatives, cost and falsifiability
   * - Authorization
     - Protocol Gate E
     - Content-addressed training authorization after A--D pass
   * - Post-training integrity
     - Protocol Gate F
     - Run manifests, completed steps, checkpoint reload and artifact hashes
   * - Capability evidence
     - Protocol Gate G
     - Pre-registered thresholds and evidence levels
   * - Independence
     - Protocol Gate H
     - Protocol hash, checkpoint hashes and M5 integrity

Pre-registered evidence levels
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Level
     - Claim allowed
   * - 0
     - Training executed; capability not demonstrated
   * - 1
     - Learning demonstrated on common validation against all controls
   * - 2
     - Syntax demonstrated on the locked partition with the required gain
   * - 3
     - Functional coding demonstrated on locked reference-execution probes
   * - 4
     - Multiple categories demonstrated
   * - 5
     - Independent locked evaluation with human review

No composite score is defined. Blending syntax, functional, documentation and
explanation rates would hide which evidence exists and which does not.

Method
------

1. Build the licence-gated dataset and pretraining token artifacts.
2. Run preflight, protocol Gates A--D, and write the authorization token.
3. Pre-train the candidate from random initialisation.
4. Fine-tune the candidate on the supervised dataset.
5. Evaluate all four models on the same locked generation tasks.
6. Evaluate the candidate on secondary locked tasks.
7. Merge records, run post-training Gates F--H, and write the outcome.
8. Verify the serving path with the fine-tuned checkpoint.
9. Publish the result, including failures and deviations.

Commands
--------

.. code-block:: powershell

   python -m data.build_m7_dataset
   python -m data.prepare_m7_pretrain
   python -m evaluation.m7_gates --phase pre

   python -m training.pretrain --config configs/train/pretrain_m7.yaml --out-dir runs/m7_pretrain
   python -m training.sft_train --config configs/train/sft_m7.yaml --out-dir runs/m7_sft

   python -m evaluation.m7_evaluate --models untrained,m3_base,m4_tuned,m7_candidate `
     --tasks code_generation --out-dir runs/m7_eval
   python -m evaluation.m7_evaluate --models m7_candidate `
     --tasks documentation,code_explanation,code_completion,sql,config_output,format_following,json_output,constrained_output `
     --max-new-tokens 128 --out-dir runs/m7_eval_secondary
   python -m evaluation.m7_combine
   python -m evaluation.m7_gates --phase post

Do not re-run the evaluation to make a failed result look different. Re-running
is allowed only to verify reproducibility, and the new output must be kept
separate from the original run.

Stop conditions
---------------

The experiment stops and reports failure rather than continuing when:

* a hard preflight gate fails;
* RAM, disk or projected runtime exceeds the documented budget;
* a training run is interrupted, fails or does not complete its planned steps;
* a checkpoint cannot be reloaded in a fresh process;
* the protocol, dataset manifest, configs or M5 benchmark hash changes;
* locked data or the M5 benchmark is used for tuning;
* serving cannot load the candidate checkpoint;
* the evidence does not reach the threshold.

A negative capability result does **not** stop integrity reporting. Gates F and
H must still pass or fail on their own evidence.

Artifacts and provenance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Artifact
     - Purpose
   * - ``data/m7/``
     - Dataset partitions, manifest, token artifacts and oracles
   * - ``configs/eval/m7_protocol.yaml``
     - Pre-registered thresholds and limitations
   * - ``reports/M7_PREFLIGHT.json``
     - Measured resource probe
   * - ``reports/M7_AUTHORIZATION.json``
     - Content-addressed authorization token
   * - ``runs/m7_pretrain``, ``runs/m7_sft``
     - Run manifests, logs and checkpoints
   * - ``runs/m7_eval``, ``runs/m7_eval_secondary``
     - Raw evaluation records and summaries
   * - ``reports/M7_GATES.json``, ``M7_OUTCOME.json``
     - Gate evidence and machine-readable outcome
   * - ``reports/M7_CANDIDATE_REPORT.rst``
     - Detailed measured result
   * - ``reports/M7_BASELINE_REPORT.rst``
     - Pre-documentation state and conflicts

What would change the conclusion
--------------------------------

Only a new, separately pre-registered experiment with more data, more compute or
a different supervision strategy could change the capability conclusion. The
current locked set must not be reused for iterative tuning. A future experiment
must keep:

* a fresh locked partition or an explicitly new evaluation;
* the same negative controls;
* a pre-registered threshold before training;
* a disclosed training budget and data scale;
* an honest separation between format compliance and semantic correctness.

Exit criteria for M7
--------------------

M7 is complete as an experiment when:

1. the dataset, licence and split gates have evidence;
2. resource feasibility is measured and recorded;
3. the experiment is justified and bounded;
4. pre-training and fine-tuning complete with reloadable checkpoints;
5. all models are evaluated under the same protocol;
6. the result is reported at its actual evidence level;
7. the fine-tuned checkpoint is compatible with the documented serving path;
8. reproducibility and manual verification instructions exist;
9. the documentation builds as HTML;
10. no release or production-readiness claim is made.
