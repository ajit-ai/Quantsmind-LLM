M7 Candidate Training and Evaluation Report
===========================================

Target version: v0.7.0 (historical experiment record -- never released)

Current release target: v1.0.0 (Apache-2.0 source release; publication tracked separately).
See ``docs/source/releases/M7_RELEASE_READINESS.rst``.

.. contents:: On this page
   :local:
   :depth: 2

Verdict
-------

**COMPLETE -- NEGATIVE RESULT, REPORTED AS MEASURED.**

The M7 engineering scope is implemented and tested. Pre-training gates A-E
passed, both training runs completed, post-training integrity gate F passed, and
the independent-evaluation gate H passed. The capability gate G did **not** pass.

The candidate achieved evidence level **0** under the pre-registered protocol:

* common-validation loss: **5.9473**, worse than M3 (4.9685) and M4 (4.9220);
* locked code generation: **0/40** syntax-valid, **0/8** functional tasks,
  **0/29** functional cases;
* the same 0/40 syntax rate as the untrained, M3 and M4 controls.

The machine-readable outcome is
``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED`` in
``reports/M7_OUTCOME.json``. This is a negative capability result, not a
training or data-integrity failure.

A. Baseline
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``; no commits made during M7
   * - M3 control
     - ``runs/m3_base/checkpoint_best.pt``, sha256
       ``56365457e5ffb280...``, 2,853,280 params
   * - M4 control
     - ``runs/m4_sft/checkpoint_best.pt``, sha256
       ``2bd86e2b9cd211bc...``, 2,853,280 params
   * - Untrained control
     - ``runs/m7_untrained/checkpoint.pt``, sha256
       ``8d8b1c3e2cb7524b...``, 9,359,104 params, seed 0, step 0
   * - Candidate architecture
     - ``configs/model/code_m3_9m.yaml``, 9,359,104 params, vocab 5,822
   * - Tokenizer
     - ``tokenizer/qm_code_bpe_v0.json``, sha256 ``90fd1c8b...``

The M3 and M4 checkpoints were opened read-only and their hashes were checked
before and after M7. Both are unchanged.

B. Data
-------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Dataset
     - ``data/m7/m7-code-supervised-v0``, transform
       ``qm.m7.source-derived/1``
   * - Train
     - **507** records: 200 generation, 100 documentation, 100 explanation,
       100 completion, 7 deterministic fixtures
   * - Validation
     - **58** aiohttp records: 15 generation, 18 documentation, 15 explanation,
       10 completion
   * - Development
     - **42** aiohttp records, not used for training or selection
   * - Locked
     - **159** records: 40 generation, 54 documentation, 48 explanation,
       14 completion, 1 SQL, 1 config, 1 format
   * - Locked source
     - ``psf-requests`` (156 records, Apache-2.0) plus 3 CC0 fixtures
   * - Functional oracles
     - **8** reference functions, **29** cases, expected values produced by
       executing the reference implementation in a subprocess
   * - Provenance
     - every repository-derived record carries source, licence, source sha1
       and source-file sha256
   * - Split isolation
     - 0 shared ids, examples, responses or source files; val/dev share the
       aiohttp source but use disjoint even/odd file buckets

The 8 oracles are ``dotted_netmask``, ``is_ipv4_address``,
``urldefragauth``, ``get_auth_from_url``, ``to_key_val_list``,
``from_key_val_list``, ``unquote_unreserved`` and ``iter_slices``.

**Scale disclosure.** This is a small, single-epoch experiment: 507
instruction records and 58,123 supervised tokens. It is not a production
instruction corpus, and the fixture tasks test deterministic formats rather
than repository-derived behaviour.

**M5 integrity.** ``evaluation/m5_benchmark.jsonl`` still has manifest hash
``52ea2ed5db839b22...``. There is no exact prompt or reference overlap with the
M7 training partition. M5 was not used as M7 evidence.

C. Training
-----------

Preflight
~~~~~~~~~

``python -m training.preflight --config configs/train/pretrain_m7.yaml
--probe-steps 3`` returned **FEASIBLE**:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Measured peak requirement
     - 309,600,256 bytes (probe, 3 real optimizer steps)
   * - Current-availability budget
     - 315,033,600 bytes; the probe fits, with no pressure warning
   * - Total-RAM budget
     - 2,939,846,246 bytes
   * - Projected pre-training
     - 40.8 minutes for 300 steps at 4,096 tokens/step
   * - Disk free
     - 190.9 GiB
   * - Device
     - CPU only; CUDA absent; 8 logical CPUs, 4 torch threads

Pre-training
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Command
     - ``python -m training.pretrain --config configs/train/pretrain_m7.yaml
       --out-dir runs/m7_pretrain``
   * - Initialisation
     - random, ``std=0.02``; no pretrained weights loaded
   * - Data
     - 193 records, 1,165,217 tokens; one pass is 1.05 epochs at 300 steps
   * - Steps
     - **300/300**, 1,228,800 tokens
   * - Final train loss
     - **4.0803** (uniform baseline 8.6694)
   * - Best validation loss
     - **5.8321** on the M7 validation stream
   * - Runtime
     - **1,633.7 s** (27.2 minutes), about 750 tokens/s
   * - Peak RSS
     - 700,182,528 bytes
   * - Checkpoint
     - ``runs/m7_pretrain/checkpoint_best.pt``, sha256
       ``5d1f61a68a04ed8a...``

The run manifest lists a soft target of final train loss < 3.0. The run finished
at 4.0803 because the pre-registered 300-step budget was exhausted. The target
was not enforced as a hard stop.

Supervised fine-tuning
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Command
     - ``python -m training.sft_train --config configs/train/sft_m7.yaml
       --out-dir runs/m7_sft``
   * - Base
     - M7 pre-trained checkpoint, step 300, 75 tensors
   * - Trainable parameters
     - 9,359,104 / 9,359,104 (full fine-tuning)
   * - Steps
     - **120/120**, 491,520 tokens
   * - Final train loss
     - **3.6300**
   * - Best validation loss
     - **4.6752**
   * - Runtime
     - **588.6 s** (9.8 minutes)
   * - Peak RSS
     - 907,993,088 bytes
   * - Checkpoint
     - ``runs/m7_sft/checkpoint_best.pt``, sha256
       ``421c218d626524ff...``

D. Evaluation protocol
----------------------

The locked partition was evaluated greedy (temperature 0), seed 7. The primary
task is ``code_generation`` at 192 new tokens for all four models. The candidate
was additionally evaluated on the other 119 locked records at 128 new tokens.

Scoring is task-aware. A generation only counts as syntax-valid if it parses and
contains a top-level function with a real body. Functional scoring executes the
8 oracles in a subprocess and compares exact outcomes. This distinction matters:
before the fix, 192 tokens of repeated prose could be counted as syntax-valid.

E. Results
----------

Primary locked generation (40 tasks, 8 functional oracles, 29 cases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 14 14 16 18

   * - Model
     - Params
     - Syntax
     - Functional
     - Cases
     - Common val loss
   * - Untrained
     - 9.36M
     - 0/40 (0.0)
     - 0/8 (0.0)
     - 0/29
     - 8.7339
   * - M3 base
     - 2.85M
     - 0/40 (0.0)
     - 0/8 (0.0)
     - 0/29
     - 4.9685
   * - M4 tuned
     - 2.85M
     - 0/40 (0.0)
     - 0/8 (0.0)
     - 0/29
     - 4.9220
   * - **M7 candidate**
     - **9.36M**
     - **0/40 (0.0)**
     - **0/8 (0.0)**
     - **0/29**
     - **5.9473**

All four models produced no parseable function definition on any of the 40
locked generation tasks. The candidate's functional failures are all
``NO_CODE``. The best baseline syntax rate is 0.0, so the protocol requires the
candidate to reach at least 0.10; it reached 0.0.

Common validation
~~~~~~~~~~~~~~~~~

The comparison uses the same M7 validation windows for every checkpoint:
``data/m7/pretrain.val.tokens.bin``, 8 fixed windows of 512 tokens, seed 1234.
M3's recorded validation loss was measured on a different aiohttp subset and is
**not** used for this comparison.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Checkpoint
     - Val loss
     - Interpretation
   * - Untrained
     - 8.7339
     - no learning
   * - M3 base (2.85M)
     - 4.9685
     - best control
   * - M4 tuned (2.85M)
     - 4.9220
     - best control
   * - M7 candidate (9.36M)
     - 5.9473
     - worse than both controls

The M7 pre-training checkpoint alone scored 5.8321 on the same stream; SFT moved
it to 5.9473. Both are worse than the 2.85M controls.

Candidate secondary tasks (119 locked records)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Task
     - Result
     - Note
   * - Documentation
     - 0/54
     - no docstring-only response passed the format check
   * - Explanation
     - 46/48
     - format only; semantic correctness is **not evaluable**
   * - Completion
     - 0/14
     - 6 prompts exceeded the 512-token context and raised
       ``GenerationError``
   * - SQL
     - 0/1
     - no ``SELECT`` statement
   * - Config output
     - 0/1
     - YAML/table format check failed
   * - Format following
     - 0/1
     - instruction check failed

The 46/48 explanation result is not a capability claim. The check only asks
whether the output has explanation shape; many outputs are visibly repetitive
and no human or semantic evaluation was performed.

Observed failure mode
~~~~~~~~~~~~~~~~~~~~~

The candidate learned local corpus style rather than function structure. A
representative locked output is a sequence of ``:param`` lines, docstring
fragments and repeated ``return None`` statements. It is not an accidental
truncation: the outputs are 400-500 characters on average, and all 40 fail the
top-level-function check. The SFT loss fell from 4.3700 to 3.6300, so the model
was optimising the training objective; it simply did not learn to emit a
function on the locked prompts. SFT also moved the common pre-training loss from
5.8321 to 5.9473: it fitted the instruction format while becoming slightly worse
at the underlying code language objective.

F. Gates
--------

.. list-table::
   :header-rows: 1
   :widths: 10 28 10 52

   * - Gate
     - Name
     - Result
     - Evidence
   * - A
     - Licence and provenance
     - PASS
     - 7 verified sources; every repository record has a corpus source sha1
   * - B
     - Dataset semantic quality
     - PASS
     - 507/58/42/159 encoded, 0 rejected; truncation and 6 completion-context
       overflows are disclosed below
   * - C
     - Split integrity and leakage
     - PASS
     - 0 shared ids/examples/responses/files; M5 hash unchanged; 0 M5 overlap
   * - D
     - Resource feasibility
     - PASS
     - FEASIBLE; 40.8 min projected; 309.6 MB measured requirement
   * - E
     - Training authorization
     - PASS
     - token ``cc43cf66b867a900...`` binds dataset, protocol, configs and gates
   * - F
     - Training integrity
     - PASS
     - 300/300 and 120/120 steps; fresh-process reload of 75 tensors each;
       M3/M4 hashes unchanged
   * - G
     - Capability evidence
     - **FAIL**
     - level 0; syntax 0/40; functional 0/8; common val worse than controls
   * - H
     - Independent evaluation
     - PASS
     - protocol hash matches; M5 unchanged; checkpoint hashes recorded; no human
       review

``reports/M7_GATES.json`` has top-level ``ok: false`` because gate G is the
capability threshold. Gates A-F and H passed. The negative outcome is recorded,
not hidden or converted into a positive claim.

G. Why the result is negative
-----------------------------

These are observations and hypotheses, not proven causes:

1. **Compute per parameter.** M7 has 3.3x M3's parameters but received 300
   steps, versus M3's 400, on the same 1.17M-token stream. The larger model may
   be substantially under-trained at this budget.
2. **Template-heavy supervision.** 100 completion, 100 documentation and 100
   explanation examples were derived from repository code. The model reproduced
   docstring fragments and ``:param`` patterns, but did not learn to emit a
   complete function on demand.
3. **The task is harder than the controls.** All four models scored 0/40 on
   locked generation, including the untrained control. The locked prompts ask
   for functions from held-out ``psf-requests`` modules; this is a difficult
   out-of-distribution test at this scale.
4. **M4 also failed.** M4 was instruction-tuned but also scored 0/40. The
   negative result is consistent with a broader data/scale limitation rather
   than a unique M7 defect.

None of these explanations changes the measured outcome. They are recorded so a
future run can test them with a larger step budget or different supervision.

H. Limitations and deviations
-----------------------------

* Baselines were evaluated on the primary locked generation task (40 records),
  not on the 119 secondary tasks. The candidate was evaluated on all 159.
* Six of 14 locked completion prompts are 741-1,197 tokens, exceeding the
  512-token model context. They raised ``GenerationError`` and are counted as
  generation errors, not as model failures with a fair chance.
* Candidate secondary tasks used 128 new tokens instead of the protocol's 192.
  The primary generation task used 192 for every model.
* Explanation and documentation semantic correctness is **not evaluated**.
  Only shape/format checks are reported.
* The 7 training fixtures and 3 locked fixtures are deterministic format tasks,
  not repository-derived code. They are disclosed in the dataset manifest.
* One seed, CPU-only. No variance estimate or confidence interval is claimed.
* The pre-training run's soft final-loss target of 3.0 was not met (4.0803);
  the run completed by the fixed step budget.
* No independent human review was performed, so evidence level 5 is not
  available.
* M5 remains a separate benchmark with an unchanged hash; it is not used as M7
  evidence and cannot rescue a failed locked result.

I. Reproduction
---------------

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

The test suite passed in full at candidate evaluation time: **892 passed** in
203 s. After the M7 documentation, gate and serving tests were added, the full
suite reached **903 passed** in 308.6 s. The M7-specific tests
are in ``tests/test_m7_evaluation.py``.

J. Artifacts
------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Artifact
     - Notes
   * - ``runs/m7_pretrain/checkpoint_best.pt``
     - sha256 ``5d1f61a68a04ed8a99c2eb4624847f8f5babf19bc765db139997cd23e4311d82``
   * - ``runs/m7_sft/checkpoint_best.pt``
     - sha256 ``421c218d626524ffa8266a5f5c5ccb06fa743d3ccf8e75cc036b1047478b4dd9``
   * - ``runs/m7_eval/summary.json``
     - merged primary and secondary evaluation
   * - ``runs/m7_eval/<model>/records.jsonl``
     - raw primary records, never overwritten
   * - ``runs/m7_eval/<model>/records_combined.jsonl``
     - merged records for inspection
   * - ``reports/M7_GATES.json``
     - A-H gate evidence
   * - ``reports/M7_OUTCOME.json``
     - machine-readable outcome and level
   * - ``reports/M7_AUTHORIZATION.json``
     - content-addressed pre-training authorization
   * - ``reports/M7_PREFLIGHT.json``
     - measured resource probe
   * - ``reports/m7_pretrain.log``, ``reports/m7_sft.log``
     - complete training logs
   * - ``reports/m7_eval_*.log``
     - complete evaluation logs
