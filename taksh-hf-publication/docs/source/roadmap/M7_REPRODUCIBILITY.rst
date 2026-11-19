M7 Reproducibility
==================

Release target: v1.0.0 (Apache-2.0 source release; publication tracked separately)

.. contents:: On this page
   :local:
   :depth: 2

Purpose
-------

This page records how the M7 experiment was produced, which inputs are bound by
hash, and what can and cannot be reproduced exactly. It is written for a reader
who wants to verify the negative result without silently retuning it.

Environment
-----------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Item
     - Measured value
   * - Python
     - 3.14.7
   * - torch
     - 2.13.0+cpu
   * - tokenizers
     - 0.23.1
   * - numpy
     - 2.5.2
   * - PyYAML
     - 6.0.3
   * - Sphinx
     - 7.4.7
   * - Platform
     - Windows-11-10.0.26300-SP0
   * - Device
     - CPU only; no CUDA; 8 logical CPUs, 4 torch threads
   * - RAM
     - 3.91 GiB total
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``
   * - Tags / release at baseline
     - 0 tags; v1.0.0 source release tracked separately

No new dependency was required for M7. The project used the existing
``requirements.txt`` stack.

Bound inputs
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Input
     - SHA-256 or identity
   * - ``configs/eval/m7_protocol.yaml``
     - ``1313462407d84b828dfb472f9142e1d542288efc500d78dd65a6437204aa6ee5``
   * - ``data/m7/m7-code-supervised-v0.manifest.json``
     - ``6ab67eddffa32adae1db71490d9e65f578f79e3397767b4bad49800504fc75da``
   * - ``configs/train/pretrain_m7.yaml``
     - ``df1c4bf011c7553eccb9c319ed9a0ac24ee955c761df0ffaf3f9259e1a6b4f30``
   * - ``configs/train/sft_m7.yaml``
     - ``0c0839fb3b5f9f29b162f0c6ebf8e1bd164bb2eebc2fea0267256fcd5e692007``
   * - ``tokenizer/qm_code_bpe_v0.json``
     - ``90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988``
   * - M5 benchmark
     - ``52ea2ed5db839b22fb6865f8fa62e52456f82be8151c45abc8f6da10731b8d87``
   * - M3 control
     - ``56365457e5ffb28061d4d98772e2d0457857e95df7db3ae4738321fd262cd078``
   * - M4 control
     - ``2bd86e2b9cd211bc6501f7719243102e69544ac6eb5755a2a06639a925968769``
   * - M7 pre-trained checkpoint
     - ``5d1f61a68a04ed8a99c2eb4624847f8f5babf19bc765db139997cd23e4311d82``
   * - M7 fine-tuned checkpoint
     - ``421c218d626524ffa8266a5f5c5ccb06fa743d3ccf8e75cc036b1047478b4dd9``
   * - Untrained control
     - ``8d8b1c3e2cb7524b190b94c0bdeb32c6bc7898c8011f9880f06cb3f710b1bfdc``

The authorization token
``cc43cf66b867a900c0d6db86465749e6c864761928b322682c0aaa7e71d05e95`` binds
the protocol, dataset manifest, training configs and pre-training gate results.
It is content-addressed, not a cryptographic signature.

Exact commands
--------------

Run from the repository root in PowerShell:

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

The pre-training command refuses to overwrite an existing run unless
``--resume`` is passed deliberately. Do not pass ``--resume`` to reproduce a
fresh run.

Suggested clean-room verification
---------------------------------

1. Verify the bound hashes in `Bound inputs`_.
2. Run ``python -m evaluation.m7_gates --phase pre`` and inspect gate evidence.
3. Do **not** train unless the authorization token is regenerated for your
   inputs. A changed protocol or dataset hash invalidates the old token.
4. If retraining, use a new output directory. Never overwrite the original
   ``runs/m7_*`` evidence.
5. Run the same evaluation commands and compare aggregate records, not a single
   cherry-picked sample.
6. Run the negative-path tests and the real-checkpoint serving test.

Expected measured results
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 54 46

   * - Item
     - Expected value
   * - Pre-training steps / tokens
     - 300 / 1,228,800
   * - Pre-training final train loss
     - 4.0803
   * - Pre-training best validation loss
     - 5.8321
   * - SFT steps / tokens
     - 120 / 491,520
   * - SFT final train loss
     - 3.6300
   * - SFT best validation loss
     - 4.6752
   * - Candidate common validation loss
     - 5.9473
   * - Locked syntax / functional
     - 0/40 / 0/8
   * - Evidence level
     - 0
   * - Outcome
     - ``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED``

Exact floating-point loss values can differ in the last decimal place across
torch builds or thread scheduling. The negative conclusion does not depend on
that difference: the candidate is substantially worse than both controls and
has zero functional successes.

Runtime observations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Stage
     - Observed runtime
   * - Pre-training
     - 1,633.7 s (27.2 minutes)
   * - Fine-tuning
     - 588.6 s (9.8 minutes)
   * - Untrained evaluation (40 tasks)
     - 1,659.5 s
   * - M3 evaluation (40 tasks)
     - 250.9 s
   * - M4 evaluation (40 tasks)
     - 250.7 s
   * - Candidate primary (40 tasks)
     - 535.2 s
   * - Candidate secondary (119 tasks)
     - 746.3 s
   * - Full test suite (candidate evaluation)
     - 892 tests in 203 s
   * - Full test suite (M7 documentation)
     - 903 tests in 308.6 s

Evaluation is CPU-only and has no KV cache; each generated token re-runs the
context window. Runtime is therefore sensitive to context length and hardware.
Do not treat these figures as a throughput benchmark.

Determinism and non-determinism
-------------------------------

Fixed by the protocol:

* training seed 42;
* evaluation seed 7 and greedy decoding;
* common-validation seed 1234 and 8 fixed windows;
* dataset split construction and source hashing;
* pre-registered thresholds and decoding limits.

Not guaranteed:

* bit-identical checkpoints across torch versions, CPU instruction sets or
  thread counts;
* bit-identical generated text after the first divergent token;
* exact runtimes under host memory pressure;
* identical loss decimals across environments.

Keep the original artifacts if comparing runs. A re-run that differs must be
reported as a re-run, not merged into the original evidence.

Serving verification
--------------------

The M7 fine-tuned checkpoint is now loadable through the same startup path used
by the OpenAI-compatible server. The compatibility check is:

.. code-block:: powershell

   python -m pytest tests/test_m7_serving.py -q

It verifies model id ``qm-code-m3-9m``, step 120, 9,359,104 parameters, vocab
5,822, context 512, `/v1/models`, and a bounded chat completion. It asserts the
contract, not output quality.

Manual tests
------------

.. code-block:: powershell

   python -m pytest tests/test_m7_gates.py tests/test_m7_evaluation.py -q
   python -m pytest tests/test_m7_serving.py -q
   python -m sphinx -W -b html docs/source docs/_build/html

On this Windows host, pytest's default temporary root may be inaccessible. Use
a writable base directory if needed:

.. code-block:: powershell

   python -m pytest tests/test_m7_gates.py -q --basetemp=C:\Temp\pytest-m7

Limits of reproduction
----------------------

* ``runs/`` and ``*.pt`` are gitignored. The checkpoints are local artifacts, not
  repository files. A clone without them cannot reproduce serving or evaluation
  without rebuilding the runs.
* The locked dataset and M5 benchmark are repository artifacts with recorded
  hashes. Do not edit them.
* No human review exists, so level 5 cannot be reproduced.
* The original protocol's soft loss target was missed; reproducing the protocol
  means reproducing that miss, not changing the target.

Release status
--------------

M7. The current release target is ``v1.0.0``; the experiment was originally
documented under the interim target ``v0.7.0``. Neither label is a model-weight
or dataset release. Packaging evidence is in
:doc:`../releases/M7_RELEASE_READINESS`.
