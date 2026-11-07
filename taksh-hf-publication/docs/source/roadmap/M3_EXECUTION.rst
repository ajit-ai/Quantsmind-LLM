M3 Base Model Training — Execution Guide
=========================================

.. contents:: On this page
   :local:
   :depth: 2

Scope
-----

M3 establishes that QuantsMind-LLM can **initialise a model from random
weights, perform real gradient updates, save a checkpoint, reload it in a fresh
process, and generate text locally using only its own trained weights**.

It does not claim to produce a useful coding model. The measured run below is a
2.85M-parameter model trained for 400 steps on 1.17M tokens of licence-verified
Python. That is far below what competent code generation requires, and the
generated output does not parse as Python. See `Known limitations`_.

What M3 is not: instruction tuning, a benchmark suite, a production API, or a
packaged release.

Dependencies and environment
----------------------------

Everything uses the existing stack; no new dependency was added.

.. list-table:: Measured environment
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Measured value
   * - Python
     - 3.14.7
   * - torch
     - 2.13.0+cpu
   * - Device
     - CPU only; no CUDA device present
   * - RAM
     - 3.91 GiB total; available memory fluctuated 31 MiB - 337 MiB during
       the milestone because other programs were running
   * - torch threads
     - 4 (8 logical CPUs)
   * - Disk free
     - 191.4 GiB

Install (unchanged from earlier milestones)::

    pip install -r requirements.txt

Architecture and strategy
-------------------------

**Strategy A - from scratch.** This is what M3 implemented and ran. Random
initialisation (``torch.nn.init.normal_``, std 0.02), next-token prediction, no
pretrained weights anywhere in the pipeline.

The architecture already existed in ``training/model.py`` (Llama family:
RMSNorm, RoPE, grouped-query attention, SwiGLU) and was reused and extended
rather than replaced.

**Strategy B - continue pretraining / fine-tune an existing model.** Assessed,
not executed. See `Strategy B assessment`_.

Model configuration
-------------------

``configs/model/code_m3_9m.yaml`` is the architecture M3 wants:
**9,359,104 parameters**, d_model 256, 8 layers, 8 heads / 4 KV heads,
ffn 1024, context 512, vocab 5822.

``configs/model/code_m3_3m.yaml`` is the host-sized fallback: **2,853,280
parameters**, d_model 160, 5 layers, 4 heads / 2 KV heads, ffn 640. **This is
the model the demonstration run used.**

Both declare ``vocab_size: 5822``, matching ``tokenizer/qm_code_bpe_v0.json``
exactly.

.. warning::

   ``configs/model/code_30m.yaml`` declares ``vocab_size: 32768``, which matches
   **neither** tokenizer in this repository (``qm_bpe_v0`` is 49152,
   ``qm_code_bpe_v0`` is 5822). It cannot train on artifacts from either without
   an override, and its 32768-wide vocabulary accounts for exactly the
   difference between its 16,257,280 parameters and the 9,359,104 of
   ``code_m3_9m.yaml``. It is documented as an orphan in that file and is not
   used by M3. ``tests/test_m3_dataset.py`` asserts no M3 config resolves to it.

Data authorisation and prerequisites
------------------------------------

Training data was **not** taken wholesale from the M2 corpus. Measured
composition of ``data/code_corpus_v2.jsonl``:

===============  =======  ==========  ================================
split            records  verified    unresolved licence
===============  =======  ==========  ================================
train            193      193         0
val              198      63          135 (``pypa-setuptools``)
test (holdout)   46       19          27 (``psf-black``)
===============  =======  ==========  ================================

162 of 437 records (37%) come from two sources whose licences need human review.
The M2 readiness gate reports this as **BLOCKED**.

``data.prepare_m3_dataset`` therefore withholds every record whose source is not
in ``licence_verification.json:verified_ids``. Training data is licensed only by
that verification report - never by a record's own ``spdx`` field, because a
record claiming MIT is not evidence that MIT was reviewed and accepted.

.. list-table:: M3 dataset
   :header-rows: 1
   :widths: 16 14 16 22 32

   * - Split
     - Records
     - Tokens
     - Repositories
     - Note
   * - train
     - 193
     - 1,165,217
     - 5
     - licence-verified sources only
   * - val
     - 63
     - 340,077
     - 1
     - licence-verified, disjoint from train
   * - holdout
     - not emitted
     -
     -
     - unresolved licence, and deliberately unspent

Train and val share no repository and no content hash. Validation tokens live in
a separate artifact, so validation data cannot reach an optimizer step.

Build the data::

    python -m data.prepare_m3_dataset \
        --corpus data/code_corpus_v2.jsonl \
        --licence data/licence_verification.json \
        --tokenizer tokenizer/qm_code_bpe_v0.json \
        --out-prefix data/m3_code_verified

Preflight
---------

Always run this before training. It separates three verdicts, and its hard gates
cannot be cleared by ``--override``:

``FEASIBLE``
    fits with the safety margin
``FEASIBLE_WITH_SMALLER_CONFIG``
    does not fit, but a named reduction would
``BLOCKED``
    licence, data-integrity or tokenizer gate failed

Memory is **measured**, not modelled: the probe runs real optimizer steps and
reads peak working set (via ``GetProcessMemoryInfo`` on Windows). Throughput is
measured the same way::

    python -m training.preflight --config configs/train/pretrain_m3.yaml --probe-steps 3
    python -m training.preflight --config configs/train/pretrain_m3_small.yaml --probe-steps 3

Training
--------

::

    python -m training.pretrain \
        --config configs/train/pretrain_m3_small.yaml \
        --out-dir runs/m3_base

Useful flags::

    --max-steps N      override the configured step budget
    --tokens PATH      override the training token artifact
    --resume           continue from <out-dir>/checkpoint_last.pt

Interrupt a run with Ctrl-C: the loop finishes the current step, saves, and
records the interruption. The manifest then reports ``status: interrupted`` and
``success: false``. An interrupted run is never marked as a successful artifact.

Checkpoint and resume
---------------------

Checkpoints are written to ``<out-dir>/checkpoint_last.pt`` and, whenever
validation improves, ``<out-dir>/checkpoint_best.pt``. Each stores model
weights, optimizer state, step, tokens processed, RNG state (torch/numpy/python),
dataset identity, tokenizer identity and the resolved config.

::

    python -m training.pretrain --config <config> --out-dir runs/m3_base --resume --max-steps 600

Resume restores RNG state, so a resumed run continues the same data stream an
uninterrupted run would have seen. Where state cannot be restored the run says
so and is explicitly labelled best-effort rather than bitwise identical.

Corrupt, truncated, schema-wrong, optimizer-less or architecturally mismatched
checkpoints are rejected with a named reason.

Local generation
----------------

::

    python -m training.generate \
        --ckpt runs/m3_base/checkpoint_last.pt \
        --tokenizer tokenizer/qm_code_bpe_v0.json \
        --prompt "def fibonacci(n):" \
        --max-new-tokens 60 --greedy \
        --record reports/m3_generation_fib.json

Options: ``--temperature``, ``--top-k``, ``--top-p``, ``--greedy``, ``--seed``,
``--stop``, ``--device``. Output is the model's own, unedited, and every run
records the checkpoint and tokenizer it used.

The tool refuses to run on a missing, corrupt or schema-wrong checkpoint, on a
tokenizer whose vocabulary disagrees with the model, and on a prompt longer than
the context window. It never falls back to random weights.

Artifact layout
---------------

.. code-block:: text

    runs/m3_base/
        checkpoint_last.pt        final weights + optimizer + RNG + identities
        checkpoint_best.pt        best validation loss
        loss_log.jsonl            append-only per-step records
        run_manifest.json         machine-readable run record

``run_manifest.json`` records architecture, parameter count, tokenizer and
dataset identities with hashes, full config, seed, library versions, hardware,
measured memory, steps, tokens, metrics, checkpoint hash, timings, and any
warnings, deviations or interruptions.

Testing
-------

::

    python -m pytest tests -q -p no:cacheprovider --basetemp=%TEMP%\pt

M3 tests: ``tests/test_m3_dataset.py`` (licence gate, split isolation, config
vocabulary), ``tests/test_m3_training_core.py`` (validation purity, perplexity
honesty, checkpoint rejection, RNG, manifest, generation), and
``tests/test_m3_end_to_end.py`` (train, checkpoint, **fresh-process** reload,
generate, and interrupt/resume). The end-to-end tests are marked ``slow``.

Known limitations
-----------------

* **The generated code is not usable.** Measured: completions from both prompts
  fail ``ast.parse`` with ``SyntaxError: unmatched ')'``. Output resembles the
  *documentation* style of the corpus (docstrings, type annotations) far more
  than executable logic.
* **Scale.** 2.85M parameters, 1.17M tokens, 1.4 epochs - far below what code
  models need for competence. The corpus itself is 2.24% of M2's 100M-token
  target.
* **The corpus is documentation-heavy.** The verified sources are requests,
  flask, urllib3, click, pytest, fastapi and aiohttp, whose files are largely
  docstrings and comments rather than dense executable statements.
* **CPU only.** About 45 minutes for 400 steps. No GPU path was exercised.
* **No KV cache.** Every generated token re-runs a full forward pass over the
  window (pre-existing, ``OPEN_ISSUES`` N6).
* **Holdout untouched.** The M2 holdout was not used for training, validation or
  any threshold decision, so it cannot corroborate the validation numbers.
* **Strategy B not executed.** No pretrained weights were trained on.

Strategy B assessment
---------------------

Continuing pretraining an existing code model is the more practical route to
useful coding capability. This milestone did **not** substitute pretrained
weights for Strategy A.

What it would require, and why it was not executed:

* **Licence review per candidate.** Each candidate model needs its licence,
  permitted use, and redistribution terms reviewed by a human. This is the same
  class of decision that currently blocks ``psf-black`` and
  ``pypa-setuptools`` in the corpus, so it cannot be self-certified in code.
* **Tokenizer compatibility.** The candidate's vocabulary must be adopted
  verbatim. Resizing it is not an option: M3 found exactly this trap with
  ``code_30m.yaml``, whose 32768-wide vocabulary matches no tokenizer shipped
  here.
* **Hardware.** Continued pretraining is compute-bound. This host is CPU-only
  with 3.91 GiB RAM and measured 45 minutes for 400 steps at 2.85M parameters;
  a useful code model is orders of magnitude beyond this.
* **Data requirements.** Meaningful continued pretraining needs far more
  licence-cleared code than the 1.17M verified tokens available here.
* **Trade-off.** Fine-tuning a strong open code model would reach useful output
  far sooner than growing QuantsMind's own from scratch. That is a strategic
  choice about what QuantsMind-LLM *is*: an independently trained model, or a
  product built on existing weights. It is a human decision, not a technical
  one.

Recommendation
--------------

Strategy A remains the correct long-term route for a sovereign, independently
trained coding model, and M3's pipeline is sufficient to run it. The binding
constraints are **data volume and compute**, not code: the licence-cleared
corpus is 1.17M tokens against a 100M target, and this host is CPU-only.

The highest-value next steps, in order:

1. Resolve the two outstanding licence decisions and authorise additional
   sources, so the corpus can grow.
2. Re-run preflight on GPU-class hardware and train ``code_m3_9m.yaml``.
3. Only then evaluate whether the from-scratch route is competitive.

Strategy B remains available as a parallel, clearly-labelled route to useful
output, and should be evaluated on its own merits rather than mixed into
Strategy A.