M4 Instruction Tuning and Model Adaptation
===========================================

.. contents:: On this page
   :local:
   :depth: 2

Outcome
-------

M4 fine-tuned the M3 base model on an instruction dataset built from
licence-verified sources, and measured the result on a frozen evaluation set
that was fixed before any tuning.

**The measured result is negative: instruction tuning did not improve
instruction following or code generation.** Both models scored 0/18 on syntax
validity and 33.3% on instruction compliance. The weights demonstrably moved --
all 18 outputs changed -- but behaviour on held-out instructions did not.

The cause is measurable, not mysterious: **61% of the evaluation prompts ask for
a task type the training data never contains**, and the single largest training
category (code completion, 51.8%) does not appear in the evaluation set at all.

This page documents the pipeline that produced that finding. Full evidence is in
``reports/M4_INSTRUCTION_TRAINING_REPORT.rst``.

What M4 built
-------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Path
     - Purpose
   * - ``data/instruction_schema.py``
     - the instruction dataset contract
   * - ``data/build_instruction_dataset.py``
     - derives instructions from the licence-verified corpus
   * - ``training/instruction_format.py``
     - formatting and assistant-only loss masking
   * - ``training/sft_dataset.py``
     - encoded dataset, shuffling, coverage accounting
   * - ``training/sft_train.py``
     - the M4 training engine
   * - ``evaluation/m4_eval_set.py``
     - the frozen evaluation set
   * - ``evaluation/code_check.py``
     - syntax, test and compliance checks
   * - ``evaluation/m4_evaluate.py``
     - before/after evaluation runner
   * - ``configs/train/sft_m4.yaml``
     - the training configuration

Dataset contract
----------------

.. code-block:: json

    {
      "id": "qm-codegeneration-1a2b3c4d5e6f",
      "instruction": "Write a Python docstring for the following function...",
      "input": "",
      "response": "\"\"\"Return the number of items...\"\"\"",
      "category": "coding",
      "task": "documentation",
      "language": "python",
      "source": "psf-requests",
      "license": "Apache-2.0",
      "source_sha": "...",
      "split": "train",
      "transform": "qm.instruction/1"
    }

Enforced by ``data/instruction_schema.py``: stable content-addressed ids, closed
vocabularies for category/task/language/split, mandatory provenance and licence,
no empty instruction or response, and a 4000-character instruction limit.

.. list-table:: Categories
   :header-rows: 1
   :widths: 34 66

   * - Category
     - Tasks
   * - coding
     - code_completion, documentation, code_explanation
   * - technical
     - sql
   * - instruction_following
     - format_following, constrained_output
   * - structured_output
     - json_output, config_output

**Not synthesised**: ``code_generation``, ``debugging``, ``refactoring``,
``test_generation``. These require judging what a correct fix is. Inventing them
would mean fabricating technical content, so they are absent rather than
plausible-looking. The absence is the direct cause of the negative result.

Building the dataset
--------------------

::

    python -m data.build_instruction_dataset \
        --corpus data/m3_code_verified.train.jsonl \
        --corpus-val data/m3_code_verified.val.jsonl \
        --out data/instruct_v1.train.jsonl \
        --out-val data/instruct_v1.val.jsonl

Every example is derived from a real source file in the licence-verified corpus.
A record's own ``spdx`` field is never treated as evidence: only
``licence_verification.json:verified_ids`` authorises a source. The two sources
M2 flagged (``psf-black``, ``pypa-setuptools``) are excluded, which is why 162
records of the M2 corpus never reach M4.

Quality pipeline: schema validation, licence gate, duplicate id / prompt /
example detection, Python syntax checking where the response must be executable,
control-character detection, and train/val isolation by id, prompt and full
example. **Nothing is silently dropped** -- every exclusion carries a reason in
``data/instruct_v1.manifest.json``.

Result: **610 train / 21 val records**, 1 excluded, split isolation verified.
The validation split is small because the licence-verified val corpus is small
(63 short files); this is stated rather than padded.

Instruction formatting
----------------------

.. code-block:: text

    <bos>### Instruction
    {instruction}

    ### Input
    {context}

    ### Response
    {response}<eos>

``### Input`` is omitted when there is no context. ``<bos>`` and ``<eos>`` are
the tokens ``qm_code_bpe_v0`` actually registers (ids 1 and 2). **No chat
template is used and no special tokens are invented**: that tokenizer has no role
markers, so rendering ``<|user|>`` would shred it into byte fragments and the
model would never learn a real turn delimiter.

Loss masking
------------

The prompt and completion are tokenized **separately** and concatenated. The
supervision boundary is therefore exact by construction -- it is the length of
the prompt id list, not a guess from re-rendering the prompt. The earlier
``training/sft.py`` estimated the boundary by re-tokenizing the prompt, which is
not guaranteed to be prefix-consistent.

.. code-block:: text

    instruction + context tokens  ->  labels = -100   (loss ignored)
    response tokens              ->  labels = id     (loss calculated)
    padding tokens               ->  labels = -100   (loss ignored)

Truncation policy: an over-long example is trimmed from the **left of the
prompt**, never the right, so the target survives. If the response alone cannot
fit, the example is **rejected** -- silently truncating a target leaves nothing
to learn from while looking like a successful dataset. In the shipped run: 170
examples had their prompt trimmed, 2 were rejected.

Note that supervision begins at the *second* response token. The first is the
model's input for predicting the second, which is correct next-token behaviour,
not an off-by-one.

Training
--------

::

    python -m training.preflight --config configs/train/sft_m4.yaml --probe-steps 3
    python -m training.sft_train --config configs/train/sft_m4.yaml \
        --out-dir runs/m4_sft

The engine reuses the M3 code rather than duplicating it:
``training.checkpoint``, ``training.pretrain`` (checkpointing, RNG capture,
``lr_at``), ``training.manifest``, ``training.instruction_format`` and
``training.sft_dataset``.

What is M4-specific: the base checkpoint is loaded from disk and opened
read-only, the freeze policy is explicit, and every M4 checkpoint records the
identity chain -- base checkpoint, base step, training strategy, instruction
format, dataset, dataset manifest, tokenizer, seed.

Resume works, including RNG state restoration.

Evaluation
----------

The evaluation set is authored in ``evaluation/m4_eval_set.py``, frozen, and
carries a content hash that a test asserts. 18 prompts across coding (9),
technical (3), instruction following (4) and structured output (2); 8 are
machine-testable.

.. warning::

   The set was revised **once**, after the first M4 run, when a test found that
   three evaluation prompts had been duplicated verbatim in the training data
   (``if-02-bullets``, ``struct-01-json``, ``struct-03-sql``) and one more was
   near-identical. Both models were then re-evaluated on the corrected set. The
   revision was driven by a contamination check, not by results, and both models
   had scored identically on the original set.

Run it identically for either model::

    python -m evaluation.m4_evaluate --ckpt <path> --tokenizer \
        tokenizer/qm_code_bpe_v0.json --label <name>

Greedy decoding is the default, so the sampler cannot decide a before/after
verdict.

Four dimensions are reported separately and never collapsed:

* **Syntax validity** -- and only for output containing real code. A block of
  ``#`` comments parses as Python, so a comment-only snippet is scored NO_CODE,
  not SYNTAX_VALID. Without this rule the M3 baseline scored a spurious 72%.
* **Execution / test success** -- authored assertions run in a subprocess with a
  timeout. Syntax-valid-but-wrong code is TEST_FAIL, which is why the two
  dimensions are kept apart.
* **Instruction compliance** -- deterministic format constraints only.
* **Extraction status** -- ambiguous output is marked, never guessed at.

Sampling coverage
-----------------

Greedy, temperature, top-k, top-p, EOS and max-length paths all have dedicated
tests in ``tests/test_m4_instruction.py``. M3's top-p defect (scattering on the
batch axis, so sampling *never* worked) is regression-tested here.

Known limitations
-----------------

* **No improvement was measured.** 0/18 syntax valid and 33.3% compliance, both
  before and after tuning.
* **Training data lacks the evaluated task types.** 61% of evaluation prompts ask
  for a task absent from training; ``code_generation`` is 44% of the eval set and
  0% of the training set.
* **Instruction data is tiny**: 610 train / 21 val records.
* **The base model is 2.85M parameters** trained on 1.17M tokens. It cannot
  express a function to copy, let alone synthesise a new one.
* **No knowledge is added by SFT.** Any correct answer would have to come from
  the base model's weights; instruction tuning only shapes behaviour.
* **CPU only**, ~3.6 minutes for 120 steps.
* **The M2 holdout remains unspent** and cannot corroborate the validation loss.

Recommendation
--------------

The from-scratch route cannot reach Level 4 (coding capability) by adding
instruction data to a 2.85M base model. Two changes would be required, and
neither is a matter of more training steps:

1. **Grow the base** -- the M2 corpus is 1.2% of its 100M-token target, and that
   is gated on licence decisions and source authorisation, not on code.
2. **Acquire instruction data for the tasks being evaluated** -- or accept that
   evaluation should measure the tasks training actually contains.

The open-weight study (in the M4 report) reaches the same conclusion from the
other direction: a suitable Apache-2.0 base exists at a size that would run on
this class of hardware, but adopting it is an architecture change, not a
drop-in.