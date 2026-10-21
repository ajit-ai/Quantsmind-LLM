M4 Instruction Tuning and Model Adaptation Report
================================================

Target version: v0.4.0 (target only -- no release was cut)

.. contents:: On this page
   :local:
   :depth: 2

Verdict
-------

**COMPLETE -- AWAITING USER ACCEPTANCE and TECHNICAL REVIEW.**

The engineering scope is implemented, tested and demonstrated end to end. The
*model experiment produced a negative result*, reported as measured: instruction
tuning did not improve instruction following or code generation.

Both halves of the central M4 question are answered: the first with a measured
"no, not at this scale", the second with a documented recommendation.

A. Baseline
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``; 0 tags; no commits made during M4
   * - Working tree
     - 115 entries before M4; existing work preserved
   * - M3 base checkpoint
     - ``runs/m3_base/checkpoint_best.pt``, sha256 ``56365457e5ffb280...``
   * - Architecture
     - decoder-only causal transformer, 2,853,280 params, vocab 5822
   * - Tokenizer
     - ``tokenizer/qm_code_bpe_v0.json``, vocab 5822

**Critical baseline finding**: ``data/code_sft_v0.jsonl`` (442 records) is
**quarantined** -- 423 responses are identical boilerplate and the remaining 19
come from a faulty regex extractor. It states "Do not train on this dataset."
M4 built a new instruction corpus rather than reusing it.

**Reuse, not duplication**: ``training/model.py``, ``training/pretrain.py``,
``training/data.py``, ``training/checkpoint.py``, ``training/validation.py``,
``training/manifest.py``, ``training/generate.py`` and ``training/preflight.py``
were reused. M4 added the instruction-specific layer around them.

B. Dataset
----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Train records
     - **610** (608 encoded, 2 rejected)
   * - Val records
     - **21**
   * - Train tokens
     - 177,354 total, 104,401 supervised (58.9%)
   * - Prompt truncated (left)
     - 170; response always preserved
   * - Categories
     - coding 602, structured_output 2, technical 3, instruction_following 3
   * - Languages
     - python 602, json 1, yaml 2, sql 3
   * - Licences
     - only sources in ``licence_verification.json:verified_ids``
   * - Excluded
     - 1 duplicate; nothing silently dropped
   * - Split isolation
     - verified: no shared id, prompt or example

162 records (37%) of the M2 corpus were withheld because ``psf-black`` and
``pypa-setuptools`` have unresolved licences.

**Training task distribution** -- this is the crux of the result:

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Task
     - Count
     - Share
   * - code_completion
     - 316
     - 51.8%
   * - code_explanation
     - 156
     - 25.6%
   * - documentation
     - 130
     - 21.3%
   * - sql / format / config / json
     - 8
     - 1.3%

**Not synthesised**: ``code_generation``, ``debugging``, ``refactoring``,
``test_generation``. These require judging what a correct fix is; inventing them
would mean fabricating technical content.

C. Training
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Base
     - M3 checkpoint, opened read-only, never modified
   * - Freeze policy
     - none (full fine-tune), 2,853,280 trainable parameters
   * - Steps
     - 120 / 120
   * - Tokens
     - 491,520
   * - Runtime
     - 3.6 min on CPU (~2,500 tok/s)
   * - Train loss
     - 4.0337 -> 2.8807
   * - Validation loss
     - 5.1812 -> 4.9096 (monotonic)
   * - Uniform baseline
     - 8.6694
   * - Checkpoints
     - ``runs/m4_sft/checkpoint_last.pt``, ``checkpoint_best.pt``

Learning rate 5e-4, deliberately below the M3 pretraining peak of 3e-3, because
tuning a converged base at the pretraining rate washes out what it learned.

Every M4 checkpoint records the identity chain: base checkpoint, base step,
training strategy, instruction format, dataset, dataset manifest, tokenizer,
seed.

D. Before / After
-----------------

Identical frozen evaluation set, identical greedy decoding, run before and
after tuning.

.. list-table::
   :header-rows: 1
   :widths: 36 24 24

   * - Metric
     - M3 base
     - M4 tuned
   * - Prompts
     - 18
     - 18
   * - Syntax valid
     - **0 (0.0%)**
     - **0 (0.0%)**
   * - Test pass
     - 0 of 0 testable
     - 0 of 0 testable
   * - Instruction compliance
     - **33.3%**
     - **33.3%**
   * - Mean output chars
     - 196.9
     - 188.6

**No measurable improvement.** All 18 outputs changed, so the weights moved; the
behaviour on held-out instructions did not.

Root cause, measured: **11 of 18 evaluation prompts (61.1%) ask for a task type
the training data never contains** -- ``code_generation`` (44.4% of the eval
set, 0% of training), ``debugging``, ``api_explanation``,
``multi_step_instruction``. Conversely the dominant training category,
``code_completion``, does not appear in the evaluation set.

Evaluation methodology
~~~~~~~~~~~~~~~~~~~~~~

Four dimensions, reported separately, never collapsed:

* **Syntax validity** -- only for output containing real code. A comment-only
  snippet parses as Python; scoring it SYNTAX_VALID produced a spurious 72%
  baseline on the first run, corrected to NO_CODE.
* **Test status** -- authored assertions in a subprocess with a timeout.
  Syntax-valid-but-wrong code is TEST_FAIL, keeping the two dimensions honest.
* **Instruction compliance** -- deterministic format constraints only; prompts
  without a machine-checkable rule are marked not-applicable, not failed.
* **Extraction status** -- multiple code blocks are marked EXTRACTION_AMBIGUOUS
  rather than resolved by picking the one that parses.

The evaluation set was revised **once**, when a test found three prompts
(``if-02-bullets``, ``struct-01-json``, ``struct-03-sql``) duplicated in the
training data verbatim and one more was near-identical. Both models were then
re-evaluated on the corrected set. The revision was driven by a contamination
check, not by results.

E. Generation
-------------

Prompt: ``Write a Python function called reverse_string...`` (greedy, 96 tokens)

M3 base -- unedited::

    # The default is used to the ```.
    # The ```.ger``.
    #
    #
    #

M4 tuned -- unedited::

    # The defaults to the given object.

    # The defaults to the
    #
    # The defaults to the
    #
    # The
    #1``, and the

Prompt: ``Respond with the single word ACKNOWLEDGED and no other text.``

Both models: non-compliant. Neither produced the required word.

Full outputs for all 18 prompts, base and tuned, are preserved unedited in
``reports/m4_eval_m3-base.json`` and ``reports/m4_eval_m4-tuned.json``.

F. Open-weight study
--------------------

Full document: ``docs/source/roadmap/M4_OPEN_WEIGHT_STUDY.rst``. **No external
model was downloaded or run**; this is analysis, not a benchmark.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Finding
   * - Qwen3-Coder licence
     - **Apache-2.0**, verified from the repository LICENSE file
   * - Qwen2.5-Coder 0.5B / 1.5B / 7B / 14B / 32B
     - Apache-2.0
   * - Qwen2.5-Coder **3B**
     - Qwen Research licence -- **not commercial**
   * - Fits this host (inference)
     - 0.5B (~380 MB Q4) and 1.5B (~941 MB Q4)
   * - Does not fit
     - 7B and above
   * - Architecture
     - **incompatible** with ``training/model.py``: Qwen2 adds biases and QKV
       norms, Qwen3-Next is MoE. Adoption is an implementation change plus a
       tokenizer change, not a weight swap

**Recommendation: hybrid, staged.** Keep the from-scratch track as the pipeline
proof and the sovereign-weights option; evaluate Qwen2.5-Coder-1.5B-Instruct as
the practical product base once hardware allows. Download nothing until the
owner decides, because adoption changes the product's identity, tokenizer
contract, checkpoint format and licence obligations.

G. Defects found and fixed
--------------------------

Each found by running the code, not by reading it:

1. **Comment-only output scored as valid Python** (measurement integrity). The
   first M3 baseline read 72% syntax-valid; every "valid" sample was ``#`` lines
   and stray fences, which parse because comments are legal Python. Added an
   executable-content check; baseline corrected to **0%**.
2. **SYNTAX_INVALID was unreachable.** ``extract_code`` returned the candidate
   only if it parsed, so "produced broken code" was reported as NO_CODE,
   indistinguishable from "produced nothing". A fenced block is now returned
   regardless, so the parse check can fail it properly.
3. **Three evaluation prompts were duplicated in the training data.** Found by
   the overlap test; set corrected and both models re-evaluated.
4. **A prompt asking for a fenced code block was penalised for producing one.**
5. **``model:`` meant two different things.** The SFT config used it for the
   freeze policy while every other config uses it for architecture, so
   checkpoint reload silently read the policy block as the architecture. Renamed
   to ``adaptation:``; the loader now requires a real ``d_model``.
6. **A wrong authored test.** ``fizzbuzz_n(5)[0]`` is ``"1"``, not ``"Fizz"``.
   The assertion, not the reference implementation, was wrong.
7. **``code_completion`` instructions blew the 4000-char limit** because the
   whole file prefix was embedded in the instruction. Moved the prefix to the
   ``input`` field, which is what it is for.

H. Limitations
--------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Item
     - Status
   * - Measured improvement
     - **None.** 0/18 syntax valid and 33.3% compliance, before and after.
   * - Train/eval task overlap
     - 61% of eval prompts ask for a task absent from training
   * - Instruction data size
     - 610 train / 21 val records; val is thin because the val corpus is small
   * - Base model
     - 2.85M params on 1.17M tokens -- cannot express a function to copy
   * - Coding capability
     - **Not demonstrated.** No output parsed as Python
   * - Generalisation
     - Not assessable; one model, one dataset, one host
   * - No external LLM used
     - Confirmed: no hosted model produced any output or any training target
   * - M2 holdout
     - Unspent; cannot corroborate the validation loss
   * - Hardware
     - CPU only, 3.91 GiB RAM; no GPU path exercised
   * - M3 base checkpoint
     - Unmodified

I. Final recommendation
-----------------------

**For M5**: proceed with capability evaluation, but evaluate the tasks the
model was actually trained on, and treat the negative result as the headline.

Concretely:

1. **Do not tune further on the current data.** More steps on a dataset that
   lacks the evaluated task cannot fix a task-coverage gap.
2. **Acquire instruction data for code generation** -- the single highest-value
   change. Everything else is secondary.
3. **Resolve the two licence decisions and authorise sources.** Corpus scale
   (1.2% of target) blocks both tracks equally.
4. **Decide the base-model question** before M5 benchmarking, because the answer
   determines what is worth benchmarking. Adopting an Apache-2.0 open-weight
   base is a larger step than any amount of additional training here.

Capability level
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 46 20 34

   * - Level
     - Status
     - Evidence
   * - 1 Training works
     - **established (M3)**
     - 4.59 nats better than random init
   * - 2 Model learns
     - **established (M3)**
     - loss 8.67 -> 3.06
   * - 3 Instruction following
     - **not established**
     - compliance unchanged at 33.3% after tuning
   * - 4 Coding capability
     - **not reached**
     - no output parsed as Python
   * - 5 Useful coding model
     - not reached
   * - 6 Competitive model
     - out of M4 scope; no comparison run

Acceptance status
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 16 44

   * - Gate
     - Status
     - Note
   * - IMPLEMENTED
     - MET
     - contract, quality pipeline, formatter, masking, trainer, evaluation
   * - AUTOMATED TESTED
     - MET
     - 723 passed, 0 failed; 46 M4 tests
   * - END-TO-END DEMONSTRATED
     - MET
     - dataset -> tokenisation -> tuning -> checkpoint -> fresh-process reload
   * - M3 BASE INSTRUCTION-TUNED
     - MET
     - 120 steps, 491,520 tokens, checkpoint written
   * - BEFORE/AFTER EVALUATED
     - MET
     - identical frozen set, identical decoding; no improvement found
   * - OPEN-WEIGHT STRATEGY ASSESSED
     - MET
     - licences verified; recommendation documented
   * - USER ACCEPTANCE
     - **outstanding**
     - requires the owner; not self-certified
   * - TECHNICAL REVIEW
     - **outstanding**
     - requires an independent reviewer; not self-certified
   * - FORMALLY CLOSED
     - not met
     - blocked on the two gates above

v0.4.0 was **not** released. No commit, tag, push, merge, publication or release
workflow was triggered.