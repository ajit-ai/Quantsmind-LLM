M5 Local Evaluation and Model Strategy Selection Report
=======================================================

Target version: v0.5.0 (target only -- no release was cut)

.. contents:: On this page
   :local:
   :depth: 2

Verdict
-------

**COMPLETE (harness and M3/M4 runs) -- EXTERNAL COMPARISON BLOCKED --
AWAITING USER ACCEPTANCE and TECHNICAL REVIEW.**

M5 delivered a reproducible evaluation framework, a frozen leakage-screened
benchmark, and honest measurements of both QuantsMind checkpoints. It did
**not** deliver the model-strategy decision the milestone was named for,
because the external reference point could not be evaluated within the
licence and resource boundaries set for this run.

Recommendation: **``INSUFFICIENT_EVIDENCE``**, with **``HYBRID_TRACK``** as the
fallback. No claim that either checkpoint is fit for code generation is
supported, and no claim about how it compares to a real code model is
refuted either.

A. Baseline
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``; 0 tags; no commits made during M5
   * - Working tree
     - 133 entries before M5; existing work preserved
   * - M3 checkpoint
     - ``runs/m3_base/checkpoint_best.pt``; 2,853,280 parameters; vocab 5,822;
       sha256 ``56365457e5ffb280...``
   * - M4 checkpoint
     - ``runs/m4_sft/checkpoint_best.pt``; 2,853,280 parameters;
       sha256 ``2bd86e2b9cd211bc...``
   * - Hardware
     - Windows, CPU-only ``torch 2.13.0+cpu``, Python 3.14.7, ~3.91 GiB RAM
   * - ``transformers``
     - 5.16.1 installed; **no external model weights present locally**

Both checkpoints were verified loadable before any measurement was taken. M3
and M4 share an architecture and differ only in weights, so the comparison
between them is exact.

B. Benchmark integrity
----------------------

The benchmark was rebuilt for M5 rather than inherited, because M5 must not
re-ask questions the M4 model was tuned against.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Check
     - Result
   * - Examples
     - 26 total; 9 automatically scorable; 17 human-review-only
   * - Prompt leakage vs M3/M4 evaluation sets
     - 0 overlapping examples (5 excluded at build time)
   * - Manifest hash
     - ``52ea2ed5db839b22fb6865f8fa62e52456f82be8151c45abc8f6da10731b8d87``
   * - Hash identical across both evaluated runs
     - yes, verified by ``evaluation/m5_report.py``
   * - Shared source documents
     - 7 examples -- risk recorded, not removed
   * - Context fit
     - token-aware truncation; every prompt fits the 512-token window
   * - Source licence
     - only ``data/licence_verification.json`` verified ids; unverified
       sources rejected even when they carry a licence string

The 7 shared-source examples are an acknowledged limitation. Same source
document does not imply identical text, but these examples are not fully
independent of training material, and the report says so rather than claiming a
clean benchmark.

C. Measured results
-------------------

Greedy decoding, one benchmark, one machine, one scoring version.

.. list-table::
   :header-rows: 1
   :widths: 26 20 20 34

   * - Metric
     - M3 base
     - M4 tuned
     - Reading
   * - Syntax valid
     - 1/20 (5.0%)
     - 0/20 (0.0%)
     - at most one in twenty outputs parsed as Python
   * - Functional pass
     - 0/0 (n/a)
     - 0/0 (n/a)
     - no executable code was produced to test
   * - Compliance
     - 3/6 (50.0%)
     - 4/6 (66.7%)
     - M4 follows format instructions more often
   * - Structured output
     - 0/0 (n/a)
     - 0/0 (n/a)
     - no scorable schema task in this benchmark
   * - Human review
     - 0/0 (n/a)
     - 0/0 (n/a)
     - 17 examples outstanding

Every denominator above is below 20 and is therefore labelled
``[PRELIMINARY]`` by the harness itself. None of these numbers supports a
capability claim.

C.1 The most important correction in M5
---------------------------------------

M4's negative result reported both models at 0/18 on syntax. M5 shows that
number was, in part, generous: a scorer that extracts "the code" from a prose
response can mark prose-with-a-code-shaped-fence as valid. M5 rejects
comment-only output as ``NO_EXECUTABLE_CODE`` and marks multiple code blocks as
ambiguous rather than guessing.

This means the M3/M4 numbers should be read as **upper bounds**. Both models
are worse than a naive read suggests.

C.2 Why no composite score
--------------------------

A weighted aggregate was deliberately not produced. M4 is more compliant than
M3 (4/6 vs 3/6) while being worse at syntax (0/20 vs 1/20), and neither produced
executable code. Any single-number score would either hide that trade-off or
manufacture an ordering the evidence does not support. The report carries a
denominator next to every numerator instead.

D. External candidate
---------------------

``Qwen/Qwen2.5-Coder-1.5B-Instruct`` was selected as the smallest
code-specialised open-weight model that a 2.85M-parameter model could plausibly
be compared against. It was assessed and **not run**.

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Check
     - Result
     - Detail
   * - Licence
     - pass
     - Apache-2.0, verified in the repository licence register
   * - Runtime
     - pass
     - ``transformers 5.16.1`` importable
   * - Local weights
     - **fail**
     - none present for this model or in the HF cache
   * - Download policy
     - **fail**
     - fetching weights is outside this run's licence boundary
   * - RAM
     - **fail**
     - ~2.9 GiB for bfloat16 weights plus KV cache, against ~3.91 GiB total

Nothing was downloaded. The attempt is recorded as a blocked run at
``runs/m5/qwen25_coder_1p5b/run_manifest.json`` with
``complete: false`` and ``evaluated: false``, and it appears in the comparison
report as ``NOT RUN``. It contributes no number anywhere.

Two independent blockers apply -- licence boundary and memory. Even with
unrestricted download, a 1.5B fp16 model on a ~3.9 GiB machine is marginal.
A smaller candidate (for example a 0.5B-class code model) would be the
sensible next external reference point.

E. Verification
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Check
     - Result
   * - Full test suite
     - **782 passed**, 0 failed
   * - New M5 tests
     - 21 in ``test_m5_scoring.py`` and ``test_m5_runner.py``, plus 15 adapter
       and syntax tests
   * - Reference oracles
     - 8/8 authored solutions pass their own functional test, so ``0/0``
       reflects the models rather than a broken harness
   * - Report integrity
     - counts cross-checked against raw ``results.jsonl``; the check result is
       stored in ``reports/M5_COMPARISON.json``
   * - Resume safety
     - a truncated final line from a killed run is ignored, not misread as a
       complete result
   * - M3 rerun
     - clean ``--no-resume`` run reproduced identical scores in 88.08 s, and
       the count verification confirms 26 raw rows against 26 reported examples

F. Limitations
--------------

1. **26 examples.** This is a smoke test. A single example moves a metric by
   five percentage points.
2. **Functional scoring never ran.** ``0/0`` proves absence of code, not
   incapacity. Neither model has been tested for whether it *can* write correct
   code.
3. **No external reference.** The comparison that would justify a strategy
   choice is missing entirely.
4. **17 human-review examples outstanding**, including the security and
   architecture tasks closest to the product's actual use.
5. **CPU-only greedy decoding.** Temperature and sampling effects unexplored.
6. **Shared source material** in 7 examples limits independence claims.

G. Recommendation
-----------------

``INSUFFICIENT_EVIDENCE``

The evidence supports one firm conclusion -- neither checkpoint currently
emits usable Python on code prompts -- and does not support any ranking,
because the alternative was never measured.

``HYBRID_TRACK`` is the fallback if a decision must be made before the
external run completes: use a general external model for code generation now,
and keep the QuantsMind checkpoints for the narrow tasks where they do behave
measurably (format compliance). This recommendation rests on the absence of
evidence for the checkpoints, not on positive evidence for the alternative,
and should be revisited once Qwen is actually run.

H. Next steps
--------------

1. **Authorise the external run**, ideally against a 0.5B-class candidate if
   memory is the binding constraint. The adapter, runner and scorers already
   exist; this requires no new code, only the weights.
2. **Review the 17 human-only examples** before any strategy is adopted.
3. **Diagnose the missing code.** Both checkpoints answered code prompts with
   something other than code. Inspecting raw outputs may reveal a prompt or
   formatting contract defect that is correctable without retraining -- the
   highest-leverage investigation available.
4. **Grow the benchmark** until denominators exceed 20 and the
   ``[PRELIMINARY]`` flag can be dropped.
5. **Re-verify leakage** if the benchmark is rebuilt for M6, since M5's
   exclusion set was computed only against M3/M4 evaluation sets.

I. Artifacts
------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Path
     - Contents
   * - ``evaluation/m5_benchmark.jsonl``
     - 26 frozen examples
   * - ``evaluation/m5_benchmark.manifest.json``
     - counts, per-task breakdown, manifest hash
   * - ``runs/m5/m3_base/``
     - ``results.jsonl``, ``summary.json``, ``run_manifest.json``
   * - ``runs/m5/m4_tuned/``
     - ``results.jsonl``, ``summary.json``, ``run_manifest.json``
   * - ``runs/m5/qwen25_coder_1p5b/``
     - ``run_manifest.json`` recording the blocked attempt
   * - ``reports/M5_COMPARISON.json``
     - machine-readable scorecard, generated from run artifacts
   * - ``reports/M5_CANDIDATE_READINESS.json``
     - external candidate readiness assessment
   * - ``docs/source/roadmap/M5_EXECUTION.rst``
     - narrative record of the milestone

All JSON artifacts are regenerated by ``python -m evaluation.m5_report`` and
``python -m evaluation.m5_candidates``; no number in this report is
hand-copied.
