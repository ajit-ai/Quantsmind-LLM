M5 Local Evaluation and Model Strategy Selection
=================================================

.. contents:: On this page
   :local:
   :depth: 2

Outcome
-------

M5 built a reproducible evaluation harness and ran it against both QuantsMind
checkpoints on one frozen benchmark, under one decoding configuration, on one
machine.

**The measured result is negative and unambiguous: neither M3 nor M4 produced
usable code on the M5 benchmark.** Functional pass rate is ``0/0`` for both
models -- not because they failed the tests, but because there was no executable
Python to hand to the tests.

.. list-table:: M5 results (greedy decoding, 26 examples)
   :header-rows: 1
   :widths: 30 18 18 18

   * - Metric
     - M3 base
     - M4 tuned
     - Denominator
   * - Syntax valid
     - 1/20 (5.0%)
     - 0/20 (0.0%)
     - 20 scorable
   * - Functional pass
     - 0/0 (n/a)
     - 0/0 (n/a)
     - 0 executable submissions
   * - Compliance
     - 3/6 (50.0%)
     - 4/6 (66.7%)
     - 6 scorable
   * - Structured output
     - 0/0 (n/a)
     - 0/0 (n/a)
     - 0 scorable
   * - Human review
     - 0/0 (n/a)
     - 0/0 (n/a)
     - 17 outstanding

**No composite score is produced, by design.** Blending these metrics would
conceal the only finding that matters: M4 is *more* compliant than M3 while
being *worse* at producing parseable Python. A single number would have made a
2.85M-parameter model that cannot write code look comparable to one that can
barely.

Strategy selection
------------------

**Recommendation: ``INSUFFICIENT_EVIDENCE``, with a ``HYBRID_TRACK`` fallback.**

The decision is deliberately not "pick a model". Two facts block any such
recommendation:

1. **The external comparison did not happen.** ``Qwen/Qwen2.5-Coder-1.5B-Instruct``
   was never evaluated. It is licence-clean and ``transformers 5.16.1`` is
   installed, but no weights are present locally and fetching them is outside
   this run's licence boundary. The run is recorded as ``blocked`` in
   ``runs/m5/qwen25_coder_1p5b/run_manifest.json`` -- never as a zero score.

2. **17 of 26 examples are human-review-only.** They are architecture,
   requirements and security tasks with no automated oracle. They are
   structurally unable to contribute to a score, and human acceptance has not
   been performed.

What M5 changed about the evidence
----------------------------------

M4's negative result was real but under-specified: both models scored 0/18 on
syntax, which could mean "the benchmark is broken" or "the models cannot write
code". M5 separates those:

- **Prose is no longer mistaken for code.** Comment-only and prose-only
  responses are rejected as ``NO_EXECUTABLE_CODE`` rather than counted as
  syntax-valid. This is the single most important correction in M5; the naive
  reading of the M3/M4 numbers flattered both models.
- **Every reported rate carries its denominator.** ``0/0`` renders as ``n/a``,
  not ``0%``. A metric with no evidence behind it is displayed as missing.
- **A zero denominator is never a pass.** ``test_zero_denominator_yields_no_score``
  pins this behaviour.
- **Reference solutions are themselves tested.** All eight authored oracles pass
  their own functional test, so a ``0/0`` reflects the models, not a broken
  harness.
- **Small denominators are labelled.** Any denominator under 20 prints
  ``[PRELIMINARY]``. Every number in the table above carries that flag.

Cross-version leakage
---------------------

M5 rebuilt the benchmark and excluded **five** examples whose prompts overlap
the M4 evaluation set, so M5 no longer re-asks questions M4 was tuned against.

.. list-table:: Benchmark integrity
   :header-rows: 1
   :widths: 40 60

   * - Check
     - Result
   * - Prompt leakage vs M3/M4
     - 0 overlapping examples
   * - Manifest hash
     - ``52ea2ed5db839b22fb6865f8fa62e52456f82be8151c45abc8f6da10731b8d87``
   * - Same hash across all runs
     - yes (verified by ``m5_report.py``)
   * - Shared source documents
     - 7 examples -- recorded as risk, not removed
   * - Prompt truncation
     - token-aware, all prompts fit the 512-token context

The 7 shared-source examples are a **known limitation, not a clean bill of
health**. Same document does not imply same text, but it does mean these
examples are not fully independent of training material. ``test_shipped_benchmark_has_zero_leakage``
and ``test_source_overlap_is_flagged_as_risk`` keep both facts visible.

What M5 built
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Path
     - Purpose
   * - ``evaluation/m5_schema.py``
     - benchmark record contract; rejects records missing licence provenance
   * - ``evaluation/m5_leakage.py``
     - normalising prompt overlap detection against prior evaluation sets
   * - ``evaluation/m5_build_benchmark.py``
     - builds the frozen benchmark; excludes cross-version leakage
   * - ``evaluation/m5_benchmark.jsonl``
     - the 26 frozen examples
   * - ``evaluation/m5_benchmark.manifest.json``
     - benchmark identity, counts and hash
   * - ``evaluation/m5_adapters.py``
     - ``QMCkptAdapter`` and ``ExternalHFAdapter`` behind one interface
   * - ``evaluation/m5_scorers.py``
     - independent syntax, functional, compliance and structured scorers
   * - ``evaluation/m5_runner.py``
     - deterministic runner with resume, selective rerun and manifests
   * - ``evaluation/m5_report.py``
     - comparison report generated from artifacts, never from memory
   * - ``evaluation/m5_candidates.py``
     - external candidate readiness assessment

External candidate status
-------------------------

``Qwen/Qwen2.5-Coder-1.5B-Instruct`` -- the smallest code-specialised
open-weight reference point -- was assessed and **not run**.

.. list-table:: Readiness
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
     - none present for this model
   * - Download policy
     - **fail**
     - fetching weights is outside this run's licence boundary
   * - RAM
     - **fail**
     - ~2.9 GiB for bfloat16 weights plus KV cache

``transformers`` being installed is not a licence to fetch weights. The
readiness report is written to ``reports/M5_CANDIDATE_READINESS.json`` and
generated without downloading anything.

How to reproduce
----------------

.. code-block:: console

   python -m evaluation.m5_build_benchmark
   python -m evaluation.m5_runner --adapter qm \
       --ckpt runs/m3_base/checkpoint_best.pt \
       --out-dir runs/m5/m3_base --label m5-m3-base
   python -m evaluation.m5_runner --adapter qm \
       --ckpt runs/m4_sft/checkpoint_best.pt \
       --out-dir runs/m5/m4_tuned --label m5-m4-tuned
   python -m evaluation.m5_candidates
   python -m evaluation.m5_report

Every run records the benchmark hash, git commit, dirty-tree state, generation
config, Python version and elapsed time in ``run_manifest.json``. Reruns resume
from completed results; a partial write left by a killed run is ignored rather
than corrupting the next one.

Verification
------------

- 782 tests pass, including 21 new M5 scoring and runner tests.
- 8 of 8 reference oracles pass their own functional test.
- Report counts are cross-checked against the raw ``results.jsonl`` and the
  verification result is recorded in the report itself.
- A run that never executed is reported as ``NOT RUN`` or ``BLOCKED`` and never
  contributes a number to the scorecard.

Limitations
-----------

- **26 examples is a smoke test, not an evaluation.** Denominators are 20, 6
  and 0. No result here supports a capability claim.
- **Functional scoring never executed.** ``0/0`` proves no code was produced;
  it says nothing about whether either model *can* write code under different
  decoding.
- **No external reference point.** The central question -- can a real
  code-specialised model do what these checkpoints cannot? -- is unanswered.
- **17 examples await human review**, including the security and architecture
  tasks most relevant to the product's actual use.
- **CPU-only decoding.** Both runs are greedy on CPU. Temperature, top-p and
  sampling effects were not explored.

Recommended next steps
----------------------

1. **Obtain Qwen weights explicitly** (user authorisation required), then run
   the existing harness against it. No new code is needed -- the adapter,
   runner and scorers are already in place. This is the highest-value next
   step because it converts a blocked comparison into a real one.
2. **Perform human review of the 17 outstanding examples** before any strategy
   is selected.
3. **Diagnose why no code is emitted.** M3's 1/20 syntax rate and M4's 0/20
   warrant inspection of raw outputs. If both models emit prose in response to
   code prompts, that is a formatting/prompt-contract defect that may be
   correctable without retraining.
4. **Expand the benchmark** so denominators can exceed 20 and
   ``[PRELIMINARY]`` flags can be dropped.
