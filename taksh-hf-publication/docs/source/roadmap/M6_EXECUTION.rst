M6 Local Capability Validation
==============================

.. contents:: On this page
   :local:
   :depth: 2

Outcome
-------

**Status: ``INSUFFICIENT_EVIDENCE``.**

M6 asked one question: *can the project demonstrate materially useful coding
capability from a model that runs locally, under its own hardware and licence
constraints?* The answer this milestone can defend is **no, not yet** -- and,
just as importantly, **the question was not answered by failure**. It was
answered by the absence of any candidate that could be run.

Three models were evaluated on the frozen M5 benchmark. All three produced
output. None produced executable code. No practical, coding-capable,
open-weight model could be obtained inside the licence and memory boundary, so
the practical layer of the hybrid strategy remains a plan rather than a result.

.. list-table:: M6 results (26 examples, one benchmark, greedy decoding)
   :header-rows: 1
   :widths: 26 14 14 14 18

   * - Metric
     - M3 base
     - M4 tuned
     - MiniLM (external)
     - Denominator
   * - Syntax valid
     - 1/20
     - 0/20
     - 0/20
     - 20 code examples
   * - Functional pass
     - 0/0
     - 0/0
     - 0/0
     - 0 executable submissions
   * - Compliance
     - 3/6
     - 4/6
     - 4/6
     - 6 checklist examples
   * - Structured output
     - 0/0
     - 0/0
     - 0/0
     - 0 scorable
   * - Evidence level
     - 3
     - 2
     - 2 (capped)
     - see `Evidence levels`_

All three runs share benchmark manifest hash ``52ea2ed5db839b22``. Every run
recorded ``hosted_api_used: false`` and ``credentials_used: false``.

The compliance column is the trap in this table, and M6 refuses to fall into it.
Every one of the 11 compliance passes across the three runs came from a
*non-substantive* rule: ``non_empty``, ``no_fences``, or a one-line stub that
satisfied ``max_lines`` by emitting ``"""``. No run recorded a single
substantive pass. Reporting "66.7% instruction following" from that data would
have been a formatting artifact presented as a capability. The
``compliance_evidence`` block in each manifest records the distinction, and
:func:`evaluation.m6_capability.assess_levels` refuses to credit level 5
without a substantive pass.

Candidates
----------

Discovery is a read-only scan of the Hugging Face cache. No model code is
executed during inspection and no weights are downloaded.

.. list-table:: Candidate inventory
   :header-rows: 1
   :widths: 34 12 12 16 26

   * - Candidate
     - Licence
     - Head
     - Acquisition
     - Blocking reason
   * - ``quantsmind/m3-base``
     - project-owned
     - causal_lm
     - PRESENT
     - evaluated
   * - ``quantsmind/m4-tuned``
     - project-owned
     - causal_lm
     - PRESENT
     - evaluated
   * - ``sentence-transformers/all-MiniLM-L6-v2``
     - Apache-2.0
     - embedding
     - PRESENT
     - evaluated; cannot follow an instruction
   * - ``google-bert/bert-base-uncased``
     - Apache-2.0
     - masked_lm
     - PRESENT
     - no free-form generation
   * - ``Qwen/Qwen2.5-Coder-1.5B-Instruct``
     - Apache-2.0
     - causal_lm
     - NOT_PRESENT
     - no local artifact; download requires authorisation
   * - ``Qwen/Qwen2.5-Coder-0.5B-Instruct``
     - Apache-2.0
     - causal_lm
     - NOT_PRESENT
     - no local artifact; download requires authorisation
   * - ``Qwen/Qwen2.5-Coder-7B-Instruct``
     - Apache-2.0
     - causal_lm
     - PARTIAL
     - empty snapshot (0 MiB)
   * - ``microsoft/Phi-3-mini-4k-instruct``
     - MIT
     - causal_lm
     - PARTIAL
     - both shards ``.incomplete``; no tokenizer, no config

Every licence above was verified from a primary source and the URL is recorded
in ``evaluation/m6_licensing.py``. **A verified licence is not the same claim as
a usable candidate**, and the inventory keeps them separate: Qwen is
licence-clean and still cannot be run, because the weights are not on disk and
fetching them is outside this run's authorization.

Two task heads are structurally ineligible. ``bert-base-uncased`` is a masked
language model and ``all-MiniLM-L6-v2`` is an embedding model. Neither can
answer a free-form coding instruction. They were evaluated anyway -- because
running the harness on an ineligible model is how you prove the harness reports
the ineligibility rather than inventing a score -- and their evidence level is
capped at 2 with an explicit note.

Evidence levels
---------------

Capability is reported on an ordered scale, never as a single number. A model
receives the highest level its persisted run data actually supports.

.. list-table:: Levels actually reached
   :header-rows: 1
   :widths: 8 34 58

   * - Level
     - Name
     - Requirement
   * - 1
     - ``RUNTIME_WORKS``
     - runtime loaded and produced a response
   * - 2
     - ``GENERATION_WORKS``
     - non-empty output on at least one example
   * - 3
     - ``SYNTAX_CAPABILITY_DEMONSTRATED``
     - at least one output parsed as valid Python
   * - 4
     - ``FUNCTIONAL_CAPABILITY_DEMONSTRATED``
     - functional numerator > 0 with denominator > 0
   * - 5
     - ``INSTRUCTION_FOLLOWING_DEMONSTRATED``
     - **substantive** compliance pass with a stated denominator
   * - 6
     - ``USEFUL_CODING_CAPABILITY_SUPPORTED``
     - multiple functional passes on at least 20 examples
   * - 7
     - ``PRODUCT_READINESS_ESTABLISHED``
     - human acceptance plus independent technical review

Level 7 is unreachable in M6 by construction: no human review round has been
run. Level 6 is unreachable for M3 and M4 because functional evidence is absent,
not because the sample is small.

Capability categories
---------------------

Categories with no eligible examples are reported ``UNAVAILABLE``, never zero.

.. list-table:: Benchmark coverage
   :header-rows: 1
   :widths: 30 12 12 46

   * - Category
     - Status
     - Examples
     - Reason
   * - code_completion
     - SCORED
     - 14
     - syntax only; functional has no test specification
   * - code_generation
     - SCORED
     - 6
     - syntax only; functional has no test specification
   * - structured_output
     - SCORED
     - 2
     - structured keys plus checklist
   * - instruction_compliance
     - SCORED
     - 1
     - two checklist rules
   * - code_explanation
     - UNAVAILABLE
     - 1
     - the only example requires human review
   * - sql
     - UNAVAILABLE
     - 2
     - both examples require human review
   * - debugging
     - UNAVAILABLE
     - 0
     - no benchmark examples
   * - refactoring
     - UNAVAILABLE
     - 0
     - no benchmark examples
   * - unit_test_generation
     - UNAVAILABLE
     - 0
     - no benchmark examples
   * - documentation
     - UNAVAILABLE
     - 0
     - no benchmark examples

Coverage is counted from the persisted example records, not from metric
denominators: a two-rule checklist makes one example contribute two rule
results, and a denominator-based count would have published "2 examples" for
``instruction_compliance`` when the benchmark contains one.

Strategy decision
-----------------

**Recommendation: ``INSUFFICIENT_EVIDENCE``. Status label:
``INSUFFICIENT_EVIDENCE``.**

Three options were assessed. None is supported by evidence at this point.

.. list-table:: Options
   :header-rows: 1
   :widths: 14 10 46 30

   * - Option
     - Status
     - Finding
     - Blocker
   * - A: sovereign only
     - NOT_SUPPORTED
     - pipeline, licence gate and checkpoints are real; no coding capability
       demonstrated
     - no supervision for any task beyond next-token completion
   * - B: practical open-weight
     - BLOCKED
     - no coding-capable open-weight model could be run
     - Qwen weights absent, Phi-3 partial, host memory-bound
   * - C: hybrid
     - PROVISIONAL
     - keep sovereign checkpoints as research; validate an external model as
       the practical layer
     - the practical layer has not been validated

The hybrid option is the one M5 recommended, and M6 does not rescind it. It
marks it truthfully as **provisional**: a hybrid strategy with an unvalidated
practical half is a plan, not a result.

Precisely why B is blocked:

1. **No local weights.** ``Qwen/Qwen2.5-Coder-0.5B`` and ``1.5B`` are
   licence-clean and completely absent. The 7B repository directory is empty
   (0 MiB). Fetching any of them requires explicit authorisation, which this
   run does not have.
2. **Phi-3 is not a model, it is two partial files.** Both shards are
   ``.incomplete`` (1000 MiB and 960 MiB of 7.12 GB), there is no tokenizer and
   no configuration. It is recorded as ``PARTIAL``, not as "almost available".
3. **The host could not run a coding model even with weights.** 3.91 GiB
   installed, of which only 0.33--0.59 GiB is free. A conservative preflight
   for a 0.5B model is 1.52 GiB and for 1.5B is 4.03 GiB. Both exceed available
   RAM before a single token is generated.

Sovereign track (Track B)
-------------------------

**No training was performed, and none is authorised.** Track B is an engineering
assessment of what a future controlled experiment would require.

Data actually available:

.. list-table:: Corpus and coverage
   :header-rows: 1
   :widths: 40 20 40

   * - Item
     - Value
     - Note
   * - train records
     - 193
     - licence-verified Python library source
   * - train tokens
     - 1,165,217
     - next-token prediction only
   * - validation records
     - 63
     - 340,077 tokens
   * - withheld sources
     - 2
     - ``psf-black`` (27), ``pypa-setuptools`` (135)
   * - task coverage
     - 1 PARTIAL, 11 ABSENT
     - code completion only, implicitly
   * - debugging / refactoring / unit tests
     - ABSENT
     - no prompt-response pairs exist

Scale proposals, with the estimator the training pipeline itself gates on
(``training.preflight``). Every figure is an ``ESTIMATE``; only the host totals
are measured.

.. list-table:: Scale estimates
   :header-rows: 1
   :widths: 18 16 16 14 14 22

   * - Scale
     - Parameters
     - Est. steady state
     - Fits total RAM
     - Fits available RAM
     - Purpose
   * - A-baseline
     - 2,853,280
     - 68.5 MiB
     - yes
     - yes
     - existing, untrained
   * - B-existing-9m
     - 9,359,104
     - 206.8 MiB
     - yes
     - yes
     - measured at ~207 MiB peak
   * - C-modest-30m
     - 23,870,592
     - 484.2 MiB
     - yes
     - no
     - lowest plausible generalisation
   * - D-125m
     - 79,987,968
     - 1,436.5 MiB
     - yes
     - no
     - GPT-2-small class
   * - E-350m
     - 276,544,512
     - 4,807.7 MiB
     - no
     - no
     - requires a GPU

Two verdicts are recorded because they genuinely differ: installed RAM is not
usable RAM. On this host only A and B pass both tests. Presenting D as "fits"
because it fits 3.91 GiB of installed memory would have been technically true
and practically false.

**Recommended next experiment: B-existing-9m.** It is the largest configuration
already authored and preflight-measured, so it isolates the data question from
the compute question. Scaling the model before fixing the data would repeat the
M4 mistake of adding capacity without adding supervision.

Preconditions before any training run is authorised:

1. licence-verified instruction data for at least ``code_generation``,
   ``debugging`` and ``unit_test_generation``;
2. a frozen benchmark covering those tasks, versioned separately from the M5
   benchmark;
3. preflight re-run on the exact training host, with a measured probe rather
   than an estimate.

Validation gates
----------------

.. list-table:: Gate status
   :header-rows: 1
   :widths: 12 58 30

   * - Gate
     - Requirement
     - Status
   * - G1
     - every target task has licence-verified supervision with an isolated
       holdout
     - **FAIL** -- 11 of 12 tasks have no supervision
   * - G2
     - preflight passes with a measured probe on the training host
     - **PASS for A/B only**
   * - G3
     - a frozen benchmark exists before training starts
     - **PARTIAL** -- M5 benchmark is frozen, but covers 4 scored categories
   * - G4
     - functional pass rate rises above the M3/M4 baseline
     - **NOT RUN** -- no training
   * - G5
     - general-code behaviour does not degrade
     - **NOT RUN** -- no training

Reproduction
------------

Every number in this document comes from a persisted artifact.

.. code-block:: text

   python -m evaluation.m6_inventory          # read-only candidate scan
   python -m evaluation.m6_resources --model-id Qwen/Qwen2.5-Coder-0.5B \
       --parameters 494000000
   python -m evaluation.m6_evaluate \
       --candidate quantsmind/m3-base \
       --checkpoint runs/m3_base/checkpoint_best.pt \
       --out-dir runs/m6/m3_base
   python -m evaluation.m6_sovereign --out reports/M6_SOVEREIGN.json
   python -m evaluation.m6_report --runs runs/m6 \
       --out reports/M6_CAPABILITY.json

The report also writes ``reports/M6_CAPABILITY.json`` with every value tagged
``MEASURED``, ``DERIVED``, ``ESTIMATE``, ``UNAVAILABLE``,
``HUMAN_JUDGMENT`` or ``RECOMMENDATION``.

Limitations
-----------

- The benchmark is 26 examples. Every rate is preliminary, and one example moves
  a percentage by several points.
- 17 examples require human review and were never scored. They are outstanding,
  not failed.
- 7 examples share source documents with training material. Leakage is recorded
  in each manifest, not hidden.
- Functional scoring is ``0/0`` for every model: no model produced executable
  code, so functional capability is **untested**, not measured as zero.
- Categories with no examples are ``UNAVAILABLE``, not zero.
- The only external model that could load is an embedding head. It cannot
  follow an instruction, and its compliance hits are formatting artifacts.
- No practical coding-capable model could be obtained inside the licence and
  memory boundary, so the central M6 question remains unanswered.
