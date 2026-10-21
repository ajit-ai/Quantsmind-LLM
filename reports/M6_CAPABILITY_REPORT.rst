M6 Local Capability Validation Report
=====================================

Target version: v0.6.0 (target only -- no release was cut)

.. contents:: On this page
   :local:
   :depth: 2

Verdict
-------

**STATUS: ``INSUFFICIENT_EVIDENCE``.**

M6 was asked whether materially useful coding capability can be demonstrated
from a model running locally under this project's hardware and licence
constraints. It cannot, and the reason is not model quality: **no practical,
coding-capable, open-weight model could be obtained inside the boundary**, and
the two project checkpoints remain unable to produce executable code.

The runtime contract, resource preflight, licence gate, evaluation bridge and
sovereign-track assessment are complete and tested. What is missing is a
candidate to point them at.

Recommendation: **``INSUFFICIENT_EVIDENCE``**. Option A (sovereign only) is
``NOT_SUPPORTED``; Option B (practical open-weight) is ``BLOCKED``; Option C
(hybrid) remains ``PROVISIONAL``.

Baseline
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Value
   * - Branch / HEAD
     - ``develop`` / ``4a5a144``; 0 tags; no commits made during M6
   * - Working tree at start
     - 149 dirty entries (37 modified, 112 untracked), all preserved
   * - RAM
     - 3.91 GiB installed; 0.33--0.59 GiB available (measured, fluctuating)
   * - CPU
     - Intel i7-8550U, 4 cores / 8 threads; CUDA absent
   * - Python / torch
     - 3.14.7 / 2.13.0+cpu
   * - transformers
     - 5.16.1
   * - Benchmark
     - ``evaluation/m5_benchmark.jsonl``; 26 examples; 23 accepted, 3 rejected
   * - Benchmark hash
     - ``52ea2ed5db839b22`` (identical for all M6 runs)

Candidate inventory
-------------------

Discovery was a read-only scan of the Hugging Face cache. No network access, no
model code executed, no weights downloaded.

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 14 30

   * - Candidate
     - Licence
     - Head
     - Acquisition
     - Finding
   * - ``quantsmind/m3-base``
     - owned
     - causal_lm
     - PRESENT
     - evaluated; level 3
   * - ``quantsmind/m4-tuned``
     - owned
     - causal_lm
     - PRESENT
     - evaluated; level 2
   * - ``all-MiniLM-L6-v2``
     - Apache-2.0
     - embedding
     - PRESENT
     - evaluated; level 2 capped
   * - ``bert-base-uncased``
     - Apache-2.0
     - masked_lm
     - PRESENT
     - ineligible: cannot generate
   * - ``Qwen2.5-Coder-1.5B``
     - Apache-2.0
     - causal_lm
     - NOT_PRESENT
     - no weights; download not authorised
   * - ``Qwen2.5-Coder-0.5B``
     - Apache-2.0
     - causal_lm
     - NOT_PRESENT
     - no weights; download not authorised
   * - ``Qwen2.5-Coder-7B``
     - Apache-2.0
     - causal_lm
     - PARTIAL
     - empty snapshot, 0 MiB
   * - ``Phi-3-mini-4k-instruct``
     - MIT
     - causal_lm
     - PARTIAL
     - both shards ``.incomplete``; no tokenizer or config

Licences were verified from primary sources and the exact URLs are recorded in
``evaluation/m6_licensing.py``. Verification is kept separate from usability:
Qwen is licence-clean and still unavailable here.

Measured results
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 14 14 20

   * - Metric
     - M3
     - M4
     - MiniLM
     - Denominator
     - Evidence kind
   * - Syntax valid
     - 1/20
     - 0/20
     - 0/20
     - 20
     - MEASURED
   * - Functional pass
     - 0/0
     - 0/0
     - 0/0
     - 0 executable
     - UNAVAILABLE
   * - Compliance
     - 3/6
     - 4/6
     - 4/6
     - 6 rules
     - MEASURED (non-substantive)
   * - Structured
     - 0/0
     - 0/0
     - 0/0
     - 0 scorable
     - UNAVAILABLE
   * - Human review
     - 17 outstanding
     - 17 outstanding
     - 17 outstanding
     - 17
     - UNAVAILABLE
   * - Evidence level
     - 3
     - 2
     - 2 (cap)
     - 1--7
     - DERIVED

Every compliance pass was vacuous: ``non_empty``, ``no_fences``, or a one-line
stub satisfying ``max_lines``. The manifests now carry a
``compliance_evidence`` block counting substantive versus vacuous passes, and
the level assessor will not credit level 5 without a substantive pass.

No composite score is produced. It would have merged three incomparable facts:
M4 is more "compliant" than M3 while producing less parseable Python, and an
embedding model matched a language model on compliance purely through bracket
formatting.

Capability coverage
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 10 46

   * - Category
     - Status
     - Examples
     - Reason
   * - code_completion
     - SCORED
     - 14
     - syntax only
   * - code_generation
     - SCORED
     - 6
     - syntax only
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
     - requires human review
   * - sql
     - UNAVAILABLE
     - 2
     - both require human review
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

Strategy decision
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Option
     - Status
     - Basis
   * - A: sovereign only
     - NOT_SUPPORTED
     - pipeline and licence gate are real; no functional pass was ever observed;
       11 of 12 tasks have no supervision
   * - B: practical open-weight
     - BLOCKED
     - Qwen absent, Phi-3 partial, 7B empty; a 0.5B model needs 1.52 GiB of a
       host with under 0.6 GiB free
   * - C: hybrid
     - PROVISIONAL
     - the split is sensible, but its practical half has never been validated

Why the practical track is blocked
----------------------------------

1. **Weights.** ``Qwen/Qwen2.5-Coder-0.5B-Instruct`` and ``1.5B`` are
   licence-clean and absent. The 7B cache directory is empty. Downloading is
   outside the authorisation for this run.
2. **Phi-3.** Both shards are ``.incomplete`` (1000 MiB and 960 MiB of 7.12 GB),
   with no tokenizer and no configuration. It is not a usable artifact.
3. **Memory.** Conservative preflight: 0.5B -> 1.52 GiB, 1.5B -> 4.03 GiB. The
   host reports 0.33--0.59 GiB available. The model would be refused before it
   loaded, and the preflight correctly records that as ``BLOCKED``.

The external run that did happen, ``all-MiniLM-L6-v2``, is an embedding model.
Its 4/6 "compliance" is a formatting artifact and is capped at level 2.

Sovereign track (Track B)
-------------------------

**No training was performed. No training was authorised.**

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Item
     - Value
     - Note
   * - Training records
     - 193
     - licence-verified Python source
   * - Training tokens
     - 1,165,217
     - next-token prediction
   * - Validation records / tokens
     - 63 / 340,077
     - held out
   * - Withheld sources
     - 2
     - ``psf-black``, ``pypa-setuptools``
   * - Task coverage
     - 1 PARTIAL, 11 ABSENT
     - completion only, implicitly
   * - Correction
     - required
     - the M4 synthetic instruction construction is not sufficient data

Scale estimates (all ``ESTIMATE`` except host totals):

.. list-table::
   :header-rows: 1
   :widths: 20 18 18 16 16 12

   * - Scale
     - Parameters
     - Steady state
     - Fits total
     - Fits available
     - Status
   * - A-baseline
     - 2,853,280
     - 68.5 MiB
     - yes
     - yes
     - ESTIMATE
   * - B-existing-9m
     - 9,359,104
     - 206.8 MiB
     - yes
     - yes
     - ESTIMATE (207 MiB measured)
   * - C-modest-30m
     - 23,870,592
     - 484.2 MiB
     - yes
     - no
     - ESTIMATE
   * - D-125m
     - 79,987,968
     - 1,436.5 MiB
     - yes
     - no
     - ESTIMATE
   * - E-350m
     - 276,544,512
     - 4,807.7 MiB
     - no
     - no
     - ESTIMATE

Recommended next experiment: **B-existing-9m**, the largest configuration
already authored and preflight-measured. It isolates the data question from the
compute question.

Preconditions: licence-verified instruction data for code generation, debugging
and unit-test generation; a frozen benchmark covering those tasks, versioned
separately from the M5 benchmark; a measured preflight on the exact training
host; explicit human authorisation.

Gates
-----

.. list-table::
   :header-rows: 1
   :widths: 10 56 34

   * - Gate
     - Requirement
     - Status
   * - G1
     - licence-verified supervision for every claimed task
     - FAIL (1 of 12)
   * - G2
     - measured preflight on the training host
     - PASS for A/B
   * - G3
     - frozen benchmark before training
     - PARTIAL (4 scored categories)
   * - G4
     - functional rate above the M3/M4 baseline
     - NOT RUN
   * - G5
     - no general-code regression
     - NOT RUN

Artifacts
---------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Path
     - Contents
   * - ``evaluation/m6_runtime.py``
     - local-model contract and two runtimes
   * - ``evaluation/m6_resources.py``
     - conservative RAM/disk preflight
   * - ``evaluation/m6_inventory.py``
     - read-only candidate discovery
   * - ``evaluation/m6_licensing.py``
     - verified licences with source URLs
   * - ``evaluation/m6_evaluate.py``
     - M5 harness bridge plus compliance evidence
   * - ``evaluation/m6_capability.py``
     - categories, evidence levels, substantive compliance
   * - ``evaluation/m6_sovereign.py``
     - Track B assessment
   * - ``evaluation/m6_report.py``
     - consolidated report and strategy decision
   * - ``evaluation/m6_cli.py``
     - stage dispatcher
   * - ``runs/m6/m3_base``, ``m4_tuned``, ``minilm_external``
     - three evaluated runs with results and manifests
   * - ``reports/M6_CAPABILITY.json``
     - machine-readable consolidated report
   * - ``docs/source/roadmap/M6_EXECUTION.rst``
     - published milestone documentation

Verification
------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Check
     - Result
   * - M6 tests
     - 87 passed
   * - M5 regression tests
     - 76 passed
   * - Full suite
     - 886 passed
   * - Sphinx ``-W``
     - build succeeded, no warnings
   * - Hosted APIs / credentials
     - none used (``hosted_api_used: false`` in every manifest)
   * - Training
     - none performed

Limitations
-----------

- 26 examples; every rate is preliminary.
- 17 examples require human review and were not scored.
- 7 examples share source documents with training material; recorded, not hidden.
- Functional capability is untested, not zero.
- Uncovered categories are ``UNAVAILABLE``, not zero.
- The central question -- whether a locally runnable open-weight model is useful
  here -- remains unanswered because no such model could be obtained.
