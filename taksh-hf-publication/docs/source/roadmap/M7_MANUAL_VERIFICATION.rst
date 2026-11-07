M7 Manual Verification
======================

Release target: v1.0.0 (Apache-2.0 source release; publication tracked separately)

This page records the pre-release baseline checks. The final tag, GitHub release
and artifact hashes are recorded in
:doc:`../releases/M7_RELEASE_READINESS`.

.. contents:: On this page
   :local:
   :depth: 2

Purpose
-------

These are the checks a human can run to confirm the M7 engineering claims. They
do not turn the negative capability result into a positive one. Passing them
means the experiment was executed and reported honestly; it does not mean the
model can code.

Run from the repository root on Windows PowerShell. On hosts where pytest's
default temporary directory is inaccessible, add a writable ``--basetemp``.

1. Repository identity
----------------------

.. code-block:: powershell

   git branch --show-current
   git rev-parse --short HEAD
   git tag
   git status --porcelain

Expected:

* branch ``develop``;
* HEAD ``4a5a144`` for the baseline record;
* no tags;
* working-tree changes are expected because M7 artifacts are uncommitted.

Recorded result: branch ``develop``, HEAD ``4a5a144``, 0 tags. The working tree
was dirty at baseline and remains dirty because M7 artifacts are local.

2. M5 benchmark integrity
-------------------------

.. code-block:: powershell

   python -m pytest tests/test_m7_gates.py::test_m5_benchmark_is_unchanged -q
   python -m pytest tests/test_m7_gates.py::test_changed_m5_hash_is_rejected -q

Expected: both tests pass. The first proves the benchmark hash matches the
pre-registered hash; the second proves a changed hash is rejected rather than
silently accepted.

Recorded result: both tests passed.

3. Negative-path and gate tests
-------------------------------

.. code-block:: powershell

   python -m pytest tests/test_m7_gates.py tests/test_m7_evaluation.py -q

Expected: all tests pass. These cover:

* a missing evaluation summary is ``NOT_RUN``, not a pass;
* perfect syntax/functional rates cannot override a worse validation loss;
* level 5 is not claimed without human review;
* missing manifests and failed runs fail Gate F;
* blocked preflight and resource overruns fail Gate D;
* the M5 hash is unchanged and changes are rejected;
* syntax scoring rejects prose and wrong function names.

Recorded result: 14 passed.

4. M7 serving compatibility
---------------------------

.. code-block:: powershell

   python -m pytest tests/test_m7_serving.py -q

Expected: 3 tests pass. The test loads
``runs/m7_sft/checkpoint_best.pt`` through
``serving.openai_server.load_state`` and checks:

* model id ``qm-code-m3-9m``;
* step 120;
* 9,359,104 parameters;
* vocab 5,822 and context 512;
* ``GET /v1/models`` metadata;
* a bounded ``POST /v1/chat/completions`` request with no quality assertion.

Recorded result: 3 passed in 12.76 s. The checkpoint loads through the serving
path after the ``base_model_config`` repair.

5. Full regression suite
------------------------

.. code-block:: powershell

   python -m pytest tests -q --basetemp=C:\Temp\pytest-m7-full

Expected: the full suite passes. This includes the existing M2--M6 tests plus
the M7 gate, negative-path and serving tests.

Recorded result: 903 passed in 308.60 s.

6. Gate report inspection
-------------------------

.. code-block:: powershell

   python -c "import json; g=json.load(open('reports/M7_GATES.json')); print([(x['gate'], x['ok']) for x in g['gates']]); print('top_ok=', g['ok'])"
   python -c "import json; o=json.load(open('reports/M7_OUTCOME.json')); print(o['outcome'], o['evidence_level'], o['gates'])"
   python -c "import json; a=json.load(open('reports/M7_AUTHORIZATION.json')); print(a['authorized'], a['inputs']['gates'])"

Expected:

* the post-training report contains Gates F, G and H;
* Gate G is ``false``; F and H are ``true``;
* top-level ``ok`` is ``false``;
* the authorization record confirms Gates A--D passed before training;
* outcome ``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED``;
* evidence level ``0``.

Recorded result:

.. code-block:: text

   [('F', True), ('G', False), ('H', True)]
   top_ok=False
   TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED 0
   {'F': True, 'G': False, 'H': True}
   True {'A': True, 'B': True, 'C': True, 'D': True}

``reports/M7_GATES.json`` is the post-training report and replaces the earlier
pre-training report file. The pre-training gate evidence is preserved in
``reports/M7_AUTHORIZATION.json`` and in the candidate report. This is a known
artifact-layout limitation, not a missing gate result.

7. Evaluation records inspection
--------------------------------

.. code-block:: powershell

   python -c "import json; s=json.load(open('runs/m7_eval/summary.json')); print([(m['model'], m['partitions']['locked']['tasks']['code_generation']['syntax_rate'], m['partitions']['locked']['tasks']['code_generation']['functional_rate']) for m in s['models']])"

Expected: all four models report ``(0.0, 0.0)`` on locked generation.

Recorded result: untrained ``(0.0, 0.0)``; m3_base ``(0.0, 0.0)``;
m4_tuned ``(0.0, 0.0)``; m7_candidate ``(0.0, 0.0)``.

8. Documentation build
----------------------

.. code-block:: powershell

   python -m sphinx -E -W -b html docs/source docs/_build/html

Expected: build succeeds with no warnings. The generated HTML must include the
five M7 pages:

* ``M7_SOVEREIGN_CAPABILITY_EXPERIMENT.html``;
* ``M7_EXPERIMENT_PLAN.html``;
* ``M7_EXPERIMENT_REPORT.html``;
* ``M7_REPRODUCIBILITY.html``;
* ``M7_MANUAL_VERIFICATION.html``.

Recorded result: see the final verification table below.

9. Release check
----------------

.. code-block:: powershell

   git tag
   git log --oneline -1

Expected at the pre-release baseline: no tag and no release commit. Current
release metadata is ``1.0.0`` (final). The tag and publication are created from
the verified release commit and recorded in the release-readiness page.

Recorded baseline result: 0 tags; the tag and publication were not yet
performed when these checks ran.

Final verification table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 48 12 40

   * - Check
     - Result
     - Evidence
   * - Repository identity
     - PASS
     - ``develop`` / ``4a5a144`` / 0 tags
   * - M5 hash unchanged
     - PASS
     - both M5 integrity tests pass
   * - M7 gate and negative-path tests
     - PASS
     - 14 passed
   * - M7 serving compatibility
     - PASS
     - 3 passed; real checkpoint loaded
   * - Gate report
     - PASS
     - G false; F/H true; A--D authorized; level 0
   * - Evaluation records
     - PASS
     - all models 0.0 syntax and functional
   * - Documentation HTML build
     - PASS
     - forced Sphinx ``-E -W`` build; five M7 HTML pages generated
   * - Release check
     - PASS
     - baseline had no tag; final publication tracked separately

If any check fails, report the failure. Do not edit a gate threshold, a locked
record, or an outcome JSON to make the table green.
