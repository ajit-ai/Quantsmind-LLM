Testing Records
===============

Milestone testing documentation lives here in RST, which is the authoritative
format for this repository's Sphinx documentation system (``docs/source``).

Naming convention
-----------------

.. code-block:: text

    M2_<SLICE>_TEST_PLAN.rst
    M2_<SLICE>_TEST_REPORT.rst

Example: ``M2_4_DATASET_SCHEMA_TEST_PLAN.rst`` /
``M2_4_DATASET_SCHEMA_TEST_REPORT.rst``.

Rules
-----

* A test report records **only evidence that actually exists** on the date it is
  written. Historical reports are never fabricated or back-dated.
* Passing automated tests is **Layer A** only. See
  :doc:`../roadmap/M2_EXECUTION` for the four validation layers and the six
  acceptance states.
* Acceptance state is stated per slice; it is never collapsed into a single
  "PASS".

Contents
--------

.. toctree::
   :maxdepth: 1

   M2_4_DATASET_SCHEMA_TEST_PLAN
   M2_4_DATASET_SCHEMA_TEST_REPORT