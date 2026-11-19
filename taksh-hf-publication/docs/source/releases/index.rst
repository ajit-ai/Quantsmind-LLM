Releases
========

Release and milestone-gate records live here in RST.

The current package/release target is ``v1.0.0``, the Apache-2.0 source release.
The Git tag and GitHub release were published on 2026-10-21. The PyPI upload is
blocked on registry credentials; no package-index release exists. The earlier
M7 experiment used ``v0.7.0`` as an interim documentation target and was never
released. Model weights and training datasets remain private.

What exists is a published research source release:

* **M2** — dataset release artifacts and the readiness gate, which reports
  ``M2 BLOCKED``. See :doc:`../roadmap/M2_RELEASE_AND_GATE`.
* **M3** — a base-model checkpoint trained from random initialisation on
  licence-verified data, plus a machine-readable run manifest. See
  :doc:`../roadmap/M3_EXECUTION`.
* **M7** — a completed, negative capability experiment with reproducible
  evidence; the v1.0.0 packaging and public-release decision are recorded in
  :doc:`M7_RELEASE_READINESS`.

Training output is not a release. The ``v1.0.0`` source release is separate from
the model: weights and datasets remain private, and the negative capability
result is disclosed in :doc:`M7_RELEASE_READINESS`.

Contents
--------

.. toctree::
   :maxdepth: 1

   M7_RELEASE_READINESS