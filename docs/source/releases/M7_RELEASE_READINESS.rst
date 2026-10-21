M7 Release Readiness
====================

.. list-table::
   :widths: 26 74

   * - Release target
     - ``v1.0.0`` (Apache-2.0 final source release)
   * - Distribution
     - ``taksh==1.0.0``
   * - Report date
     - 2026-10-21
   * - Decision
     - **PUBLICATION AUTHORIZED**
   * - Verification status
     - **RELEASE GATES PASSED**
   * - Repository state
     - branch ``develop``, release commit recorded below
   * - Capability result
     - negative; evidence level 0

This report closes the M7 packaging and public-release milestone. M7 is the
final milestone for the completed sovereign capability experiment; there is no
M8. The result is a research release, not a model release.

Decision
--------

The package, source tree and documentation passed the release gates and are
approved for publication. Publication is authorized by the owner. The negative
capability result is preserved: M7 does not demonstrate usable code generation
and must not be presented as a production or capability release.

.. list-table:: Release decisions
   :header-rows: 1
   :widths: 30 20 50

   * - Item
     - Decision
     - Basis
   * - Package and source release
     - GO
     - Apache-2.0 wheel and sdist; isolated install; tests and docs build
   * - Capability claim
     - NO-GO
     - 0/40 locked syntax-valid; 0/8 functional; evidence level 0
   * - Production deployment
     - NO-GO
     - no functional coding evidence; security and operational gates unmet
   * - Model weights and data release
     - WITHHELD
     - no release licence for weights; M2 remains ``BLOCKED``
   * - Git tag, package publication, GitHub release
     - AUTHORIZED
     - explicit owner authorization; executed only after the release-gate run

Scope and version reconciliation
--------------------------------

The current release target is ``v1.0.0``. The same version is defined once in
``taksh/__init__.py`` and consumed by package metadata, Sphinx and the
OpenAI-compatible server.

The experiment was originally documented under the interim target ``v0.7.0``.
That label was never tagged or published. It is retained only in historical
experiment records; current release metadata and current M7 documentation use
``v1.0.0``. Historical artifacts keep their original labels so the experiment
remains auditable.

What ``v1.0.0`` contains:

* the Taksh source package and command-line entry points;
* the training, evaluation and serving code;
* the M7 experiment records and machine-readable evidence;
* this release-readiness record;
* the Apache-2.0 ``LICENSE`` and ``NOTICE`` copyright record.

What it does **not** contain or authorise:

* model weights or a weight licence;
* a dataset release or training-data licence;
* a capability, production or security claim.

The source code is Apache-2.0. Third-party runtime dependencies remain under
their own licences; no third-party code is vendored. Model weights and training
datasets remain private and are not relicensed.

Capability evidence
-------------------

.. list-table:: M7 locked result
   :header-rows: 1
   :widths: 42 58

   * - Measure
     - Result
   * - Outcome
     - ``TRAINING_COMPLETED_CAPABILITY_NOT_DEMONSTRATED``
   * - Evidence level
     - **0**
   * - Locked syntax-valid generations
     - **0/40**
   * - Functional tasks
     - **0/8**
   * - Functional cases
     - **0/29**
   * - Candidate common-validation loss
     - **5.9473**
   * - Best control common-validation loss
     - 4.9220 (M4 tuned); M3 base 4.9685; untrained 8.7339
   * - Gate F, training integrity
     - PASS
   * - Gate G, capability evidence
     - **FAIL**
   * - Gate H, independent evaluation
     - PASS for protocol and checkpoint binding; no human review, level 5 unavailable

All four models, including the untrained control, produced 0/40 parseable
top-level functions. The candidate's functional failures were ``NO_CODE``. The
candidate did not beat the controls on the common validation windows and did
not reach evidence level 2.

The full record is in :doc:`../roadmap/M7_EXPERIMENT_REPORT`; the candidate
report is ``reports/M7_CANDIDATE_REPORT.rst``. The result does not support claims
of code generation, instruction following, production readiness or release
readiness for the model.

Packaging and release automation
--------------------------------

Release preparation is reproducible through one command::

   python scripts/prepare_release.py

The command runs the version-consistency check, builds a clean wheel and source
distribution, inspects archive contents, validates metadata and licence files,
runs ``twine check``, installs the wheel into isolated environments, checks the
installed version and CLIs, runs the real-checkpoint serving smoke test when
assets are available, runs regression tests, validates Sphinx with warnings as
errors, and writes evidence to ``reports/release-evidence/``.

It never tags, commits, pushes, merges, uploads to a package index or creates a
GitHub release. The procedure and failure behaviour are documented in
``docs/RELEASING.md``.

Packaging evidence
------------------

Packaging is defined in ``pyproject.toml``:

* distribution name ``taksh``;
* dynamic version from ``taksh.__version__``;
* Apache-2.0 licence metadata and the ``LICENSE`` and ``NOTICE`` files;
* runtime dependencies declared in the package metadata;
* console scripts ``taksh`` and ``taksh-openai``;
* package discovery for ``taksh``, ``data``, ``evaluation``, ``serving``,
  ``tokenizer`` and ``training``.

``MANIFEST.in`` includes the configuration directory, changelog and test
support files in the source distribution. The wheel contains code and metadata;
it does not contain model weights, datasets, checkpoints or tokenizer assets.

Build command::

   python -m build --outdir <temporary-directory> .

.. list-table:: Build artifacts
   :header-rows: 1
   :widths: 34 14 16 36

   * - Artifact
     - Bytes
     - SHA-256
     - Contents
   * - ``taksh-1.0.0-py3-none-any.whl``
     - 336,332
     - ``047ad9278dfc95cb8b6eff9b21299dd7cd7ecfe83ea117dc26e237617955d8c9``
     - importable code, entry points, licence metadata
   * - ``taksh-1.0.0.tar.gz``
     - 418,840
     - ``341225b9cf0f08ce5accef5f843e24009bc12d54395ce383643294c2f1d15a28``
     - source, tests, configuration files, changelog and build metadata

Isolated wheel check
~~~~~~~~~~~~~~~~~~~~

The wheel was installed with ``--no-index --no-deps`` into a clean temporary
virtual environment with no inherited dependencies. ``taksh`` imported from
that environment's ``site-packages``, not from the repository, and
``taksh --version`` printed ``Taksh 1.0.0 (final)``.

A second temporary virtual environment with ``--system-site-packages`` was used
for the dependency-bearing serving check. This is an installation and serving
check, not a dependency-resolution test.

Observed from the installed environments:

* ``taksh --version`` printed ``Taksh 1.0.0 (final)``.
* ``taksh-openai --help`` exposed ``--ckpt``, ``--tokenizer``, host, port and
  remote-binding options.
* ``taksh`` and ``serving`` imported from the environment's ``site-packages``,
  not from the repository.
* the real M7 SFT checkpoint loaded: step 120, 9,359,104 parameters, vocabulary
  5,822, context 512, CPU.
* ``GET /health`` returned 200 with ``version: 1.0.0`` and
  ``model_loaded: true``.
* ``GET /v1/models`` returned 200 with the loaded model ID
  ``qm-code-m3-9m`` and metadata.
* ``POST /v1/chat/completions`` returned 200 with four completion tokens. The
  generated text was not usable code, consistent with the negative capability
  result.

Serving limitations are recorded rather than hidden:

* the wheel does not include a checkpoint, tokenizer or model configuration;
  the serving check used the repository's checkpoint and a copy of
  ``configs/`` because the checkpoint records a relative configuration path;
* the serving environment inherited system dependencies, so this is an
  installation check, not a dependency-resolution test;
* loading a checkpoint may fall back from ``weights_only=True`` and must use a
  trusted startup path; checkpoints are executable pickle data;
* the completion was a serving smoke test only and is not capability evidence.

Verification
------------

Tests and documentation were checked before publication approval.

.. list-table:: Verification results
   :header-rows: 1
   :widths: 38 62

   * - Check
     - Result
   * - Version consistency
     - ``python scripts/check_version_consistency.py``: PASS
   * - Full test suite
     - ``python -m pytest tests -q``: **912 passed** in 220.59 s
   * - Clean source-tree tests
     - exported release tree without private data/weights: **874 passed, 38 skipped**
   * - Release-focused tests
     - **56 passed** for release metadata, M7 gates, M7 serving and OpenAI serving
   * - Package build
     - sdist and wheel built successfully with setuptools 84.0.0
   * - Archive inspection
     - wheel 105 members, sdist 223 members; no forbidden weights, data, caches or secrets
   * - Artifact metadata
     - ``python -m twine check``: PASSED for wheel and sdist
   * - Clean wheel installation
     - installed with no dependencies; version and CLI identity passed
   * - Real checkpoint serving
     - health, model discovery and chat completion returned 200
   * - Sphinx HTML build
     - ``python -m sphinx -W -b html docs/source <temporary-output>``: passed
   * - Evidence report
     - ``reports/release-evidence/RELEASE_EVIDENCE.md`` and ``.json``

Licence and data status
-----------------------

* The repository source is Apache-2.0; the licence file is included in the
  wheel metadata and the source distribution.
* The M7 training data used seven licence-verified sources; 162 records were
  withheld pending licence review.
* The broader M2 dataset release remains ``BLOCKED``. Two nested-licence
  exclusions and human source-authorisation decisions remain open; see
  :doc:`../roadmap/M2_RELEASE_AND_GATE`.
* No licence is granted here for model weights. No weights or datasets are
  published by this source release.
* The M2, M3, M4, M5 and M6 records retain their original licences and
  limitations.

Release scope and remaining withholdings
----------------------------------------

The Apache-2.0 source release is approved and authorized. The following remain
withheld and are not released by ``v1.0.0``:

* model weights and checkpoints;
* training datasets and a training-data licence;
* the broader M2 dataset release, which remains ``BLOCKED`` on two nested-licence
  exclusions and human source-authorisation decisions;
* any capability, production or security claim. The model decision remains
  **NO-GO** because M7 produced 0/40 locked syntax-valid generations, 0/8
  functional tasks and evidence level 0.

These dataset and weight restrictions do not block the Apache-2.0 source-code
release. They block any weight or dataset publication.

Git and release actions
-----------------------

At the start of release verification the repository was on branch ``develop``
at commit ``4a5a144``, with no tags and a large pre-existing uncommitted
working tree. Release preparation added the package metadata, version module,
licence notice, release tests, naming updates and documentation directly to the
working tree without rewriting history.

The release commit, ``v1.0.0`` tag, GitHub release and any package-registry
publication are created only from the verified release commit after the final
gate run. Their exact identifiers and artifact hashes are recorded in the final
release report and `reports/release-evidence/`.

``reports/M7_AUTHORIZATION.json`` authorises the M7 training inputs. It is not
itself a publication approval; publication is authorized separately by the
owner.

No model capability, production readiness or weight-release claim is made by
this decision.
