# Changelog

All notable changes to Taksh are recorded here. Dates are UTC.

## [1.0.0] - 2026-10-21

### Added

- Unified package metadata for the `taksh` distribution with Apache-2.0
  licensing, a `NOTICE` copyright record, declared runtime dependencies and the
  `taksh` / `taksh-openai` console entry points.
- Version-consistency and release-preparation checks.
- Source-distribution configuration files and release evidence automation.
- Release-readiness documentation and the M7 negative-result record.

### Verified

- Wheel and source distribution build, archive inspection and `twine check`.
- Isolated wheel installation, version identity and CLI checks.
- OpenAI-compatible serving smoke test with the actual M7 SFT checkpoint.
- Regression tests and a Sphinx warnings-as-errors documentation build.

### Capability status

The M7 sovereign capability experiment completed with a negative result:
0/40 locked syntax-valid generations, 0/8 functional tasks, evidence level 0.
Serving HTTP success does not establish output correctness. The software package
and the trained model have different readiness statuses.

### Not included

- Model weights or checkpoints.
- Training or private datasets.
- A model licence or dataset-release licence.
- Production, security or useful-code-generation claims.

### Publication

The release is published only after explicit owner authorisation and a passing
release-gate run. No model weights, training data or dataset-release licence are
included. The earlier `0.7.0` label was an interim M7 documentation target and
was never tagged or released.
