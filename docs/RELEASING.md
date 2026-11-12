# Taksh Release Procedure

This procedure prepares and verifies a package release. It never publishes.
A Git tag, GitHub release, package-index upload, push or merge requires separate
explicit owner authorisation after the evidence is reviewed. For `v1.0.0`, that
authorisation was granted and the published identifiers are recorded in
`source/releases/M7_RELEASE_READINESS.rst`.

## Authoritative version

`taksh.__version__` in [`../taksh/__init__.py`](../taksh/__init__.py) is the only
authoritative version. Packaging, Sphinx and the OpenAI-compatible server consume
it. Release metadata is deliberately dynamic to prevent drift.

The current release is `1.0.0` with status `final`. Historical
`0.7.0` M7 documentation is not a released version. Version policy is documented
in [VERSIONING.md](VERSIONING.md).

## Prerequisites

- Python 3.11 or newer for release tooling.
- Git available on `PATH`.
- `build`, `twine` and `sphinx` installed in the release environment.
- Runtime dependencies installed for the isolated serving check.
- The M7 checkpoint and tokenizer if the serving smoke test is required:
  - `runs/m7_sft/checkpoint_best.pt`
  - `tokenizer/qm_code_bpe_v0.json`

The checkpoint is not shipped in the package. The serving check is skipped when
those assets are unavailable, and the evidence report records the skip.

## One-command preparation

From the repository root:

```bash
python scripts/prepare_release.py
```

This command is read-only with respect to Git and publication services. It:

1. Runs the version-consistency check.
2. Removes derived `build/` and `taksh.egg-info/` directories.
3. Builds a wheel and source distribution in a temporary directory.
4. Inspects both archives for weights, private data, caches, secrets and local
   paths.
5. Validates package name, version, licence metadata and licence files.
6. Runs `twine check`.
7. Copies the artifacts to `dist/`.
8. Installs the wheel into a clean virtual environment and checks the installed
   version and `taksh --version`.
9. Installs the wheel into a dependency-inheriting virtual environment and checks
   `taksh-openai --help`.
10. Runs the installed-package serving check against the actual M7 checkpoint
    when available.
11. Runs `python -m pytest tests -q`.
12. Runs `python -m sphinx -W -b html docs/source <temporary-output>`.
13. Writes machine-readable and Markdown evidence to
    `reports/release-evidence/`.

The script exits non-zero and records the verdict `BLOCKED` on any required
failure. It never tags, commits, pushes, merges, uploads or creates a GitHub
release.

The clean virtual environment contains only the wheel and standard library, so
it proves installation and version identity independently of the source tree.
The serving environment uses `--system-site-packages` because runtime
dependencies are already installed and are not re-downloaded; this limitation is
recorded in the evidence.

## Faster checks

Skip expensive stages during development:

```bash
python scripts/prepare_release.py --skip-tests --skip-docs --skip-serving
```

Skipped required stages produce `PARTIALLY COMPLETE`, not a full release
verdict. CI uses a reduced run after the full unit suite and uploads the evidence
artifact.

Require a clean tree before preparing an actual release:

```bash
python scripts/prepare_release.py --require-clean
```

## Version consistency only

```bash
python scripts/check_version_consistency.py
```

It fails if `taksh.__version__`, dynamic package metadata, Sphinx configuration,
server declarations, CLI output, README, changelog, version history, current M7
docs or release-readiness records disagree. Historical `0.7.0` references are
checked only as historical records and must state that they were never released.

## Artifacts and evidence

- Wheel and source distribution: `dist/`
- Evidence JSON: `reports/release-evidence/RELEASE_EVIDENCE.json`
- Evidence summary: `reports/release-evidence/RELEASE_EVIDENCE.md`

The package excludes model weights, checkpoints, tokenizer assets, private
datasets and caches. The source distribution includes configuration files,
tests and release metadata.

## Publication gate

For `v1.0.0`, the owner authorized publication after reviewing the evidence.
The publication steps are executed only from the verified release commit:

```text
git tag -a v1.0.0 <verified-commit> -m "Taksh v1.0.0"
git push origin develop
git push origin v1.0.0
gh release create v1.0.0 --title "Taksh v1.0.0" --notes-file <notes>
python -m twine upload dist/*
```

The preparation script itself never tags, pushes or uploads. The Git tag,
package version and GitHub release must all be `v1.0.0` / `1.0.0`. The package
release must not be described as a model capability or production release.
Model weights and private datasets remain unpublished.

## Distribution channels beyond PyPI

PyPI and GitHub source releases cover the **package**, not a model or dataset.
The channel-by-channel plan for Hugging Face, Cline, OpenCode, OpenRouter and the
local OpenAI-compatible server is in
[roadmap/TAKSH-DISTRIBUTION-AND-PUBLICATION-PLAN.md](roadmap/TAKSH-DISTRIBUTION-AND-PUBLICATION-PLAN.md).
Its machine-readable evidence is generated by
`python -m evaluation.publication_readiness`.

As of 2026-11-12, no model, dataset or hosted-API channel is ready. The canonical
checkpoint is missing, tool calling is not implemented, and publication is not
authorized for this branch.
