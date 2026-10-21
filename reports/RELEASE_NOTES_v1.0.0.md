# Taksh v1.0.0

Apache-2.0 source release for the completed M7 sovereign capability experiment.

## What this is

- The Taksh (formerly Quantsmind-LLM) source package, training/evaluation/serving
  code, tests, configuration and M7 experiment records.
- An honest negative research result: 0/40 locked syntax-valid generations,
  0/8 functional tasks and evidence level 0.
- Source code licensed under Apache-2.0. See `LICENSE` and `NOTICE`.

## What this is not

- Not a model release. Model weights and checkpoints are not included.
- Not a dataset release. Training and evaluation data remain private.
- Not evidence of coding capability, production readiness or security readiness.
  Serving HTTP success does not make generated output correct.

## Install

PyPI publication is currently blocked on registry credentials, so the package
is not yet available from PyPI. Install the wheel attached to this GitHub
release, or build from the source archive:

```bash
pip install taksh-1.0.0-py3-none-any.whl
taksh --version
```

The wheel contains code and metadata only. It does not contain weights, data,
checkpoints or tokenizer assets.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `taksh-1.0.0-py3-none-any.whl` | 336,332 | `047ad9278dfc95cb8b6eff9b21299dd7cd7ecfe83ea117dc26e237617955d8c9` |
| `taksh-1.0.0.tar.gz` | 418,840 | `341225b9cf0f08ce5accef5f843e24009bc12d54395ce383643294c2f1d15a28` |

## Verification

Release preparation passed: version consistency, wheel/sdist build, archive
inspection, `twine check`, isolated wheel installation, CLI checks, real-checkpoint
serving smoke test, 912 local tests and a warnings-as-errors Sphinx build. A clean
export without private data or weights passed 874 tests with 38 expected skips.

Full evidence is in `reports/release-evidence/RELEASE_EVIDENCE.md` and the release
readiness record.

## Third-party licences

Runtime dependencies remain under their own licences. No third-party code is
vendored. Model weights and training datasets are not relicensed by this release.
