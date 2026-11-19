# TAKSH — I7 Release Qualification (Developer Preview)

**Date:** 2026-11-19
**Status:** `CONDITIONAL_PASS` — packaging qualified, capability NOT_EVALUATED, publication NOT authorized.
**Target channel:** developer preview (source release only); no model weights, no capability claim.
**Owner decision required:** publication decision (owner-only, and *not* exercised).

---

## 1. Why this, and why now

Release-qualification preparation was explicitly chosen as the next serial step. It is the only work that is honest today because TAKSH cannot produce a coder yet, and no responsible release can pretend otherwise. It is the work that would let TAKSH become usable later.

This document qualifies the **engineering plumbing** and documents honest limits. It does not qualify the model as a coder.

## 2. Release identity

- Package: `taksh` 1.0.0
- Licence: Apache-2.0 (source only; weights and datasets excluded)
- Authoritative version: `taksh.__version__`
- Provenance: `git` branch/commit to be recorded at release time by the owner
- Configuration: `pyproject.toml`, `pytest.ini`, `conftest.py`, `MANIFEST.in`

## 3. Verified integrity

| Item | Evidence |
|---|---|
| Frozen Python benchmark | SHA-256 `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark | SHA-256 `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| `runs/v180_sft/checkpoint_last.pt` | SHA-256 `9b541036…cb7f666` — unchanged |
| `tokenizer/taksh_code_bpe_v1.json` | SHA-256 `554c8e75…ffcd92`, git blob `fbedf003…` — unchanged |
| `configs/foundation/l060_foundation.yaml` | SHA-256 `01a90464…` — unchanged |
| Local `main` vs `origin/main` | Equal |

## 4. Release-governance compliance (artifact/artifact level)

| Pillar | State |
|---|---|
| Exact release commit | owner records at decision time |
| Branch/remote verification | recorded by release script |
| Inventory with provenance classes | maintained by release tooling |
| Eight-subsystem + DLCAE integrity | preserved; reports unedited |
| Focused test evidence | CI green on merged PRs #4–#8 (see below) |
| Regression evidence | verified on merge into `main` |
| CI evidence + limitations | bare-runner-only; full suite is local/full-environment qualification |
| Frozen benchmark results | 0/30 admitted and public |
| Security & safety | safe checkpoint load, loopback-only default, no external calls; reviewers: owner |
| Resource measurements | measured on the author's CPU host only (load 42.1s, gen ~2.9s) — not a platform claim |
| Known issues & deferrals | recorded in `docs/OPEN_ISSUES.md`, `TAKSH-FOUNDATION-PROGRAMME-STATUS.md` |
| Capability claims | **none**; field status is `NOT_EVALUATED` |
| Release decision | **not exercised** by this document |

## 5. PR evidence to date (main)

| PR | Scope | CI |
|---|---|---|
| #4 | L0/ENG foundation + docs + CI gate | green |
| #5 | Native provider wiring; immutable v0.1/v0.2 artifacts | green |
| #6 | Provider-wired Gradio UI with identity display | green |
| #7 | Aligned-format SFT + negative capability result | green |
| #8 | Native serving qualification (provenance, guards) | green |

## 6. Qualification failures and honest gaps

1. **No capability.** The model cannot generate syntactically valid code. Not revocable by documentation.
2. **Full suite not bare CI-runnable** — requires artefacts deliberately excluded from git. Honest local gate from now on.
3. **Streaming is not shipped** from the native provider path.
4. **Karkain remains DISCOVERED**; `kcc.exe` `0xC0000135`; no tasks executed.
5. **FIM sentinels absent** from tokenizer v1; FIM requires tokenizer v2 (owner decision).
6. **Cross-language benchmarks: zero tasks executed** for non-Python languages.
7. **No model publication authorized** — irrespective of this document.

## 7. Release verdict (this document)

- **Conditional: PASS** — engineering readiness for a source-only developer-preview pipeline is real, testable and reproducible.
- **Capability verdict: NOT_EVALUATED** — frozen benchmark 0/30 stands; native v0.1 0/20 stands; nothing here changes that.
- **GA verdict: BLOCKED** — no capability, no cross-language qualification, no full-suite bare CI, no publication authorization.
- **Publication verdict: NOT AUTHORIZED** — release artefacts (wheel, sdist) are built, inspected and verified but **not uploaded** by this process. Any upload, tag, or GitHub release is owner-only.

## 8. If the owner later chooses to publicize

The minimum authorised sequence (and nothing more):

1. Owner reads and signs the release-readiness decision record.
2. Owner sets version string and declares the edition (e.g., `v1.0.0-developer-preview1`) without touching `taksh.__version__` unless a version bump is intended.
3. Re-run `scripts/prepare_release.py` against the exact commit. Archive inspection must pass.
4. Owner performs the artefact upload (to PyPI/other registry) themselves or under an out-of-band instruction recorded by the owner.
5. Owner publishes release notes that state the capability evidence publicly (not hyperbole).
6. Owner updates `reports/release-evidence/` with exact SHA-256s at publication time.

## 9. Evidence pointer

The authoritative machine-readable release evidence is regenerated at release time by `scripts/prepare_release.py` into `reports/release-evidence/`. This document records the qualification verdict only; the artefact evidence itself lives there.
