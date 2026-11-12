# TAKSH — CI/CD and Documentation Recovery (2026-11-11)

**Integration note (2026-11-12).** This document is the recovery record for
`feature/l060-foundation-model-qualification`. The present branch
(`integration/taksh-cicd-recovery`, later merged into `develop`) is a minimal,
isolated port: CI fixes + recovery docs + seven historical-failure tests + workflow
corrections. Milestone reports referenced below (`docs/L100ENG*`, the master roadmap,
the machine registry) do **not** live on this branch; they live on the feature branch.
The seven Historical Failure Set 001 tests were re-verified on this branch at develop
base HEAD (`a654e5d` + ported fixes): **7 passed**.
**Frozen benchmarks, the multilingual benchmark, `runs/v180_sft/checkpoint_last.pt`,
`configs/foundation/l060_foundation.yaml` and `tokenizer/taksh_code_bpe_v1.json` do not
exist on this branch** — they are feature-branch evidence artifacts; `BLOCKED_ARTIFACT_ABSENT`
is reported, not FAIL.
See the PR body for the definitive current verification statement.

**Scope:** recovery of TAKSH's engineering delivery infrastructure and documentation. **Not** a product release, not an ENG milestone, not a capability qualification. No ENG4 is created.
**Baseline:** `d2f0d1443ddb959429a2232a8d240457c2f90590` on `feature/l060-foundation-model-qualification` (verified == remote).
**Owner decisions enforced:** `TAKSH-OWNER-DECISION-2026-11-11` — A1.1-C (preparation-only), A2-E (no compute). No training, checkpoints, data, compute, spending, publication or merge is performed or authorized here.

---

## 1. Baseline (verified)

| Item | Result |
|---|---|
| Branch | `feature/l060-foundation-model-qualification` |
| HEAD | `d2f0d1443ddb959429a2232a8d240457c2f90590` (== `origin/…` via `git ls-remote`) |
| Upstream tracking | none configured (pushes use explicit branch) |
| Pre-existing owner changes | six CRLF files preserved untouched: `serving/code_chat.py`, `serving/local_inference.py`, `taksh-hf-publication/upload_manifest.json`, `training/manifest.py`, `training/pretrain.py`, `training/sft.py` |
| Host | Python 3.14.7 · Node v24.18.1 · git 2.51.0 · Sphinx 7.4.7 · ~4.2 GB RAM (very low at audit time, ~97 MB free) |
| Repo | 84.13 MiB; no remote tags modified |
| Tags | `v1.0.0` exists; this recovery creates/modifies none |
| develop / main | untouched |

## 2. Workflow inventory (complete)

| File | Name | Triggers | Permissions | Dispatchable? |
|---|---|---|---|---|
| `.github/workflows/ci.yml` | CI | push→develop/main + PRs | **added `contents: read` this recovery** | no |
| `.github/workflows/ci-l0704-manual-qualification.yml` | CI (L0.7.4 manual qualification) | `workflow_dispatch` only (default ref = feature branch) | `contents: read` | **no** — file not on default branch (verified via API and `gh workflow list`) |

`gh workflow list --all` → `CI` only; via API `403`-free anonymously as well. `main` and `develop` contain **only** `ci.yml` (sha `27cf4649c2`, 6137 bytes). `.github/` contains nothing else. `CODEOWNERS`/`CONTRIBUTING.md` present but no release workflows exist. No Dependabot. No reusable workflows or referenced composite actions.

## 3. Historical CI review (from Actions history; real run IDs)

Actions history (66 total runs) terminates in failure on `develop` (2026-10-21 and 2026-11-06).
Sample run `42144960532` (2026-11-06T01:36:29Z, PR #1) — job `unit-and-smoke` (32m23s):

```
✓ guard (3s)
✗ Unit tests (failure → whole job cancelled at the 30m budget)
✗ everything else (skipped: release tooling, version check, smoke pipeline,
   resume-from-checkpoint, multimodal fit-check, tokenizer round-trip,
   model config validation, SFT dataset validation, eval harness import,
   serving checks, OpenAI route checks, legacy /load exposure checks)
```

The **7 failing tests are exactly Historical Failure Set 001** (owner-supplied). The fixes landed on this feature branch (L0.7.3/L0.7.4, 2026-11-11) but **never reached `develop`** — `ci.yml` triggers `push→develop` only, and no merge exists (owner restriction). Hence every historical failure on `develop` reproduces the pre-fix state until the owner merges.

## 4. Remote CI execution evidence (honest)

| State | Evidence |
|---|---|
| Workflow file exists | yes — both files in this branch |
| Workflow syntactically valid | yes — `actionlint`-grade YAML parse; assertions in `tests/test_ci_workflows.py` |
| Local equivalent passes | Historical Set 001: 7/7 at HEAD (132 s); ENG1–3 focused chain: 162/162 (170 s) |
| Triggered remotely | **no** — `CI` fires only on push develop/main + PRs; the L0.7.4 manual workflow is invisible to `workflow_dispatch` (verified: `gh workflow list --all`,
| `gh workflow view`, anonymous + authenticated API |
| Completed successfully remotely | not for this branch |
| Required CI checks qualified | **no** |

**REMOTE_CI_QUALIFICATION = BLOCKED_OWNER_ACTION_REQUIRED.**

Owner action needed (exact): on the **default branch**, either (a) add `feature/l060-foundation-model-qualification` to
the `push:` trigger list in `ci.yml`, or (b) merge/dispatch `ci-l0704-manual-qualification.yml` onto the default branch, or
(c) enable branch workflows from this branch (owner decision — modifying `main` is not within agent authority).
No probe of scope `workflow` was used to alter protection or dispatch anything; only read-only inspection was performed.
A failed `gh workflow view "CI (L0.7.4 manual qualification)"` (workflow not found) is the recorded proof, as is the
absence from `gh workflow list --all`.

## 5. CI architecture

**Today (post-recovery):** `CI` (guard + unit-and-smoke + release packaging + smoke training + route checks; default branches only) — unchanged semantics, now least-privilege (`contents: read`).
**`CI (manual qualification)`** (owner-dispatchable gate): focused L0.6.0 → L1.0.0-ENG3 chain, frozen-benchmark hash pinning, 45-minute budget, explicit concurrency non-cancellation. **Dispatch requires owner action** (see §4).
**Static regression coverage:** `tests/test_ci_workflows.py` — parses both workflow files and enforces: parse, least privilege, dispatch-only semantics of the manual workflow, branch targets, guard dependency, concurrency, timeouts, absence of publish/deploy/release steps, frozen hash visibility, and that the manual chain covers every milestone suite.
**Recommended target (owner decision):** one `push`-triggered fast workflow for the feature branch (or default after merge), with the ENG3 suites in the focused chain; avoid proliferating overlapping workflows.

## 6. Python/runtime compatibility

| Runtime | Evidence |
|---|---|
| 3.14.7 (this host) | executed everywhere in this milestone; full L0→L1 focused chain green |
| 3.11 (ci.yml target) | **never executed on this host**; only stdlib-static evidence existed; `requirements.txt` and current code have no syntax beyond 3.11 support per focused runs on 3.14 |
| `pyproject.toml` | `requires-python` not declared or stale — check during release preparation |

**Unsupported claim prohibition:** no version here is declared "runtime-qualified on 3.11". Documentation will keep this explicit.

## 7. Documentation audit

Audited: README, STATUS, ROADMAP chain (git trees), master roadmap + release governance, `docs/source` (RST/Sphinx index), all `docs/*report*.md` (L0.6→ENG3), RULEBOOK, OPEN_ISSUES, PAUSED, changelogs, taksh release metadata.

**Verified-accurate (kept unchanged):**
- `docs/TAKSH-RELEASE-GOVERNANCE.md` (package version ≠ milestone gates — correct and explicit).
- `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md` + `TAKSH-ROADMAP-CHANGELOG.md` + `configs/roadmap/taksh-v3-roadmap.yaml` (ENG1/2/3 entries honest; no false claims).
- `docs/L100ENG*_*.md` milestone reports — ENG1/2/3 explicitly state engineering-only qualification, MODEL UNAVAILABLE, capability NOT_EVALUATED.
- `docs/source/*.rst` — M7 negative-result record accurate; nothing to invent.

**Corrected (this recovery):**

| File | Correction |
|---|---|
| `README.md` | Additive 2026-11-11 TAKSH roadmap status banner: v1.0.0 GA **NOT established**; engineering-qualified milestones listed; canonical 111M missing; 0/30 stands; `v180_sft` never substituted; Karkain DISCOVERED; A1.1-C/A2-E enforced; develop/main untouched |
| `docs/STATUS.md` | Additive 2026-11-11 TAKSH status banner with the same invariants; historical M7-era banners preserved verbatim |

**Documentation build (Sphinx):** `sphinx-build -b html docs/source <out>` — **exit 0**, 98 pages, zero warnings; output in `C:\Windows\Temp\opencode\sphinx-out3` (outside repo; nothing committed). No documentation deploy is performed.

**Unfixed (documented as defects; owner actions):**

| Item | Status |
|---|---|
| Sphinx TOC covers only M2→M7 narrative; none of L0.6.x→ENG3/DLCAE embedded in `docs/source` | DEFERRED (owner content decision) |
| `docs/ROADMAP.md` (M0–M7) vs `configs/roadmap/taksh-v3-roadmap.yaml` authority — the latter is current-milestone program; the former is historical authority for v1.0.0 baseline | accepted as-designed by roadmap changelog Revision 1.0; no rewrite |
| `pyproject.toml` requires-python hygiene | OPEN (owned by release tooling) |

## 8. Version and release consistency

| Check | Result |
|---|---|
| `scripts/check_version_consistency.py` | **PASS** — `taksh.__version__` = 1.0.0 ("final"), authoritative, consistent |
| Changelog | `[1.0.0] - 2026-10-21` only; roadmap history lives in `TAKSH-ROADMAP-CHANGELOG.md` (accepted practice) |
| Package release state | `release_status="final"` (packaging only; explicitly NOT a milestone gate — see release governance) |
| tags / releases | `v1.0.0` exists; **none created here**; no PyPI publish |

**No version bump, tag, release, publication or merge was performed or permitted in this recovery.**

## 9. Test evidence (commands + results)

| Command | Result |
|---|---|
| `python -m pytest tests/test_m5_adapters_syntax.py::test_external_adapter_does_not_download_by_default tests/test_m6_sovereign.py::test_available_ram_is_the_binding_constraint tests/test_v120_python_data.py::test_g7_manifest_has_no_timestamp_or_absolute_path tests/test_v150_coding_foundation.py::test_geometry_inspector_reports_a_matching_triple tests/test_v150_coding_foundation.py::test_tokenizer_artifact_and_model_geometry_are_unchanged tests/test_v160_authorization_and_baseline.py::test_g10_geometry_inspector_accepts_the_s2_triple tests/test_v180_corpus_qualification.py::test_sampled_decoding_is_reproducible_with_a_seed -q` | **7 passed in 132.29s** |
| `python -m pytest tests/test_l100eng_dlcae_runtime.py tests/test_l100eng2_semantic_validation.py tests/test_l100eng3_end_to_end.py -q` | **162 passed in 170.00s** |
| `python -m pytest tests/test_ci_workflows.py -q` | see §13 receipt |
| Full `pytest tests` | **not executed** (host memory ~97 MB free and ~170 s+ per ENG chain; multi-hour native execution impossible here; remote CI is blocked — see §4) |
| `scripts/check_version_consistency.py` | PASS |
| `python -m sphinx -b html docs/source <out>` | PASS (exit 0, 98 pages) |

## 10. CD/publication/deployment status

**Unauthorized and unperformed:** no GitHub Pages deploy, no PyPI, no Hugging Face, no artifacts published, no release-evidence uploaded as a distribution. Sphinx build occurred in a local `C:\Windows\Temp\opencode\sphinx-out3` only. CI runs artifacts stay internal.

## 11. Remaining blockers and exact owner actions

1. **Remote CI on the feature branch** — owner must add a `push` trigger (or default-branch merge of the manual workflow). Evidence in §4 (`gh workflow list --all`, `gh workflow view`, run history IDs in §3).
2. **develop failures** — the seven Historical Failure Set 001 remain unmerged into `develop`; owner must authorize the merge (or direct the CI-fix push) to re-verify remotely.
3. **Required status checks / protection** — repository protection settings are outside agent authority (documented only).
4. **Sphinx coverage** — owner must integrate milestone docs into `docs/source` or approve a different publication story; deploys remain blocked.
5. **Canonical 111M checkpoint** — unchanged: MISSING (owner authorization + compute required). L1.0.0 GA NOT_ESTABLISHED. Actual coding capability NOT_EVALUATED.
6. **Karkain** — DISCOVERED; KARK-BENCH 0 tasks; KCC 0xC0000135.
7. **Python 3.11 runtime** — unverified locally.

## 12. Actions not performed

No training, fine-tuning, weight updates or checkpoint creation/substitution; no external data acquisition/download; no GPU/compute/spend/paid APIs; no publication/tag/release/PR; no `main`/`develop` merges; no Karkain/MicroQuantum changes; no branch-protection changes; no force-push; no frozen benchmark touched; no test weakened/removed.

## 13. Deliverable verification

- Audit document: this file.
- Defect register: [`TAKSH-RECOVERY-DEFECT-REGISTER.md`](TAKSH-RECOVERY-DEFECT-REGISTER.md).
- Machine-readable report: `reports/taksh_cicd_documentation_recovery.json`.
- Static CI regression suite: `tests/test_ci_workflows.py`.
- Workflow corrections: `ci.yml` (least-privilege), `ci-l0704-manual-qualification.yml` (trigger/dispatch docs + ENG1–3 chain + hash pinning retained).
- README/STATUS additive banners.

## 14. Final qualification decision

- **LOCAL ENGINEERING RECOVERY QUALIFIED** — all defects within the authorized scope are fixed and regression-covered; artifacts/checks verified locally.
- **REMOTE CI QUALIFICATION = `BLOCKED_OWNER_ACTION_REQUIRED`** — GitHub policy prevents any execution on this branch without a default-branch owner action.
- Product: MODEL_UNAVAILABLE · ACTUAL_CODING_CAPABILITY NOT_EVALUATED · LANGUAGE QUALIFICATION unchanged (Karkain DISCOVERED) · **L1.0.0 GA NOT_ESTABLISHED**.
