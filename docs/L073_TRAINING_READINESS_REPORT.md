# TAKSH L0.7.3 — Training Readiness & CI Regression Closure Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `68dc5b7422ffe7740368191ee62faf04c8995ced` (L0.7.2)
**Milestone type:** corrective engineering & readiness — **NOT a training milestone.**
**Package version:** remains `1.0.0` — not bumped.

---

## 1. Verdict

**L0.7.3 COMPLETE (engineering).** All seven inherited CI failures now carry evidence-backed
dispositions (all seven RESOLVED, none skipped/deleted/weakened). The authoritative
training-readiness gate is in place and **fails closed: BLOCKED** — because owner training
authorization is MISSING, the canonical `taksh-foundation-111m` artifact does not exist, and
the host cannot hold a training state. CI cannot execute on this feature branch; that is
reported honestly, not papered over.

**TAKSH is not ready to train.** Nothing in this milestone authorizes training, and the 0/30
frozen-benchmark ceiling stands.

---

## 2. Baseline verification

Branch `feature/l060-foundation-model-qualification` · HEAD == remote == `68dc5b7422ffe7740368191ee62faf04c8995ced` · working tree 100 entries (6 pre-existing user modifications preserved byte-identically; nothing else pre-staged).

---

## 3. PR #63 investigation (mandatory)

**PR #63 — `fix(taksh): make Seed-Coder adapter loader availability independent of transformers` — does not exist in `ajit-ai/Quantsmind-LLM`.**

Evidence gathered 2026-11-11 (owner: `ajit-ai`; repo is not a fork; zero forks):

- `gh pr view 63` → *GraphQL: Could not resolve to a PullRequest with the number of 63*
- `gh pr list --state all --search 'Seed-Coder adapter'` → only **PR #1** (OPEN, `develop`→`main`, provider architecture + Seed-Coder qualification)
- `gh pr list --state all --search 'adapter loader'` → ∅; `gh search prs --owner ajit-ai 'adapter loader'` → ∅
- `gh repo view` → `isFork: false, forkCount: 0` — it cannot be hiding upstream
- `gh search issues` cross-checks → ∅

**Disposition: NOT_ACCESSIBLE.** It must not be treated as a merged fix, and no current code depends on it. The underlying defect (adapter discovery requiring the optional `transformers` package) was reproduced and fixed **in this branch directly**, with attribution to the reported intent (see ledger-001). PR #1 remains OPEN and untouched. "PR #63's history" cannot be preserved because no such object exists in the accessible repository — this is documented, not silently substituted.

---

## 4. Seven-failure regression ledger

Machine-readable: `configs/data/l073_regression_ledger.yaml` (versioned, schema `taksh.ci_regression_ledger/1`).

| # | Test | Root cause | Corrective change | Disposition |
|---|---|---|---|---|
| 1 | `test_m5_adapters_syntax.py::test_external_adapter_does_not_download_by_default` | preflight required `transformers` **before** weights discovery; discovery was never dependency-free | discovery reordered: licence → weights (dependency-free) → RAM → transformers; one stable combined reason string; offline default untouched | **RESOLVED** |
| 2 | `test_m6_sovereign.py::test_available_ram_is_the_binding_constraint` | assertion depended on live host RAM; environment-dependent | `estimate_scales(memory=None)` — injectable; test rewritten with deterministic synthetic memory (4 GiB total / 128 MiB available); live-measurement test kept as informational-only | **RESOLVED** |
| 3 | `test_v120_python_data.py::test_g7_manifest_has_no_timestamp_or_absolute_path` | benchmark provenance embedded the *resolved absolute* local path into `coverage.suites[].origin` (→ manifest.json) leaking `/tmp/pytest-of-runner/pytest-0/python_v10/bench.jsonl` | provenance now records the **declared** registry path (portable), or `local://<basename>` for absolute declarations; `content_sha256` added; suite identity + integrity preserved | **RESOLVED** |
| 4 | `test_v150_coding_foundation.py::test_geometry_inspector_reports_a_matching_triple` | consumed untracked `runs/m7_pretrain/checkpoint_last.pt`; `runs/` is gitignored → absent on CI | synthetic S2-equivalent fixture in `tmp_path` (exact `code_m3_9m` geometry, step 300, labelled synthetic); inspector assertions unchanged | **RESOLVED** |
| 5 | `test_v150_coding_foundation.py::test_tokenizer_artifact_and_model_geometry_are_unchanged` | same untracked-artifact dependency | same fixture; same assertions | **RESOLVED** |
| 6 | `test_v160_authorization_and_baseline.py::test_g10_geometry_inspector_accepts_the_s2_triple` | same untracked-artifact dependency | same fixture | **RESOLVED** |
| 7 | `test_v180_corpus_qualification.py::test_sampled_decoding_is_reproducible_with_a_seed` | same untracked-artifact dependency + legacy unsafe-load path | fixture loads via default safe path; equal seeds reproduce identical ids; no capability claim | **RESOLVED** |

Counting: RESOLVED = 7 · BLOCKED = 0 · FORMALLY_DEFERRED = 0. No failing test was skipped, deleted, weakened or renamed to force green.

Residual risk documented: three *other* test files reference untracked `runs/` checkpoints (`test_l04_pipeline.py`, `test_openai_server.py`, `test_l05_feasibility.py`) — these pass on hosts with artifacts (like this one and presumably the historical runner) and would fail on a bare CI checkout; they are NOT part of the inherited seven and were not altered.

---

## 5. Verification

```
pytest tests/test_m5_adapters_syntax.py tests/test_v120_python_data.py tests/test_m6_sovereign.py   → 97 passed, 1 skipped
pytest tests/test_v150_coding_foundation.py tests/test_v160_authorization_and_baseline.py tests/test_v180_corpus_qualification.py
                                                                                                          → 100 passed
pytest tests/test_l060_foundation.py tests/test_l060r1_architecture_alignment.py tests/test_m6_resources.py \
       tests/test_l0601_pipeline.py tests/test_l062_inference.py tests/test_l070_data_readiness.py \
       tests/test_l071_corpus.py tests/test_l072_freeze.py tests/test_l073_readiness.py             → 350 passed
pytest tests/test_l073_readiness.py                                                                   → 36 passed
python -m evaluation.l073_readiness                                                                     → verdict BLOCKED
```

**CI status — reported honestly:** `.github/workflows/ci.yml` triggers on pushes to `develop`/`main` and on pull requests. This feature branch therefore **cannot** run GitHub CI without a PR or an owner-granted `workflow_dispatch` addition; no push/PR was made to `develop` or `main` (forbidden). The historical "v1.0.0 CI run" (7 failed / 1,295 passed / 39 skipped / exit 1) is documented in the ledger for provenance only; current evidence comes from the focused suites above on Python 3.14.7 + torch 2.13.0+cpu. **A passing focused suite is not a full CI pass**; the owner must trigger CI (e.g., via a PR or by permitting `workflow_dispatch` on this branch) and then re-run it before merging anywhere.

---

## 6. Training-readiness gate — 12 dimensions (authoritative, fail-closed)

`evaluation/l073_readiness.py` → `reports/l073_training_readiness.json`. Verdict is the most restrictive dimension, never the most favourable.

| Dimension | Status | Evidence (live/pinned) |
|---|---|---|
| corpus_snapshot_integrity | **PASS** | `l072-l04-fdb1fee2beaae9c2`; records_sha256 recomputed and matched; 4 files present |
| source_provenance | **PASS** | l04 record VERIFIED (7/7); ACQUIRED |
| technical_sample_eligibility | **PASS** | 1,050 TECHNICALLY_ELIGIBLE / 2 excluded / **training_eligible = 0** |
| legal_training_permission | **PARTIAL** | APPROVED_WITH_CONDITIONS; internal research pretraining only; external/commercial release needs fresh counsel review |
| owner_training_authorization | **MISSING** | no authorization exists or is creatable by tooling; none granted by this milestone |
| language_coverage | **PARTIAL** | Python only (1,380,611 tokens); other 52 + Karkain zero |
| tokenizer_model_compatibility | **PASS** | tokenizer committed blob `fbedf003…` + foundation config `01a90464…` match; vocab 5822; context 2048 |
| checkpoint_availability_and_identity | **MISSING** | canonical `taksh-foundation-111m` has NO artifact anywhere; `v180_sft` (`9b541036…`) exists but is a different ~28.8M architecture (d_model 384) — must not be substituted |
| hardware_resource_feasibility | **BLOCKED** | measured: 4.2 GiB total, ~0.13–0.16 GiB available under load (~97 % used), free disk 170 GiB; 111M training state ≈ 1.65 GiB fp32+Adam (16 B/param; estimate), weights alone ≈ 0.41 GiB (measured checkpoint 0.33 GiB); host cannot hold the training state; **no authorization regardless** |
| benchmark_thresholds | **PASS** | frozen bench `854c7ca8…` intact; multilingual `7a839060…` intact; tasks/thresholds unchanged; 0/30 stands |
| ci_regression_status | **PARTIAL** | 7/7 dispositions evidence-backed (all RESOLVED); CI cannot run on this branch; focused suites green; residual untracked-artifact risk documented |
| reproducibility_and_recovery | **PASS** | content-derived snapshot id (byte-reproducible rebuilds); atomic writes; L0.6.1 resume gates unchanged and CLOSED |

Aggregate: **BLOCKED** (`owner_training_authorization` + `checkpoint_availability_and_identity` + `hardware_resource_feasibility`). Training-eligible tokens: **ZERO.** Frozen snapshot: `l072-l04-fdb1fee2beaae9c2` (FROZEN_RESEARCH_ONLY, preserved byte-identically).

---

## 7. Host resource measurement (point-in-time, labelled)

The host (4 CPUs, 3.91 GiB RAM — 4,199,780,352 B) is chronically near capacity: available RAM measured ranged 128–156 MiB at request time in this milestone (≈96–97 % used); free disk ≈ 170–183 GiB. Estimates are labelled as estimates (params×16 B fp32+Adam; params×4 B weights-only). No model was downloaded, no cloud or paid resource was provisioned, no GPU was used, no price or duration claims are made. **No optimization can make training feasible on this host, and none was applied.**

## 8. L0.7.2 snapshot immutability — verified

`snapshot.json` · `records_meta.jsonl` · `quarantine.jsonl` · `eligibility.json` all unchanged; parent chain (`artifacts/l071_l04_snapshot/…`) verified byte-identical; `train.jsonl` re-hashed. The gate recomputes integrity live every run (PASS).

## 9. Preservation evidence (before and after)

| Artifact | Method | Result |
|---|---|---|
| Frozen Python benchmark | SHA-256 | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark (64 tasks) | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| `runs/v180_sft/checkpoint_last.pt` | streaming SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged |
| Tokenizer committed blob | git blob hash | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` — unchanged |
| L0.7.0/1/2 reports and artifacts | byte-identical | re-verified; one volatile `l070_resource_readiness.json` re-overwrite by L0.7.0's aggregation test was **restored byte-identical** |
| taksh-hf-publication/ | only pre-existing CRLF modification | preserved, not staged |
| develop / main | untouched | ✓ |

No pretraining, no fine-tuning, no benchmark re-run-as-capability, no data acquisition, no network fetch beyond the documented GitHub metadata queries, no publication, no merge, no tag, no release, no force-push. Package version `1.0.0`.

## 10. Exact L0.7.3-owned files

`configs/data/l073_regression_ledger.yaml` ·
`evaluation/l073_readiness.py` ·
`evaluation/m5_adapters.py` (preflight reorder, fix-1) ·
`evaluation/m6_sovereign.py` (injectable memory) ·
`data/python_v1/benchmarks.py` (logical origin + content_sha256) ·
`tests/test_m6_sovereign.py` ·
`tests/test_v150_coding_foundation.py` ·
`tests/test_v160_authorization_and_baseline.py` ·
`tests/test_v180_corpus_qualification.py` ·
`tests/fixture_s2_checkpoint.py` (new synthetic fixture) ·
`tests/test_l073_readiness.py` (36 tests) ·
`reports/l073_training_readiness.json` ·
`docs/L073_TRAINING_READINESS_REPORT.md` (this file)

Modifications to pre-existing milestone reports: none. Modification to package: none.

## 11. Remaining blockers and owner decisions

1. **Owner training authorization** — the only path forward to L0.8.0. Missing; this milestone grants nothing and cannot create it.
2. **Canonical foundation-111m artifact does not exist** — it must be produced by an explicitly authorized and *affordable* pretraining run; this host cannot complete one.
3. **CI execution** — `.github/workflows/ci.yml` fires only on `develop`/`main` pushes and PRs; owner action is required to get a green CI signal (e.g., authorize a PR or add `workflow_dispatch` for this branch).
4. **Karkain `kcc.exe 0xC0000135`** — unchanged; no Karkain capability; repository untouched.
5. **External candidates (Stack v1/v2, CodeSearchNet)** — remain `HUMAN_LEGAL_REVIEW_REQUIRED`; nothing acquired.
6. **`ci.yml` python-version 3.11 vs local 3.14.7** — untested interplay; full suite on 3.11 unchecked.

## 12. Confirmations

- No training was executed; no fine-tuning; no GPU used; no external data acquired or downloaded; no paid API/storage; nothing published; no merges to `develop`/`main`; no tags/releases; no force-push; L0.8.0 NOT started; TAKSH remains not-a-coding-capability at 0/30.
- The L0.7.2 frozen snapshot is byte-identical; the 53-language catalogue, Karkain first-class status and L0–L3 ambition are preserved unreduced.
