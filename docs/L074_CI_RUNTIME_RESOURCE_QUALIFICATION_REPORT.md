# TAKSH L0.7.4 — CI Execution, Runtime Compatibility & Resource Strategy Qualification Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `bc3d296dd680924fd3b6fa4ea362be3a33eaff80` (L0.7.3)
**Ending HEAD:** `6c8177f71566996da019371675a59d9033daf8f1`
**Milestone type:** engineering qualification — **NOT a training milestone.**
**Package version:** remains `1.0.0` — not bumped, not a release.

---

## 1. Verdict

**L0.7.4 COMPLETE (engineering), with CI execution BLOCKED by a documented owner-dependent step.**
All other workstreams passed; all seven inherited CI failures remain resolved and covered by
focused tests; the readiness report is internally consistent and fail-closed (BLOCKED); the
frozen corpus is preserved; no training authority was created.

**TAKSH is still not ready to train.** Zero training-eligible tokens, owner authorization
missing, canonical 111M artifact absent, host incapable of the training state. 0/30 frozen
benchmark ceiling stands.

---

## 2. Baseline verification

Branch `feature/l060-foundation-model-qualification` · HEAD `bc3d296…` == remote at start ✓ ·
100 working-tree entries (6 pre-existing user CRLF modifications preserved) · L0.7.3 ledger and
readiness report present and internally consistent · snapshot `l072-l04-fdb1fee2beaae9c2` present
with artifacts intact · `taksh-hf-publication/` untouched.

---

## 3. Workstream A — CI execution: exact blocker, with verbatim evidence

**Goal:** a genuine GitHub Actions CI execution against this branch without unauthorized merges.
**Outcome: NOT AVAILABLE on this branch — blocked by the repository's workflow trigger policy, not by any red test.**

Evidence collected 2026-11-11:

| Probe | Result |
|---|---|
| `gh workflow list --all` | only `CI` (active, id 340947744, path `.github/workflows/ci.yml`) |
| `gh api repos/…/actions/workflows` | only `CI` active; `dispatch_supported: null` |
| `gh workflow run CI --ref feature/l060-foundation-model-qualification` | **HTTP 422** — `Workflow does not have 'workflow_dispatch' trigger` |
| `gh workflow run CI --ref 6c8177f71566996da019371675a59d9033daf8f1` | **HTTP 422** — `No ref found for: <sha>` |
| `.github/workflows/ci-l0704-manual-qualification.yml` (added this milestone, read-only, tests-only, `workflow_dispatch`) | **invisible to the Actions API on the default branch** — `workflow not found on the default branch (HTTP 404)` |

GitHub only exposes/dispatches workflows that exist on the **default branch (`main`)**. The
feature branch cannot self-authorize its own dispatch path. Modifying `main` or any workflow
there is outside this milestone's mandate — **owner action required**.

### Required owner actions (either one)

**Option A — recommended.** Owner adds `.github/workflows/ci-l0704-manual-qualification.yml`
to `main` (push or a reviewed PR). Then:
```
gh workflow run "CI (L0.7.4 manual qualification)" --repo ajit-ai/Quantsmind-LLM --ref feature/l060-foundation-model-qualification
```
The workflow runs on the feature branch's exact commit and reports pass/fail of the full
suite (minus three documented untracked-artifact tests: `test_warmstart_l03_checkpoint_loads_with_strict_compatibility`, `test_overfit_checkpoint_greedy_determinism`, `test_preservation_block_records_all_ten_checkpoints` — these are machine-dependent, not code regressions).

**Option B — minimal edit.** Owner adds `workflow_dispatch:` to the production `CI` workflow on `main` (one line).
Then `gh workflow run CI --repo ajit-ai/Quantsmind-LLM --ref feature/l060-foundation-model-qualification` executes the existing production pipeline against this branch.

**Option C — PR path.** Owner authorizes opening a PR from this branch (to `main` or `develop`). GitHub Actions would run `pull_request` CI on it automatically.

### What a dispatch would verify

The CI-only workflow (`ci-l0704-manual-qualification.yml`) is read-only (`contents: read`), tests-only,
has no release/publish steps, runs the focused L0.6.0→L0.7.3 chain, the full suite minus the
three documented untracked-artifact tests, version consistency, readiness-gate informational
output, and frozen-benchmark hash checks. It does not replace, alter or bypass production CI,
and it does not trigger on push/PR to any branch (dispatch only).

**The new workflow file is present on this branch and is unusable until it lands on the default
branch. It was not merged to `main` — doing so is the owner's decision.**

---

## 4. Workstream B — Python 3.11 compatibility: CONDITIONAL (static only; runtime NOT EXECUTED)

- `pyproject.toml` declares `requires-python = ">=3.10"` (3.10+ supported by metadata).
- 479 files statically parsed at `feature_version=(3,11)` — **zero incompatible syntax in
  real code** (the one flagged file, `scripts/eval_v210_agent.py`, is an intentional f-string
  template, invalid on every Python version and never executed).
- Every declared dependency supports cp311 wheels (torch, tokenizers, numpy, fastapi, pydantic, etc.).
- CI matrix targets 3.11.
- **Runtime NOT EXECUTED:** no Python 3.11 interpreter on this host; downloading one is
  unauthorized. Outcome = **CONDITIONAL** (static evidence only), not QUALIFIED.

## 5. Workstream C — Resource strategy: BLOCKED on this host

Full report: `docs/L074_RESOURCE_STRATEGY.md` + `reports/l074_resource_strategy.json`.

- Host (measured, 16:03Z): 8 CPUs · RAM 4,199,780,352 B total · **207,020,032 B available (95.1% used)** · swap 8.59 GB (6.38 free) · disk free ~170 GiB · MX150 4,096 MiB (driver 457.49, CUDA 11.1) present but **CUDA unavailable** (torch 2.13.0+cpu).
- Canonical 111M (110,658,816 params): fp32 + Adam ≈ **1.77 GiB** (estimate, upper bound); fits nowhere here. bf16 + factorized optimizer path ≈ 0.5–0.7 GB — theoretical, unmeasured, impractical on 8 CPU threads at 95% RAM occupancy. **Not a qualification path.**
- MX150: 4 GiB VRAM, Pascal-class, no bf16, CUDA 11.1 driver vs torch 2.13.0+cpu — **unusable with the installed stack**, unusable regardless of comfort for 111M.
- Strategies A (local CPU — rejected) and B (CPU memory-efficient — unmeasured, impractical) are **not viable**; C (owner GPU) **not available**; D (external, owner-authorized) **possible only with written authorization**.
- **Minimum viable environment (recommendation, estimate):** single GPU with ≥ 16 GiB bf16-capable VRAM + ≥ 16 GiB system RAM + ≥ 80 GiB disk + current CUDA toolchain + owner-authorized compute and data-handling policy. Nothing provisioned.
- **No prices, vendors, GPU models, or durations were invented** (the report contains none); no GPU was used; no compute rented or purchased; no prices paid.

## 6. Workstream D — Canonical checkpoint readiness: qualified, artifact still absent

`configs/foundation/l074_canonical_checkpoint_readiness.yaml` — full artifact contract for a
future authorized run (model/tokenizer/training-config fingerprints, dataset snapshot and
eligibility record, training code revision, optimizer/scheduler config, seeds, step, epoch,
precision, environment, integrity hash, resume compatibility, lifecycle rules).
`evaluation/l074_checkpoint_readiness.py` validates it fail-closed:

| Candidate | Classification |
|---|---|
| canonical 111M artifact (expected present) | `MISSING_ARTIFACT` |
| `runs/v180_sft/checkpoint_last.pt` | **`GEOMETRY_INCOMPATIBLE`** (resolved d_model 384 ≠ 768; 12 vs 12 layers but d_model mismatch) |
| synthetic S2 fixture | `SYNTHETIC_REJECTED` (never production) |

| Requirement class | Status |
|---|---|
| Identity checks (vocab parity 5822, embedding rows 5822, context 2048, tied embeddings) | defined and enforced fail-closed |
| Lifecycle (atomic write, interrupted-write detection, fingerprint tamper rejection, resume compatibility) | defined; enforced by L0.6.1 machinery (verified green) |
| Owner authorization gate | provenance-required; never satisfiable without an owner record |
| Existing checkpoints | untouched, hashes unchanged |

**The spec is qualified; no artifact exists; nothing fabricated; training not authorized.**

## 7. Workstream E — Readiness gate reconciliation (18 dimensions)

`evaluation/l074_readiness.py` → `reports/l074_training_readiness.json`. Wraps L0.7.3 unchanged and adds six dimensions:

| Dimension | Status |
|---|---|
| (12 original) corpus_snapshot_integrity / source_provenance / technical_sample_eligibility / legal_training_permission / owner_training_authorization / language_coverage / tokenizer_model_compatibility / checkpoint_availability_and_identity / hardware_resource_feasibility / benchmark_thresholds / ci_regression_status / reproducibility_and_recovery | PASS ×7, PARTIAL ×2, MISSING ×2, BLOCKED ×1 (unchanged from L0.7.3) |
| focused_regression_qualification (new) | **PASS** |
| full_github_actions_ci_execution (new) | **BLOCKED** — owner action required (§3) |
| python_311_compatibility (new) | **CONDITIONAL** — static only |
| canonical_checkpoint_readiness (new) | **BLOCKED** — spec qualified, artifact missing |
| resource_strategy_availability (new) | **BLOCKED** — host incapable; minimum viable GPU environment undefined until owner authorizes |
| actual_training_authorization (new) | **MISSING** — grants nothing |

Aggregate: **BLOCKED** — all three mandatory prerequisite classes (owner authorization, canonical artifact, host capability) are absent. **Training-eligible tokens: ZERO.** Invariants preserved: technical ≠ training eligibility; conditional legal permission ≠ authorization; CI-passable focused suites ≠ full CI; resource strategy ≠ available compute; frozen benchmark 0/30 stands.

## 8. Workstream F — Seven inherited CI regressions remain covered

Re-executed: `pytest tests/test_m5_adapters_syntax.py tests/test_m6_sovereign.py tests/test_v120_python_data.py tests/test_v150_coding_foundation.py tests/test_v160_authorization_and_baseline.py tests/test_v180_corpus_qualification.py` → **197 passed, 1 skipped** (Python 3.14.7). The seven dispositions in `configs/data/l073_regression_ledger.yaml` are unchanged: all RESOLVED; none weakened, skipped, or renamed. The three untracked-artifact tests are documented separately (out of the inherited seven, machine-dependent; excluded from the CI-only path with explicit note). A local focused pass is not a full CI pass; CI requires owner action (§3).

## 9. Testing

```
pytest tests/test_l074_readiness.py                     → 26 passed (new)
pytest <L0.6.0 → L0.7.4 focused chain>                   → 376 passed (no regressions)
pytest <all seven inherited-failure files>               → 197 passed, 1 skipped
python -m evaluation.l074_readiness                        → 18 dimensions; verdict BLOCKED
python -m evaluation.l074_checkpoint_readiness             → spec valid; candidates all rejected as required
python -m evaluation.l073_readiness (unchanged)            → BLOCKED
```

## 10. Preservation evidence (before and after)

| Artifact | Method | Result |
|---|---|---|
| Frozen Python benchmark | SHA-256 (streaming) | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| `runs/v180_sft/checkpoint_last.pt` | streaming SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged |
| Tokenizer committed blob | git blob hash | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` — unchanged |
| `artifacts/l072_l04_freeze/` (snapshot `l072-l04-fdb1fee2beaae9c2`) | SHA-256 + parent-chain verification (re-derived) | preserved |
| L0.7.0 / L0.7.1 / L0.7.2 / L0.7.3 artifacts | byte-identical; one volatile L0.7.0 resource report re-overwritten by its own test run was **restored byte-identical** | preserved |
| taksh-hf-publication/ | untouched | preserved |
| develop / main | untouched | ✓ |

## 11. Commit / push

`6c8177f71566996da019371675a59d9033daf8f1` — `feat(l0704): qualify CI runtime and training resources`
— pushed to `origin/feature/l060-foundation-model-qualification`; **remote HEAD verified equal** (`git ls-remote`). No force-push; no merge to `develop`/`main`; no tags, releases, or publications; `package.json` remains `1.0.0`.

## 12. Remaining blockers and owner decisions

1. **Owner action required to enable CI execution** (§3: push the CI-only workflow to `main`, or add `workflow_dispatch` to production CI, or authorize a PR). Until then there is no genuine GitHub Actions signal on this branch.
2. **Owner training authorization** — absent everywhere this milestone; canonical 111M artifact absent; host incapable of the training state.
3. **Python 3.11 runtime** — `CONDITIONAL` only; an interpreter is not installed and cannot be downloaded without authorization.
4. **Minimum viable training environment** — required: GPU ≥ 16 GiB VRAM (bf16), ≥ 16 GiB RAM, ≥ 80 GiB disk, CUDA toolchain current. Owner must authorize provisioning (budget, data residency, checkpoint policy, external-vs-internal distribution) or supply it.
5. Karkain `kcc.exe` `0xC0000135` — unchanged; no Karkain work started.
6. External candidates — still `HUMAN_LEGAL_REVIEW_REQUIRED` (L0.7.0).
7. Frozen 0/30 ceiling stands; not executed here.

## 13. Attestation

No training or fine-tuning; no external data acquired, downloaded, or ingested; no GPU used;
no paid compute, API or storage; nothing published; no merges to `develop`/`main`; no tags or
releases; no force-push; no history rewrite; package version unchanged; L0.7.2 frozen snapshot
preserved; the owner's pre-existing working-tree modifications were never touched.

**L0.7.4 stops here. L0.8.0 is not started. Nothing in this milestone entitles TAKSH to train.**
No capability claim is made anywhere; TAKSH remains not-a-proven coding intelligence.
