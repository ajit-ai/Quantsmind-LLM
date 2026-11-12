# TAKSH — Develop Integration Inventory (L1.0.0, 2026-11-12)

**Purpose:** comprehensive, evidence-based integration and product-readiness baseline for TAKSH.
**Scope:** `develop` (current, verified) vs `feature/l060-foundation-model-qualification` — commit-level, file-level.
**Baseline:** `develop @ 0215c6d039fceffd334f8931aa96a3287010a580` (verified == remote 2026-11-12).
**Planning branch:** `planning/l1000-integration-and-ga-readiness` (created from develop `0215c6d`).
**Owner decisions in force:** `TAKSH-OWNER-DECISION-2026-11-11` — A1.1-C (preparation-only), A2-E (no compute). No training, checkpoints, data, compute, releases, merges to `main`, tags, deploys, or publication is performed or authorized by this milestone.

---

## 1. Verified repository state

| Item | Result |
|---|---|
| `develop` HEAD | `0215c6d039fceffd334f8931aa96a3287010a580` (Merge PR #2 `reconcile-recovery-register` + PR #3; parents `61e2dba` × `8f1fbda`) |
| develop == remote | verified via `git ls-remote origin refs/heads/develop` (equal, 2026-11-12) |
| `main` HEAD | `d50de3ea220cbf18eee070e191a507ba70a758bb` — "Merge pull request #1 from ajit-ai/develop" (merged 2026-11-12T04:56:18Z by `ajit-ai`) — **trees of main and develop are identical; main = develop + the merge commit** |
| `feature/l060-foundation-model-qualification` HEAD | `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` (verified == remote) |
| Feature-branch delta vs develop | **19 unique commits**; content delta **+1023 files** (993 added, 30 modified, **0 deleted**) — ≈ +407,822 / −347 |
| Tags | `v1.0.0` (pre-existing) — none created by any integration milestone |
| `v1.0.0` GA | **NOT ESTABLISHED** |
| Owner working-tree | 6 pre-existing CRLF files preserved (`M serving/code_chat.py`, `serving/local_inference.py`, `taksh-hf-publication/upload_manifest.json`, `training/manifest.py`, `training/pretrain.py`, `training/sft.py`) — untouched here |
| `integration/taksh-cicd-recovery` | `90eba0a` (develop + first half of the CI recovery; superseded by develop PRs #2/#3, retained — branch retention, not deletion) |
| `reconcile-recovery-register` | `61e2dba` (register reconciliation; merged into develop as part of PR #3) |

## 2. Integration history so far (verified, in order)

| Date | Action | Result |
|---|---|---|
| 2026-10-21 | `v1.0.0` tag published (pre-existing); PyPI upload blocked (registry credentials) | develop/main carry release governance, no GA claim |
| 2026-11-06 | develop CI red since October (7 failed, 1295 passed) | Historical Failure Set 001 |
| 2026-11-11 | L0.6.x–L1.0.0-ENG3 all executed on feature branch; isolated recovery commit `78b6cdf` | engineering qualification, zero training |
| 2026-11-12 03:23Z | PR #2 opened (`integration/taksh-cicd-recovery` → develop) | CI green after 2 fix cycles (runs `43147929246` ✗, `43150748557` ✗, `43151304728` ✓) |
| 2026-11-12 03:57Z | PR #2 **MERGED** into develop (`8f1fbda`) | develop first green CI since 2026-09-28 (`43151613260` ✓, unit 1m52s) |
| 2026-11-12 04:04Z | PR #3 (register reconciliation) **MERGED** into develop (`0215c6d`) | `43152075567` ✓, `43152403348` ✓ |
| 2026-11-12 04:56Z | **Owner merged develop → main** in PR #1 (`d50de3e`) | main now holds develop's tree (CI recovery integrated) |

## 3. What develop actually contains (qualified today)

| Capability area | Status | Evidence |
|---|---|---|
| Packaging & release machinery (`taksh` v1.0.0) | IMPLEMENTED; v1.0.0 source released; GA NOT established | `pyproject.toml`, `MANIFEST.in`, `NOTICE`, `LICENSE`, `taksh/` (`kv_cache.py`, `v2_3_*.py`), `scripts/prepare_release.py`, `scripts/check_version_consistency.py` (PASS) |
| Base v1.x.0→v2.x.0 milestone qualification work | IMPLEMENTED upgrades; historical artifacts | v1.1.0→v2.5.0 milestones qualified (documentation corpus shipped in repo): `docs/v1.1.0_TRAINING_INTEGRITY.md` … `docs/v2.5.0_B1_COMPLETION_REPORT.md`; reports at root (`TAKSH_V120_*`, `TAKSH_V130_*`, …); learning infrastructure code (`training/*`) present and covered by focused tests |
| CI & delivery | **GREEN** (non-training path) | `.github/workflows/ci.yml` (+`blocked-training-smoke` gated off default); `ci-l0704-manual-qualification.yml` (dispatch-only); runs `43151304728`, `43151613260`, `43152075567`, `43152403348`, `43151620156` |
| Historical Failure Set 001 | RESOLVED remotely | 5 fix paths integrated via PR #2; see `docs/audit/TAKSH-RECOVERY-DEFECT-REGISTER.md` §H |
| DLCAE runtime foundation (ENG1) | **ABSENT** (lives on feature branch only) | `taksh/dlcae/` exists only on feature branch (`78b6cdf`) |
| Semantic/execution validation foundation (ENG2) | **ABSENT** on develop | `taksh/dlcae/semantic/` |
| End-to-end coding-intelligence pipeline (ENG3) | **ABSENT** on develop | `taksh/dlcae/coding/` |
| L0.6.x/L0.7.x/L0.8.0-PRE milestone artifacts | **ABSENT** on develop | L0.6.x…L1.0.0-ENG3 reports/checks/tests — feature branch only |
| Canonical `taksh-foundation-111m` checkpoint | **NEVER EXISTED** (missing everywhere) | see §4 of the model availability assessment |
| `v180_sft` checkpoint | Present (owner-owned), GEOMETRY_INCOMPATIBLE with the canonical 111M (d_model 384 ≠ 768) — never substitutable | hash `9b541036…` |
| Frozen Python benchmark | **Absent on develop; present on feature branch** (`854c7ca8…`), 0/30 at 2026-10-21 | see Feature-Branch-Commit-Matrix |
| Frozen corpus | **Absent on develop** (1,050 technical-eligible Python records, 1,380,611 tokens — `FROZEN_RESEARCH_ONLY`) | `artifacts/l072_l04_freeze/snapshot.json` (feature branch copy) |
| Karkain 54th language | **Absent on develop** (fine — feature branch `DISCOVERED`, kcc `0xC0000135`) | see Karkain section of GA roadmap |
| Owner decision record (A1.1-C / A2-E) | **Absent on develop** (only on feature branch) | branch-specific; present in `docs/L075_L080_OWNER_DECISION_RECORD.md` on feature |
| Master roadmap / release governance | **Absent on develop** | feature: `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md`, `docs/TAKSH-RELEASE-GOVERNANCE.md`, `configs/roadmap/taksh-v3-roadmap.yaml` |

## 4. What develop still lacks for product GA (top-level)

1. The canonical `taksh-foundation-111m` model artifact (**does not exist anywhere**).
2. Any working DLCAE runtime (ENG1–3, feature branch only).
3. Any demonstrated coding-capability evaluation baseline (0/30 stands).
4. Any legally authorized training corpus in develop (Python-frozen corpus exists only on the feature branch, research-only, inedible by the owner decisions A1.1-C/A2-E).
5. The actual release itself: no tag/release/packaging evidence exists beyond the pre-existing `v1.0.0` GA-not-established status.

## 5. What the feature branch uniquely contains

| Layer | Files (approx.) | Contents |
|---|---|---|
| Publication-ready v3.0.0 package | **790** (`8c55346`) + merge commit | Entire untracked owner tree snapshot: `taksh-hf-publication/` (453 files, complete HF publication staging with configs/tests/reports), all root-level `TAKSH_V*` milestone reports, full experiment lineage (`data/l02_overfit`, `data/l03_corpus`, `data/l04_code_corpus`, `evaluation/*`, `scripts/*`, `artifacts/*`, `hf_export/`, `dist/`, `build/`, `rebuild_manifest.py`, `verify_*.py`) |
| Milestone program (L0.6.0→L1.0.0-ENG3, 17 commits; `45f0dc9`…`78b6cdf`) | ~253 | Code (`taksh/dlcae/…`), evaluators, focused tests (70 + 54 + 38), reports, docs |
| Roadmap & governance | ~9 (`84888e6`, `7b7e5f8`) | v3.0.0 master roadmap + blank owner decision record |
| L0.8.0-PRE preflight | 25 files (`262ae69`) | preflight pipeline, resource estimates, readiness gate |
| CI recovery (`78b6cdf`) | 9 files | **already integrated into develop via PR #2/#3 (different SHAs)** |

## 6. Divergence map

```
a654e5d (base)
├── develop: a654e5d → 61ae12a → 30facf5 → 90eba0a → 8f1fbda → 61e2dba → 0215c6d (merged into main at d50de3e)
└── feature: a654e5d → 8c55346 → c601a73 → 45f0dc9 → 1fef89d → 9721aa6 → da16051
              → 27c7b3a → 68dc5b7 → bc3d296 → 6c8177f → b47fafa → 84a25ba → 7b7e5f8
              → 84888e6 → 262ae69 → efcda29 → 6065394 → d2f0d14 → 78b6cdf
```

The branches share history only up to `a654e5d`. Any feature integration proposal must **cherry-pick**; a merge wholesale is not viable (would import the 790-file publication-ready snapshot, ~100k lines of tracked owner experiment artifacts, and would drag in what the owner keeps behind owner-only visibility).

## 7. Subsystem-by-subsystem capability audit

| Subsystem | Implementation | Tests & exact results | Integration | Demonstrated capability | Status |
|---|---|---|---|---|---|
| Learning Engine | feature branch: `training/learning_protocol.py`, `training/replay.py`, `training/binding_curriculum.py` etc. | 218 focused tests in L0.6.x/L1 suites (QUALIFIED at feature branch) | NOT in develop (feature-only) | none — never trained | QUALIFIED as engineering only |
| Knowledge Engine | feature branch L0.6.x stores by construction | focused tests (ENG1: `tests/test_l100eng_dlcae_runtime.py`) — 70 passed | feature-only | none — no verified facts yet | IMPLEMENTED_AND_TESTED |
| Code Intelligence Engine | `training/model.py`, `serving/python_codegen.py` (develop) + feature branch coding engines `taksh/dlcae/coding/*.py` | develop has its own suites; feature ENG1–3 162 passed | not in develop yet (PR sequence blocked) | none | PARTIAL |
| Engineering Engine | `training/supervision*.py`, `training/supervised_train.py` (develop); DLCAE planning code (feature) | develop tests; no training execution | PARTIAL: system path unverified end-to-end | none | PARTIAL |
| DLCAE (6 engines) | feature branch: `taksh/dlcae/comprehension.py`, `sequences.py`, `knowledge.py`, `adaptation.py`, `validation.py`, `learning.py` + `taksh/dlcae/semantic/*` | L1.0.0-ENG 70/70, ENG2 54/54, ENG3 38/38, combined 162/162 | feature branch only | none; engineering only | IMPLEMENTED_AND_TESTED |
| Memory Engine | `taksh/v2_3_*.py` + `training/sequence.py` etc. | develop tests | develop | none | PARTIAL |
| Evaluation Engine | `evaluation/*.py` (feature + develop both) | frozen benchmark 0/30 at 2026-10-21; never re-scored | split across branches | none | BLOCKED_ARTIFACT_ABSENT on develop; 0/30 historical |
| Agent Engine | `serving/agent_profile.py`, `serving/domain_router.py`, `training/agent.py` | develop tests; synthetic only | develop (static) | none | PARTIAL |
| Model loading & inference | `serving/inference_engine.py`, `serving/local_inference.py` — L0.6.2 qualified | `tests/test_l062_inference.py` 11 passed; develop tests | develop (int); feature reuses same code | none | IMPLEMENTED_AND_TESTED |
| Tokenizer & vocabulary | `tokenizer/taksh_code_bpe_v1.json` (feature); `tokenizer/qm_code_bpe_v0.json` (develop-safe existence) | `test_code_tokenizer.py`, ENG suites | split ownership | none; tokenizer correctness unverified on external corpus | QUALIFIED as engineering |
| Training configuration & checkpoint compatibility | `configs/foundation/l060_foundation.yaml`, `configs/train/*.yaml`, `evaluation/l060*.py` | all ENG suites pass; L0.7.5 readiness gate BLOCKED | feature-only | none | PREPARATION_QUALIFIED (NO-GO preserved) |
| Provider/adapters | `serving/providers.py`, `evaluation/m5_adapters.py` — feature has the reordered preflight; develop now ports it fixed | develop CI green (PRs #2/#3) + feature tests pass | develop ✓ (CI-verified) | none | QUALIFIED as engineering |
| Task orchestration | `taksh/dlcae/sequences.py` (feature); `training/sequence.py` (develop) | ENG tests 10/10; develop tests | feature-only DLCAE stack; develop sequences | none | PARTIAL |
| Semantic analysis | `taksh/dlcae/semantic/*` — real Python AST analysis (precision/recall 1.0 on synthetic ground truth) | ENG2 tests 19 groups, all pass | feature-only | none | IMPLEMENTED_AND_TESTED |
| Sandboxed execution | `taksh/dlcae/semantic/tools.py::Sandbox`, `PythonToolchain`, mypy/ruff/node/bash/go/javac adapters | ENG2 toolchain matrix tests | feature-only | none (fixtures only) | IMPLEMENTED_AND_TESTED |
| Validation & feedback | `taksh/dlcae/validation.py`, `taksh/dlcae/semantic/feedback.py` | ENG2 correlation tests | feature-only | none | IMPLEMENTED_AND_TESTED |
| Suggestion-only repair | `taksh/dlcae/semantic/feedback.py::repair_candidates` | ENG2 tests; `source_mutation == False` asserted | feature-only | none | IMPLEMENTED_AND_TESTED |
| Bounded/resumable task ledger | `taksh/dlcae/sequences.py` + ledger persistence (`taksh/dlcae/persistence.py`) | ENG1 interruption/resume tests | feature-only | none | IMPLEMENTED_AND_TESTED |
| Language adapters | `taksh/dlcae/registry.py`, dynamic registry (`configs/foundation/l060r1_dynamic_language_registry.yaml`) | ENG1 registry tests | feature-only | none | IMPLEMENTED_AND_TESTED |
| Karkain integration | `taksh/dlcae/karkain.py` | ENG1/2/3 Karkain blocked-state tests | feature-only | DISCOVERED only; KARK-BENCH 0 tasks; KCC 0xC0000135 | BLOCKED (toolchain) |
| Benchmark & evaluation infra | `evaluation/eval_code.py` (develop); `evaluation/l02_python_bench_v1.json` (feature); `evaluation/l051_multilingual_benchmark_v1.json` (feature) | never re-scored on any host | split | none; 0/30 stands; absent on develop | BLOCKED_ARTIFACT_ABSENT on develop |

**Capability claims summary:** zero across all subsystems. Every claim above is engineering/verification only. All capability evaluation is future work, gated by Stage 2–4 of the GA roadmap.

## 8. Duplicated/obsolete/superseded patterns

| Item | Where | State |
|---|---|---|
| CI recovery commit `78b6cdf` (feature) | feature | **superseded by develop's PRs #2/#3** — do not port wholesale; feature keeps the provenance |
| L0.8.0-PRE verdict (PREPARATION_QUALIFIED/NO-GO) | feature | matches develop reality post-recovery; no port needed |
| `docs/L076_MASTER_ROADMAP_REPORT.md` | feature | report lives on feature; roadmap registry itself needs a dedicated integration PR |
| `configs/l060r1_dynamic_language_registry.yaml` (Karkain `DISCOVERED` + base-53) | feature | required by DLCAE stack; integration PR 2 carries it |
| Base 53-language registry `configs/feasibility/l051_languages.yaml` | feature working tree ONLY | **untracked owner artifact** — absent on all branches; integration must include or explicitly exclude; owner decision |

## 9. Commit ancestry & merge-base record

- `git merge-base(develop, feature) = a654e5dc13dc77e925b3981d09792bade43a970a` — feature builds on develop's exact history.
- 19 feature-only commits; 6 develop/main-only commits (CI recovery integration + SFT fix + reconciliation).
- Content equivalence main↔develop (trees identical); main is an ancestor of the feature lineage via `c601a73` (merge commit of main-into-v3.0.0-package) — c601a73 is *not* on develop.

## 10. Required deliverables

| # | File |
|---|---|
| 1 | `docs/audit/TAKSH-DEVELOP-INTEGRATION-INVENTORY.md` (this file) |
| 2 | `docs/audit/TAKSH-FEATURE-BRANCH-COMMIT-MATRIX.md` |
| 3 | `docs/audit/TAKSH-MODEL-AVAILABILITY-ASSESSMENT.md` |
| 4 | `docs/audit/TAKSH-CODING-CAPABILITY-EVALUATION-PLAN.md` |
| 5 | `docs/audit/TAKSH-PRODUCT-GA-READINESS-ROADMAP.md` |
| 6 | `docs/audit/TAKSH-OWNER-DECISIONS-REQUIRED.md` |
| 7 | `reports/taksh_integration_ga_readiness.json` |
