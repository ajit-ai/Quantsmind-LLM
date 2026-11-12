# TAKSH — Feature-Branch Commit-Level Matrix (L1.0.0, 2026-11-12)

**Branches:** `feature/l060-foundation-model-qualification @ 78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` → `develop @ 0215c6d` (== `main @ d50de3e`).
**Divergence:** 19 feature-only commits; 6 develop/main-only commits (CI recovery integration).
**Content delta:** +1023 files (993 added, 30 modified, 0 deleted).

Legend:

- **A** Product architecture (DLCAE runtime engines, engineering pipeline)
- **B** Required for model availability (foundation artifacts, preflight targeting the 111M canonical model)
- **C** Required for actual coding-capability evaluation (DLCAE runtime, semantic/validation foundation, dev-task harness)
- **D** Engineering infrastructure & CI / packaging / delivery
- **E** Documentation & governance
- **F** Experimental / exploratory / publication staging
- **G** Superseded, duplicated, or administrative
- **H** Blocked by owner authorization (training CI, training runs, data acquisition)
- **I** Not suitable for integration

---

## 1. Per-commit matrix

| # | Commit | What it does (as-written) | Files | Category | Develop? | Notes / dependency |
|---|---|---|---|---|---|---|
| 1 | `8c55346` "TAKSH v3.0.0 - Complete publication-ready publication-ready source-code package" | **Full owner-tree snapshot commit** — the entire publication-ready package (`taksh-hf-publication/` 453 files, all root `TAKSH_V*` reports, full l0x→v2.x experiment lineage, `hf_export/`, publication docs, 54 config files, 50 test files, HF readme/provenance/license files). Based on develop `a654e5d`; superseded in content by develop `0215c6d` + main `d50de3e` for the recovery parts but **not for the 767 extra files** | 790 (+/- ~336k lines) | **F / I** | No | Not integratable wholesale. Cherry-pick individual files only with owner sign-off; no tests reference these files; owner must decide if publication staging is ever to be consumed downstream. Contains the `TAKSH_EXISTING_MODEL_RELEASE_READINESS.md`, `TAKSH_V141_FINAL_QUALIFICATION_REPORT.md`, publication license/provenance trees and the complete experiment archive — these are not develop-ready artifacts |
| 2 | `c601a73` "Merge develop into main: TAKSH v3.0.0 complete publication-ready source-code package" | Merge commit of main(`cb9b81e`) into the feature lineage — effectively the union of 8c55346 + main | 0 (tree-equal to union) | **G** | No | Pure merge — no reviewable content to port |
| 3 | `45f0dc9` feat(l0600-r1) DLCAE and dynamic language architecture | L0.6.0-R1: DLCAE architecture docs (architecture alignment), 53+54 dynamic language registry, KARK-BENCH schema, Karkain association, TAKSH-ARCHITECTURE.md, DLCAE-ARCHITECTURE, DYNAMIC-LANGUAGE-REGISTRY, KARKAIN-INTEGRATION, DLCAE-RESEARCH-PLAN reports, 19 tests | 12 | **A** | No | **FOUNDATION E** of ENG1–3; required for A/C. Carries the registry/schema foundation the DLCAE runtime depends on. Dependency: none outside itself |
| 4 | `1fef89d` feat(l0601) DLCAE-aware training pipeline | L0.6.1: DLCAE-aware training pipeline config; provenance manifest; orchestration; replay; DLCAE-to-learning interface; Karkain learning contract; checkpoint lifecycle (atomic write); incremental snapshot/replay qualification | 22 | **A/D** | No | Depends on 45f0dc9. Includes verified `v180_sft` GEOMETRY_INCOMPATIBLE policy pointer. Required for B (training readiness if/when authorized) |
| 5 | `9721aa6` feat(l0602) local inference architecture | L0.6.2: local inference qualification; ModelLoader/TokenizerAdapter/Sampling/Checkpoint lifecycle contracts; memory preflight; frozen benchmark; the canonical-111M-missing policy | 12 | **A/B/D** | No | Depends on 1fef89d. Defines the canonical-missing contract every later milestone honours |
| 6 | `da16051` feat(l0700) multilingual data acquisition readiness | L0.7.0: data acquisition readiness qualification (H-series): source registry, license qualification, provenance, contamination screen, corpus planning, DLCAE sources registration, acquisition workflow (0 authorizations) | 34 | **D/B** | No | Depends on 1fef89d/9721aa6. Required for B (any authorized data stage) |
| 7 | `27c7b3a` feat(l0701) multilingual corpus engineering | L0.7.1: python-corpus pipeline; multilingual sources; karkain strategy; corpus engineering evidence; frozen corpus engineering | 31 | **D/B** | No | Depends on da16051. Corpus frozen to Python in L0.7.2 (not publishable) |
| 8 | `68dc5b7` feat(l0702) training corpus freeze | L0.7.2: frozen corpus `l072-l04-fdb1fee2beaae9c2` (FROZEN_RESEARCH_ONLY); 1,050 technical-eligible Python records / 1,380,611 tokens; giant-class AST reslice; verdict computation | 17 | **B/D** | No | Depends on 27c7b3a. Frozen evidence; cannot enter develop without owner decision |
| 9 | `bc3d296` feat(l0703) training readiness and close CI debt | L0.7.3: readiness gate; 7 CI regression dispositions; PR #63 existence check (none); preflight constraints; readiness receipt format | 13 | **D** | No | Depends on 68dc5b7. Historical CI evidence. **Mostly superseded in develop by PRs #2/#3 (different SHA)** — ports exist there as green; full port would re-perform the same fix with conflicting paths — DO NOT port wholesale |
| 10 | `6c8177f` feat(l0704) CI runtime and training resources | L0.7.4: CI runtime & resources; workflow dispatch visibility check; MX150 ineffectiveness; minimum viable GPU recommendation; Python 3.11 static-only verdict | 9 | **D** | No | Depends on bc3d296. Captures CI-roadmap evidence; recovery already covered its substance in develop — keep in feature as provenance |
| 11 | `b47fafa` docs(l0704) CI execution, runtime and resource qualification report | L0.7.4 human-readable report | 1 | **E** | No | Docs-only; different content from recovery audit docs (recovery is develop's) |
| 12 | `84a25ba` feat(l0705) pretraining readiness decision | L0.7.5: 18-dimension readiness gate; BLOCKED verdicts; blank owner decision record filed | 9 | **D/H** | No | Depends on bc3d296. Cataloged for provenance only — its output drives the A1.1-C/A2-E decisions already in force |
| 13 | `7b7e5f8` docs(l0705) owner L0.8.0 decision record (blank) | The blank owner decision record as-filed. A1.1-C + A2-E recorded 2026-11-11 in `262ae69` | 1 | **E/H** | No | Owner-controlled artifact. Porting to develop would bypass owner choice-of-lane; leave on feature |
| 14 | `84888e6` docs(roadmap) TAKSH v3.0.0 master development roadmap | L0.7.6: v3.0.0 master roadmap + machine registry (`taksh-v3-roadmap.yaml`), TAKSH-RELEASE-GOVERNANCE.md, roadmap changelog, validator | 9 | **E** | No | Self-contained docs/config/validator; **cherry-pick-able with review** — valuable for develop control of milestones, but see §4 order |
| 15 | `262ae69` feat(l080pre) foundation training preflight (preparation only) | L0.8.0-PRE: PREPARATION_QUALIFIED + TRAINING NO-GO verdict (owner's A1.1-C/A2-E). 10 evaluation gates; resource estimates; checkpoint-recovery readiness; machine-readable reports | 25 | **D/H** (mostly I for develop) | No | Depends on 84888e6, bc3d296+L0s. Preparation verdict ≠ authorization; its value is provenance. If ported, must carry no shadow authorization |
| 16 | `efcda29` feat(l100eng) DLCAE runtime foundation | L1.0.0-ENG1: all six DLCAE engines IMPLEMENTED_AND_TESTED (engineering qualification only); 70 focused tests; sequence orchestration/resume; knowledge OBSERVED→VERIFIED binding; Karkain first-class adapter (KCC-blocked); developer docs | 18 | **A/C/D** | No | **Required, but as a small integration series**, not one-shot. Depends on 45f0dc9. Its code imports feature-branch-only modules: `evaluation/l060r1_dynamic_registry.py`, `evaluation/l060r1_kark_bench_schema.py`, `configs/foundation/l060r1_dynamic_language_registry.yaml` — all feature-branch-only — and relies on the base 53-language registry (`configs/feasibility/l051_languages.yaml`) and `evaluation/l051_language_registry.py` which exist **only in the owner's working tree (untracked) — committed on neither branch**; also reuses `training/checkpoint_lifecycle.py` (feature-branch only) and `training/model.py` (both). Any integration PR must carry those files or equivalents — this is a cross-branch dependency; see matrix §"Untracked owner artifacts" |
| 17 | `6065394` feat(l100eng2) semantic/execution validation foundation | L1.0.0-ENG2: semantic contract (six statuses), AST symbol analysis (precision/recall 1.0/1.0), type-relation records, dependency engine (cycles/missing), sandboxed toolchain adapters (python/mypy/ruff/node/bash/go/javac), feedback/correlation, 54 focused tests | 19 | **A/C** | No | Depends on efcda29 |
| 18 | `d2f0d14` feat(l100eng3) end-to-end coding intelligence integration | L1.0.0-ENG3: coding-task pipeline; model availability gate (v180_sft rejected GEOMETRY_INCOMPATIBLE — never substituted); bounded repair loop; `configs/l100eng3/dev_tasks.json`; 38 focused tests | 11 | **A/C** | No | Depends on efcda29 + 6065394 |
| 19 | `78b6cdf` chore(recovery) CI/CD & documentation recovery | Isolated CI recovery commit — the base of PR #2 → develop | 9 | **D** | Develop has an equivalent (different SHAs — `61ae12a`, `30facf5`, `90eba0a`) | **Superseded by develop**. DO NOT integrate (would revert recovery work). Keep as provenance on the feature branch |

## 2. Dependency graph (feature branch)

```
8c55346 (v3.0.0 snapshot) → c601a73 (merge)
     45f0dc9 (L0.6.0-R1)
         ├→ 1fef89d (L0.6.1 training pipeline)
         │      ├→ 9721aa6 (L0.6.2 inference)
         │      │      ├→ da16051 (L0.7.0 data acquisition)
         │      │      │      ├→ 27c7b3a (L0.7.1 corpus engineering)
         │      │      │      ├→ 68dc5b7 (L0.7.2 corpus freeze)
         │      │      │      ├→ bc3d296 (L0.7.3 readiness/CI)
         │      │      ├→ 6c8177f (L0.7.4 CI/runtime<->resources)
         │      │      ├→ b47fafa (docs)
         │      │      ├→ 84a25ba (L0.7.5)
         │      │      └→ 7b7e5f8 (docs)
         ├→ 84888e6 (roadmap)
         │      └→ 262ae69 (L0.8.0-PRE preflight)
         ├→ efcda29 (L1.0.0-ENG1 runtime)
         │      └→ 6065394 (L1.0.0-ENG2 semantic)
         │               └→ d2f0d14 (L1.0.0-ENG3 integration)
         └→ 78b6cdf (recovery → integrated into develop via PRs #2/#3, different SHAs)
```

## 3. What must cross into develop to make the canonical-111M model work

(Not integration advice — the full list of required artifacts.)

1. A **trained/checkpoint `taksh-foundation-111m` artifact** — none exists, anywhere (would be born from an owner-authorized training run; currently impossible under A1.1-C/A2-E and absent even with them).
2. **DLCAE runtime** (efcda29, 6065394, d2f0d14) — gated integration PRs (see §6).
3. **L0.6.x milestone infrastructure** (45f0dc9, 1fef89d, 9721aa6) — needed for home of the `MODEL_UNAVAILABLE` contract.
4. **L0.7.x qualification artifacts** — carry evidence (frozen corpus, benchmarks) the honest capability story references; develop owns none today.
5. **The owner decisions** (`7b7e5f8` + `262ae69`) — the A1.1-C/A2-E canonical record lives on feature; develop has none.

## 4. Recommended integration order (if/when owner authorizes each PR)

| # | PR | Content | Category | Size | Risk | Dependency |
|---|---|---|---|---|---|---|
| 1 | Roadmap registry & governance docs | 84888e6 (selectively: registry, governance, changelog, validator) | E | small | low | none |
| 2 | DLCAE architecture & schema docs + registry | 45f0dc9 (docs + validator + tests) | A | medium | low | none |
| 3 | DLCAE-aware training pipeline | 1fef89d | A/D | medium | low | PR2 |
| 4 | Local inference | 9721aa6 | A/B/D | medium | low (BLOCKED v180_sft preflight by design) | PR3 |
| 5 | L0.7.x evidence package (freeze + reports; **not** corpus duplicates) | 27c7b3a, 68dc5b7 (evidence only) | B/D | medium | medium (one artifact ≥ maybe dozens MB; verify) | PR3-5 |
| 6 | L0.8.0-PRE preflight | 262ae69 (preflight only) | D/H | medium | medium (must stay NO-GO) | PR3-6 |
| 7 | DLCAE runtime (ENG1) | efcda29 | A/C | medium | low | PR2-3 |
| 8 | Semantic validation (ENG2) | 6065394 | A/C | medium | low | PR6 |
| 9 | End-to-end integration (ENG3) | d2f0d14 | A/C | small | low | PR7 |
| 10 | Recovery duplicate (78b6cdf) | **DO NOT PORT** — integrated | D | — | revert risk | — |

Orderings are evidence-based only; every PR needs its own owner green-light, scope review, and CI qualification on the target branch.

## 5. Classification summary

| Category | Commits |
|---|---|
| A (product architecture) | 45f0dc9, 1fef89d*, 9721aa6*, efcda29, 6065394, d2f0d14 |
| B (model availability) | da16051, 27c7b3a, 68dc5b7, 1fef89d, 9721aa6, 262ae69* |
| C (coding-capability evaluation) | efcda29, 6065394, d2f0d14, 45f0dc9* |
| D (infra & CI) | 1fef89d, 9721aa6, da16051, 27c7b3a, 68dc5b7, bc3d296, 6c8177f, 84a25ba, 262ae69, efcda29, 78b6cdf |
| E (docs & governance) | b47fafa, 7b7e5f8, 84888e6, 262ae69, 45f0dc9* |
| F (experimental / publication snapshot) | 8c55346, c601a73 |
| G (superseded/administrative) | c601a73 |
| H (blocked) | 262ae69 (preparation NO-GO), 84a25ba, 7b7e5f8, 68dc5b7 |
| I (unsuitable) | 8c55346, c601a73 |

* = dual-category.

## 6. Files unique to feature, grouped by destination (not migrated)

| Bucket | Example files | Destination intent |
|---|---|---|
| Code (product) | `taksh/dlcae/*.py` (21 files), `evaluation/l0*.py`, `evaluation/l1*.py` (Milestone validators) | integration PRs 2–9 |
| Tests | `tests/test_l060*.py` … `tests/test_l100eng3*.py` | follow-on PRs |
| Evidence/machine reports | `reports/l*.json` (~80 files) | do-not-port by default; owner decides |
| Frozen corpus | `artifacts/l072_l04_freeze/*` (1,050 records, 1,380,611 tokens) | B/H |
| Publication bundle | `taksh-hf-publication/*` (453 files) | **never integrate** — publication-only, owner-only |
| Benchmarks | `evaluation/l02_python_bench_v1.json`, `evaluation/l051_multilingual_benchmark_v1.json` | required for honest capability evaluation later (B/C) |

## 7. Untracked owner artifacts committed on NO branch (important)

The owner's working tree on this host carries these files, **none of which exists on develop, main, or the feature branch** (verified by `git ls-tree` / `git cat-file` on all three refs, 2026-11-12):

- `configs/feasibility/l051_languages.yaml` — the base 53-language registry the feature DLCAE stack reads (hashing `075d4b56…` recorded by earlier milestones).
- `evaluation/l051_language_registry.py` — its validator.
- plus the 6 pre-existing owner CRLF-modified files (`serving/code_chat.py`, `serving/local_inference.py`, `taksh-hf-publication/upload_manifest.json`, `training/manifest.py`, `training/pretrain.py`, `training/sft.py`) and the **~94 other untracked entries** (data/l02_overfit/…, scripts/*, configs/*, evaluation/* validators, root-level TAKSH_V* reports, taksh-hf-publication/ content) — see the original working tree in `F:/Codes/Git/Quantsmind-LLM`.

**Honest integration consequence:** the feature-branch L0.6.x milestone infrastructure runs against this untracked-but-present substrate. Any clean-tree migration PR that lacks it will fail; any PR that swept it in would be a publication decision the owner has not authorized.
