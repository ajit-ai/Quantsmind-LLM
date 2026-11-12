# TAKSH — L1.0.1 Integration Authorization Matrix (2026-11-12)

**Baseline:** planning branch `6662cce` (== remote); source branch `feature/l060-foundation-model-qualification @ 78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1`; develop `0215c6d`; main `d50de3e`.
**This matrix recommends; it does not authorize.** The owner decides each row. No row is approved by default.
**All content copied/adapted under each PR must preserve:** frozen benchmarks (hashes `854c7ca8…`, `7a839060…`), the canonical-111M-missing reality, `v180_sft` GEOMETRY_INCOMPATIBLE, A1.1-C/A2-E, the 0/30 Python benchmark history, Karkain DISCOVERED, and zero training of any kind.

---

## Requirements A–A9 vs reality (recap)

| Row | Commit | Class | Independent? | Verdict |
|---|---|---|---|---|
| A-01 | `84888e6` | Docs/governance only (9 files) | Yes (but needed first) | PROPOSED for approval |
| A-02 | `45f0dc9` | Docs + schema/config + registry validator + 19 tests | Yes (needs A-01) | PROPOSED |
| A-03 | `1fef89d` | `training/` + `evaluation/l0601_*` + tests; 22 files | Needs A-02 | PROPOSED |
| A-04 | `9721aa6` | `serving/inference_engine.py` + CRLF-owner `serving/local_inference.py` base + `evaluation/l062_*`; 12 files | Needs A-03 | PROPOSED |
| A-05 | `27c7b3a` + `68dc5b7` | Evidence-only subset (freeze artifact + reports; **no corpus port**); ~14 files | Needs A-02 | PROPOSED |
| A-06 | `262ae69` | Preflight checker + aggregate gate, 25 files; **preservation-only** | Needs A-02, A-03 | PROPOSED |
| A-07 | `efcda29` | `taksh/dlcae/*`; 18 files; 38 tests; DLCAE runtime gated on A-02..A-04 | Highly conditional — **depends on untracked `l051_languages.yaml`/`l051_language_registry.py`** | BLOCKED on C-decision |
| A-08 | `6065394` | `taksh/dlcae/semantic/*`; 19 files; needs A-07 | chain | DEFERRED until A-07 authorized |
| A-09 | `d2f0d14` | `taksh/dlcae/coding/*`; 11 files; needs A-08 | chain | DEFERRED until A-08 |
| — | `78b6cdf` | recovery commit | n/a | DO NOT RE-PORT (superseded by develop PR#2/#3) |
| — | `8c55346`, `c601a73` | v3.0.0 publication-ready snapshot | n/a | DO NOT PORT (publication staging) |

---

## A-01 — Roadmap & governance (eligible for a documentation-only PR)

**Provenance:** commit `84888e6` (L0.7.6 manifest); content-free of code.

**Exact source files (cherry-pick candidates):**
- `configs/roadmap/taksh-v3-roadmap.yaml` (validators read this file; validator exists in commit)
- `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md`
- `docs/TAKSH-RELEASE-GOVERNANCE.md`
- `docs/TAKSH-ROADMAP-CHANGELOG.md`
- `evaluation/roadmap/roadmap_validator.py` (root prune path: `evaluation/roadmap/` — new dir)
- `tests/test_l076_roadmap.py`
- `docs/L076_MASTER_ROADMAP_REPORT.md` — optional (historical record; owner may exclude)
- **NOT** `docs/L075_L080_OWNER_DECISION_RECORD.md` — owner-edited, intentionally empty when committed; the L0.8.0-PRE version is included with A-06 (already recorded decisions are in that file).

**Purpose/value:** gives develop the machine-readable milestone registry + release governance invariants already proven on feature. Establishes "what counts as done" inside the repo.

**Dependencies:** none (root `evaluation/roadmap/` new dir). Does **not** import anything from the QA path.

**Evidence:** `84888e6` passes 28/28 roadmap tests (`test_l076_roadmap.py`) plus roadmap validator PASS.

**Missing tests:** none specific — focused suite is adequate; validator self-checks entries against roadmap YAML.

**Risks:**
- Doc conflict: develop README/STATUS banners already exist; docs must stay additive, not contradictory.
- A validator is required for CI use; must not pull owner-controlled paths (validator must validate without requiring feature-branch absolute paths).

**Independent:** YES.

**Risk rating:** LOW.

**Agent recommendation:** **approve in principle, conditionally — only as a small docs-only PR reviewed by the owner; never bundled with anything else.** Recommend: owner approves.

---
## A-02 — DLCAE architecture and dynamic language registry

**Provenance:** `45f0dc9`; 12 files.

**Exact source files:**
- `docs/architecture/TAKSH-ARCHITECTURE.md` · `docs/architecture/DLCAE-ARCHITECTURE.md`
- `docs/architecture/DYNAMIC-LANGUAGE-REGISTRY.md` · `docs/architecture/KARKAIN-INTEGRATION.md`
- `docs/architecture/DLCAE-RESEARCH-PLAN.md`
- `configs/foundation/l060r1_dynamic_language_registry.yaml`
- `evaluation/l060r1_dynamic_registry.py` · `evaluation/l060r1_kark_bench_schema.py`
- `reports/l060r1_dynamic_language_registry.json` · `reports/l060r1_kark_bench_schema.json`
- `tests/test_l060r1_architecture_alignment.py`
- `docs/L0600_R1_ARCHITECTURE_ALIGNMENT_REPORT.md`

**Purpose:** DLCAE conceptual frame + first registry encoding; frozen 0/30 baseline acknowledgment; Karkain pinned DISCOVERED with full limitation record.

**Assets NOT ported by this PR:** base registry `configs/feasibility/l051_languages.yaml` (untracked — decision C), and nothing from L0.6.1+.

**Dependencies:** A-01 recommended (for roadmap appendix references) — not required in code paths.

**Tests evidence (feature branch):** 19/19 `test_l060r1_architecture_alignment.py` pass; also 236 prior green.

**Missing tests:** none.

**Risks:**
- Registry schema places `karkain` at `.kark` DISCOVERED with zero corpus — fine for docs, must never be presented as capability;
- validator expects `evaluation.l051_language_registry.validate_registry` — the base registry validator file is **absent** on feature? No: present at `evaluation/l051_language_registry.py` in the working tree but **untracked** on feature too — dependency on C-decision.
- Carriage of the v180_sft resource files must not reference disks/… it doesn't.

**Independent:** NO (hard dependency on registry decision).

**Risk:** LOW-MEDIUM.

**Agent recommendation:** **approve conditionally, contingent on C-01 or C-02** (or make schema validator tolerate absence of base registry YAML, gated). Do not approve before registry file provenance is documented.

---
## A-03 — L0.6.1 DLCAE-aware training pipeline

**`1fef89d`**; 22 files.

**Files:** `evaluation/l0601_*.py` (11 modules) · `training/tokenizer_contract.py` · `training/incremental.py` · `training/determinism.py` · `configs/example_operations/.gitkeep`? · `training/checkpoint_lifecycle` related files: `training/checkpoint_lifecycle.py` · `training/manifest.py` · `training/replay.py` · `reports/l0601_*.json` · `tests/test_l0601_pipeline.py` · `docs/L061_TRAINING_PIPELINE_QUALIFICATION_REPORT.md`.

**Purpose:** DLCAE-aware training-readiness pipeline, manifest/checkpoint-lifecycle contracts, replay — every artifact fails closed; no training of any kind to be performed at ANY point by merging this PR.

**Dependencies:** A-02; share `training/replay.py` with `training/replay.py` additions unique to this commit — check exact diff before merging.

**Evidence:** L0.6.1 qualified; 236 tests green at its time (retained through L0.8.0-PRE, 466 green overall later).

**Missing tests:** none identified.

**Risks:**
- `training/checkpoint_lifecycle.py` contains atomic-write tests used by later milestones (exact-API compatibility required by A-06/A-07);
- touches `training/sft.py` / `training/pretrain.py` — owner CRLF files; cherry-picking conflicts may include **mis-applied** owner CRLF versions; owner must diff before merge.

**Independent:** NO (needs A-02).

**Risk:** LOW-MEDIUM.

**Agent recommendation:** **approve with care** — merge only with exact cherry-pick review; **reject without owner diff review** given CRLF-ownership conflicts.

---
## A-04 — L0.6.2 local inference qualification

**`9721aa6`**; 12 files.

**Files:** `evaluation/l062_*.py` (inference/tokenizer/…/karkain) · `serving/inference_engine.py` (full production model loader — CRLF-owner file? **no** — `serving/inference_engine.py` is not modified; only appended adapters: `serving/local_inference.py` owner-CRLF) · `reports/l062_*.json` · `tests/test_l062_inference.py` · `docs/L062_LOCAL_INFERENCE_QUALIFICATION_REPORT.md`.

**Purpose:** enforced `MODEL_UNAVAILABLE` / canonical-substitution policy at inference boundary; checkpoints GEOMETRY_INCOMPATIBLE refused; owner authorization boundaries locked (§"GEOMETRY" 9b541036…).

**Dependencies:** A-03. Also reads `configs/foundation/l060_foundation.yaml` (**feature-branch-only** — must be ported with this or earlier PR).

**Evidence:** v180_sft rejected three ways by L0.6.2/L0.6.4/L0.8.0-PRE; 42 qualified local inference tests; `tests/test_l062_inference.py` 18 passed.

**Missing tests:** none.

**Risks:**
- `serving/local_inference.py` — **owner CRLF file; the owner's working-tree version is the only truthful one; merging may overwrite it** — must be excluded from cherry-pick, retained CRLF, or individually re-applied after diff review.
- GPU/CPU boundary remains: no compute spend authorized.

**Independent:** NO.

**Risk:** MEDIUM (CRLF risk).

**Agent recommendation:** **approve conditionally; exclude `serving/local_inference.py` from the patch or port only additive hunks the owner explicitly accepts.** POSSIBLE: owner chooses to bind the two CRLF serving files as a personal decision and merges them in one step personally.

---
## A-05 — L0.7.x evidence bundle (evidence only)

**`27c7b3a` + `68dc5b7`**; selected files only.

**Port subset:** `docs/L070…L073_*_REPORT.md` (4) · `artifacts/l072_l04_freeze/snapshot.json` (+ `records_meta.jsonl` if owner approves) · `evaluation/l071_corpus_record.py`/`l071_dedup.py`/`l071_snapshot.py`/`l072_*.py` · `tests/test_l072_freeze.py` · `tests/test_l071_corpus.py` · `evaluation/l070_*.py` (12) · `configs/roadmap|snapshots/*.yaml` (only if owner approves) · `reports/l070_*.json` · `configs/feasibility/l051_languages.yaml` (**EXCLUDED** — untracked, decision C) · `evaluation/l051_language_registry.py` (**EXCLUDED** for same reason).

**Do NOT port:** corpus rows/content (`data/l02_overfit/`, `data/l03_corpus/`, `artifacts/l072_l04_freeze/` — the frozen corpus is **not** re-written; technical-eligibility is not legal/training eligibility).

**Purpose:** full qualification evidence in the develop tree: the frozen corpus is frozen-ready research evidence for any future authorized training decision.

**Dependencies:** A-02 recommended; A-03 required for the preflight chain it feeds.

**Evidence:** L0.7.x milestones qualified (L0.7.0 136, L0.7.1 148, L0.7.2 121 pass; full chain 466 green at L0.8.0-PRE).

**Missing tests:** none.

**Risks:**
- Bulk content in `artifacts/`; distributing research-corpus provenance documents expands leakage surface — owner must classify information sensitivity before merge.
- Some files (contamination registry) contain provenance of Hugging Face dataset URLs — owner may not want all provenance public.

**Independent:** semi-independent (needs A-02 for naming only).
**Risk:** LOW-MEDIUM (information disclosure, not behavior).
**Agent recommendation:** **approve with scope restriction + sensitivity review; no corpus payload; reports only; tests mandatory green.**

---
## A-06 — L0.8.0-PRE preflight

**`262ae69`**; 25 files.

**Files:** `evaluation/l080pre/*.py` (9) · `configs/l080pre/l080pre_preflight.yaml` · `tests/test_l080pre_preflight.py` · `tests/test_l076_roadmap.py` (PR #2 saved only the owner-agreement alignment tests — different semantics, don't re-port) · `docs/L080PRE_EXECUTION_BOUNDARY.md` · `docs/L080PRE_PREFLIGHT_REPORT.md` · `reports/l080pre_*.json`.

**Purpose:** fail-closed preflight for a future training decision; **PREPARATION_QUALIFIED but NOT AUTHORIZED** stays the only acceptable outcome.

**Dependencies:** A-01..A-04; READS `L075_L080_OWNER_DECISION_RECORD.md` (owner edits in place — feature tree has it; develop doesn't).

**Evidence:** 29 focused tests pass; aggregate reports survive two rebase cycles; readiness validator correctly reports NO-GO for training under A1.1-C/A2-E.

**Missing tests:** none.

**Risks:**
- The preflight verdict says NO-GO; importing it cannot yank the leash. Keep this property.
- `tests/test_l076_roadmap.py` differs semantically between branches (this was fixed in develop already);
- `docs/L075_L080_OWNER_DECISION_RECORD.md` on develop lacks A1.1-C/A2-E marks — the decision lives on feature; port only if owner rewrites/consents (or references the feature copy by hash in develop).

**Independent:** NO. **Risk:** LOW. **Agent recommendation:** **approve conditionally — only if it lands with the NO-GO semantics intact and the owner re-signs the decision record link; no test may be weakened.**

---
## A-07 — L1.0.0-ENG1 runtime

**`efcda29`**; 18 files.

**Files:** `taksh/dlcae/__init__.py` + 12 modules · `evaluation/l100eng_qualification.py` · `tests/test_l100eng_dlcae_runtime.py` · `docs/L100ENG_DLCAE_RUNTIME_FOUNDATION.md` · `docs/architecture/DLCAE-ARCHITECTURE.md` (status edit only) · `reports/l100eng_runtime_qualification.json`.

**Purpose:** the six-engine DLCAE runtime (qualify engineering, never capability). Real, testable, resumable comprehension engine with honest BLOCKED paths.

**Dependencies:** A-02/A-03 (engines consume training/replay, checkpoint_lifecycle, registry ABI); **and** on the untracked `l051_languages.yaml` (registry read at import). The base registry validator (`evaluation/l051_language_registry.py` — also untracked) is imported by `l060r1_dynamic_registry` lazily, guarded by try/catch — runs without it but the base check fails loud.

**Evidence:** 70 focused tests pass; focused chain 536 total at milestone; Karkain contract honored; DLCAE runtime receives handoffs refused.

**Missing tests:** none.

**Risks:**
- `training/manifest.py`, `training/pretrain.py`, `training/sft.py` — owner CRLF files, possibly touched by ENG1? grep: ENG1's 18 files do not touch them. Good.
- Registry ABI: if A-02/A-03 not present at merge time, engines fail closed — acceptable but CI must reflect that state deliberately.

**Independent:** NO. **Risk:** LOW-MEDIUM. **Agent recommendation:** **blocked until C-decision resolves in owner's favour; then approve. INTEGRATION MUST NOT SHIP WITH SILENT FALLBACKS FOR MISSING FILES — fail loud is the only acceptable failure mode.**

---
## A-08, A-09 — ENG2 (semantic) & ENG3 (end-to-end) — chained to A-07

A-08 (`6065394`) and A-09 (`d2f0d14`) are strictly downstream of A-07. No independent value. Same rules and risks.

**Agent recommendation:** **defer until A-07 is decided and CI stays green on develop**; when approved, port in their own PRs with exact file lists and nothing else.

---
## What this changes for the owner (decision points the matrix cannot answer)

1. **Registry files provenance** (C-01..C-04) is a hard gate for DLCAE integration; nothing else proceeds cleanly.
2. **Information sensitivity** of A-05 determines whether ANY evidence ships to develop (public repo).
3. **CRLF serving files** (`serving/local_inference.py`) risk being overwritten without owner attention — owner must diff.
4. **`test_l076_roadmap.py` semantic divergence** — whichever version lands must reflect A1.1-C/A2-E;
5. **Publication files** must never appear — A-classification decisions on 8c55346/c601a73 are REJECTED.

## Summary matrix

| Item | Files | Depends | Risk | Independent | Recommended decision |
|---|---|---|---|---|---|
| A-01 docs/governance | 8-9 | — | LOW | YES | OWNER TO DECIDE — proposed |
| A-02 DLCAE arch/registry | 12 | C-decision | LOW-MED | NO | OWNER TO DECIDE — conditional |
| A-03 L0.6.1 training pipeline | 22 | A-02 | LOW-MED | NO | OWNER TO DECIDE |
| A-04 L0.6.2 inference | 12 | A-03 | MEDIUM (CRLF) | NO | OWNER TO DECIDE |
| A-05 evidence only | ~14 | A-03 | LOW-MED | semi | OWNER TO DECIDE |
| A-06 preflight | 25 | A-02/03 | LOW | NO | OWNER TO DECIDE |
| A-07 ENG1 runtime | 18 | A-02/03/04+C | LOW-MED | NO | OWNER TO DECIDE — blocked on C |
| A-08 ENG2 semantic | 19 | A-07 | LOW-MED | NO | DEFER |
| A-09 ENG3 e2e | 11 | A-08 | LOW | NO | DEFER |
| A-recovery `78b6cdf` | 9 | n/a | n/a | **DO NOT PORT** | OWNER_REJECTED (recommended) |
| A-publication `8c55346`/`c601a73` | 790 | n/a | n/a (release content) | **DO NOT PORT** | OWNER_REJECTED (recommended) |
