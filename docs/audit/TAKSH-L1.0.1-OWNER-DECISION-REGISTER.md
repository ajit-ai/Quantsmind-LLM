# TAKSH — L1.0.1 Owner Decision Register (2026-11-12)

**Baseline:** planning branch `6662cce2765d47b5a9f173fc9d4c9ca09dba37a9` (== remote).
**Status legend:** PROPOSED (by agent, not authorized) · OWNER_APPROVED · OWNER_REJECTED · DEFERRED · BLOCKED · NOT_DECIDED.
**Only explicit owner decisions recorded are marked OWNER_APPROVED.** Silence is never consent; a recommendation is never an authorization.
**Hard restrictions in force:** TAKSH-OWNER-DECISION-2026-11-11 — A1.1-C (preparation-only) · A2-E (no compute). No training, no checkpoint generation, no data acquisition, no spending, no publication, no tags/releases, no branch protection changes, no Karkain/MicroQuantum modifications, no `TAKSH_ENABLE_TRAINING_CI`.
**Product status preserved:** v1.0.0 GA NOT ESTABLISHED · actual coding capability NOT_EVALUATED · no language qualified · benchmark 0/30 stands · KARK-BENCH 0 tasks · Karkain DISCOVERED · `v180_sft` never substituted.

---

## Reference owner decisions (explicitly recorded, preserved as-is)

| ID | Decision | Status |
|---|---|---|
| D-0001 | A1.1-C — Authorize **preparation and preflight only; no training**. Date 2026-11-11; reference `TAKSH-OWNER-DECISION-2026-11-11` | **OWNER_APPROVED** |
| D-0002 | A2-E — **No compute authorization** of any kind. Date 2026-11-11 | **OWNER_APPROVED** |
| D-0003 | L0.8.0 entry contract defined (`docs/L075_L080_ENTRY_CONTRACT.md`); the owner decision is preparation-only — training explicitly not authorized | **OWNER_APPROVED** |
| D-0004 | v1.0.0 source release published 2026-10-21 (tag exists); PyPI upload blocked on credentials — publication not completed, release not GA | **OWNER_APPROVED** (historical act) |
| D-0005 | CI recovery integrated into develop via PRs #2/#3; develop→main merge executed by owner 2026-11-12T04:56Z (d50de3e) | **OWNER_APPROVED** (owner action) |

## Decision A — Feature milestone integration (per-PR decisions)

Source: `docs/audit/TAKSH-FEATURE-BRANCH-COMMIT-MATRIX.md` + `TAKSH-DEVELOP-INTEGRATION-INVENTORY.md` (in this branch). Merged into develop would transfer 19 commits, including a 790-file publication package. `78b6cdf` is **superseded by develop PRs #2/#3** — must not be re-ported.

| ID | Item | Source commit | Proposal | Status |
|---|---|---|---|---|
| A-01 | Roadmap docs & governance PR | `84888e6` (subset) | Cherry-pick docs + `configs/roadmap/taksh-v3-roadmap.yaml` + `TAKSH-RELEASE-GOVERNANCE.md` + `TAKSH-ROADMAP-CHANGELOG.md` + validator; zero code commit | NOT_DECIDED |
| A-02 | DLCAE architecture & dynamic registry | `45f0dc9` | docs (DLCAE-ARCHITECTURE, DYNAMIC-LANGUAGE-REGISTRY, KARKAIN-INTEGRATION, DLCAE-RESEARCH-PLAN, TAKSH-ARCHITECTURE), registry configs, validator, tests | NOT_DECIDED |
| A-03 | L0.6.1 DLCAE-aware training pipeline | `1fef89d` | `training/lifecycle|manifest|geometry|…` changes, `evaluation/l0601_*` validators, tests | NOT_DECIDED |
| A-04 | L0.6.2 local inference qualification | `9721aa6` | `serving/inference_engine.py`, `serving/local_inference.py` (CRLF-preserved files; owner's working-tree copies differ), `evaluation/l062_*` | NOT_DECIDED |
| A-05 | L0.7.x evidence package (freeze + preflight evidence) | `27c7b3a`, `68dc5b7` | Port documents/reports only; do **not** port corpus artifacts by default | NOT_DECIDED |
| A-06 | L0.8.0-PRE preflight | `262ae69` | Port preflight validator + aggregate gate (preparation-only, NO-GO honored) | NOT_DECIDED |
| A-07 | L1.0.0-ENG1 DLCAE runtime | `efcda29` | `taksh/dlcae/` engines; requires A-02 + A-03 to exist or equivalent in develop | NOT_DECIDED |
| A-08 | L1.0.0-ENG2 semantic validation | `6065394` | `taksh/dlcae/semantic/*` | NOT_DECIDED |
| A-09 | L1.0.0-ENG3 end-to-end integration | `d2f0d14` | `taksh/dlcae/coding/*` + dev tasks | NOT_DECIDED |
| A-10 | Recovery duplicate re-port | `78b6cdf` | **DO NOT PORT — superseded by develop (PRs #2/#3)** | **OWNER_REJECTED** (agent recommendation; owner must still concur) |
| A-11 | Publication package `8c55346` + `c601a73` |
| — | — | **NOT for integration — publication staging snapshot; 790 files; not product code** | **OWNER_REJECTED** (agent recommendation; owner must concur) |

**Inter-milestone dependency chain (hard):** A-02 → A-03 → A-04 → A-06 → A-07 → A-08 → A-09. A-01 can run independently. A-05 can run after A-03.

**Agent recommendation (non-binding):** approve A-01..A-09 as small sequential PRs **only in develop**, each with focused tests and this checkpoint's authorization; none authorized by this task. Port **nothing** else.

## Decision B — Benchmark artifact policy

| ID | Option | Status |
|---|---|---|
| B-01 | Freeze python benchmark **integrated into develop now** | NOT_DECIDED |
| B-02 | Frozen benchmark stays feature-branch scoped; **isolated qualification workflow only** inside owner-dispatched runs | NOT_DECIDED |
| B-03 | **Benchmark integration deferred until model availability exists** | NOT_DECIDED |
| B-04 | Owner rejects existing `l02_python_bench_v1.json` and demands replacement (new bench, fresh counsel-approved provenance) | NOT_DECIDED |

**Identities preserved:** frozen Python benchmark SHA `854c7ca833…` (0/30); multilingual benchmark SHA `7a839060…` (64 tasks, separate artifact, separate decision).

**English task inventory (l02_bench):** 30 problems; in frozen form since L0.2 milestone; never rescored; 0/30 stands until a model artifact is measured and owner authorizes a qualification re-run.

**Agent recommendation (non-binding):** prefer **B-02 or B-03** today: keep the artifact in feature branch until model availability; **do not** import a benchmark into develop for demonstration/CI purposes — a benchmark habitually green is a benchmark that was cheated.

## Decision C — Language-registry dependencies

**Untracked owner artifacts required by the A-07 path:**

| File | Exists? | SHA-256 | Committed anywhere? |
|---|---|---|---|
| `configs/feasibility/l051_languages.yaml` | ✓ working tree | `075d4b568f05…` (30,924 B) | no |
| `evaluation/l051_language_registry.py` | ✓ working tree | `6f1610b5ff5c…` (6,755 B) | no |

| ID | Option | Status |
|---|---|---|
| C-01 | Owner traces the origin of these files (repo provenance, license, author); then supplies and approves them | NOT_DECIDED |
| C-02 | Owner authorizes a separately-scoped recovery milestone to reconstruct/recreate them under audit | NOT_DECIDED |
| C-03 | A-07 (`efcda29`) integration deferred indefinitely; alternative milestones proceed without it | NOT_DECIDED |
| C-04 | A-07 path excluded; alternate DLCAE integration strategy adopted (manual re-implementation under review) | NOT_DECIDED |

**Recommendation by the L1.0.0 matrix:** investigation-only confirmation required — `efcda29` needs these files to validate `python` entries (or the base registry must be excluded by design). *None of these options have been approved.* Registry-integrity damage from fabricated entries poisons every later qualification; **do not reconstruct without owner foresight and counsel review.**

## Decision D — Canonical model and compute path

**Canonical configuration reconfirmed:** `taksh-foundation-111m` — vocab 5,822 · d_model 768 · 12 layers · 12 heads · 6 KV heads · FFN 3,072 · context 2,048 · 110,658,816 params (formula) · RoPE 10000 · RMSNorm · tied embeddings · `configs/foundation/l060_foundation.yaml` (`01a90464…`).

**Canonical checkpoint:** MISSING (never created; no artifact is admissible today).
**`v180_sft`:** GEOMETRY_INCOMPATIBLE (`d_model` 384, `n_heads` 8, `n_kv_heads` 4, `ffn_dim` 1536); `9b541036…`; NEVER re-targetable. Do not re-anchor or rename-and-point; do not run or measure it.

| ID | Option | Status |
|---|---|---|
| D-01 | Owner-provided compatible checkpoint (sponsored weights created under owner's separate authorization) | NOT_DECIDED |
| D-02 | Owner-authorized training on suitable infrastructure (A1.1-A + A2-A/A2-B/A2-C/A2-D per compute class) | NOT_DECIDED |
| D-03 | Separately funded external compute + artifact production (fresh contracts, provider + budget + audited plan) | NOT_DECIDED |
| D-04 | Continue infrastructure development without a model (ENG runtime only; capability tests never executed; product not viable) | NOT_DECIDED |
| D-05 | Pause model-dependent product work (no DLCAE integration, no capability evaluation, roadmap frozen at L0/L1) | NOT_DECIDED |

**Compute envelope estimates (clearly marked estimates, not measured performance):**
- fp32+AdamW optimizer state ≈ 1.77 GB (params+grads+`m`+`v` at fp32); weights fp32 ≈ 443 MB.
- `bf16` mixed-precision working set ≈ 1.33 GB optimizer + activations (unmeasured).
- Training feasibility requires a practical GPU ≥ 16 GiB VRAM (minimum), ideally ≥ 24 GB; host RAM ≥ 16 GB; disk ≥ 80 GB; owner authorization — **none of which A2-E provides**.
- MX150 (4 GiB) is not viable; CUDA is unavailable in the installed PyTorch build (`torch.cuda.is_available()==False`).

**Prerequisites (all required):** authorized data (L0.8.0 entry contract + owner selection), authorized environment (A2 decision), A1.1-A-type training decision, external evaluation/auditor approval for what is published (or not), and a fresh capability decision.

## Decision E — Karkain and multilingual scope

Reconfirmation:

| Fact | Value |
|---|---|
| Language id | `karkain` (54th) |
| Registration status | `DISCOVERED` |
| `.kark` extension | supported via adapter (`taksh/dlcae/karkain.py`, feature branch) |
| Karkain SPEC pin | v1.1.0 (windows/amd64 Stable 1.1.0; commit e612ede; branch feature/v1.2.0-native-independence) |
| KCC execution on this host | **0xC0000135** — cannot validate; every capability verdict = BLOCKED |
| KARK-BENCH tasks | **0** |
| Karkain corpus | **0** tokens |
| KCC-as-oracle stance | preserved (adapter always returns BLOCKED) |
| Multilingual benchmark | `7a839060…` (64 tasks; language ids determined, none qualified) |
| Languages | 53 base + Karkain = 54; registry preserved intact; no language qualified |

| ID | Option | Status |
|---|---|---|
| E-01 | Defer Karkain until KCC reproducible (host repair by owner, alternate host, or owner-sanctioned build) AND TAKSH-KARK-BENCH ≥ 1 authored task approved by owner | NOT_DECIDED |
| E-02 | Authorize a **separate adapter-readiness assessment only** (no capability claim, no benchmark instantiation, no host repair) | NOT_DECIDED |
| E-03 | Remove Karkain from the immediate product scope; keep it exclusively as a future track on the feature branch | NOT_DECIDED |
| E-04 | Preserve Karkain as a documented future language-extension target; freeze current integration assets; revisit at GA review | NOT_DECIDED |

**Agent recommendation (non-binding):** E-01 or E-04. **E-02** only if owner treats it as a learning/diagnostic step. Do not claim any Karkain support — that requires functional qualification on the bench.

**Multilingual policy (paired decision):** the multilingual benchmark remains a **frozen, unmeasured artifact**; 64 tasks stay unverified; no multilingual claim may exist before capability work. Choice stands: treat as E-01/B-02 pattern — isolated, unrun, undistributed.

## Decision F — Next milestone proposal

See `docs/audit/TAKSH-L1.0.1-NEXT-MILESTONE-PROPOSAL.md` for the full proposal (§F). Only the *smallest useful* item is proposed, optionally + no implementation work; nothing is authorized here.

## Summary: agent recommendations (NOT DECISIONS)

| Decision | Recommendation (non-binding) |
|---|---|
| A | Approve A-01…A-09 as a sequence of PRs (develop only, per-PR owner review/CI); A-10 and A-11 refused |
| B | Prefer B-02 or B-03; preserve 0/30; no score claims |
| C | Owner must first decide on disposition of untracked registry files before A-07 is viable |
| D | D-04 (continue tooling) or D-05 (pause) are the only options available today; D-01..D-03 need explicit A1.1/A2 actions |
| E | E-01 or E-04; E-02 optional, diagnostic only |
| F | Scoring at proposal stage; no execution authorized |

## Immediate owner answers required

1. `INTEGRATION_SEQUENCE` — yes/no on A-01…A-09 as a sequence of PRs (develop + CI only, no tags/releases)?
2. `BENCHMARKS` — B-01 / B-02 / B-03 / B-04? (and the same for the multilingual artifact)
3. `L051_DEPENDENCIES` — C-01 / C-02 / C-03 / C-04 (and where the original `l051_languages.yaml` came from)?
4. `CANONICAL_MODEL` — D-01 / D-02 / D-03 / D-04 / D-05?
5. `KARKTAIN_SCOPE` — E-01 / E-02 / E-03 / E-04?
6. `NEXT_MILESTONE` — confirm (or replace) the proposal; authorize nothing beyond it.
7. `PRODUCT_RELEASE` — explicit reiteration that no tags/releases/publication may occur inside TAKSH without a separate written decision.
