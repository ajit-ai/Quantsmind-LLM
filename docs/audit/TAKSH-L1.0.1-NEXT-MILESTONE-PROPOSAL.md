# TAKSH — L1.0.1 Next-Milestone Proposal (2026-11-12)

**Authorization:** none granted by this document. Every item here requires a specific owner decision recorded first. No implementation work has been started under this proposal.
**Basis:** verified branch `6662cce` (planning), `0215c6d` (develop), `d50de3e` (main), `78b6cdf` (feature).

---

## Decision the proposal needs first (frozen silence-preserved)

The owner must answer the A–F questions in `TAKSH-L1.0.1-OWNER-DECISION-REGISTER.md` before this proposal becomes actionable. **If the owner provides no answers, nothing below proceeds.**

## Candidate next milestones — ordered by risk and value

### Candidate 1 (recommended): **L2.0.0-DOCS-GOV** — documentation and governance integration

A single, tiny documentation+governance PR to develop containing only the roadmap registry and release governance:

- `configs/roadmap/taksh-v3-roadmap.yaml` (cherry-picked from `84888e6`)
- `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md`
- `docs/TAKSH-RELEASE-GOVERNANCE.md`
- `docs/TAKSH-ROADMAP-CHANGELOG.md`
- `evaluation/roadmap/roadmap_validator.py` (new)
- `tests/test_l076_roadmap.py` (subset adjusted to develop baseline; the L0.8.0-PRE-only owner-decision record is NOT ported)
- `.pytest_a` entry: none — focused tests already exist

**Why first:** smallest content; zero code change; gives develop the machine-readable milestone ledger; unblocks later PRs that reference the registry; exercises the cherry-pick discipline needed for everything after.

**Scope:** documentation + validator + one test file. **Not in scope:** any DLCAE code, any training target/config, any benchmark import, any Python 3.11 validation, any master feature-branch merge.

**Required starting branch:** `develop` @ `0215c6d` (re-verify) with its isolated CI recovery intact.
**Starting pipeline:** `feature/l060-foundation-model-qualification @ 78b6cdf` as cherry-pick source only.

**Files affected:** the six files above (new paths `evaluation/roadmap/roadmap_validator.py`, `tests/test_l076_roadmap.py`).

**Focused tests:** `tests/test_l076_roadmap.py` (28 tests expected green) plus `git diff --check`.

**Deliverables:** one PR into develop, full CI green; machine-readable validator output attached; zero capability claims.

**Required owner authorization:** explicit "Candidate 1 is authorized" (and a merge decision in its own PR; owner still merges).
**Exclusions explicit:** no CHANGELOG re-entry claiming v1.0.0 GA; no claims about any milestone; no registry edits outside owner review; TAKSH_ENABLE_TRAINING_CI unset; no benchmark ports; no tags/releases.

**Exit criteria:** PR merged by owner after green CI; develop CI remains green; v1.0.0 GA not established; no other branch touches.

**Estimated cost:** ≤ 1 h. **Risk:** LOW. **Independent:** yes.

---

### Candidate 2 (only after Candidate 1 is green and owner approves): **L2.0.1-DLCAE-ARCH** — DLCAE architecture port

Cherry-picked docs + registry + validator + KARK-BENCH schema (A-02): docs/architecture/*, `configs/foundation/l060r1_dynamic_language_registry.yaml`, `evaluation/l060r1_dynamic_registry.py`, `evaluation/l060r1_kark_bench_schema.py`, `tests/test_l060r1_architecture_alignment.py`, `reports/l060r1_*.json` — **excludes** `configs/feasibility/l051_languages.yaml` and `evaluation/l051_language_registry.py` (blocked on C-decision).

**Risk:** registry contracts must allow base registry absence without failing silently; the validator already tolerates it via guarded imports on feature branch.
**Requires:** Candidate 1 merged; owner explicitly reaffirms A1.1-C/A2-E; CI green on develop.

---

### Candidate 3 (optional, add-on if prior steps pass): **L2.0.2-CI-HARDEN** — CI hardening for 3.11

Add Python 3.11 to the CI matrix **without** changing the training-CI gate semantics:

- `ci.yml`: `strategy.matrix.python-version: ['3.11', '3.14']` on the non-training job only.
- `ci-l0704-manual-qualification.yml`: no change (manual govern).
- Fixed, reproducible install of torch CPU and requirements.
- No benchmark downloads; `--no-deps` where safe; timeout guard kept.

**Why:** current host is 3.14; 3.11 never executed anywhere. Documenting `3.14` only = untruth; adding 3.11 is the honest fix.

**Risk:** low; honest evidence could reveal failures — each failure must be logged, not hidden.
**Requires:** explicit owner go for a CI change, plus whatever else is in flight is stable.

---

### Explicitly OUT of scope for this proposal

- Integration of `8c55346`/`c601a73` (publication staging — never without an owner decision of a release nature).
- Any Karkain work beyond the preservation case; any `v180_sft` execution; any Python 3.11 runtime training; any external data; any triggering of `TAKSH_ENABLE_TRAINING_CI`; any new milestone codes (L2.* allowed but L2.0.3+ requires owner ordering).

## Cost/effort estimates

| Candidate | Scope | Est. time | Dependencies | Owner actions |
|---|---|---|---|---|
| 1 — DOCS-GOV | ~6-9 files | ≤ 1 h | none | review + merge |
| 2 — DLCAE-ARCH | ~12 files | 2–4 h | candidate 1 + C-decision | review + merge |
| 3 — CI-HARDEN | `.github/workflows/*` (+maybe pytest) | ≤ 2 h | candidate 1 | review + merge |

## What the owner needs to say to start

1. Any of: "Authorize Candidate 1", "Candidate 1 + 2", "Candidate 1+2+3", or "nothing".
2. Explicit yes/no on `v1.0.0 GA` status remaining not established (presumed no action).
3. Explicit "I do not authorize training" — restating A1.1-C/A2-E in the decision record.
