# TAKSH Roadmap Changelog

**Document ID:** `TAKSH-ROADMAP-CHANGELOG`
**Rule:** the roadmap may only change through an explicit, recorded revision. Nothing is ever silently rewritten; every record here stands for history.

---

## Revision 1.0 — 2026-11-11 (L0.7.6)

**Action:** initial consolidation of the v3 roadmap (this revision).

**Changes:**

- Established `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md` as the authoritative forward roadmap to TAKSH v3.0.0, re-based onto the L0–L3 milestone program (25 L0 milestones, 10 v1.0.0/L1 milestones, 11 v2.0.0/L2 milestones, 8 v3.0.0/L3 milestones).
- Retired forward authority of `docs/roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md` for this milestone program (its history and its v1.0.0 record remain authoritative for the v1.0.0–v3.0.0 version-responsibility definition).
- Created machine-readable registry `configs/roadmap/taksh-v3-roadmap.yaml` (schema `taksh.v3_roadmap_registry/1`).
- Created `docs/TAKSH-RELEASE-GOVERNANCE.md` (versioning, evidence thresholds, authorization boundaries, GA gates).
- Added `evaluation/roadmap/roadmap_validator.py` (read-only consistency validation: unique IDs, statuses, dependency integrity, release mapping, evidence existence, markdown cross-reference).

**Preserved unchanged:**

- `docs/ROADMAP.md` (M0–M7 record, v1.0.0 baseline 2026-10-21)
- `docs/roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md` (v1.1.0+ forward reference as of 2026-10-21; now superseded for milestone-program planning by this roadmap)
- Every L0 milestone qualification record through L0.7.5.
- Every frozen artifact, benchmark, checkpoint, tokenizer hash and registry.

**Authority note:** no milestone state, no owner decision, and no capability is changed by this revision. The seven L0.7.x blockers (owner authorization, compute, canonical 111M checkpoint, CI execution on this branch, Python 3.11 runtime, frozen Python benchmark 0/30, Karkain toolchain) remain exactly as recorded. L0.8.0 remains BLOCKED. The blank owner decision record remains blank and unmodified.

**Owners decision made this revision:** none. Records exist only to describe the path; they confer nothing.

---

## Reading the changelog

| Revision | Date | Milestone | Effect |
|---|---|---|---|
| 0.x (historical) | 2026-09..10 | M0–M7; v1.0.0, v1.1.0..v3.0.0 | forward planning baseline; superseded by this document |
| 1.0 | 2026-11-11 | L0.7.6 | v3 milestone program established |
