# TAKSH — Foundation Programme Status Register

**Document ID:** `TAKSH-FOUNDATION-PROGRAMME-STATUS`
**Date:** 2026-11-20
**Branch:** `feature/coding-genius-cap-local`
**Scope:** central register for the coding-only genius programme. Statuses are never self-assigned upward; they follow the gates they record.

Status vocabulary: `NOT_STARTED` · `IN_PROGRESS` · `BLOCKED` · `PRESERVED_UNTRACKED` · `IMPLEMENTED_NOT_QUALIFIED` · `RECOVERED_NOT_INTEGRATED` · `READY_FOR_OWNER_CONSOLIDATION_APPROVAL` · `OWNER_APPROVAL_REQUIRED` · `QUALIFIED`

---

## Register

| Unit | Title | Status | Evidence | Gate / next action |
|---|---|---|---|---|
| V0 | Recovery, branch & baseline audit | **READY_FOR_OWNER_CONSOLIDATION_APPROVAL** | `docs/audit/TAKSH-STEP0-BASELINE-CONSOLIDATION.md`, `TAKSH-STEP0.1-BRANCH-CONSOLIDATION-MANIFEST.md`, `TAKSH-STEP0.2-NATIVE-RECOVERY-AND-CI-LINEAGE.md`, `TAKSH-STEP0.3-CONSOLIDATION-PLAN.md` | Owner selects baseline; I0 |
| I0 | Owner baseline acceptance | **OWNER_APPROVAL_REQUIRED** | Current branch `78b6cdf…`, closure worktree `87b8761…` verified present with 14 recoverable native artifacts; `git worktree list` confirms 3 linked worktrees | Owner must approve baseline selection (closure `87b8761…` vs current) before any I2 integration |
| I1 | Quick correctness fixes (Gradio stream, tokenizer default, KV-cache default, top_k DoS) | **QUALIFIED_WITH_LIMITATIONS** | `serving/code_chat.py`, `serving/local_inference.py`, `serving/completion_server.py`; `pytest tests/test_m7_serving.py tests/test_openai_server.py tests/test_legacy_serving_security.py tests/test_kv_cache.py tests/test_openai_real_checkpoint.py tests/test_architect_package.py tests/test_l060_foundation.py -q` → 68 + 79 passed | I1 gate met (focused suite green). C3 flipped to KV cache on (equivalence tested). Real per-token streaming deferred to I5. Full host suite not re-run (environment constraints) |
| I2 | Native artifact integration | **RECOVERED_NOT_INTEGRATED** | 14 files inventoried in closure worktree; checkpoint `9b54…` & tokenizer `554c…` hashes verified | Owner approval + closure baseline selection |
| I3 | Recovery/CI selective review | **NOT_STARTED** | `61ae12a`, `30facf5`, `90eba0a`, `61e2dba` assessed; none integrated | Review against L0.7.4 CI |
| V1 | Native diagnostic closure | **IN_PROGRESS (evidence present)** | `evaluation/native_v02/*` and `TAKSH-NATIVE-V0.2-DIAGNOSIS.md` integrated in closure worktree; this branch + #5 integrate v0/v0.1/v0.2 evidence | Accept/reject v0.2 diagnosis as authoritative |
| V2 | Capability recovery | **NEGATIVE RESULT (0/20 syntax/executable/correct)** — aligned-format SFT on v180/l02 with frozen seeds, evaluated against immutable Native v0.1; empty-output drop 16→14 with no syntax/correctness gain. See `docs/audit/TAKSH-STEP0.5-CAPABILITY-RECOVERY.md` | Owner decision required for scale-up; no capability claim | No progress; re-evaluation for I4 recommended |
| V3 | Native serving qualification | **IMPLEMENTED_NOT_QUALIFIED (product)** — serving/integrity only | `serving/foundation_inference.py` qualification, `tests/test_i5_native_serving.py` | Not a capability result; see `docs/audit/TAKSH-STEP0.6-NATIVE-SERVING-QUALIFICATION.md` |
| V4 | Engineering intelligence | **NOT_STARTED** | `taksh/dlcae/coding/orchestrator.py` exists (engineering-qualified only) | After V2/V3 gates |
| V5 | Gradio product integration | **IN_PROGRESS (partial)** | `serving/code_chat.py` works; defects C1/C2 found | I1 fixes then I6 |
| V6 | Release qualification developer preview | **CONDITIONAL_PASS** — packaging/pipeline qualified; capability NOT_EVALUATED; publication NOT authorized | See `docs/audit/TAKSH-STEP0.9-RELEASE-QUALIFICATION.md` | Publication is owner decision; evidence regenerated per release |
| V7 | Dense scale-up (100M→3B, tokenizer v2) | **BLOCKED (owner gates)** | Configs/model cards exist (v190/…/v300), none trained | Training authorization + GPU |
| V8 | Repository intelligence | **NOT_STARTED** | Roadmap L2.0–L2.10 | After V7 |
| V9 | MoE scale-up | **DEFERRED (premature)** | No MoE implementation yet | Only after dense proof |
| V10 | World distribution | **NOT_STARTED** | `hf_export/` scaffolding exists (not authorized) | Owner publication authorization |
| V11 | Local free-resource capability experiment (Step 0.10) | **NEGATIVE RESULT — 0/20 correct** — benchmark-clean local corpus; raw/template mismatch isolated; 1,000-step SFT improved syntax only; higher-LR continuation regressed | `docs/audit/TAKSH-STEP0.10-LOCAL-CAPABILITY-EXPERIMENT.md`, `TAKSH-CAP-CORPUS-PROVENANCE.md`, `evaluation/capability/*` | Owner decision on authorised GPU/compute and larger corpus; no local CPU path remains |

---

## Verified identities (as of 2026-11-14)

- Checkpoint `runs/v180_sft/checkpoint_last.pt`: 345,596,421 bytes, SHA-256 `9B54103695054ECB30D8521FFCEB62F443D2D1FD8D064993EDC2D5070CB7F666` — match
- Tokenizer `tokenizer/taksh_code_bpe_v1.json`: 411,743 bytes, SHA-256 `554C8E7572D58923A0766F114B5F419B14E68BDFF3A738803B30728EC3FFCD92` — match
- Frozen Python benchmark: **0/30** (unchanged)
- Native v0.1: **0/20 correct** (unchanged)

## Protected exclusions (never commit/redistribute without owner + legal)

- `data/l02_overfit/`, `data/l02_sft/`, `data/l03_corpus/` (91.22 MB, CC-BY-SA mixed), `data/l04_code_corpus/` (32.81 MB, mixed)
- `runs/v180_sft/checkpoint_last.pt` (untracked)
- `taksh-hf-publication/upload_manifest.json` and `hf_export/` publication scaffolding
- `runs/` checkpoints generally

## Non-claims

No capability, no GA, no publication is claimed by this register. Every status above is evidence-bounded; upgrades require the gate named next to them to pass first.
