# TAKSH Step 0 — Baseline Branch Audit and Stabilization Report

**Date:** 2026-11-14
**Mode:** local engineering audit only; no merge, tag, push, release, training or publication performed
**Branch:** `feature/l060-foundation-model-qualification` @ `78b6cdf`
**Remote state:** local refs only; no fetch was performed by this audit

---

## 1. Verdict

The working tree is **not release-ready and not safely consolidatable by automation**.

It contains substantial local source, documentation, test and experiment work plus **~124 MB of untracked corpus/data output**. The current branch is a valid engineering foundation to continue work, but it must not be merged to `develop` or `main` without owner review, selective commits and explicit approval.

No capability is created by this report. The frozen Python benchmark remains 0/30.

---

## 2. Branch topology

| Branch | Relationship to current HEAD | Notes |
|---|---|---|
| `feature/l060-foundation-model-qualification` (HEAD) | baseline | 18 commits ahead of local `develop`; 17 ahead of local `main` |
| `develop` | ancestor of HEAD | HEAD has the L0.6/L0.7/L0.8-PRE/ENG work; local `develop` is stale as an implementation branch |
| `main` | ancestor of HEAD | L0 work continues after the v3.0.0 source-release merge |
| `feature/l060-1-integration-closure` | HEAD + 4 commits | descendant; can be evaluated as a fast-forward candidate after review and tests |
| `feature/l060-unified-provider-integration` | HEAD + 1 commit | same provider lineage; compare with l060-1 before combining |
| `feature/taksh-five-version-ga-program` | diverged at `a654e5d`; 37 commits ahead of HEAD | contains recovery, platform, security, agent and GA-hardening work; not a safe automatic merge |
| `integration/taksh-cicd-recovery` | diverged at `a654e5d`; 3 commits ahead | recovery CI fixes; valuable but need review against current CI and local work |
| `planning/l1000-integration-and-ga-readiness` | HEAD lineage? no; diverged at `a654e5d`; 7 commits ahead | planning/audit documents only plus shared ancestor commits |
| `reconcile-recovery-register` | diverged at `a654e5d`; 5 commits ahead | recovery/reconciliation docs/tests |
| `feature/l110-native-model-foundation` / `l111-native-inference-mechanics` | diverged at `a654e5d`; 7–8 commits ahead | native foundation/inference work; separately owned lineage |
| `checkpoint/l101-owner-decisions` | diverged at `a654e5d`; 8 commits ahead | owner-decision checkpoint lineage; not merge material |

Current branch and `feature/l060-1-integration-closure` share the exact recent provider/coding-workflow lineage. The five-version program and recovery branches passed through `a654e5d`, before the L0.7/L0.8 work on the current branch.

---

## 3. Working-tree inventory

| Metric | Count/size |
|---|---:|
| Total porcelain entries | 107 |
| Modified tracked files | 6 |
| Untracked files/paths | 101 |
| Largest untracked path | `data/l03_corpus/` — 91.22 MB |
| Second largest | `data/l04_code_corpus/` — 32.81 MB |
| Approximate untracked data size | ~124 MB |

Modified files include:

- `serving/code_chat.py` — loopback-only local UI binding;
- `serving/local_inference.py` — narrow NumPy-only checkpoint allowlist;
- `training/manifest.py` — initialization provenance field;
- `training/pretrain.py` — explicit `--init-from` warmstart and grad-norm logging;
- `training/sft.py` — tokenizer/seq_len/model-config path resolution and vocab mismatch guard;
- `taksh-hf-publication/upload_manifest.json` — publication metadata (not authorized by this report).

Untracked corpus/data paths include `data/l03_corpus/`, `data/l04_code_corpus/`, `data/l02_overfit/`, `data/l02_sft/` and numerous generated reports. These should not be committed or published without an explicit review of provenance, licensing and size policy.

---

## 4. What must not be merged blindly

1. **Five-version program branch** — substantial agent/platform/security work, but it diverged before current L0 qualification work and contains publication/GA artifacts that cannot imply capability or release authorization.
2. **Recovery CI branch** — contains fixes for CI on the recovery lineage; these must be reviewed rather than replayed mechanically.
3. **Large generated corpora** — `data/l03_corpus/` and `data/l04_code_corpus/` are not source-only changes and may carry license/provenance obligations.
4. **HF publication artifacts** — `taksh-hf-publication/upload_manifest.json` and `hf_export/` are publication surfaces; local edits never authorize publication.
5. **Owner decision records** — blank records must remain blank until the owner acts.

---

## 5. Recommended consolidation sequence

### Phase A — OWNER REVIEW (no Git integration yet)

1. Owner reviews `git status --short` and classifies each untracked path:
   - `source/tests/eval` — commit candidate;
   - `generated reports` — explicit keep/archive decision;
   - `data corpus` — legal/provenance review required;
   - `publication` — explicitly excluded from baseline commit.
2. Owner confirms whether `feature/l060-1-integration-closure` should become the consolidated provider baseline.
3. Owner chooses whether to keep the five-version platform branch as a separate lineage or authorize selective cherry-picks.

### Phase B — LOCAL SOURCE COMMIT

1. Stage only reviewed source/config/evaluation/test/report paths.
2. Exclude `data/l03_corpus/`, `data/l04_code_corpus/` and publication surfaces unless separately approved.
3. Run:
   - `taksh-env\Scripts\python.exe -m pytest tests/test_l060_foundation.py tests/test_architect_package.py -q`
   - `taksh-env\Scripts\python.exe -m evaluation.roadmap.roadmap_validator`
   - full regression suite once the environment is clean.
4. Create one local commit on a consolidation branch; do not push without owner approval.

### Phase C — INTEGRATE DESCENDANT PROVIDER WORK

1. Inspect `56916b6..feature/l060-1-integration-closure` for provider integration and coding workflow hardening.
2. Fast-forward or cherry-pick as appropriate after tests.
3. Verify the unified provider contract; do not silently replace native TAKSH behavior.

### Phase D — SELECTIVE RECOVERY/PLATFORM CHERRY-PICKS

1. Recovery CI fixes from `integration/taksh-cicd-recovery` may be candidates for manual review.
2. Five-version platform commits require explicit owner authorization, particularly anything touching release, publication or deployment.
3. No protected branch is updated without owner sign-off.

---

## 6. Verification performed in this audit

| Check | Result |
|---|---|
| `pytest tests/test_l060_foundation.py tests/test_architect_package.py -q` | **42 passed** |
| `pytest tests/test_token_contract_and_chat.py tests/test_v110_contract.py tests/test_data_integrity.py tests/test_eval_run_manifest.py tests/test_validate_manifest.py -q` | **129 passed** |
| `python -m evaluation.roadmap.roadmap_validator` | valid; milestones=54 |
| Branch topology | current branch is 18 commits ahead of local develop |
| Working tree size check | ~124 MB untracked data identified |
| Protected artifacts | not intentionally modified by this report; `reports/l076_roadmap_validation.json` had no diff after validator re-run |

The integrity layer already exists in source: `training/chat_template.py`, `training/tokenizer_contract.py`, `training/checkpoint_schema.py` and related serving/training paths. Step 0 therefore verified it rather than re-implementing it.

The full suite was not re-run because the host has a documented history of environmental temp-directory failures and the working tree contains large untracked generated data. It should be run after the owner classifies untracked files and the source set is frozen.

---

## 7. Immediate Step 0 decisions needed from owner

1. Is `feature/l060-1-integration-closure` the preferred provider baseline?
2. Should the five-version platform branch remain a separate lineage, or are selective cherry-picks authorized?
3. Which untracked corpus/data paths are acceptable to commit, and which must be excluded?
4. Should generated report files be committed or stored outside Git?
5. Is local commit authorization granted for the coding-genius and Design Architect source/docs?

Until those decisions arrive, the correct disposition is: **do not merge, do not push, do not publish; keep the audit and engineering work local.**
