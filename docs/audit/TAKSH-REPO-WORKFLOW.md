# TAKSH — Repository, Coding, Testing and CI Plan

**Document ID:** `TAKSH-REPO-WORKFLOW-v1`
**Date:** 2026-11-19
**Owner:** ajit-ai
**Parent plan:** `docs/audit/TAKSH-IMPLEMENTATION-PLAN.md`
**Status:** adopted by owner instruction; Git mutations still require explicit per-step approval.
**PR policy (2026-11-19):** all changes to `main` (and `develop`) land **only via pull request**. Direct or local merges to protected branches are not permitted by the agent, even with owner approval of the content; the agent stops at PR creation and asks the owner to review/merge.

---

## 1. Branch strategy

| Branch class | Role | Rules |
|---|---|---|
| `main` | Release lineage | No direct commits; changes arrive **only via PR**. No local merges, no push to `origin/main` by the agent. |
| `develop` | Integration lineage | Same: PR-only; owner-gated. |
| `feature/<track>-<slug>` | Unit of work, e.g. `feature/coding-genius-i1-serving-fixes` | Branch from the approved baseline. Keep one logical unit per branch. Lands in `main`/`develop` via PR. |
| `integration/*`, `planning/*` | Pre-existing lineages | Preserve; do not delete or rename. |

Default baseline for active work is the current foundation lineage tip (`feature/l060-foundation-model-qualification` @ `78b6cdf…`) until the owner authorizes another baseline (e.g. `feature/l060-1-integration-closure`).

## 2. Commit discipline

- **Atomic commits**: one logical change per commit; tests in the same commit or immediately before it.
- **Message format**: `type(scope): summary` — `feat|fix|docs|test|chore|refactor|build|ci(scope): summary`. Scope examples: `serving`, `architect`, `roadmap`, `eval`, `train`.
- **Never commit**:
  - checkpoints / model weights (`runs/**`);
  - protected corpora (`data/l02_overfit/`, `data/l02_sft/`, `data/l03_corpus/`, `data/l04_code_corpus/`) without legal review;
  - publication artifacts (`hf_export/`, `taksh-hf-publication/upload_manifest.json`) without owner authorization;
  - generated token-streams or large artifacts not previously tracked.
- **Owner's pre-existing uncommitted work** is never swept into a commit without explicit owner listing/approval of the paths.
- **Commit body** records evidence: command output, hashes of affected artifacts, and known limitations.

## 3. Testing gates (run in this order)

**Gate G1 — focused, for touched modules only**
```
python -m pytest tests/test_<module_under_change>.py -q
```

**Gate G2 — serving/inference safety (any serving change)**
```
python -m pytest tests/test_m7_serving.py tests/test_openai_server.py tests/test_legacy_serving_security.py tests/test_kv_cache.py tests/test_openai_real_checkpoint.py -q
```

**Gate G3 — baseline integrity**
```
python -m pytest tests/test_architect_package.py tests/test_l060_foundation.py tests/test_token_contract_and_chat.py tests/test_v110_contract.py tests/test_data_integrity.py tests/test_eval_run_manifest.py tests/test_validate_manifest.py -q
```

**Gate G4 — repository integrity**
```
python -m evaluation.roadmap.roadmap_validator
# expected: roadmap valid; milestones=54
```

**Gate G5 — capability (only when a benchmark changes)**
Native v0.1 is immutable; any new benchmark version is `evaluation/native_v03/` and must record suite + output hashes. No run without explicit authorization and bounded resources.

Host constraint: full suite historically exceeds memory/time on this machine; focused suites + recorded limitations are the accepted minimum until CI is authorized.

## 4. CI plan

| Layer | Reality |
|---|---|
| `.github/workflows/ci.yml` | Production trigger policy (push to `develop`/`main`). |
| `.github/workflows/ci-l0704-manual-qualification.yml` | Manual qualification workflow; dispatch blocked from non-default branch. |
| Local gates G1–G4 | Stand-in until CI from the feature branch is authorized by owner (requires default-branch action). |
| Merge to `main` | Only via a **pull request** after all local gates pass **and** CI evidence exists or CI limitation is explicitly recorded. The agent creates the PR (if authorized to push) and stops; merging is an owner action. |

CI authorization is an owner action; this document does not grant it.

## 5. Merge-to-main procedure (PR-required)

1. Owner designates the feature branch and full SHA.
2. All gates G1–G4 green; failures recorded, not hidden.
3. Push the feature branch to `origin` only with explicit owner authorization (network operation).
4. Open a PR from the feature branch into `main` (or `develop` if agreed) describing scope, evidence, test results and limitations.
5. The agent stops at PR creation and explicitly asks the owner to review and merge. **The agent never clicks/executes the merge.**
6. Owner merges via PR (or authorizes a specific merge action, still never a direct push).
7. Tag/release only with separate owner authorization; publication is never implied by a PR merge.

## 6. Rollback

- Before merge: delete the feature branch (owner decision); working tree unchanged.
- After merge: revert via a new `fix(revert):` commit or owner-authorized branch reset; never history rewrite.

## 7. Mapping to the implementation plan

| Unit | Work unit / candidate branch | Merge target |
|---|---|---|
| I1 | `feature/coding-genius-i1-serving-fixes` | `main` (owner approval) |
| I2 | `feature/coding-genius-i2-native-integration` | `main` (owner approval) — after closure-baseline decision |
| I3 | `feature/coding-genius-i3-ci-review` | `main` (owner approval) |
| I4 | `feature/coding-genius-i4-capability-recovery` | `main` (owner approval) — after training authorization |
| I5/I6 | `feature/coding-genius-i5-serving` / `-i6-product` | `main` (owner approval) |
| I7 | release unit | release tag (owner authorization) — not a merge |

## 8. Standing prohibitions

No force-push · no push without explicit owner command · no local commits/merges to `main`/`develop` (PR-only) · no agent-side PR merge · no tag/release/publication without authorization · no committing protected corpora/checkpoints/publication artifacts · no claiming CI exists when it is manually blocked · no claiming capability from unexecuted benchmarks.
