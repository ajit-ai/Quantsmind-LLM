# TAKSH Step 0.3 — Controlled Baseline Consolidation Plan

**Date:** 2026-11-14  
**Mode:** read-only evidence-based integration audit; **no Git or source mutation performed**  
**Provisional baseline candidate:** `feature/l060-1-integration-closure` @ `87b876114a698b334ac7d15c6d1eee32f9138ffd`  
**Primary branch under comparison:** `feature/l060-foundation-model-qualification` @ `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1`

## Completion status

`READY_FOR_OWNER_CONSOLIDATION_APPROVAL`

This is approval readiness, not authorization to merge, commit, push, publish or train. No consolidation was performed.

---

## 1. Executive summary

1. The current foundation branch is not an authoritative product baseline by itself; it is a verified engineering foundation with no demonstrated coding capability.
2. `feature/l060-1-integration-closure` is a direct, clean descendant of current HEAD and is the lowest-risk provisional source baseline for the L0/provider lineage.
3. Native Model v0/v0.1/v0.2 work exists as untracked evidence in the closure worktree. It is not committed on the closure branch or any inspected branch.
4. The six recovery/CI commits are real and independently useful, but they are absent from both current HEAD and closure. They overlap L0.7.x CI/release qualification work and must be reviewed individually.
5. Checkpoint and tokenizer identities match the reported Native v0.1 values. This verifies file identity only; it does not verify model capability or load safety beyond the current code paths.
6. Protected corpora remain untracked, duplicated byte-for-byte by file list between primary and closure worktree, and must be excluded from any consolidation.
7. The Native v0.1 benchmark remains immutable and reports 0/20 correct generations. No code-capability claim is permitted.
8. `docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md` does not exist and was **not** created or modified. Its proposed status is included below instead.

**Recommended direction:** adopt `feature/l060-1-integration-closure` as the provisional source baseline only after owner approval; preserve native artifacts as untracked evidence until integration is separately authorized; keep corpora and publication surfaces excluded.

---

## 2. Exact repository and worktree identities

| Worktree | Repository | Branch | Full HEAD | State |
|---|---|---|---|---|
| `F:/Codes/Git/Quantsmind-LLM` | TAKSH primary | `feature/l060-foundation-model-qualification` | `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` | tracked modifications and untracked files preserved; 109 status entries after prior audits |
| `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-l060-1-closure` | same repository | `feature/l060-1-integration-closure` | `87b876114a698b334ac7d15c6d1eee32f9138ffd` | 2 modified tracked files, 11 untracked paths; same repository and same closure worktree as Step 0.2 |
| `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-l060-unified-provider` | same repository | `feature/l060-unified-provider-integration` | `820624e9d8a27587af228be23d6c90716a734702` | clean |
| `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-cicd-recovery` | same repository | `feature/taksh-five-version-ga-program` | `978826bfc031e46afbdcf71792da3baebb5a3c48` | 3 untracked paths: `.taksh/`, `reports/qualification/`, `reports/taksh_cicd_documentation_recovery.json` |

`git worktree list --porcelain` confirms:

- primary worktree at `F:/Codes/Git/Quantsmind-LLM`;
- `integration/taksh-l060-1-closure`;
- `integration/taksh-l060-unified-provider`;
- `integration/taksh-cicd-recovery` — note the path is misleading: the branch there is `feature/taksh-five-version-ga-program`, not an integration branch.

No worktree was created or modified by this audit.

### Closure worktree status (observed, not altered)

```text
 M serving/foundation_inference.py
 M serving/native_provider.py
?? data/l02_overfit/
?? data/l02_sft/
?? data/l03_corpus/
?? data/l04_code_corpus/
?? docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md
?? docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md
?? evaluation/native_v01/
?? evaluation/native_v02/
?? tests/test_native_v01_evaluation.py
?? tests/test_native_v02_diagnosis.py
?? tests/test_native_v160_inference.py
```

### Unified-provider worktree status

Clean.

### Five-version worktree status

```text
?? .taksh/
?? reports/qualification/
?? reports/taksh_cicd_documentation_recovery.json
```

Untracked; not promoted as baseline evidence.

---

## 3. Branch and commit comparison

### Local branch relationships

| Branch | Merge-base with HEAD | Ahead of HEAD | Behind HEAD | Interpretation |
|---|---|---:|---:|---|
| `feature/l060-1-integration-closure` | `78b6cdf...` | +4 | 0 | direct descendant; provisional baseline |
| `feature/l060-unified-provider-integration` | `78b6cdf...` | +1 | 0 | provider-only lineage |
| `develop` (local) | local `develop` is ancestor | +18 HEAD ahead | 0 | local develop stale |
| `main` (local) | local `main` is ancestor | +17 HEAD ahead | 0 | local main stale |
| `feature/l110-native-model-foundation` | `a654e5d` | HEAD 19 ahead; l110 7 ahead | — | native lineage, divergent |
| `feature/l111-native-inference-mechanics` | `a654e5d` | HEAD 19 ahead; l111 8 ahead | — | native mechanics lineage |
| `feature/taksh-five-version-ga-program` | `a654e5d` | HEAD 19 ahead; five 37 ahead | — | large divergent platform line |
| `integration/taksh-cicd-recovery` | `a654e5d` | HEAD 19 ahead; recovery 3 ahead | — | recovery CI lineage |
| `origin/develop` | `a654e5d` | HEAD 19 ahead; origin/develop 6 ahead | — | recovery/reconciliation lineage absent locally |
| `origin/main` | `a654e5d` | HEAD 19 ahead; origin/main 7 ahead | — | remote main lineage absent locally |

### Five-version relationship

Five-version contains the recovery lineage and a reimplemented/copied native lineage. It is not a safe wholesale merge into the L0 foundation because:

- it diverged from `a654e5d`, before `c601a73` and the L0.6/L0.7 workchain;
- `bb3311f` contains native-foundation/native-inference content with the same blob hashes found in `l111`;
- it contains separate publication, deployment, IDE and GA surfaces;
- it contains a root-cause/attention-mask repair and broad platform changes that need independent qualification;
- release/GA documentation cannot imply current native-model capability.

### Closure baseline relationship

`feature/l060-1-integration-closure` contains current HEAD plus:

- `56916b6` unified provider architecture integration;
- `da7c885` L0.6/provider close-out;
- `ecd30fe` unified provider contract/workflow hardening;
- `87b8761` closure ledger/qualification report.

Current primary's six modified tracked files match the closure HEAD for all except the publication manifest. Most pre-existing untracked L0/L0.5/L0.6 files also match closure HEAD, with only the already-identified current-local differences remaining.

---

## 4. Six recovery/CI commit assessments

| SHA | Subject/date | Parent(s) | Changed files | Purpose | Recommendation |
|---|---|---|---|---|---|
| `61ae12a9fe18863bdef963b7deae3b6011713a32` | fix(ci): integrate TAKSH recovery and historical regression fixes<br>2026-11-12 | `a654e5dc13dc77e925b3981d09792bade43a970a` | `.github/workflows/ci.yml`, `.github/workflows/ci-l0704-manual-qualification.yml`, README/STATUS, recovery audit docs, `evaluation/m6_sovereign.py`, `evaluation/taksh_cicd_recovery_check.py`, S2/M6/v150/v160/v180 tests/fixture | repair CI against historical regressions and bring recovery fixes forward | **PARTIAL — review carefully**. Do not cherry-pick wholesale. Extract only CI/testing safeguards after conflict review with current L0.7.4 qualification |
| `30facf553f0f498e7f994688f69da1c289224f59` | fix(ci): port adapter preflight reorder and manifest origin sanitization from recovery<br>2026-11-12 | `61ae12a` | `data/python_v1/benchmarks.py`, `evaluation/m5_adapters.py` | benchmark/adapter correctness and manifest-origin sanitization | **PARTIAL**. Likely independently useful but must be compared against current L0.7.1/L0.7.2 corpus/manifest work before integration |
| `90eba0acc217ecb507b8c20f373f06e9a2101c40` | fix(ci): load repository tokenizer in SFT dataset validation step<br>2026-11-12 | `30facf5` | `.github/workflows/ci.yml` | prevent `None` tokenizer in CI validation | **DEFER/PARTIAL**. Depends on final CI shape after `61ae12a`; useful targeted fix, low standalone risk but wrong in isolation |
| `8f1fbda2d25cf19c0acbd1c835c9efc903981eea` | Merge pull request #2 from integration/taksh-cicd-recovery<br>2026-11-12 | `a654e5d 90eba0a` | same as `61ae12a`+`30facf5`+`90eba0a`, plus merge metadata | recovery PR merge | **DEFER**. Merge commit, not an integration unit; use the individual commits |
| `61e2dba8b880df445101ea05001ca605d2e57241` | docs(recovery): reconcile defect register post-merge (develop green, PR #2)<br>2026-11-12 | `8f1fbda` | `docs/audit/TAKSH-RECOVERY-DEFECT-REGISTER.md` | reconcile defect register and PR state | **PARTIAL/DOCUMENTARY**. Integrate only after reviewing whether register remains authoritative for current baseline |
| `0215c6d039fceffd334f8931aa96a3287010a580` | Merge pull request #3 from reconcile-recovery-register<br>2026-11-12 | `8f1fbda 61e2dba` | defect register merge | recovery reconciliation PR merge | **DEFER**. Merge commit; not an independent change |

### Independence assessment

- `90eba0a` depends on `30facf5` and `61ae12a`; it is not useful as an isolated cherry-pick.
- `8f1fbda` and `0215c6d` are merge commits and should not be cherry-picked as code changes.
- `61ae12a` has the broad blast radius and conflicts most with the current L0/ENG lineage.
- Recovery CI commits are valuable release/regression safeguards, but they do not unblock native model capability.

**No recovery/CI commit should be integrated wholesale.** Any integration must be file-level, conflict-reviewed and followed by focused CI/regression tests.

---

## 5. Native v0/v0.1/v0.2 artifact inventory

All files below are in the closure worktree unless marked otherwise. None is committed on an inspected branch.

| Artifact | Worktree path | Status | Bytes | SHA-256 | Purpose | Complete/tested? | Duplicates? | Destination proposal |
|---|---|---|---:|---|---|---|---|---|
| `foundation_inference.py` | `.../serving/foundation_inference.py` | modified tracked vs closure HEAD | 4,902 | `1839053F46DDB0B7A8CFC375061D76CD0F97DCF04FCA102386EC83A94C47292D` | safe foundation loading, bounded generation, metadata reporting | code complete; tests associated only after v0/0.1/0.2 review | primary has older `3623dc70...`; closure HEAD has `3623dc70...` | PRESERVE for approval; then INTEGRATE |
| `native_provider.py` | `.../serving/native_provider.py` | modified tracked vs closure HEAD | 10,931 | `D3FEFDAD4CA157FC0FDC52A41AED90E6CB5833B907CAAA4B4A33B77BD82E1559` | native provider identity/capabilities/probe wiring | code complete; task-specific provider tests exist | closure HEAD and unified have different versions | PRESERVE; choose closure variant |
| `test_native_v160_inference.py` | `.../tests/test_native_v160_inference.py` | untracked | 9,320 | `CB31C9C79B7422825BC79E3171727E8C1F255EF48FE5545BE8758894134BB31F` | native v0 mechanics/loading tests | present; not executed by this audit | none found | PRESERVE; INTEGRATE only on owner approval |
| `native_v01/__init__.py` | `.../evaluation/native_v01/__init__.py` | untracked | 242 | `BD780569398906D2910855F0969F92026E5EB6FF860E7B0D629616FED74A5DE0` | benchmark package boundary | present | no | PRESERVE; INTEGRATE later |
| `native_v01/tasks.py` | `.../evaluation/native_v01/tasks.py` | untracked | 14,525 | `152A368FFF9B20EBF005902FEC91AFCE7AAE98C236C7A6F192B0107E39867773` | immutable 20-task benchmark definitions | present; record/hash preserved from Step 0.2 | no | PRESERVE; never alter |
| `native_v01/runner.py` | `.../evaluation/native_v01/runner.py` | untracked | 20,962 | `9C06F7F11DB26A454ED4E78F39FD29BC8C2CFD9D9CD2BACE766A9A3211237C3A` | real native inference evaluator with objective checks | present; benchmark results recorded; not rerun here | no | PRESERVE; INTEGRATE later |
| `native_v01/baseline_results.json` | `.../evaluation/native_v01/baseline_results.json` | untracked | 32,223 | `C8DBAF6F56E7EC950A817D055E741DB65D35B3B5613C734E8996773C51B5CBAE` | recorded 0/20 benchmark evidence | present | no | PRESERVE; do not regenerate |
| `test_native_v01_evaluation.py` | `.../tests/test_native_v01_evaluation.py` | untracked | 6,222 | `386FE21870272633347E94B2544E5C49F5BE086558FD7CA651CB60F78ADEE88B` | tests for benchmark runner/scoring/sandbox | present | no | PRESERVE; INTEGRATE later |
| `TAKSH-NATIVE-V0.1-BASELINE.md` | `.../docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` | untracked | 15,137 | `8A7CA2AC5560CFFA9B5D9E13EE63F422BD3AB18CBCDFADA49727A4E8CDD53813` | baseline report and limits | present | no | PRESERVE; commit candidate only after review |
| `native_v02/__init__.py` | `.../evaluation/native_v02/__init__.py` | untracked | 148 | `DD2B671089E64AB425FF9DE60DD8448AB6A725966457457C37AA74CFC4B435AD` | diagnostic package boundary | present | no | DEFER/PRESERVE |
| `native_v02/diagnose.py` | `.../evaluation/native_v02/diagnose.py` | untracked | 30,895 | `25EE03EEFC36E61081402849F24458DA96F68246C44AF3A71D1F4EB9B6D9C99C` | prompt/logit/decoding ablation diagnostics | present; not rerun here | no | DEFER; not part of immutable baseline |
| `native_v02/diagnosis_results.json` | `.../evaluation/native_v02/diagnosis_results.json` | untracked | 1,125,939 | `F88CEEDBB5C3E21E44567B9C8016210D9A6379434B33F689BFB41FF3F858D055` | recorded diagnosis results | present; large generated evidence | no | PRESERVE; external archival decision |
| `test_native_v02_diagnosis.py` | `.../tests/test_native_v02_diagnosis.py` | untracked | 4,747 | `4E8D8E2314F8D76187F1F8E0E6BBFD0420A502B171FA0B2603AD2397B8943D35` | diagnostic test harness | present | no | DEFER |
| `TAKSH-NATIVE-V0.2-DIAGNOSIS.md` | `.../docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md` | untracked | 21,054 | `F736368220CBB017059E08D3461A61038212BDFFAF0D64000E7B15DC46017051` | diagnosis findings | present | no | PRESERVE; not baseline release evidence |

### Native artifact count and roles

- Native v0: 3 files.
- Native v0.1: 6 files.
- Native v0.2: 5 files.
- Total native-related files accounted for: **14**.

The Native v0.1 benchmark tasks/results were checked against Step 0.2 hashes and were not changed:

- `tasks.py`: `152A...`
- `baseline_results.json`: `C8DB...`
- report: `8A7C...`

---

## 6. Checkpoint and tokenizer identity verification

### Checkpoint

| Identity | Value | Status |
|---|---|---|
| path | `F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt` | exists |
| bytes | `345596421` | **match** |
| SHA-256 | `9B54103695054ECB30D8521FFCEB62F443D2D1FD8D064993EDC2D5070CB7F666` | **match** |
| loading | not performed | not verified |
| model geometry | v180_sft / code_v160_15m family, d_model 384, 12L, 8H, 4KV, FFN 1536, context 2048 | copied from prior recorded baseline; not independently reloaded |
| model quality | 0/20 benchmark correct | **no capability claim** |

### Tokenizer

| Identity | Value | Status |
|---|---|---|
| primary path | `F:/Codes/Git/Quantsmind-LLM/tokenizer/taksh_code_bpe_v1.json` | exists |
| primary bytes | `411743` | recorded |
| primary SHA-256 | `554C8E7572D58923A0766F114B5F419B14E68BDFF3A738803B30728EC3FFCD92` | **match** |
| closure path | closure worktree same tokenizer file | exists |
| closure SHA-256 | `554C8E7572D58923A0766F114B5F419B14E68BDFF3A738803B30728EC3FFCD92` | **match** |
| vocab | 5,822 | recorded identity; not re-tokenized here |
| BOS/EOS | 1/2 | recorded in prior evidence; not independently decoded |
| tokenizer behavior | not loaded by this audit | not verified |

---

## 7. Corpus preservation and license status

### Directory inventory

| Path | Primary count/size | Closure count/size | Same filenames? | Status |
|---|---:|---:|---|---|
| `data/l02_overfit/` | 3 / 0.01 MB | 3 / 0.01 MB | yes | protected; no rights review |
| `data/l02_sft/` | 3 / 0.09 MB | 3 / 0.09 MB | yes | protected; CC0 indicated by prior evidence but not independently licensed for release |
| `data/l03_corpus/` | 8 / 91.22 MB | 8 / 91.22 MB | yes | protected; mixed CC-BY-SA/public-domain/code licenses |
| `data/l04_code_corpus/` | 16 / 32.81 MB | 16 / 32.81 MB | yes | protected; mixed CC-BY-SA/public-domain/MIT/BSD/Apache |

### Hash equivalence evidence

The following representative files match byte-for-byte between primary and closure:

- `data/l03_corpus/dataset_manifest.json`
- `data/l03_corpus/train.jsonl`
- `data/l04_code_corpus/dataset_manifest.json`
- `data/l04_code_corpus/train.jsonl`
- `data/l02_sft/sft.jsonl`
- `data/l02_overfit/sft.jsonl`

Name inventory matches for all four directories; representative hash matches confirm the closure worktree duplicates the protected data. This audit did not hash every file individually after a tooling error, and therefore does **not** claim universal hash equivalence.

### Provenance and licenses

- `data/l03_corpus/PROVENANCE.md` records CC-BY-SA-3.0 Wikipedia, public-domain Gutenberg texts, MIT/BSD/Apache code and Taksh-owned source. It explicitly notes share-alike/license obligations and limits.
- `data/l04_code_corpus/dataset_manifest.json` records m3 verified code, Taksh repository code and Wikipedia/Gutenberg content, plus benchmark-leak quarantine.
- `l02_sft` and `l02_overfit` contain CC0/cleaned synthetic-style SFT data per prior manifest evidence; their status is still not a publication authorization.

**None of the four directories may enter Git integration, a release, or publication without a separate legal/provenance gate.** Duplicates in the closure worktree must also remain excluded.

### Checkpoint and token streams

- `runs/v180_sft/checkpoint_last.pt` is untracked and must not enter Git.
- Tracked token streams under `data/v160_tokens`, `data/v170_tokens`, `data/v180_tokens` already exist in repository history; they are not part of this proposed integration.

---

## 8. Architecture and dependency map

### Canonical responsibilities

| Responsibility | Canonical implementation now | Status | Notes |
|---|---|---|---|
| Model implementation | `training/model.py` in closure/current; l111 variant also exists | closure/current equivalent; native variant differs | keep closure variant until native decision |
| Model configuration | `configs/model/code_v160_15m.yaml`; proposed `configs/foundation/l060_foundation.yaml`; native `configs/model/foundation_111m.yaml` | multiple candidates | closure current candidate for native v0; l110/l111 for native v1; no automatic choice |
| Checkpoint schema | `training/checkpoint_schema.py` | stable across inspected branches | retain |
| Tokenizer contract | `training/tokenizer_contract.py` | stable | retain; do not retrain tokenizer here |
| Chat template | `training/chat_template.py` | stable | v1 contracts valid for historical artifact; v2 requires reserved tokens |
| Training pipeline | `training/pretrain.py`, `training/sft.py`, `training/train.py` | closure/current variants equivalent | retain baseline |
| Checkpoint loading | `training/checkpoint.py`, `serving/local_inference.py` | current/closure | safe loader with narrow NumPy retry |
| Native inference | `serving/foundation_inference.py` | older uncommitted in primary; modified in closure | preserve/closure decision |
| Native provider | `serving/native_provider.py` | absent in primary; closure committed+modified; unified variant exists | closure provisional |
| Provider abstraction | `serving/providers.py`, `serving/provider_diagnostics.py` | closure/unified/five differ | review, don't merge wholesale |
| Native evaluation | `evaluation/native_v01/*` | untracked only | preserve; immutable |
| Native diagnostics | `evaluation/native_v02/*` | untracked only | defer/preserve |
| Engineering workflow | `taksh/engineering.py`, `taksh/specialists.py`, provider code | closure/unified/five differ | keep separate pending provider decision |
| Recovery/CI | `.github/workflows/*`, `evaluation/taksh_cicd_recovery_check.py`, recovery docs | absent locally; present remote/five | selective review only |
| Release safeguards | checkpoint lifecycle, manifests, roadmap validator | partial; local engineering only | not GA |
| Gradio product integration | `serving/code_chat.py`, `serving/local_inference.py` | local worktree modified to loopback-only | not yet product-qualified |

### Duplicated implementations

- Native model config: `code_v160_15m.yaml` vs `foundation_111m.yaml` — different product lines; keep separate.
- Provider identity: closure vs unified — closure records native identity more completely; unified is a thinner provider-integrated branch.
- Native diagnostics vs immutable baseline — v0.2 is diagnostic and must not replace v0.1.
- Recovery CI docs vs L0.7.4 reports — both are evidence about CI; neither may claim release readiness.

### Missing or unqualified components

- Native v0.1 implementation is not committed.
- Native v0.1 model capability is 0/20.
- Native v0.2 diagnostics are untracked and not independently rerun.
- No verified claim that the native model is generally usable.
- No verified release/serving SLO on the target machine.
- No proof that current CI passes on remote/protected branches.

### Circular dependencies

No circular code dependency was established by this audit. The primary architectural tension is governance/evidence, not import cycles: baseline release artifacts, checkpoints and benchmarks must not be generated by the same unproven code path without explicit qualification.

### Version incompatibilities

- `training/sft.py` has different hashes across current/closure vs l110/l111/five.
- `training/model.py` differs across native lineage and five-version.
- Five-version has modified `training/model.py` and serving paths, while closure/native paths differ again.
- Provider diagnostics differ across closure/unified/five.

---

## 9. File-level consolidation manifest

Classifications: **KEEP** = retain as authoritative; **INTEGRATE** = safe candidate after approval; **PARTIAL** = selected files/changes only; **PRESERVE** = keep as-is, not yet integrated; **DEFER** = await prerequisite; **EXCLUDE** = must not enter baseline; **OWNER DECISION** = explicit owner choice required.

| Path or capability | Source | Destination proposal | Classification | Dependencies | Conflict risk | Focused tests after integration | Reversible? | Reason |
|---|---|---|---|---|---|---|---|---|
| `training/chat_template.py` | current/closure | same | **KEEP** | tokenizer contract | none | tokenizer/chat contract tests | yes | stable integrity layer |
| `training/tokenizer_contract.py` | current/closure | same | **KEEP** | tokenizer | none | contract tests | yes | stable |
| `training/checkpoint_schema.py` | current/closure | same | **KEEP** | manifest/provenance | none | checkpoint schema tests | yes | stable |
| `training/train.py` | current/closure | same | **KEEP** | integrity layer | none | v1.1 integrity tests | yes | equivalent |
| `training/pretrain.py` | closure/current | baseline candidate | **KEEP/PARTIAL** | safe load, manifest | medium | foundation + manifest tests | yes | contains current local warmstart work |
| `training/sft.py` | closure/current | baseline candidate | **KEEP/PARTIAL** | config resolution, vocab guard | medium | SFT config/tokenizer tests | yes | closure hash matches current |
| `training/manifest.py` | closure/current | baseline candidate | **KEEP/PARTIAL** | checkpoint schema | low | manifest tests | yes | provenance field |
| `serving/code_chat.py` | closure/current | baseline candidate loopback UI | **INTEGRATE** after review | local_inference | low | serving/security tests | yes | loopback-only fix |
| `serving/local_inference.py` | closure/current | baseline candidate | **INTEGRATE** after review | safe loading/allowlist | medium | safe checkpoint tests | yes | checkpoint safety |
| `configs/foundation/l060_foundation.yaml` | closure/current | baseline candidate engineering config | **INTEGRATE** after review | L0.6 qualification | medium | foundation config tests | yes | L0 foundation candidate |
| L0/L0.5/L0.6 evaluation/tests/reports in working tree | closure/current | external evidence archive or reviewed baseline | **PARTIAL** | qualification artifacts | high | foundation/l060 tests | yes | evidence, not code capability |
| `serving/foundation_inference.py` | closure modified | closure baseline | **PRESERVE/INTEGRATE** | safe loading, token contract | medium | native v160/native v01 tests after approval | yes | native v0 implementation |
| `serving/native_provider.py` | closure modified | closure baseline | **PRESERVE/INTEGRATE** | provider architecture | medium/high | native provider tests | yes | native provider identity |
| `tests/test_native_v160_inference.py` | closure untracked | closure baseline | **PRESERVE/INTEGRATE** | foundation_inference | low/medium | native v160 test | yes | v0 mechanics |
| `evaluation/native_v01/*` | closure untracked | closure baseline immutable | **PRESERVE** | native inference, checkpoint | low | native v01 evaluation tests | yes | immutable benchmark; do not edit |
| `tests/test_native_v01_evaluation.py` | closure untracked | closure baseline | **PRESERVE** | runner | low | native v01 tests | yes | benchmark test |
| `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` | closure untracked | evidence docs | **PRESERVE** | benchmark files | none | link cleanup only after owner | yes | evidence, not capability claim |
| `evaluation/native_v02/*` | closure untracked | external/deferred evidence | **DEFER** | v0.1 baseline | medium | native v02 diagnostic tests after authorization | yes | diagnostic, not release gate |
| `tests/test_native_v02_diagnosis.py` | closure untracked | deferred | **DEFER** | diagnostic code | unknown | native v02 tests | yes | not required for baseline |
| `docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md` | closure untracked | evidence docs | **DEFER/PRESERVE** | diagnosis outputs | none | review only | yes | diagnosis, not release |
| `evaluation/native_v01/baseline_results.json` | closure untracked | immutable evidence | **PRESERVE** | runner | none | checksum verification after copy | yes | immutable Native v0.1 result |
| Recovery CI `61ae12a` | recovery refs/five | no branch until review | **PARTIAL/OWNER DECISION** | CI review | high | CI workflow qualification | yes | broad blast radius |
| Recovery CI `30facf5` | recovery refs/five | no branch until review | **PARTIAL/OWNER DECISION** | benchmark/adapters | medium | adapter/benchmark tests | yes | must compare with L0.7 corpus work |
| Recovery CI `90eba0a` | recovery refs/five | no branch until review | **DEFER** | CI shape from `61ae12a` | medium | CI workflow tests | yes | targeted but dependent |
| Recovery docs `61e2dba` | recovery refs | no branch until review | **PARTIAL/OWNER DECISION** | defect register relevance | low | document review | yes | documentation |
| Merge commits `8f1fbda`, `0215c6d` | recovery refs | do not integrate as commits | **EXCLUDE as commits** | parent commits | high | none | n/a | merge metadata, not units |
| `data/l02_overfit/` | working tree/closure | exclude | **EXCLUDE** | provenance/license | high/legal | none | n/a | not release material |
| `data/l02_sft/` | working tree/closure | exclude | **EXCLUDE** | provenance/license | high/legal | none | n/a | corpus exclusion |
| `data/l03_corpus/` | working tree/closure | exclude | **EXCLUDE** | CC-BY-SA obligations | high/legal | none | n/a | corpus exclusion |
| `data/l04_code_corpus/` | working tree/closure | exclude | **EXCLUDE** | mixed licenses | high/legal | none | n/a | corpus exclusion |
| `runs/v180_sft/checkpoint_last.pt` | working tree only | external artifact store only | **EXCLUDE** | none | high | none | n/a | checkpoint not Git artifact |
| `data/v160_tokens/*`, `v170_tokens/*`, `v180_tokens/*` | already tracked | unchanged | **KEEP/EXCLUDE from new commits** | repository history | n/a | n/a | yes | no new baseline action |
| `taksh-hf-publication/upload_manifest.json`, `hf_export/*`, `export_hf*`, `verify_pub.py` | working tree | exclude | **EXCLUDE** | release authorization | high | none | yes | publication surface |
| `reports/*` generated by L0/L0.5/L0.6/native | working tree/closure | separate evidence archive or reviewed commit | **OWNER DECISION** | artifact provenance | medium | none | yes | evidence policy required |
| Five-version GA/platform/IDE/security work | five-version | separate lineage | **DEFER/OWNER DECISION** | recovery CI, five-version contracts | high | five-version focused tests if ever selected | yes | not part of native baseline |
| Native model lineage `l110`/`l111`/`foundation_111m.yaml` | native branches | separate experiment/product line | **DEFER/OWNER DECISION** | native programme | high | l110/l111 tests | yes | do not fold into current consolidation |
| Coding Genius docs and architect package | current working tree | retain separately | **DEFER/OWNER DECISION** | product direction | low/medium | architect package tests | yes | keep separate as owner directed |
| `docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md` | absent | do not create here | **OWNER DECISION** | status register | low | n/a | yes | proposed status below; not created |

---

## 10. Ordered integration sequence for future owner-approved work

No step below is executed by this report.

### Unit 1 — Baseline infrastructure and safeguards

**Source:** closure baseline + integrity layer.  
**Dependencies:** current L0/ENG code, checkpoint schema, tokenizer contract.  
**Actions:**

1. Owner chooses closure as baseline.
2. Owner decides which recovery CI pieces are relevant.
3. Retain checkpoint/tokenizer/integrity layer unchanged.
4. Document excluded corpus and publication paths in baseline commit message if authorized.

**Focused verification:**

```text
pytest tests/test_token_contract_and_chat.py tests/test_v110_contract.py tests/test_data_integrity.py tests/test_eval_run_manifest.py tests/test_validate_manifest.py -q
pytest tests/test_l060_foundation.py tests/test_architect_package.py -q
python -m evaluation.roadmap.roadmap_validator
```

### Unit 2 — Native inference implementation

**Source:** closure `serving/foundation_inference.py` and related safe loader.  
**Actions:** integrate only if owner authorizes a future Git baseline; no model execution by default.  
**Focused verification:**

```text
pytest tests/test_native_v160_inference.py -q
pytest tests/test_l062_inference.py -q
pytest tests/test_v141_safe_checkpoint.py -q
```

### Unit 3 — Native provider integration

**Source:** closure `serving/native_provider.py`.  
**Dependencies:** provider abstraction, identity rules, no provider fallback.  
**Focused verification:**

```text
pytest tests/test_native_provider.py tests/test_coding_providers.py tests/test_unified_provider_diagnostics.py -q
```

### Unit 4 — Native v0.1 benchmark

**Source:** closure `evaluation/native_v01/*`, `tests/test_native_v01_evaluation.py`, `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`.  
**Actions:** commit as immutable evidence only if owner approves; do not rerun inference unless separately authorized and bounded.  
**Focused verification:**

```text
pytest tests/test_native_v01_evaluation.py -q
python -m evaluation.native_v01.runner --help
```

### Unit 5 — Native v0.2 diagnostics

**Source:** closure `evaluation/native_v02/*`, diagnostic report.  
**Actions:** retain separately; do not treat as a new benchmark; run only with explicit resource authorization.  
**Focused verification:**

```text
pytest tests/test_native_v02_diagnosis.py -q
```

### Unit 6 — Documentation/audit evidence

**Source:** current coding-genius docs, Step 0 audits, native reports.  
**Actions:** commit only reviewed docs; keep large corpora/publication files excluded.  
**Focused verification:** markdown review; no model inference.

### Unit 7 — Optional Gradio/product integration

**Source:** `serving/code_chat.py` and `serving/local_inference.py`.  
**Dependencies:** native inference, provider selection, explicit UI provider labels.  
**Focused verification:**

```text
pytest tests/test_m7_serving.py tests/test_local_assistant_l01.py tests/test_legacy_serving_security.py -q
```

No unit is executed here.

---

## 11. Conflicts and risks

1. **Checkpoint safety:** any future integration must preserve `weights_only=True` and narrow allowlists; unrestricted pickle is forbidden.
2. **Native capability:** 0/20 correct generations; integration cannot imply a working native coding assistant.
3. **Recovery overlap:** recovery CI conflicts with current L0.7.4/CI qualification files; line-level review required.
4. **Provider identity:** replacing native provider with a generic provider silently is unacceptable; UI/tests must identify the provider.
5. **Benchmark contamination:** Native v0.1 must stay unchanged; v0.2 is diagnostic only.
6. **Corpus leakage:** `l03`/`l04` contain CC-BY-SA data and derived tokens; no publication or redistribution without legal review.
7. **Evidence duplication:** closure worktree duplicates large protected data; hashes/name inventory match but licensing does not.
8. **Publication surfaces:** HF export/upload files cannot be used as release readiness evidence.
9. **Five-version entanglement:** its native files match l111, but GA/platform work is not the native baseline.
10. **Git state:** local `develop`/`main` are stale; remote tracking inspection was read-only and remote current state remains unverified without a fetch explicitly authorized by owner.

---

## 12. Owner decisions required

1. Confirm `feature/l060-1-integration-closure` as provisional baseline.
2. Authorize preservation or integration of the 14 native v0/v0.1/v0.2 files.
3. Decide whether native files enter Git, or remain local/external evidence.
4. Decide file-level treatment of recovery/CI commits `61ae12a`, `30facf5`, `90eba0a`, and recovery documentation `61e2dba`.
5. Decide whether to ignore merge commits `8f1fbda` and `0215c6d` as integration units.
6. Decide whether reports belong in Git or external evidence storage.
7. Decide the fate of the closure worktree's duplicated corpora; no deletion by this audit.
8. Confirm checkpoint and corpora remain outside Git/release.
9. Decide whether coding-genius/Design Architect source enters a future baseline or stays separate.
10. Authorize or withhold any future local commit/merge. No Git mutation has occurred.

---

## 13. Proposed status entry for absent programme file

`docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md` does not exist and was not created. If the owner later asks to create/update it, the appropriate entry is:

> `TAKSH-STEP0.3-CONSOLIDATION-PLAN.md`: read-only consolidation audit complete; provisional baseline identified as `feature/l060-1-integration-closure` @ `87b8761`; native artifacts accounted for as untracked closure-worktree evidence; checkpoint/tokenizer identities matched; protected corpora excluded; no merge/commit authorized; status `READY_FOR_OWNER_CONSOLIDATION_APPROVAL`, pending explicit owner decision.

---

## 14. Explicit confirmation of no repository changes

This phase performed read-only Git inspection and hashing only. It did **not**:

- switch branches;
- create, delete, rename or move branches/worktrees;
- fetch or pull;
- stage, commit, merge, cherry-pick, rebase, reset, stash or clean;
- modify source, tests, configs, documentation, checkpoints, tokenizers or datasets;
- delete/move/publish data;
- run training or inference;
- create `docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md`.

Only this report file was created: `docs/audit/TAKSH-STEP0.3-CONSOLIDATION-PLAN.md`.
