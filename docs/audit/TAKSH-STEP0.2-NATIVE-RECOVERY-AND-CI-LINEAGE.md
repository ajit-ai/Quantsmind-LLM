# TAKSH Step 0.2 — Native Model Recovery and Recovery/CI Lineage Reconciliation

**Date:** 2026-11-14  
**Mode:** READ-ONLY investigation; no branch switch, fetch, merge, cherry-pick, staging, commit, push, publication, training or inference.  
**Primary repository:** `F:/Codes/Git/Quantsmind-LLM`  
**Secrets:** contents were not inspected or printed; no credential values are exposed here.

## Final status

`PARTIAL_REQUIRES_OWNER_DECISION`

Native Model v0/v0.1 artifacts were located as **uncommitted work in an existing linked worktree**, not in the foundation branch. The remote recovery/CI lineage is identified, but it is absent from the foundation branch and overlaps existing L0.7.4 CI qualification work. No consolidation is authorized by this report.

---

## 1. Repository and worktree inventory

### Primary repository

| Repository | Path | Branch / HEAD | Purpose |
|---|---|---|---|
| TAKSH primary | `F:/Codes/Git/Quantsmind-LLM` | `feature/l060-foundation-model-qualification` @ `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` | audit target |

### Existing linked worktrees discovered under the repository's Git metadata

| Worktree | Path | Branch / HEAD | Status observed |
|---|---|---|---|
| Closure worktree | `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-l060-1-closure` | `feature/l060-1-integration-closure` @ `87b876114a698b334ac7d15c6d1eee32f9138ffd` | modified native provider/inference files; untracked native v0/v0.1/v0.2 artifacts |
| Unified provider worktree | `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-l060-unified-provider` | `feature/l060-unified-provider-integration` @ `820624e9d8a27587af228be23d6c90716a734702` | native provider source; no native v0.1 evaluation artifacts found |
| Recovery/five-version worktree | `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-cicd-recovery` | `feature/taksh-five-version-ga-program` @ `978826bfc031e46afbdcf71792da3baebb5a3c48` | provider-agnostic/GA worktree; no native v0.1 evaluation artifacts found |

The worktrees were **not created or modified by this audit**. Their uncommitted and untracked files were inventoried read-only only.

### Other local TAKSH-named repositories inspected

| Repository | Finding |
|---|---|
| `taksh-hf-publication/` inside the primary repository | no native v0/v0.1/v0.2 artifact matches found |

A whole-drive recursive search was attempted on this host, but it timed out; no conclusion is drawn about inaccessible or differently named repositories. The accessible TAKSH locations inspected are the primary repository and its three linked worktrees.

---

## 2. Exact branch and HEAD inventory

| Ref | Full HEAD / commit | Parent(s) | Date | Subject | Local status |
|---|---|---|---|---|---|
| `feature/l060-foundation-model-qualification` | `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` | `d2f0d14` | 2026-11-12 | chore(recovery): qualify CI/CD and documentation recovery | current primary branch |
| `feature/l060-1-integration-closure` | `87b876114a698b334ac7d15c6d1eee32f9138ffd` | `ecd30fe` | 2026-11-13 | docs(l060-1): reconcile closure ledger | direct descendant of current HEAD |
| `feature/taksh-five-version-ga-program` | `978826bfc031e46afbdcf71792da3baebb5a3c48` | `768ea2d` | 2026-11-13 | feat(agent): integrate provider-agnostic coding intelligence | contains recovery/native copies |
| `develop` | `8c553465216dcaa5d36f73022b0db55aa3254e1f` | `a654e5d` | 2026-11-07 | TAKSH v3.0.0 publication-ready source package | stale locally |
| `main` | `c601a73716d527dfa100291461c06cefde760a05` | `cb9b81e 8c55346` | 2026-11-07 | Merge develop into main | stale locally |
| `origin/develop` | `0215c6d039fceffd334f8931aa96a3287010a580` | `8f1fbda 61e2dba` | 2026-11-12 | Merge pull request #3 from reconcile-recovery-register | remote-tracking ref; not fetched |
| `origin/main` | `d50de3ea220cbf18eee070e191a507ba70a758bb` | merge of develop into main | 2026-11-12 | Merge pull request #1 from develop | remote-tracking ref; not fetched |
| `origin/HEAD` | `d50de3ea220cbf18eee070e191a507ba70a758bb` | — | 2026-11-12 | Merge pull request #1 | remote-tracking ref; not fetched |
| `feature/l110-native-model-foundation` | `c7443f91c7537d8a46d67afaa0679a50f2504f53` | `0215c6d` | 2026-11-12 | feat(l110): native model foundation core | remote-tracking ref also present |
| `feature/l111-native-inference-mechanics` | `8c896b2516f3063a0037c55464ab0aa10aff59fb` | `c7443f9` | 2026-11-12 | feat(l111): native inference mechanics | remote-tracking ref also present |
| `feature/l060-unified-provider-integration` | `820624e9d8a27587af228be23d6c90716a734702` | `78b6cdf` | 2026-11-13 | feat(provider): unified provider | one commit above current HEAD |
| `integration/taksh-cicd-recovery` | `90eba0acc217ecb507b8c20f373f06e9a2101c40` | `30facf5` | 2026-11-12 | fix(ci): load repository tokenizer in SFT validation | recovery CI lineage |
| `planning/l1000-integration-and-ga-readiness` | `6662cce2765d47b5a9f173fc9d4c9ca09dba37a9` | `0215c6d` | 2026-11-12 | docs(plan) | planning lineage |
| `reconcile-recovery-register` | `61e2dba8b880df445101ea05001ca605d2e57241` | `8f1fbda` | 2026-11-12 | docs(recovery) | recovery lineage |

### Merge bases against current HEAD

- HEAD vs `develop`: `8c553465216dcaa5d36f73022b0db55aa3254e1f` (local `develop` is ancestor).
- HEAD vs `main`: `c601a73716d527dfa100291461c06cefde760a05` (local `main` is ancestor).
- HEAD vs `feature/l060-1-integration-closure`: current HEAD is the direct ancestor base (`78b6cdf...`).
- HEAD vs `integration/taksh-cicd-recovery`: `a654e5dc13dc77e925b3981d09792bade43a970a`.
- HEAD vs `origin/develop`: `a654e5dc13dc77e925b3981d09792bade43a970a` — current branch is not integrated with the recovery lineage visible on `origin/develop`.

---

## 3. Native Model v0 artifact recovery table

| Artifact | Current primary | Current working tree | Closure HEAD | Closure worktree uncommitted | Other branches | Recovery status |
|---|---|---|---|---|---|---|
| `serving/foundation_inference.py` | missing | untracked | committed blob `3623dc70ec5d2c3f05691c7b853fd5691409bab2` | modified to `91c32d343d602daf0e29d61b6a9c8070396bdab2` | unified/five: missing (commit history differs by lineage) | **Recovered** in current working tree and closure worktree; current is older than closure worktree |
| `serving/native_provider.py` | missing | missing | committed blob `5cef1c032b62f8bf6f657095fc480f31b8ee6eaf` | modified to `63419ebae06628c9ad8d6f981fd3da9acf81d298` | unified: `8d4f21282607b1f3c21df5da4ebdf07dcd2744b9`; five: missing | **Recovered** from closure; unified contains a related but distinct version |
| `tests/test_native_v160_inference.py` | missing | missing | not committed in closure | untracked, SHA-256 `CB31C9C79B7422825BC79E3171727E8C1F255EF48FE5545BE8758894134BB31F` | no branch containing it found | **Recovered** as untracked closure-worktree evidence; uncommitted |
| Native v0 recovery documentation | missing | missing | not committed | no dedicated v0 report found | no | **Recovered only as code/report references inside native v0.1 baseline** |

### Current primary hashes vs closure HEAD/worktree

| File | Current primary | Closure HEAD | Closure worktree |
|---|---|---|---|
| `serving/foundation_inference.py` | `3623dc70ec5d2c3f05691c7b853fd5691409bab2` | `3623dc70ec5d2c3f05691c7b853fd5691409bab2` | `91c32d343d602daf0e29d61b6a9c8070396bdab2` |
| `serving/native_provider.py` | absent | `5cef1c032b62f8bf6f657095fc480f31b8ee6eaf` | `63419ebae06628c9ad8d6f981fd3da9acf81d298` |

Current primary therefore contains an **older uncommitted copy** of native inference but not the provider implementation.

---

## 4. Native Model v0.1 artifact recovery table

All v0.1 artifacts are **untracked in the closure worktree**, not committed on any inspected branch:

| Artifact | Path in closure worktree | SHA-256 | Git object hash | Recovery status |
|---|---|---|---|---|
| `evaluation/native_v01/__init__.py` | `C:/.../taksh-l060-1-closure/evaluation/native_v01/__init__.py` | `BD780569398906D2910855F0969F92026E5EB6FF860E7B0D629616FED74A5DE0` | `ceed2092977f016be6119452bbf40ff443806141` | recovered; untracked |
| `evaluation/native_v01/tasks.py` | `.../evaluation/native_v01/tasks.py` | `152A368FFF9B20EBF005902FEC91AFCE7AAE98C236C7A6F192B0107E39867773` | `eb3b2efeaf7a24c14a1a2e576e770dfc34afbc6e` | recovered; untracked |
| `evaluation/native_v01/runner.py` | `.../evaluation/native_v01/runner.py` | `9C06F7F11DB26A454ED4E78F39FD29BC8C2CFD9D9CD2BACE766A9A3211237C3A` | `5376af71e446b9712a8e84553b7c397fb4404634` | recovered; untracked |
| `evaluation/native_v01/baseline_results.json` | `.../evaluation/native_v01/baseline_results.json` | `C8DBAF6F56E7EC950A817D055E741DB65D35B3B5613C734E8996773C51B5CBAE` | `c928612c9ac2ae2f83c3175035244ebe05b9bb13` | recovered; untracked |
| `tests/test_native_v01_evaluation.py` | `.../tests/test_native_v01_evaluation.py` | `386FE21870272633347E94B2544E5C49F5BE086558FD7CA651CB60F78ADEE88B` | `567bb30e11e282987e23f11e42774eb694235d7c` | recovered; untracked |
| `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` | `.../docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` | `8A7CA2AC5560CFFA9B5D9E13EE63F422BD3AB18CBCDFADA49727A4E8CDD53813` | `d299e1201f59aadbd419d65e9bf7c976d7e94032` | recovered; untracked |

### Additional discovered v0.2 artifacts (not required to be restored, but must be preserved)

| Artifact | SHA-256 | Status |
|---|---|---|
| `evaluation/native_v02/__init__.py` | `DD2B671089E64AB425FF9DE60DD8448AB6A725966457457C37AA74CFC4B435AD` | untracked |
| `evaluation/native_v02/diagnose.py` | `25EE03EEFC36E61081402849F24458DA96F68246C44AF3A71D1F4EB9B6D9C99C` | untracked |
| `evaluation/native_v02/diagnosis_results.json` | `F88CEEDBB5C3E21E44567B9C8016210D9A6379434B33F689BFB41FF3F858D055` | untracked |
| `tests/test_native_v02_diagnosis.py` | `4E8D8E2314F8D76187F1F8E0E6BBFD0420A502B171FA0B2603AD2397B8943D35` | untracked |
| `docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md` | `F736368220CBB017059E08D3461A61038212BDFFAF0D64000E7B15DC46017051` | untracked |

No native v0/v0.1/v0.2 artifact was found committed on `feature/l060-1-integration-closure`, current primary, `feature/l060-unified-provider-integration`, `feature/taksh-five-version-ga-program`, `develop` or `main`.

---

## 5. Hash comparison and provenance findings

### Native inference/provider

- Closure HEAD contains committed native provider and native inference files.
- Closure worktree has uncommitted diagnostic enhancements to both files.
- Current primary contains only the older uncommitted `serving/foundation_inference.py` copy matching closure HEAD.
- Current primary does **not** contain `serving/native_provider.py`.
- Unified-provider has a distinct provider implementation and no native v0.1 runner/results.

### Token streams and v180 artifacts

- Tokenizer in closure worktree is SHA-256 `554C8E7572D58923A0766F114B5F419B14E68BDFF3A738803B30728EC3FFCD92`, matching the reported expected tokenizer identity.
- Checkpoints are not Git artifacts and were not copied or modified.

### Protected data duplicates

The closure worktree contains data copies with the same file hashes as the primary for representative files:

| Path | Primary SHA-256 | Closure worktree SHA-256 |
|---|---|---|
| `data/l03_corpus/dataset_manifest.json` | `C3E84F79D523055C97A99E6D58C154F5C2F979DF33B3BDBFFA1E2414A80472BA` | identical |
| `data/l03_corpus/train.jsonl` | `51915236AE5C2925F579F1A31C246031D90F605BD9FDB330213B7C5EAC94B30B` | identical |
| `data/l04_code_corpus/dataset_manifest.json` | `B43745AF14B9DBA8D24E68862863FAE9451582BC4F452C9FFBC5FDD854E3378F` | identical |
| `data/l04_code_corpus/train.jsonl` | `CF89B8BC803B7363D2643E337A561524644E4924318C90332399F1625DFA1D5F` | identical |
| `data/l02_sft/sft.jsonl` | `549E4643326BECB7B620B0C4C6F75B614121EF8027FF4D054858343AACE40FD2` | identical |
| `data/l02_overfit/sft.jsonl` | `734372A28255FE05F4B38FF4F76580DD734B1BEAC0BC009750D793DA772C98AF` | identical |

The closure worktree carries a second copy of about 124 MB of protected data/corpus. These are duplicates, not evidence of redistribution rights, and should not be staged or published.

---

## 6. Benchmark evidence verification

The following was verified **without rerunning inference**:

- File exists: `evaluation/native_v01/baseline_results.json` in closure worktree.
- Schema: `taksh.native_v01_baseline/1`.
- Task count: 20.
- `syntax_ok`: 0/20.
- `executable`: 0/20.
- `correct`: 0/20.
- `empty_output`: 16/20.
- `degenerate_output`: 20/20.
- `inference_failed`: 0/20.
- Categories: arithmetic/string manipulation, basic function generation, code-correction/debugging, conditional logic/edge cases, function completion, instruction following, lists/dicts/iteration, simple algorithms.
- `baseline_results.json` matches the reported native v0.1 baseline counts.

The baseline report states:

- checkpoint path `F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt`;
- checkpoint SHA-256 `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666`;
- tokenizer SHA-256 `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92`;
- 28,787,328 parameters;
- `qm-code-v160-15m`, d_model 384, 12 layers, 8 heads, 4 KV heads, FFN 1536, context 2048, tied embeddings;
- load time 19.322 s and total generation 2.932 s for the baseline run;
- no capability claim is made.

These are **recorded execution artifacts**, not a new execution by this audit. The v0.2 diagnosis adds a 120-experiment prompt/decoding matrix, but this audit did not independently reproduce those results either.

---

## 7. Remote recovery/CI commit inventory

### Current branch versus recovery lineage

Current foundation branch and closure are absent from all six recovery/reconciliation commits visible on `origin/develop`; `git cherry HEAD integration/taksh-cicd-recovery` reports all three recovery commits as `+` (not present). Five-version contains the same lineage.

| SHA | Date | Parent(s) | Subject | Files changed | Disposition assessment |
|---|---|---|---|---|---|
| `61ae12a9fe18863bdef963b7deae3b6011713a32` | 2026-11-12 08:52 | `a654e5dc13dc77e925b3981d09792bade43a970a` | fix(ci): integrate TAKSH recovery and historical regression fixes | CI workflows, README/STATUS, recovery audit docs, `evaluation/m6_sovereign.py`, `evaluation/taksh_cicd_recovery_check.py`, S2/M6/v150/v160/v180 tests/fixture | CI/release-safeguard repair; absent from current; may overlap L0.7.4/L0.7.3 CI qualification work |
| `30facf553f0f498e7f994688f69da1c289224f59` | 2026-11-12 09:19 | `61ae12a9fe18863bdef963b7deae3b6011713a32` | fix(ci): port adapter preflight reorder and manifest origin sanitization from recovery | `data/python_v1/benchmarks.py`, `evaluation/m5_adapters.py` | targeted adapter/data benchmark repair; absent from current; could duplicate L0.7.x recovery-era fixes |
| `90eba0acc217ecb507b8c20f373f06e9a2101c40` | 2026-11-12 09:24 | `30facf553f0f498e7f994688f69da1c289224f59` | fix(ci): load repository tokenizer in SFT dataset validation step (no None tokenizer) | `.github/workflows/ci.yml` | CI tokenizer-loading fix; absent from current; intersects current L0.7.4/L0.7.3 CI qualification |
| `8f1fbda2d25cf19c0acbd1c835c9efc903981eea` | 2026-11-12 09:27 | `a654e5dc13dc77e925b3981d09792bade43a970a 90eba0acc217ecb507b8c20f373f06e9a2101c40` | Merge pull request #2 from integration/taksh-cicd-recovery | same as above plus workflows | merge commit, not an independently safe cherry-pick |
| `61e2dba8b880df445101ea05001ca605d2e57241` | 2026-11-12 09:31 | `8f1fbda2d25cf19c0acbd1c835c9efc903981eea` | docs(recovery): reconcile defect register post-merge (develop green, PR #2) | `docs/audit/TAKSH-RECOVERY-DEFECT-REGISTER.md` | documentation reconciliation; absent from current; should be reviewed with recovery docs |
| `0215c6d039fceffd334f8931aa96a3287010a580` | 2026-11-12 09:34 | `8f1fbda2d25cf19c0acbd1c835c9efc903981eea 61e2dba8b880df445101ea05001ca605d2e57241` | Merge pull request #3 from reconcile-recovery-register | defect register | merge commit; not a direct integration unit |

### Presence matrix

| Commit/capability | current foundation | closure | unified provider | five-version | `integration/taksh-cicd-recovery` | `origin/develop` |
|---|---|---|---|---|---|---|
| `61ae12a` recovery CI integration | absent | absent | absent | present | present | present |
| `30facf5` adapter/benchmark fix | absent | absent | absent | present | present | present |
| `90eba0a` SFT tokenizer validation fix | absent | absent | absent | present | present | present |
| `8f1fbda` recovery merge | absent | absent | absent | present | present | present |
| `61e2dba` defect register reconciliation | absent | absent | absent | present | absent as standalone branch tip | present |
| `0215c6d` reconciliation merge | absent | absent | absent | present | absent as standalone branch tip | present |

### Cherry-pick conflict assessment

- `61ae12a` touches broad CI and historical regression tests. Current foundation contains newer L0.7.4/L0.8.0-PRE/L0.6/ENG documentation and CI qualification work, so conflicts are likely in `.github/workflows/ci.yml`, `.github/workflows/ci-l0704-manual-qualification.yml`, `docs/STATUS.md` and the recovery audit documents.
- `30facf5` modifies `data/python_v1/benchmarks.py` and `evaluation/m5_adapters.py`. Current foundation has newer L0.7.1/L0.7.2/L0.7.4 corpus and benchmark provenance work; unverified overlap.
- `90eba0a` is a single workflow tokenizer fix but depends on the exact CI file state after `61ae12a`/`30facf5`.
- `8f1fbda` and `0215c6d` are merge commits; cherry-picking them without their parents would duplicate the same recovery content.
- The recovery lineage is not required for native v0.1 execution; it is relevant to future CI validation and release safeguards.

---

## 8. File-level capability and dependency matrix

| Capability | Source commit/branch | Required dependencies | Present in foundation? | Present in closure? | Recommendation |
|---|---|---|---|---|---|
| Checkpoint schema `quantsmind.checkpoint/2` | current HEAD / closure | tokenizer contract, provenance hashing | yes | yes | retain foundation/closure implementation |
| Safe checkpoint loading | current/closure and l111/five variants | `weights_only=True`, NumPy allowlist, validation by name | yes, current local modified copy matches closure | yes | owner must choose closure baseline before integrating native artifacts |
| Tokenizer contract | current HEAD / closure | tokenizer artifact and v1/v2 markers | yes | yes | retain; do not alter tokenizer during consolidation |
| Chat template rendering | current HEAD | checkpoint schema, tokenizer contract | yes | yes | retain; test before native integration |
| Native `foundation_inference.py` | closure committed + closure worktree diagnostics | tokenizer contract, checkpoint loading, model implementation | older uncommitted copy | yes, both committed and newer uncommitted | integrate only via closure baseline decision, not by recreating |
| Native provider `native_provider.py` | closure committed + closure worktree diagnostics | provider architecture, role/identity handling | no | yes | owner decision: closure or unified-provider variant |
| Native inference v0 tests | closure worktree untracked | `foundation_inference.py`, checkpoint/tokenizer | no | untracked only | preserve; decide commit policy separately |
| Native v0.1 benchmark | closure worktree untracked | native inference, checkpoint, tokenizer, runner | no | untracked only | preserve immutable baseline; do not regenerate |
| Native v0.2 diagnosis | closure worktree untracked | v0.1 prompt format evidence, checkpoints, tokenizer | no | untracked only | preserve as diagnostic evidence; do not rerun |
| Provider-agnostic coding intelligence | five-version `978826b` | recovery lineage, five-version program, security/auth | no | no direct equivalent | selective review only; do not fold wholesale |
| Recovery CI fixes | recovery/origin/develop | workflow/CI validation | no | no | selectively integrate after owner decision and conflict review |
| Native model source config `foundation_111m.yaml` | l110/l111/five | no training authorized | no | no | keep separate pending native programme decision |
| Training infrastructure | current HEAD | checkpoint schema, integrity layer, data manifests | yes | yes | retain; no retraining authorized |
| CI/release safeguards | recovery lineage | current L0.7.4 CI work | partial/local engineering only | partial/local engineering only | treat as dependency gate before any future baseline, not as automatic merge |

**Native Model v0/v0.1 does not depend on the recovery CI commits**. It depends on the native inference/provider lineage present in closure. Recovery/CI is required for trustworthy CI and release validation, not for the historical native v0.1 run.

---

## 9. Duplicate and conflict analysis

- `serving/foundation_inference.py` in primary working tree equals closure HEAD `3623dc70...`, not closure worktree's diagnostic copy `91c32d34...`.
- `serving/native_provider.py` is absent in primary working tree, present in closure and unified with different hashes.
- `evaluation/native_v01/*` and `evaluation/native_v02/*` are duplicated only in the closure worktree; no committed copy found elsewhere.
- Large corpora are duplicated byte-for-byte between primary and closure worktree.
- Five-version contains a copy of native foundation/native inference content with identical hashes to l111, but it is not the committed closure lineage and is not a native v0.1 baseline.
- `taksh-hf-publication/` has no native artifacts and must not be treated as recovery evidence.
- The v0.1 baseline, v0.2 diagnosis and current frozen Python benchmark are distinct artifacts; none can be used to claim coding capability.

---

## 10. Checkpoint and tokenizer identity status

| Identity | Expected | Actually checked | Result |
|---|---|---|---|
| `runs/v180_sft/checkpoint_last.pt` path | present | yes, read-only `Test-Path`/metadata | exists |
| file size | — | 345,596,421 bytes | recorded |
| SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` | yes, `Get-FileHash` | **match** |
| tokenizer `taksh_code_bpe_v1.json` in closure worktree | SHA-256 `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` | yes, `Get-FileHash` | **match** |
| tokenizer vocab | 5,822 | confirmed by file hash; vocab 5,822 recorded in baseline artifacts | match with recorded identity |
| BOS / EOS | 1 / 2 | recorded in baseline artifacts; not independently decoded | recorded as evidence, not independently validated |
| Model geometry | 28,787,328 parameters; d_model 384; 12L; 8H; 4KV; FFN 1536; context 2048; tied embeddings | not loaded/tested by this audit; copied from baseline metadata | **not independently verified** |
| Training step | step 200 reported by baseline | not independently recovered | not independently verified |
| Canonical checkpoint | absent by project records | not checked by this audit | still missing per existing reports |

The hash verification is limited to file identity. It does not authorize loading the checkpoint with unrestricted pickle, training, inference, copying, redistribution or publication.

---

## 11. Proposed selective integration sequence

1. **Freeze and inventory closure worktree** as the native artifact source of truth, under owner control. Preserve:
   - `serving/foundation_inference.py` (uncommitted diagnostic changes),
   - `serving/native_provider.py` (uncommitted diagnostic changes),
   - native v0/v0.1/v0.2 source/tests/docs/results,
   - protected corpus exclusions.
2. **Decide whether closure is the baseline**. If yes, preserve the existing working tree; do not overwrite it.
3. **Choose native artifact integration unit**: commit policy for native v0/v0.1/v0.2 must be explicitly approved before any staging.
4. **Resolve checkpoint safety and tokenizer compatibility tests** using focused tests, not broad inference unless authorized.
5. **Review recovery/CI commits separately**. If CI repair is wanted, perform a conflict review of `61ae12a`, `30facf5`, `90eba0a` against current L0.7.4 workflow/docs; do not cherry-pick blindly.
6. **Reconcile native v0/v0.1/v0.2 with the closure baseline** in a read-only integration review before any future merge.
7. **Only after owner approval**, create a consolidation branch/commit and run focused regression tests.
8. **Separately decide** whether to pursue Phase 0.3 planning, Phase 0.4 diagnostics, Phase 0.5 capability recovery or a hold.

No Git mutation is authorized here.

---

## 12. Unresolved questions and owner decisions

1. Is `feature/l060-1-integration-closure` the authoritative source for native artifacts?
2. Should the closure worktree's uncommitted diagnostic changes be preserved as-is, copied by owner-approved process, or discarded?
3. Should `evaluation/native_v01/` and `evaluation/native_v02/` be committed as immutable evidence or stored externally?
4. Are duplicates of `data/l03_corpus` and `data/l04_code_corpus` intended to be retained in the worktree, or archived/removed outside this audit's scope by owner/legal decision?
5. Should recovery CI commits be integrated into the foundation branch, or should L0.7.4 remain the accepted CI-local engineering qualification?
6. Is native v0.1 baseline accepted as the immutable regression baseline, or should a new benchmark be versioned first?
7. Is any native inference diagnostic run authorized in the future, and under what resource limits?
8. Should the closure branch be used for native artifact preservation, or should artifacts be recovered into the primary working tree?
9. Does owner authorize any future local commit, and if so, what files are in or out?
10. No training, compute spend, model redistribution, dataset publication, Git merge or release is authorized by this report.

---

## 13. Exact commands used

Representative read-only commands:

```text
Get-Location
git rev-parse --show-toplevel
git status -sb
git status --porcelain
git rev-parse --verify refs/heads/<ref>
git for-each-ref --format="%(refname:short)|%(objectname)" refs/remotes/origin
git show -s --format="%ci|%s" <ref>
git merge-base <ref-a> <ref-b>
git rev-list --count <from>..<to>
git log --oneline --reverse <from>..<to>
git ls-tree -r <branch> -- <path>
git ls-tree -r --name-only <branch>
git diff --stat <branch-a> <branch-b> [-- <paths>]
git cherry <current> <branch>
git worktree list --porcelain
git -C <linked-worktree> status --porcelain
git -C <linked-worktree> diff --stat [-- <paths>]
Get-ChildItem -Recurse -File
Get-FileHash -Algorithm SHA256 -LiteralPath <path>
Get-Item <path>
python -c <read-only JSON summary of baseline_results.json>
```

Whole-drive recursive search was attempted and timed out; no files were changed by that attempt. No temporary files created by this audit were added to the repository.

---

## 14. Explicit limitations

- Remote-tracking refs are not confirmed current remote heads; no network operation or fetch was performed.
- Native model execution was not rerun; all performance and correctness numbers are recorded evidence, not independently reproduced.
- Model geometry and step count are not independently verified in this audit.
- No checkpoint, tokenizer, dataset, training corpus, generated result or protected artifact was modified, copied, moved or redistributed.
- Large-corpus copies duplicate protected untracked files; hashes match but licensing and redistribution status remain unresolved.
- Presence of `.pyc` files and cache directories in native folders is not a certification of reproducibility.
- No capability claim is made; Native v0.1 records 0/20 syntax, 0/20 execution and 0/20 correctness.
