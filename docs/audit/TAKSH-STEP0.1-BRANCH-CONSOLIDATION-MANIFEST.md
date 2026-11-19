# TAKSH Step 0.1 — Read-Only Branch Comparison and Consolidation Manifest

**Date:** 2026-11-14  
**Branch under audit:** `feature/l060-foundation-model-qualification` @ `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1`  
**Mode:** READ-ONLY Git comparison; no branch switch, merge, cherry-pick, staging, commit, fetch, push, publish, training or filesystem deletion/move.  
**Only file created by this audit:** `docs/audit/TAKSH-STEP0.1-BRANCH-CONSOLIDATION-MANIFEST.md`

## Completion status

`READY_FOR_OWNER_CONSOLIDATION_APPROVAL`

The repository has enough evidence for a consolidation decision. This status does **not** authorize consolidation, merging, pushing, publishing, model training or any protected-branch action.

---

## 1. Executive verdict

1. The current foundation branch is an engineering foundation, not a release. It is 18 commits ahead of the stale local `develop` and 17 commits ahead of stale local `main`, but it is **behind the remote-tracking recovery lineage visible as `origin/develop` by six commits**.
2. `feature/l060-1-integration-closure` is a direct descendant of current HEAD: it contains the current local foundation work and four provider/coding-workflow/closure commits. The current working tree matches it for all six tracked modifications and almost every pre-existing untracked source/evaluation path inspected.
3. `feature/taksh-five-version-ga-program` is not a safe wholesale integration candidate. It diverged from a pre-L0 base, contains 37 unique commits, includes a squashed copy of native-model/native-inference content whose blob hashes match `feature/l111-native-inference-mechanics`, plus a much broader platform/security/GA/IDE line, and its GA/publication docs must not be allowed to imply capability or release readiness.
4. `evaluation/native_v01/`, `tests/test_native_v01_evaluation.py`, `tests/test_native_v160_inference.py` and `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` are **not present in any inspected branch or the current working tree**.
5. Large untracked data and generated publication surfaces must remain excluded from any baseline. Corpus provenance and redistribution rights are unresolved until owner/legal review.

---

## 2. Exact branch inventory

### Local branches

| Branch | Full HEAD | Parent(s) | Date | Subject |
|---|---|---|---|---|
| `feature/l060-foundation-model-qualification` | `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` | `d2f0d14` | 2026-11-12 | chore(recovery): qualify CI/CD and documentation recovery (local engineering only) |
| `feature/l060-1-integration-closure` | `87b876114a698b334ac7d15c6d1eee32f9138ffd` | `ecd30fe` | 2026-11-13 | docs(l060-1): reconcile closure ledger and qualification report |
| `feature/taksh-five-version-ga-program` | `978826bfc031e46afbdcf71792da3baebb5a3c48` | `768ea2d` | 2026-11-13 | feat(agent): integrate provider-agnostic coding intelligence |
| `develop` | `8c553465216dcaa5d36f73022b0db55aa3254e1f` | `a654e5d` | 2026-11-07 | TAKSH v3.0.0 - Complete publication-ready source-code package |
| `main` | `c601a73716d527dfa100291461c06cefde760a05` | `cb9b81e 8c55346` | 2026-11-07 | Merge develop into main: TAKSH v3.0.0 complete publication-ready source-code package |
| `feature/l110-native-model-foundation` | `c7443f91c7537d8a46d67afaa0679a50f2504f53` | `0215c6d` | 2026-11-12 | feat(l110): implement native model foundation core (source preparation only; no training) |
| `feature/l111-native-inference-mechanics` | `8c896b2516f3063a0037c55464ab0aa10aff59fb` | `c7443f9` | 2026-11-12 | feat(l111): implement native inference mechanics and contract qualification (no training) |
| `feature/l060-unified-provider-integration` | `820624e9d8a27587af228be23d6c90716a734702` | `78b6cdf` | 2026-11-13 | feat(provider): integrate unified coding provider architecture |
| `integration/taksh-cicd-recovery` | `90eba0acc217ecb507b8c20f373f06e9a2101c40` | `30facf5` | 2026-11-12 | fix(ci): load repository tokenizer in SFT dataset validation step (no None tokenizer) |
| `planning/l1000-integration-and-ga-readiness` | `6662cce2765d47b5a9f173fc9d4c9ca09dba37a9` | `0215c6d` | 2026-11-12 | docs(plan): develop integration and product GA readiness assessment (planning only) |
| `reconcile-recovery-register` | `61e2dba8b880df445101ea05001ca605d2e57241` | `8f1fbda` | 2026-11-12 | docs(recovery): reconcile defect register post-merge (develop green, PR #2) |
| `checkpoint/l101-owner-decisions` | `11f0ed9449e05a87faa35a95ff7df01b411157b8` | `6662cce` | 2026-11-12 | docs(checkpoint): L1.0.1 owner decisions and integration authorization register (planning-only) |

### Remote-tracking references, as last recorded locally; no fetch performed

| Remote ref | Full SHA | Relationship to current HEAD |
|---|---|---|
| `origin/HEAD` | `d50de3ea220cbf18eee070e191a507ba70a758bb` | remote lineage not integrated locally |
| `origin/main` | `d50de3ea220cbf18eee070e191a507ba70a758bb` | remote main is not local `main`; current branch is behind remote by seven commits and ahead by nineteen |
| `origin/develop` | `0215c6d039fceffd334f8931aa96a3287010a580` | current branch is behind remote by six commits and ahead by nineteen |
| `origin/feature/l060-foundation-model-qualification` | `78b6cdf76ccbd416d1208eb5e71e01a03d59c5b1` | same as current HEAD |
| `origin/feature/l060-1-integration-closure` | `87b876114a698b334ac7d15c6d1eee32f9138ffd` | same closure lineage |
| `origin/feature/taksh-five-version-ga-program` | `978826bfc031e46afbdcf71792da3baebb5a3c48` | same five-version lineage |

**Critical warning:** local `develop` and local `main` are stale as consolidation authorities. Remote-tracking refs show that `origin/develop` contains the recovery PR lineage (`61ae12a`, `30facf5`, `90eba0a`, `8f1fbda`, `61e2dba`, `0215c6d`) which is not present in the current foundation branch. No fetch was performed; this observation is based solely on the local remote-tracking refs already on disk.

---

## 3. History comparison and merge bases

### Against stale local `develop` and `main`

| Branch | Merge-base with local develop | Ahead local develop | Behind local develop | Merge-base with local main | Ahead local main | Behind local main |
|---|---|---:|---:|---|---:|---:|
| `feature/l060-foundation-model-qualification` | `8c55346` | 18 | 0 | `c601a73` | 17 | 0 |
| `feature/l060-1-integration-closure` | `8c55346` | 22 | 0 | `c601a73` | 21 | 0 |
| `feature/taksh-five-version-ga-program` | `a654e5d` | 37 | 1 | `a654e5d` | 37 | 2 |
| `develop` | `8c55346` | 0 | 0 | `8c55346` | 0 | 1 |
| `main` | `8c55346` | 1 | 0 | `c601a73` | 0 | 0 |
| `feature/l110-native-model-foundation` | `a654e5d` | 7 | 1 | `a654e5d` | 7 | 2 |
| `feature/l111-native-inference-mechanics` | `a654e5d` | 8 | 1 | `a654e5d` | 8 | 2 |
| `feature/l060-unified-provider-integration` | `8c55346` | 19 | 0 | `c601a73` | 18 | 0 |
| `integration/taksh-cicd-recovery` | `a654e5d` | 3 | 1 | `a654e5d` | 3 | 2 |
| `planning/l1000-integration-and-ga-readiness` | `a654e5d` | 7 | 1 | `a654e5d` | 7 | 2 |
| `reconcile-recovery-register` | `a654e5d` | 5 | 1 | `a654e5d` | 5 | 2 |
| `checkpoint/l101-owner-decisions` | `a654e5d` | 8 | 1 | `a654e5d` | 8 | 2 |

### Against remote-tracking refs

| Ref pair | Ahead | Behind | Meaning |
|---|---:|---:|---|
| current HEAD vs `origin/develop` | 19 | 6 | current branch lacks the six-commit recovery/reconciliation lineage |
| current HEAD vs `origin/main` | 19 | 7 | remote main/HEAD has additional recovery-lineage commits |
| closure vs `origin/develop` | 23 | 6 | closure likewise lacks that remote recovery lineage |
| five-version vs `origin/develop` | 31 | 0 | five-version contains the recovery lineage; local develop is stale |
| five-version vs `origin/main` | 31 | 1 | five-version contains the recovery lineage and diverges from current remote main |

---

## 4. Unique commit summaries

### `feature/l060-foundation-model-qualification` (18 unique vs local develop)

- `c601a73` Merge develop into main: TAKSH v3.0.0 source release merge
- `45f0dc9` L0.6.0-R1 DLCAE/dynamic language architecture qualification
- `1fef89d` L0.6.1 DLCAE-aware training pipeline qualification
- `9721aa6` L0.6.2 local inference architecture qualification
- `da16051` L0.7.0 data acquisition/qualification readiness
- `27c7b3a` L0.7.1 multilingual corpus engineering
- `68dc5b7` L0.7.2 training corpus freeze
- `bc3d296` L0.7.3 training readiness and CI regression closure
- `6c8177f` L0.7.4 CI runtime/resource qualification
- `b47fafa` docs L0.7.4 report
- `84a25ba` L0.7.5 pretraining readiness decision
- `7b7e5f8` blank owner decision record
- `84888e6` v3.0.0 master development roadmap
- `262ae69` L0.8.0-PRE preflight qualification, preparation only
- `efcda29` ENG DLCAE runtime foundation
- `6065394` ENG2 semantic/execution validation
- `d2f0d14` ENG3 end-to-end coding intelligence integration
- `78b6cdf` recovery qualification current HEAD

### `feature/l060-1-integration-closure` (four commits above current HEAD)

- `56916b6` feat(provider): integrate unified coding provider architecture
- `da7c885` feat: close L0.6 foundation and unified provider integration
- `ecd30fe` feat(l060-1): harden unified provider contract and coding workflow
- `87b8761` docs(l060-1): reconcile closure ledger and qualification report

It is a clean descendant of current HEAD. It contains the current foundation work and the provider integration lineage. It is the lowest-risk candidate if the owner wants to preserve the current local source snapshot rather than reconstruct it file by file.

### `feature/l060-unified-provider-integration` (one commit above current HEAD)

- `820624e` feat(provider): integrate unified coding provider architecture — same provider lineage as `56916b6` in closure, but without the later L0.6.1 close-out commit or closure report.

Comparison between unified-provider and closure shows closure adds:

- `da7c885` L0.6 foundation + provider integration close-out
- `ecd30fe` unified provider contract and coding workflow hardening
- `87b8761` closure ledger/qualification report

### `feature/taksh-five-version-ga-program` (37 unique vs local develop)

Shared recovery/reconciliation lineage with its remote base:

- `61ae12a` CI integration recovery
- `30facf5` CI adapter preflight reorder and manifest origin sanitization
- `90eba0a` tokenizer loading fix in SFT CI validation
- `8f1fbda` merge recovery PR
- `61e2dba` defect register reconciliation
- `0215c6d` merge reconciliation PR

Five-version source/foundation work:

- `bb3311f` five-version program v1.1.0–v2.0.0 source foundation
- `8214f31` distribution readiness assessment and capability statement
- `8add2fe` gated Hugging Face export preparation
- `e10e870` L1.1.0 config hash drift reconciliation
- `f81be01` version gate matrix
- `9f0358c` Python 3.11 compatibility evidence
- `db526a9` blocked 1.2.0 training authorization packet
- `09a44e1` frozen benchmark manifest
- `7af309b` tool-calling contract design
- `f1e59a9` capability evaluation contract
- `f738b47` multi-file engineering contract
- `153a490` GA security and deployment contract

Platform/GA/engineering work:

- `e97e95d` broadcast batch-invariant attention masks fix
- `3fdf96b` Phase A-C verification and release-readiness audit
- `8cd3f2d` coding platform and multi-file workflows integration
- `daba1e5` engineering platform audit
- `32aa850` tool runtime, sandbox, memory, agent workflow and service layer
- `dcb77ee` deployment helpers and performance baselines
- `864aed6` keep recovery report untracked
- `1316185` IDE client, REST auth and deployment configuration
- `1e7e711` IDE/REST security audit update
- `fbfa88d` structured logging, recovery and release evidence
- `d5490c7` GA hardening audit
- `65558cb` live engineering-service integration test
- `fb40f97` VS Code Extension Development Host tests
- `95e3a33` dependency vulnerability assessment
- `d61a4af` host and security qualification audit
- `f8a286d` isolation/deployment/review security hardening
- `73a81bc` security hardening audit
- `768ea2d` credential storage and operational-control remediation
- `978826b` provider-agnostic coding intelligence integration

`bb3311f` is not an empty/common base: it contains native-model/native-inference work copied or reimplemented with identical blob hashes to the l111 lineage.

### `feature/l110-native-model-foundation` / `feature/l111-native-inference-mechanics`

- l110: native model foundation core, `configs/model/foundation_111m.yaml`, `docs/L110_NATIVE_MODEL_FOUNDATION.md`, `evaluation/l110_native_foundation.py`, tests and report.
- l111: adds native inference mechanics, `evaluation/l111_native_inference.py`, `serving/generation_contract.py`, `tests/test_l111_native_inference.py`, and updated `training/model.py`/`serving/local_inference.py`.

### `integration/taksh-cicd-recovery`

- `61ae12a`, `30facf5`, `90eba0a`: CI/tokenizer/adapter fixes from the recovery lineage. These are absent from current HEAD. Direct replay would need conflict review against current L0.7.4 CI qualification work.

---

## 5. File-level capability matrix

Legend: `SAME` means identical blob hash across inspected refs; `DIFF` means hash differs; `MISSING` means absent.

| Path / capability | develop | main | current HEAD | current worktree | l060-1 closure | l110 | l111 | unified provider | five-version | Classification |
|---|---|---|---|---|---|---|---|---|---|---|
| `training/chat_template.py` | 531a3ac3501b | 531a3ac3501b | 531a3ac3501b | same | 531a3ac3501b | 531a3ac3501b | 531a3ac3501b | 531a3ac3501b | 531a3ac3501b | Present and equivalent |
| `training/tokenizer_contract.py` | f4d362e0c152 | f4d362e0c152 | f4d362e0c152 | same | f4d362e0c152 | f4d362e0c152 | f4d362e0c152 | f4d362e0c152 | f4d362e0c152 | Present and equivalent |
| `training/checkpoint_schema.py` | fae98c07ba35 | fae98c07ba35 | fae98c07ba35 | same | fae98c07ba35 | fae98c07ba35 | fae98c07ba35 | fae98c07ba35 | fae98c07ba35 | Present and equivalent |
| `training/train.py` | 193ab4a34a67 | 193ab4a34a67 | 193ab4a34a67 | same | 193ab4a34a67 | 193ab4a34a67 | 193ab4a34a67 | 193ab4a34a67 | 193ab4a34a67 | Present and equivalent |
| `training/pretrain.py` | e360d75926ea | e360d75926ea | e360d75926ea | 81b57042599c | 81b57042599c | e360d75926ea | e360d75926ea | e360d75926ea | e360d75926ea | Present but different; current/closure match |
| `training/sft.py` | 7ba71ac858b4 | 7ba71ac858b4 | 7ba71ac858b4 | 7f73107eaa45 | 7f73107eaa45 | 79c839fdb33f | 79c839fdb33f | 7ba71ac858b4 | 79c839fdb33f | Present but different; current/closure match |
| `training/model.py` | 91bb9f39ac10 | 91bb9f39ac10 | 91bb9f39ac10 | same | 91bb9f39ac10 | 42c3e02f4597 | 42c3e02f4597 | 91bb9f39ac10 | a26beac17c9e | Present but different; native branches differ |
| `serving/local_inference.py` | e1a595beed8f | e1a595beed8f | e1a595beed8f | 4725bd9fcd38 | 4725bd9fcd38 | 3981397f2031 | 116ba9298e19 | e1a595beed8f | 116ba9298e19 | Present but different; current/closure match |
| `serving/foundation_inference.py` | MISSING | MISSING | MISSING | 3623dc70ec5d | 3623dc70ec5d | MISSING | MISSING | MISSING | MISSING | Unique to working tree/closure |
| `serving/native_provider.py` | MISSING | MISSING | MISSING | MISSING | 5cef1c032b62 | MISSING | MISSING | 8d4f21282607 | MISSING | Provider/native lineage; closure/unified variants |
| `serving/providers.py` | 7cee4ebcde6e | 7cee4ebcde6e | 7cee4ebcde6e | same as HEAD | 531486153a40 | 7cee4ebcde6e | 7cee4ebcde6e | 7cee4ebcde6e | 7cee4ebcde6e | Present but different; closure provider-integrated |
| `serving/provider_diagnostics.py` | MISSING | MISSING | MISSING | MISSING | 9771c8ecac43 | MISSING | MISSING | 14c63c8db865 | 5bbaaa1c1c00 | Unique provider diagnostics variants |
| `serving/generation_contract.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | a98335198510 | MISSING | a98335198510 | Native branch and five-version copy; same hash |
| `serving/engineering_api.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 8f14f651761d | Unique to five-version |
| `serving/engineering_app.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 465e3fa8bb6c | Unique to five-version |
| `serving/engineering_service.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | Missing everywhere inspected |
| `taksh/agent.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 50e0f253e443 | Unique to five-version |
| `taksh/coding_engine.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | d0dc36088934 | Unique to five-version |
| `taksh/engineering.py` | MISSING | MISSING | MISSING | MISSING | 3f0148c0c68c | MISSING | MISSING | 3ec33ea4e606 | 3ec33ea4e606 | Provider/platform overlap; closure differs |
| `taksh/specialists.py` | MISSING | MISSING | MISSING | MISSING | f3d6a3a9341d | MISSING | MISSING | f3d6a3a9341d | f3d6a3a9341d | Shared closure/unified/five content |
| `taksh/tools.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | f87f812b20ab | Unique to five-version |
| `configs/foundation/l060_foundation.yaml` | MISSING | MISSING | MISSING | 6b4adb86b288 | 6b4adb86b288 | MISSING | MISSING | MISSING | MISSING | Unique to working tree/closure |
| `configs/model/code_v160_15m.yaml` | 4a392d44d6bf | 4a392d44d6bf | 4a392d44d6bf | same | 4a392d44d6bf | MISSING | MISSING | 4a392d44d6bf | MISSING | Present; absent from native/five lineages |
| `configs/model/foundation_111m.yaml` | MISSING | MISSING | MISSING | MISSING | MISSING | 998606e27b90 | 998606e27b90 | MISSING | 998606e27b90 | Native lineage; same hash l110/l111/five |
| `configs/roadmap/five_version_program.yaml` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 9a34ea224c60 | Unique to five-version |
| `evaluation/native_v01/` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | **Missing everywhere inspected** |
| `tests/test_native_v01_evaluation.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | **Missing everywhere inspected** |
| `tests/test_native_v160_inference.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | **Missing everywhere inspected** |
| `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | **Missing everywhere inspected** |
| `docs/L110_NATIVE_MODEL_FOUNDATION.md` | MISSING | MISSING | MISSING | MISSING | MISSING | present | present | MISSING | present | Native lineage; same content lineage across l110/l111/five |
| `docs/L111_NATIVE_INFERENCE.md` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | b56703d8f792 | MISSING | b56703d8f792 | l111 and equal copy in five-version |
| `docs/roadmap/TAKSH-CODING-AGENT-ARCHITECTURE.md` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 81eaef09ed06 | Unique to five-version |
| `ide/vscode/package.json` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 5540e2b12e95 | Unique to five-version |
| `evaluation/frozen_benchmarks.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 6ab07ee8c3ab | Unique to five-version |
| `evaluation/capability_evaluation.py` | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 162a8a22ae5e | Unique to five-version |

### Capability classification summary

- **Foundation integrity:** present and equivalent across all inspected branches.
- **Current local source work:** mostly identical to `feature/l060-1-integration-closure` by blob hash.
- **Provider integration:** present in closure, unified-provider and five-version; closure/unified/five differ in small ways and ordering; no automatic winner.
- **Native model foundation:** present in l110/l111 and copied into five-version with identical hashes; absent from current HEAD.
- **Native v0.1 evaluation surface:** not present anywhere inspected; do not claim it exists.
- **Five-version platform/GA/IDE/security line:** unique to five-version; much of it is valuable but depends on divergent recovery and publication surfaces; selective integration only.

---

## 6. Working-tree inventory

The working tree was not modified by this audit.

### Counts

| Metric | Value |
|---|---:|
| Porcelain entries | 107 |
| Modified tracked files | 6 |
| Untracked paths | 101 |
| Largest untracked path | `data/l03_corpus/` — 91.22 MB |
| Second largest | `data/l04_code_corpus/` — 32.81 MB |
| Additional untracked data | `data/l02_overfit/`, `data/l02_sft/` |
| Tracked token-stream files | `data/v160_tokens/*`, `data/v170_tokens/*`, `data/v180_tokens/*` (`.bin`) |
| Untracked checkpoint/weights | none detected by filename in status output |
| Untracked tokenizer files | none confirmed by filename in this audit |

### Modified tracked files

| Path | Working tree hash | Match | Classification |
|---|---|---|---|
| `serving/code_chat.py` | `150ee646b057` | closure | loopback-only local UI binding |
| `serving/local_inference.py` | `4725bd9fcd38` | closure | narrow NumPy checkpoint allowlist |
| `taksh-hf-publication/upload_manifest.json` | `9585498e285e` | closure differs | publication surface; exclude from baseline decision |
| `training/manifest.py` | `24a95d2f4600` | closure | initialization provenance |
| `training/pretrain.py` | `81b57042599c` | closure | `--init-from` and grad-norm reporting |
| `training/sft.py` | `7f73107eaa45` | closure | tokenizer/seq-len/model-config resolution and vocab guard |

### New untracked coding-genius work created during this session

Not present in any inspected branch and not included in closure:

- `docs/CODING_GENIUS_PLAN.md`
- `docs/CAPABILITY_STATEMENT.md`
- `docs/COMPETITOR_MATRIX.md`
- `docs/POSITIONING_STATEMENT.md`
- `docs/architecture/DESIGN-ARCHITECT.md`
- `docs/audit/TAKSH-STEP0-BASELINE-CONSOLIDATION.md`
- `taksh/architect/`
- `tests/test_architect_package.py`

These are source/documentation work; whether to commit them is an owner decision.

### Pre-existing untracked L0/L0.5/L0.6 lineage

The working-tree blobs for the following files match `feature/l060-1-integration-closure`:

- `configs/feasibility/*`
- `configs/foundation/l060_foundation.yaml`
- `configs/train/overfit.yaml`, `pretrain_l03*.yaml`, `pretrain_l04_*.yaml`, `sft_l02*.yaml`
- `evaluation/l02*`, `l04*`, `l051*`, `l05*`, `l060*`, `resource_guard.py`
- `reports/*` except `reports/L060_FOUNDATION_MODEL_QUALIFICATION_REPORT.md` (working tree differs)
- `scripts/build_l02*`, `build_l03*`, `build_l04*`, `compare_checkpoint_params.py`, `run_taksh_local.ps1`
- `serving/foundation_inference.py`, `serving/local_assistant.py`
- `tests/test_l02*`, `tests/test_l03*`, `tests/test_l04*` (l04 differs), `tests/test_l051*`, `tests/test_l05*`, `tests/test_l060*` (l060 differs)
- `docs/L060_STRATEGY_C_INTERFACE.md`

This means the current working-tree snapshot is substantially a captured copy of the closure branch, not a unique divergence. Remaining differences must be reviewed per file before baseline consolidation.

### Generated reports

Untracked generated reports under `reports/` and `docs/` are evidence artifacts. They are not automatically source release material. They should be separately archived or excluded according to owner policy.

### Publication surfaces

Exclude from any baseline consolidation absent explicit owner authorization:

- `taksh-hf-publication/upload_manifest.json`
- `hf_export/`
- `export_hf.py`
- `export_hf.ps1`
- `verify_pub.py`
- `rebuild_manifest.py`

### Data and corpora exclusions

Do not stage, commit, merge, push or publish without a separate legal/provenance decision:

- `data/l02_overfit/`
- `data/l02_sft/`
- `data/l03_corpus/` — includes CC-BY-SA-3.0 Wikipedia content; share-alike/attribution obligations must be reviewed
- `data/l04_code_corpus/` — contains CC-BY-SA-3.0/public-domain/code records and is a generated corpus, not source
- `data/l03_corpus/train.jsonl`, token streams and manifests
- `data/l04_code_corpus/*`

The manifests indicate mixed licenses and “see-MANIFEST” handling. This is insufficient to assume redistribution rights. Treat them as unresolved.

### Token streams and model artifacts

Tracked `data/v160_tokens/*`, `data/v170_tokens/*` and `data/v180_tokens/*` are committed token streams, not model weights. They are part of the repository history but should not be casually included in new experimental commits without size/provenance review.

### Secrets and credential-bearing files

No file name in the working-tree status output is an obvious private key, `.env` or credential file. A non-content filename scan found tokenizer-report/audit paths matching words such as `token` and `api`, not credential material. A lightweight content-pattern scan matched 44 file paths, but these matches are overwhelmingly test fixtures, documentation terminology, mock environment examples and code paths mentioning `password`, `token`, `secret` or `api_key`; no literal secret value was printed or verified. Therefore: **no confirmed secret is established by this audit, but this is not a security certification.** Formal secret scanning remains required before any commit/publication.

---

## 7. Conflicts and dependencies

### Direct dependency findings

1. **Current HEAD is an ancestor of closure.** `feature/l060-1-integration-closure` can be evaluated as a fast-forward-like source candidate; combining it with current working tree would require resolving the small current differences (`reports/L060_FOUNDATION_MODEL_QUALIFICATION_REPORT.md`, `tests/test_l04_pipeline.py`, `tests/test_l060_foundation.py`, `taksh-hf-publication/upload_manifest.json`) rather than four-way merging.
2. **Current HEAD does not contain recovery CI lineage visible on `origin/develop`.** The six remote-only commits are recovery/CI fixes. Integrating them after L0 work would need conflict review with `.github/workflows/ci-l0704-manual-qualification.yml`, `evaluation/taksh_cicd_recovery_check.py` and tests such as `tests/test_ci_workflows.py`.
3. **Five-version contains five-version + recovery + native-model content.** Its native files match `l111`, but its platform/GA work additionally depends on divergent publication and deployment artifacts. This is not one coherent upstream to merge.
4. **Closure and five-version both contain provider work.** `taksh/engineering.py` matches between unified-provider and five-version (`3ec33ea4e606`) but closure has a different hash (`3f0148c0c68c`). `taksh/specialists.py` is identical across closure/unified/five (`f3d6a3a9341d`). `serving/provider_diagnostics.py` differs across all three. Provider reconciliation must be semantic, not hash-based.
5. **Native `l111` changes `training/model.py` and `serving/local_inference.py`.** These interact with current local-inference allowlist and any consolidation of native inference must include tests for `tests/test_l111_native_inference.py` and `tests/test_l110_native_foundation.py`.
6. **Five-version modifies `training/model.py` differently from both current and native lineages.** Do not assume native and five-version model files are interchangeable.
7. **Large corpora are not source.** The corpora may be legally usable only under their licenses and for the stated purposes; publication or model-output licensing does not follow automatically.

### Potential conflicts

| Area | Branches | Risk |
|---|---|---|
| `training/sft.py` | current/closure vs l110/l111/five | overlapping tokenizer/config resolution changes |
| `training/model.py` | native vs current vs five | architecture/mask/attention behavior changes |
| `serving/local_inference.py` | current/closure vs l110/l111/five | checkpoint safety, inference mechanics and native contracts |
| `serving/providers.py` | current vs closure vs five | provider integration semantics |
| `.github/workflows/*` and `evaluation/taksh_cicd_recovery_check.py` | current vs remote recovery | CI validation expectations |
| `reports/*` | current/closure vs provider/native | generated evidence should not conflict semantically but should not be committed wholesale |
| `data/l03_corpus`, `data/l04_code_corpus` | all | excluded until legal decision |

---

## 8. Proposed consolidation manifest

| Path or capability | Source branch/ref | Destination proposal | Reason | Dependencies | Risk | Owner approval |
|---|---|---|---|---|---|---|
| Integrity layer: `training/chat_template.py`, `training/tokenizer_contract.py`, `training/checkpoint_schema.py`, `training/train.py` | present on all | Retain as frozen integrity surface | Equivalent and core to reproducibility | None | Low | Required before baseline commit |
| Integrity layer tests (`test_token_contract_and_chat`, `test_v110_contract`, data/manifest tests) | current working tree | Retain and re-run after any future integration | historical 129-test result must be re-proven | None | Low | Required for baseline authorization |
| Current checkpoint allowlist changes in `serving/local_inference.py` | working tree/closure | Retain selectively | safe checkpoint loading; current and closure match | NumPy allowlist; closure provider work may affect file | Medium | Required |
| Current training provenance changes in `training/manifest.py`, `training/pretrain.py`, `training/sft.py` | working tree/closure | Retain selectively | initialization/warmstart/tokenizer binding | SFT config/tokenizer conventions | Medium | Required |
| `configs/foundation/l060_foundation.yaml` | working tree/closure | Retain selectively as engineering config | canonical foundation config candidate | current L0/L0.6 qualification evidence | Medium | Required |
| L0.5/L0.6/L0.7/L0.8-PRE evaluation and tests | working tree/closure | Retain selectively; closure contains equivalent committed copies | evidence-gated foundation artifacts | no capability claims | High if blindly committed | Required |
| `reports/*` generated by L0/L0.5/L0.6 | working tree/closure | Keep separate or archive intentionally | generated evidence, not source | none | Medium | Required |
| `docs/CODING_GENIUS_PLAN.md`, `docs/CAPABILITY_STATEMENT.md`, `docs/COMPETITOR_MATRIX.md`, `docs/POSITIONING_STATEMENT.md` | current working tree | Retain as docs if owner wants the product direction documented | product direction; no capability claim | none | Low | Required |
| `docs/architecture/DESIGN-ARCHITECT.md` | current working tree | Retain selectively | coding-adjacent design feature P0 | architect package | Low/Medium | Required |
| `taksh/architect/*`, `tests/test_architect_package.py` | current working tree | Retain selectively | Design Architect engineering foundation | no model/training authorization | Medium | Required |
| `feature/l060-1-integration-closure` as a whole | closure branch | Consider selective integration or owner-approved local baseline branch | direct descendant; contains current source work and provider fixes | closure report/test evidence | High if automatic | Mandatory owner decision |
| `feature/l060-unified-provider-integration` as a whole | unified branch | Keep separate; closure is richer descendant | one provider commit without close-out | closure overlaps | Medium | Mandatory owner decision |
| Five-version platform/GA/IDE/security work | five-version | Selective integration only; keep line separate by default | valuable but divergent and release-governance sensitive | recovery lineage, native files, deployment/publication artifacts | High | Mandatory owner decision |
| Native model foundation (`foundation_111m.yaml`, l110 evaluation/tests/docs) | l110/l111/five | Keep separate pending native-model decision | source preparation only; no training authorized | l110 docs/eval/tests | High | Mandatory owner decision |
| Native inference mechanics (`l111`, `generation_contract.py`) | l111/five | Keep separate pending native-model decision | contract qualification; no training authorized | l110 foundation | High | Mandatory owner decision |
| Native v0.1 evaluation and baseline docs | **none found** | Do not invent or scaffold from audit | not present; absence must remain visible | external decision/new milestone | High | Mandatory owner decision |
| Recovery CI lineage (`61ae12a`, `30facf5`, `90eba0a`, merges/reconciliation) | `integration/taksh-cicd-recovery`, remote refs | Selectively evaluate after owner decision; do not replay blindly | current branch lacks the six-commit recovery lineage | CI/workflow conflict review | High | Mandatory owner decision |
| Remote `origin/develop`/`origin/main` history | remote refs | Do not fetch or switch; owner must authorize inspection/integration | local develop/main stale | network/authentication/owner approval | High | Mandatory owner decision |
| `data/l03_corpus/` | current working tree | Exclude from baseline; external archive/legal review only | CC-BY-SA obligations and mixed licenses | legal review | High/legal | Mandatory owner + legal decision |
| `data/l04_code_corpus/` | current working tree | Exclude from baseline; external archive/legal review only | mixed CC-BY-SA/PD/code licenses | legal review | High/legal | Mandatory owner + legal decision |
| `data/l02_overfit/`, `data/l02_sft/` | current working tree | Exclude; review provenance first | generated experiment data | source decision history | High | Mandatory owner decision |
| `taksh-hf-publication/*`, `hf_export/*`, `export_hf*`, `verify_pub.py`, `rebuild_manifest.py` | working tree | Exclude from baseline | publication/release surfaces | owner release authorization | High | Mandatory owner decision |
| Tracked token streams `data/v160_tokens`, `v170_tokens`, `v180_tokens` | repository history/current | Do not touch in consolidation | existing committed artifacts | none | Medium | Owner awareness required |
| Checkpoints/weights (`*.pt`, `*.gguf`, `*.safetensors`) | none detected in status | Exclude unconditionally from baseline | no release artifact authorized | training authorization | High | Owner decision required for any future addition |
| Frozen benchmarks and tokenizer v1 | tracked protected artifacts | Retain unchanged; verify by hash | protected evidence | none | Low if untouched | Owner approval for any change |
| Reports under `reports/l060*`, `reports/L05*`, `reports/L051*` | working tree/closure | Archive separately; do not treat as merge requirement | generated evidence | report generation tools | Medium | Owner decision |
| Existing large untracked `build/` directory | working tree/repo | Inspect before any commit; exclude unless explicitly intended | generated package/build output | none | Medium | Owner decision |

---

## 9. Recommended consolidation order

### Stage 0 — Owner freeze decision

**Prerequisites:** this manifest reviewed; owner chooses whether closure is the baseline.  
**Rollback:** no Git operation yet; current branch remains untouched.  
**Gate:** explicit owner approval.

1. Classify every working-tree path into retain/separate/exclude.
2. Sign off that data/corpora and publication surfaces are excluded.
3. Decide whether new coding-genius/architect files are in scope for the future baseline.

### Stage 1 — Integrity layer freeze

**Prerequisites:** Stage 0 approval.  
**Rollback:** revert working-tree changes if baseline rejected; no branch switching.  
**Gate:** rerun in the frozen context:

```text
pytest tests/test_token_contract_and_chat.py tests/test_v110_contract.py tests/test_data_integrity.py tests/test_eval_run_manifest.py tests/test_validate_manifest.py -q
pytest tests/test_l060_foundation.py tests/test_architect_package.py -q
python -m evaluation.roadmap.roadmap_validator
```

Do not proceed if any gate fails.

### Stage 2 — Provider baseline selection

**Prerequisites:** provider owner decision.  
**Recommended evaluation order:** `feature/l060-1-integration-closure` → `feature/l060-unified-provider-integration` → current working tree.  
**Gate:** compare semantic provider contracts, diagnostics, native-provider behavior and tests; never accept hash differences blindly.  
**Rollback:** keep all three refs; do not delete or rename branches.

### Stage 3 — Native-model decision, not merge

**Prerequisites:** Stage 2 complete; owner chooses whether native v0/v0.1 work belongs in this baseline.  
**Finding:** no native_v01 surface exists anywhere inspected.  
**Gate:** owner decides whether `l110`/`l111` remain research/source-preparation branches or are brought into the baseline in a separate, atomic, test-backed operation.  
**Rollback:** retain independent branches; no cross-merge with current L0 graph until this gate passes.

### Stage 4 — Five-version selective integration review

**Prerequisites:** Stages 1–3 complete.  
**Action:** review only the categories the owner explicitly wants:

- recovery CI fixes;
- native foundation/inference;
- provider/security changes;
- IDE/deployment surfaces;
- capability evaluation/contracts;
- GA hardening.

**Gate:** no release, publication or deployment artifacts enter the baseline without separate authorization.  
**Rollback:** discard candidate changes locally; refs remain available.

### Stage 5 — Future baseline source set

**Prerequisites:** owner-approved file set.  
**Action (future, not authorized by this document):** create a local baseline branch or commit, stage only reviewed source/tests/config/docs, run the full qualification set, and only then request protected-branch review.

**Gate:** explicit authorization for any commit/merge/push/release.

---

## 10. Required owner decisions

1. Is `feature/l060-1-integration-closure` the preferred provider/source baseline?
2. Should the existing working-tree L0/L0.6 untracked lineage be replaced by the closure branch, or staged separately file-by-file?
3. Is selective cherry-pick from `origin/develop`/recovery lineage authorized, despite current L0 work being ahead in some files and behind in CI files?
4. Should native-model `l110`/`l111` enter the baseline, or remain separate?
5. Should five-version platform/GA/IDE/security work enter any future consolidation, and under what slice?
6. Which generated `reports/*` files belong in Git, and which are local evidence only?
7. Are `data/l03_corpus/`, `data/l04_code_corpus/`, `data/l02_*` corpora legally cleared for redistribution, or permanently excluded from release/publication?
8. Is the new coding-genius/Design Architect documentation and architect package in scope for the future baseline?
9. Is any future local commit authorized? If yes, on which branch/ref and with what exclusions?
10. Does owner confirm that no model training, compute spend, publication, tag, release, or protected-branch update is authorized by this audit?

---

## 11. Exact read-only commands executed

This audit did not fetch, pull, merge, switch, stage, commit, clean, stash, push, publish or train. Representative commands:

```text
git rev-parse --verify refs/heads/<branch>
git for-each-ref --format="%(refname:short)|%(objectname:short)|%(committerdate:short)|%(authorname)" refs/heads
git for-each-ref --format="%(refname:short)|%(objectname)" refs/remotes/origin
git show -s --format="%ci %s" <branch>
git show -s --format="%p" <branch>
git merge-base <branch-a> <branch-b>
git rev-list --count <from>..<to>
git rev-list --left-right --count <branch-a>...<branch-b>
git log --oneline --reverse <from>..<to>
git ls-tree -r <branch> -- <path>
git ls-tree -r --name-only <branch>
git diff --stat <branch-a> <branch-b> [-- <paths>]
git diff --name-status/--name-only/--shortstat <branch-a> <branch-b>
git status --porcelain
git hash-object <path>
python -m evaluation.roadmap.roadmap_validator
pytest tests/test_l060_foundation.py tests/test_architect_package.py -q
pytest tests/test_token_contract_and_chat.py tests/test_v110_contract.py tests/test_data_integrity.py tests/test_eval_run_manifest.py tests/test_validate_manifest.py -q
Get-ChildItem
```

Some large shell output was redirected by the environment to temporary storage; no temporary artifacts were copied into the repository by this audit.

---

## 12. Known gaps and unverified claims

- No `fetch` or `pull` was run. Remote-tracking refs are only as current as the last fetch already present in the repository.
- Full regression was not re-run as part of this audit; it is too expensive for an exploratory branch comparison and the working tree must first be classified.
- No legal opinion has been given for CC-BY-SA or other corpus content.
- No secret scan certification is implied; filename/pattern inspection is insufficient for formal security review.
- No capability claim is made. Frozen Python benchmark remains 0/30.
- Native v0.1 evaluation artifacts are absent, not "implemented but undocumented".
- Remote-tracking `origin/develop`/`origin/main` may differ from actual remote heads if the local remote-tracking refs are stale.
- The current working tree contains a local modification to `taksh-hf-publication/upload_manifest.json`; publication remains unauthorized.
- The repository contains pre-existing CRLF/BOM/encoding warnings from Git; this audit did not normalize line endings.

---

## 13. Final status

`READY_FOR_OWNER_CONSOLIDATION_APPROVAL`

Owner action is required before any consolidation, commit, merge, push, release or publication. The safest near-term path is to treat `feature/l060-1-integration-closure` as the candidate baseline for existing source work, keep five-version and native branches separate, and exclude all corpora and publication surfaces pending explicit decisions.
