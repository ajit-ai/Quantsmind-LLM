# TAKSH_EXISTING_MODEL_RELEASE_READINESS.md

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Decision: **NOT READY**

This report uses existing checkpoints and the existing Taksh inference implementation only. No training, external weight downloads, paid services, Git history changes, or package-version changes were performed.

---

## 1. Current branch, HEAD and working-tree status

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a`
- Remote: `origin` = `https://github.com/ajit-ai/Quantsmind-LLM.git`
- Package identity: `taksh`, version `1.0.0` from `taksh/__init__.py`

At the start of this readiness review, the working tree contained pre-existing modified and untracked work, including:

- Modified tracked files: `serving/code_chat.py`, `serving/completion_server.py`, `tests/test_v200_training_stability.py`, `tests/test_v210_supervision_learning.py`
- Untracked v2.2/v2.3/v2.5/C*/D* research, benchmark, provider, evaluation, and report files
- `reports/live-lab/` historical artifacts

Those files were preserved. This report is the only file added by this task.

---

## 2. Existing checkpoint inventory

### 2.1 Repository-local checkpoint files

| Checkpoint | Size | Release relevance |
|---|---:|---|
| `runs/code_pretrain/checkpoint_last.pt` | 195,178,207 | Historical 16.3M-class code pretrain; trained steps/limited capability require separate audit. |
| `runs/code_pretrain_tiny/checkpoint_last.pt` | 5,975,331 | Historical tiny code model; useful smoke artifact, not a product candidate. |
| `runs/code_sft/checkpoint_last.pt` | 5,975,331 | SFT smoke artifact; SFT corpus was quarantined, so not recommended for release. |
| `runs/code_sft_smoke/checkpoint_last.pt` | 5,971,491 | Smoke artifact; reported embedding-row mismatch in prior analysis. |
| `runs/code_smoke/checkpoint_last.pt` | 195,178,207 | Smoke/training artifact; historical only. |
| `runs/corporate_pretrain/checkpoint_last.pt` | 5,975,395 | Corporate synthetic-demo run; not a release candidate. |
| `runs/corporate_sft/checkpoint_last.pt` | 5,975,331 | Corporate synthetic-demo run; not a release candidate. |
| `runs/local_15m/checkpoint_last.pt` | 93,280,943 | General-text related 7.77M-class artifact; uses the 49,152-token tokenizer; not the Python code candidate. |
| `runs/m3_base/checkpoint_best.pt` | 34,311,773 | Historical M3 artifact; not frozen for release without requalification. |
| `runs/m3_base/checkpoint_last.pt` | 34,311,773 | Historical M3 artifact. |
| `runs/m4_sft/checkpoint_best.pt` | 34,311,773 | Historical M4 artifact. |
| `runs/m4_sft/checkpoint_last.pt` | 34,311,773 | Historical M4 artifact. |
| `runs/m7_pretrain/checkpoint_best.pt` | 112,414,437 | M7 pretraining checkpoint; associated with the negative M7 capability record. |
| `runs/m7_pretrain/checkpoint_last.pt` | 112,414,437 | M7 pretraining checkpoint. |
| `runs/m7_sft/checkpoint_best.pt` | 112,414,501 | M7 SFT checkpoint; capability evidence is negative. |
| `runs/m7_sft/checkpoint_last.pt` | 112,414,501 | M7 SFT checkpoint; capability evidence is negative. |
| `runs/m7_untrained/checkpoint.pt` | 37,464,309 | Baseline/untrained artifact; not a trained release candidate. |
| `runs/mm_fit_check/checkpoint_last.pt` | 114,301,607 | Multimodal fit-check; not the Python code path. |
| `runs/mm_m1_local/checkpoint_450.pt` | 114,301,773 | Multimodal experiment artifact. |
| `runs/mm_m1_local/checkpoint_last.pt` | 114,302,759 | Multimodal experiment artifact. |
| `runs/mm_m1_smoke/checkpoint_last.pt` | 114,302,823 | Multimodal smoke artifact. |
| `runs/mm_m1_v2/checkpoint_last.pt` | 52,110,647 | Multimodal experiment artifact. |
| `runs/smoke/checkpoint_last.pt` | 1,528,419 | Toy smoke artifact. |

Repository token datasets are not model checkpoints and are not listed above.

### 2.2 Evaluation-chain checkpoints outside the tracked repository tree

| Checkpoint | Size | SHA-256 | Evidence / status |
|---|---:|---|---|
| `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt` | 37,463,045 | `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` | Frozen S2 checkpoint; **selected candidate** for readiness testing. |
| `C:\Users\Lenovo\AppData\Local\Temp\opencode\v220\cellC\checkpoints\trained.pt` | 37,463,173 | `8cf0abe331d1c104155176f202d5be9839fa1b29dcb61ac9c66c43b5760d882b` | v2.2 continued cell C; reports better aggregate paraphrase exact counts but still fails binding/composition/mapping generalization. |
| `C:\Users\Lenovo\AppData\Local\Temp\opencode\v220\cellC_repeat\checkpoints\trained.pt` | 37,463,173 | identical to cell C | Deterministic repeat of cell C. |

---

## 3. Recommended checkpoint and evidence

### Recommended existing candidate: frozen S2 checkpoint

- Path: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt`
- Size: **37,463,045 bytes**
- SHA-256: **`02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`**
- Architecture: `qm-code-m3-9m`
- Parameters: **9,359,104**
- Vocabulary: **5,822**
- Context: **512**
- Weight SHA and frozen-dataset provenance are established in the v2.1/v2.2 experiment records.

Evidence for selecting S2:

1. It is the checkpoint used by the frozen AD-1, C1/C2/C3, and provider qualification suites.
2. Its hash is frozen and reproducible.
3. The implemented Taksh loader loads it successfully.
4. It is original Taksh-trained weights, not an external pretrained checkpoint.
5. v2.2 cell C is not selected because its gains are narrow, binding remains poor, and the S2 checkpoint has the most direct current-architecture evidence.

Tokenizer compatibility:

- Required tokenizer: `tokenizer/qm_code_bpe_v0.json`
- Size: **410,135 bytes**
- SHA-256: **`90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`**
- Vocabulary: **5,822**
- The companion general-text tokenizer is `tokenizer/qm_bpe_v0.json` with 49,152 vocabulary and is **not** the S2 code tokenizer.

---

## 4. Actual inference test results

Actual execution evidence from the existing implementation at review time:

| Check | Result |
|---|---|
| Checkpoint load | **PASS** — `loaded checkpoint step=? params=9,359,104 tensors=75` |
| Tokenizer load | **PASS** — `tokenizer/qm_code_bpe_v0.json` loaded successfully |
| Prompt tokens | **23** |
| Completion tokens | **6** |
| Finish reason | **stop** |
| Generated text | `'    return left + right\n'` |
| Second deterministic run | **Identical output** |
| Deterministic | **True** |
| Parsed as Python | **True** |

The sample prompt was:

```text
Write a function that returns the sum of two integers.
def compute_total(x, y):
```

The model produced syntactically valid Python but used `left` and `right` rather than the requested `x` and `y`. It is therefore constructible and loadable, but not functionally correct for that case.

No crash, missing dependency, configuration error, or memory failure occurred in this bounded test.

---

## 5. Small Python capability check

Existing frozen evidence remains the controlling capability record:

| Suite/configuration | Result |
|---|---|
| AD-1 raw S2 | **2/29 exact**, **12/29 structural**, **6/29 binding**, **24/29 parsed**, **4/4 negative safe** |
| AD-1 conservative binding repair | **5/29 exact**, **12/29 structural**, **10/29 binding** |
| C1 qualification raw | **2/46 exact**, **9/40 structural**, **3/46 binding**, **2/37 functional** |
| C1 repaired | **2/46 exact**, **9/40 structural**, **4/46 binding**, **2/37 functional** |
| C2 qualification raw | **1/35 exact**, **5/30 structural**, **1/35 binding**, **1/28 functional** |
| C3 development benchmark | **0/38 exact**, **0/38 binding**, **0/33 functional** |
| C3 held-out benchmark | **0/54 exact**, **0/54 binding**, **0/45 functional** |

Known failure modes from the frozen suites:

1. **Parameter-name binding failure** — emits training-time names instead of requested parameter names.
2. **Template/operator confusion** — structurally plausible operations replace the requested operation.
3. **Composition failure** — multi-operation and mapping tasks are not generally solved.
4. **Syntax corruption** — many composition/mapping outputs do not parse.
5. **Negative-control template hits** — malformed, side-effecting, undefined-name, empty, or unsupported outputs occur, although the scorer does not credit them.

The model can be used for bounded research demonstrations and regression tests. It should not be represented as a general coding model.

---

## 6. Hugging Face readiness checklist

| Requirement | Status | Evidence / missing item |
|---|---|---|
| Model weights | **Partial** | S2 weights exist locally, but the official release path and packaging are not prepared. |
| Model configuration | **Partial** | Architecture/config is embedded in the checkpoint (`qm-code-m3-9m`, 9,359,104 params, 5,822 vocab, 512 context), but a standard HF-facing config package is not prepared. |
| Tokenizer files | **Partial** | `qm_code_bpe_v0.json` exists and is compatible; a standard HF tokenizer package is not prepared. |
| Loading and inference code | **Partial** | Taksh local inference exists, but it is not a standard HF `from_pretrained`/pipeline integration. |
| Dependency requirements | **Partial** | PyTorch/tokenizers/numpy/PyYAML are declared; Transformers/Accelerate-style foundation loading is not applicable to the original S2 format. |
| Model card and usage examples | **Missing** | No HF experimental model card is ready in the repository. |
| Model usage examples | **Missing** | Only internal prompts and evaluation scripts exist; no public example card with clear limitations. |
| License | **Blocker** | Repository source is Apache-2.0, but the S2 weights are not covered by a dedicated weights license and are not cleared for redistribution. |
| Data/model provenance | **Blocker** | Corpus provenance exists in part, but the released artifact lacks a complete model/data rights and training-data provenance declaration. |
| Hardware requirements | **Partial** | CPU runtime is possible for this small model, but no canonical minimum RAM/disk table is published in a model card. |
| Known limitations | **Missing** | Would need explicit declaration: weak binding, weak composition/mapping, no general coding capability. |
| Security boundary | **Missing** | Generated code is not executed by the scorer, but an HF release card must state that this is not a certified code-execution service. |

---

## 7. Exact release blockers

1. **No model-weights license decision.** Source-code Apache-2.0 does not license the S2 weights for redistribution.
2. **No complete model/data provenance package.** Training corpus, selection criteria, decontamination, and published model-card provenance are not packaged for release.
3. **No HF-facing model/tokenizer package.** The checkpoint is a PyTorch/Taksh format, not a standard HF model directory.
4. **No HF model card.** Capability, hardware, license, limitations, and reproducibility details are not assembled as a release artifact.
5. **Capability is too weak for a normal experimental model claim.** C3 held-out functional capability is `0/45`; the model must be described as a negative-result research artifact unless the release purpose is explicitly experimental archaeology/regression research.
6. **Checkpoint custody risk.** The selected S2 checkpoint is currently outside the tracked repository tree in a temporary experiment directory; it needs a governed, hash-verified release artifact location.
7. **Package/version and release identity remain v1.0.0 source-release oriented.** No model artifact release metadata exists.

---

## 8. Minimum implementation changes needed before publication

Do not implement these in this task. The minimum path to an experimental release would be:

1. Choose and record a model-weights license after legal review.
2. Copy the selected S2 checkpoint and `qm_code_bpe_v0.json` into a governed release artifact directory.
3. Record file sizes and SHA-256 hashes for weights, tokenizer, config, and any inference code.
4. Add a minimal HF model card stating:
   - original Taksh architecture and parameter count
   - tokenizer and checkpoint hashes
   - load/generation examples
   - hardware requirements
   - exact capabilities and negative-result limitations
   - data/model provenance and license terms
5. Add a standalone loader/inference smoke test for the released artifact format.
6. Run a final small frozen-suite evaluation against the packaged artifact, not an informal local path.
7. Obtain owner approval for the exact model revision, weights license, and publication target.

No additional training, quantization, checkpoint conversion, or external model substitution is required to satisfy an **experimental negative-result** release, but none of that work has been done.

---

## 9. Recommended next version and next implementation task

### Recommended next version

**v1.0.0 experimental model artifacts** only after the blockers in §7 are cleared.

This should not be described as v1.1.0 capability work. A separate capability milestone should not begin from this release report.

### Recommended next implementation task

**TAKSH-EXPERIMENTAL-ARTIFACT-PACKAGING**

Scope:

1. Create a governed artifact directory for the frozen S2 checkpoint.
2. Record SHA-256 hashes and tokenizer metadata.
3. Draft the minimal HF model card with explicit negative-result limitations.
4. Add one reproducible loading/generation smoke test.
5. Stop before publication for owner/license approval.

### Decision

**NOT READY**

The existing Taksh S2 model is loadable, deterministic, and supported by a strong internal evaluation record. It is not ready for Hugging Face publication because the weights are not legally cleared for redistribution, the release artifact/tokenizer are not packaged in HF form, the model card and provenance record are missing, and demonstrated functional coding capability is effectively absent.

If those specific legal and packaging fixes are completed, the appropriate next milestone would be a **READY AFTER SPECIFIC FIXES** experimental release, not a capability release.
