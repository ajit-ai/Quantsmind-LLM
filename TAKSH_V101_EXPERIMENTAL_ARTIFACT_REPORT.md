# Taksh v1.0.1 — Experimental Artifact Packaging Report

Date: 2026-11-06
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`

Status: **locally staged experimental artifact candidate; NOT APPROVED FOR DISTRIBUTION.**

This task did not publish anything, log in to Hugging Face, upload weights, start training, download external models, use paid services, commit, push, merge, tag, or change the package version.

---

## 1. Files created and changed

Created under the repository:

- `artifacts/taksh-s2-experimental/taksh-s2.pt`
- `artifacts/taksh-s2-experimental/tokenizer/qm_code_bpe_v0.json`
- `artifacts/taksh-s2-experimental/config.json`
- `artifacts/taksh-s2-experimental/artifact_manifest.json`
- `artifacts/taksh-s2-experimental/LICENSE_STATUS.txt`
- `artifacts/taksh-s2-experimental/README.md`
- `tests/test_taksh_s2_experimental_artifact_smoke.py`
- `TAKSH_V101_EXPERIMENTAL_ARTIFACT_REPORT.md`

No existing checkpoint, training run, experiment, branch, tag, or tracked file was modified by this task. Pre-existing working-tree modifications and untracked research files remain preserved.

The original checkpoint was copied, never moved or deleted.

---

## 2. Source checkpoint and tokenizer verification

| Source | Exists | Size | SHA-256 | Expected | Result |
|---|---:|---:|---|---|---|
| `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt` | yes | 37,463,045 | `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` | same | **MATCH** |
| `tokenizer/qm_code_bpe_v0.json` | yes | 410,135 | `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` | same | **MATCH** |

Artifact copies were rechecked after copying:

- `artifacts/taksh-s2-experimental/taksh-s2.pt`
  - SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- `artifacts/taksh-s2-experimental/tokenizer/qm_code_bpe_v0.json`
  - SHA-256: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`

### Artifact metadata

- Model ID: `taksh-s2`
- Architecture: `qm-code-m3-9m`
- Parameters: **9,359,104**
- Vocabulary size: **5,822**
- Context length: **512**
- Weight SHA-256: **`02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`**
- Tokenizer SHA-256: **`90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`**
- Distribution status: **NOT APPROVED FOR DISTRIBUTION**

---

## 3. Smoke test command and results

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_taksh_s2_experimental_artifact_smoke.py -q -p no:cacheprovider
```

Result: **2 passed in 26.04s**

The test verifies:

1. Artifact checkpoint hash.
2. Tokenizer hash.
3. The existing Taksh loader loads `taksh-s2.pt`.
4. Parameter count is **9,359,104**.
5. Context length is **512**.
6. The tokenizer loads.
7. A fixed deterministic prompt produces output.
8. A repeated deterministic run produces the same output.
9. The completed source parses as Python.

Observed output for the fixed prompt:

```text
    return left + right
```

This is the known binding failure: the prompt asks for `x` and `y`, while the model emits `left` and `right`. The smoke test records this and does not claim coding competence.

Compilation of the smoke test also passed.

---

## 4. Draft model-card contents

`artifacts/taksh-s2-experimental/README.md` contains a draft HF-style model card with:

- `license: unknown`
- Draft/non-publication status
- Taksh S2 architecture and specifications
- Verified checkpoint and tokenizer hashes
- Installation/inference instructions using the existing Taksh loader
- Verified CPU-only inference note
- Existing evaluation results:
  - AD-1 raw: **2/29 exact**, **12/29 structural**, **6/29 binding**
  - C3 held-out: **0/54 exact**, **0/54 binding**, **0/45 functional**
- Explicit statement that coding capability is not demonstrated
- Explicit statement that weights are not cleared for distribution
- Limitations: weak binding, weak composition/mapping, 512-token context, no certified code execution

The model card does **not** invent or declare a license.

---

## 5. License and provenance questions requiring owner approval

Before any Hugging Face publication or distribution, the owner must decide:

1. **Weights license:** choose and record the license that governs S2 weights. Source-code Apache-2.0 does not by itself license the checkpoint for redistribution.
2. **Training-data rights:** document the corpus sources, rights basis, deduplication, PII/secret controls, and decontamination posture for the exact training run.
3. **Model-card provenance:** confirm that the card may describe the model as original Taksh-trained weights.
4. **Release custody:** move or copy the artifact into a governed, hash-verified distribution directory.
5. **Publication channel:** confirm the Hugging Face organization/repository name and visibility.
6. **Example and limitation review:** approve the usage example and the explicit negative-capability statement.
7. **Version label:** decide whether `v1.0.1-experimental-candidate` is a package label, artifact label, or release tag; no version change was made here.
8. **Security boundary:** approve the model card’s statement that generated code is not execution-certified.

---

## 6. Exact remaining steps before Hugging Face publication

1. Obtain a recorded weights-license decision.
2. Obtain a model/data provenance and rights review.
3. Confirm the final artifact directory and exact uploaded files.
4. Confirm organization/repository target and visibility.
5. Finalize the model-card license field only after approval.
6. Run the smoke test against the final staged artifact.
7. Publish separately; this task did not upload, log in, publish, or distribute anything.

## Final state

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a`
- Package version: `1.0.0`
- Artifact status: **NOT APPROVED FOR DISTRIBUTION**
- Upload/publish status: **not performed**
- Training/download/paid-service status: **not performed**
