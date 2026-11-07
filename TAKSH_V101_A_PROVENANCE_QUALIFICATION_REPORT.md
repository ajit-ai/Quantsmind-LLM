# Taksh v1.0.1-A — Model Provenance Qualification Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Experimental artifact package inspected: `artifacts/taksh-s2-experimental/`

Status: **DISTRIBUTION BLOCKED — IDENTIFIED RESTRICTIONS**

This is an engineering provenance assessment, not a legal opinion. No model license, data license, contact, or publication decision was made.

---

## 1. Repository baseline and Git status

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a`
- Remote: `origin` = `https://github.com/ajit-ai/Quantsmind-LLM.git`
- Package identity: `taksh`, version `1.0.0` from `taksh/__init__.py`
- Original checkpoint: copied to `artifacts/taksh-s2-experimental/taksh-s2.pt`; original checkpoint was not moved or modified.
- Selected artifact:
  - `taksh-s2.pt`: **37,463,045 bytes**, SHA-256 `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
  - `tokenizer/qm_code_bpe_v0.json`: **410,135 bytes**, SHA-256 `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`
  - `config.json`: architecture `qm-code-m3-9m`, **9,359,104 parameters**, vocabulary **5,822**, context **512**

At review, the working tree contained pre-existing modified and untracked research files. They were preserved. This report is the only file created by this task.

---

## 2. Evidence sources inspected

Repository evidence only; no external lookup, model download, or training was used.

- `README.md`
- `taksh/__init__.py`
- `pyproject.toml`
- `LICENSE` and `NOTICE`
- `configs/model/code_m3_9m.yaml`
- `training/model.py`
- `training/pretrain.py`
- `training/checkpoint.py`
- `training/checkpoint_schema.py`
- `tokenizer/train_tokenizer.py`
- `tokenizer/qm_code_bpe_v0.report.json`
- `configs/data/approved_code_sources.yaml`
- `data/DATA_CARD_code_v1.md`
- `data/code_corpus_v1.manifest.json`
- `data/code_corpus_v2.manifest.json`
- `data/licence_verification.json`
- `data/contamination_audit.json`
- `docs/OPEN_ISSUES.md`
- `docs/VERSIONING.md`
- `docs/STATUS.md`
- `docs/source/roadmap/M3_EXECUTION.rst`
- `docs/v1.2.0_PYTHON_DATA_TOKENIZER.md`
- `docs/v1.3.0_PYTHON_PRETRAINING.md`
- `docs/v1.4.0_PYTHON_CAPABILITY_AGENT_INTEGRATION.md`
- `docs/v1.5.0_PYTHON_CODING_INTELLIGENCE_FOUNDATION.md`
- `docs/v1.8.0_AUTHORIZED_CORPUS_QUALIFICATION_AND_LEARNING_EXPERIMENT.md`
- `docs/v2.1.0_INSTRUCTION_TO_OPERATION_LEARNING_AND_SUPERVISION_ALIGNMENT.md`
- `docs/v2.2.0_COMPLETE_FUNCTION_GENERATION_AND_GENERALIZATION_RELIABILITY.md`
- `TAKSH_EXISTING_MODEL_RELEASE_READINESS.md`
- `TAKSH_V101_EXPERIMENTAL_ARTIFACT_REPORT.md`

No evidence was inferred from filenames or website claims.

---

## 3. Dataset/source inventory

### 3.1 `code_corpus_v1`

Purpose: licence-verified external Python code corpus for M1/M2 lineage.  
Records: **447**. Tokens with `qm_code_bpe_v0.json`: **2,277,473**.  
Sources: 9 repositories pinned to full 40-character commit SHAs.

| Source | Repo | Declared licence | Records kept |
|---|---|---:|---:|
| psf-requests | psf/requests | Apache-2.0 | 19 |
| pallets-flask | pallets/flask | BSD-3-Clause | 23 |
| psf-black | psf/black | MIT | 37 |
| urllib3-urllib3 | urllib3/urllib3 | MIT | 39 |
| pallets-click | pallets/click | BSD-3-Clause | 17 |
| pytest-dev-pytest | pytest-dev/pytest | MIT | 79 |
| aio-libs-aiohttp | aio-libs/aiohttp | Apache-2.0 | 63 |
| pypa-setuptools | pypa/setuptools | MIT | 135 |
| tiangolo-fastapi | tiangolo/fastapi | MIT | 35 |

Evidence status: **PARTIALLY VERIFIED**.

- The corpus has pinned commits, per-record provenance fields, exact and near-duplicate controls, and a licence-verification gate.
- `data/licence_verification.json` checked 9 sources: **7 verified**, **2 rejected/needs human review**.
- `psf-black` is MIT at the root but contains a GPL-2.0 `src/blib2to3/` subtree.
- `pypa/setuptools` is MIT at the root but contains AGPL-3.0 text under `setuptools/config/_validate_pyproject/`.
- Both conflicting subtrees are quarantined and counted in the manifest, but the data card explicitly leaves them pending human review.
- The project’s own config notes: “this project keeps no model-weights or training-data licence yet.”
- Only in-repo benchmark subsets were checked. Published HumanEval/MBPP were not checked; lexical audit only.

License call: **PARTIALLY VERIFIED**. Per-record license metadata is strong, but redistribution requires human confirmation and cannot be inferred from SPDX strings.

### 3.2 `code_corpus_v2` and M3 verified split

Purpose: M3/M7 lineage from `code_corpus_v2`, followed by `data.prepare_m3_dataset` withholding unresolved-licence records.

- `code_corpus_v2`: **437 kept records**
- Counts by source include `psf-black` **27** and `pypa-setuptools` **135**; manifest records **12** `quarantined_licence_conflict` rejections.
- M3 execution evidence shows train **193 records / 1,165,217 tokens** from licence-verified sources, validation **63 records / 340,077 tokens**, and holdout withheld because unresolved licence records were deliberately not spent.
- The M3 execution guide says training data is “never” licensed by a record’s SPDX field alone; it is licensed only by `data/licence_verification.json:verified_ids`.

License call: **PARTIALLY VERIFIED**. This is a strong provenance pattern, but it is **not** a project-level model/data license or a legal clearance.

### 3.3 Other corpora and artifacts

| Dataset/artifact | Evidence status |
|---|---|
| `code_corpus_v0` | **PARTIALLY VERIFIED.** 77 chunks, mostly this repository’s own source; retained as legacy self-sourced shard. |
| `corpus_v0` | **PARTIALLY VERIFIED.** General-text corpus of 9,993,983 tokens; older provenance and schema conformance are incomplete. |
| `internal_corpus_v0` | **UNKNOWN.** Synthetic/demo/internal-style corpus; not a provenance-checked external dataset. |
| `code_sft_v0` | **Not selected.** Quarantined SFT corpus; not a basis for S2 weights. |
| Synthetic datasets | **PARTIALLY VERIFIED/UNKNOWN.** Diagnostic and synthetic runs exist, but selected S2 capability/continued-training evidence is not traceable to a fully cleared synthetic corpus in the tracked evidence reviewed. |

### 3.4 Tokenizer provenance

- `tokenizer/qm_code_bpe_v0.json` is a byte-level BPE tokenizer with NFKC normalization, byte-level pre-tokenization, byte-level decoding, and special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- `qm_code_bpe_v0.report.json` records vocabulary **5,822** and round-trip sanity samples.
- The tokenizer file hash is verified: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`.
- The exact corpus input file used to generate this tokenizer is **not recorded in the tokenizer report**. README and docs state the intended command, but provenance is therefore **PARTIALLY VERIFIED**, not complete.

---

## 4. Training-run and checkpoint traceability

### 4.1 Selected checkpoint lineage

The selected artifact is the frozen v2.1.0 S2 checkpoint, referenced by the v2.2 report as:

`C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt`

Its embedded config is:

- `name`: `qm-code-m3-9m`
- `vocab_size`: `5822`
- `d_model`: `256`
- `n_layers`: `8`
- `n_heads`: `8`
- `n_kv_heads`: `4`
- `ffn_dim`: `1024`
- `max_seq_len`: `512`
- `rope_theta`: `10000.0`
- `norm_eps`: `1e-5`
- `tie_embeddings`: `true`

Parameter count is **9,359,104**, consistent with `configs/model/code_m3_9m.yaml`.

### 4.2 Original Taksh implementation

**VERIFIED:** The repository contains an original decoder implementation in `training/model.py` using RMSNorm, RoPE, GQA, and SwiGLU. No `AutoModel` or `from_pretrained` model initialization was found in the training package.

The pretraining loop constructs `QMModel` from config and trains it. `training/continued_training.py` can load an existing Taksh checkpoint for continued fine-tuning and records the base checkpoint hash, but that is for v2.2 continued cells; the selected v2.1 checkpoint itself predates that path and is treated as the frozen base.

### 4.3 Run traceability

**PARTIALLY VERIFIED.** The v2.1 and v2.2 reports identify `runE` and the checkpoint SHA, and the M3/M7 evidence traces the architecture and data pipeline. However, the exact S2 run manifest and per-run dataset/tokenizer identities for the selected `runE` checkpoint are outside the tracked repository evidence reviewed here. The selected checkpoint is traceable as a frozen Taksh checkpoint, but not as a complete public training-record bundle.

### 4.4 External initialization

**VERIFIED absence:** no Hugging Face/transformers `from_pretrained` initialization was found in the training code. The selected model is not a downloaded pretrained foundation checkpoint.

---

## 5. Third-party component inventory

| Component | License/obligation evidence | Status |
|---|---|---|
| Taksh source code | Repository `LICENSE` is Apache-2.0; `NOTICE` recorded. | **VERIFIED** |
| Taksh model code | `training/model.py` and related training code are original Taksh implementation; no external copied architecture component identified. | **VERIFIED**, subject to project source license. |
| Runtime dependencies declared in `pyproject.toml` | PyTorch, tokenizers, NumPy, PyYAML, Gradio, Pillow, FastAPI, Uvicorn, Pydantic and OpenAI client under their own licences. | **PARTIALLY VERIFIED**; dependency obligations exist but are not a weights license. |
| Tokenizer implementation | `tokenizers` library is a third-party runtime dependency; the trained tokenizer data is a Taksh-generated artifact. | **PARTIALLY VERIFIED**; tokenizer artifact provenance is incomplete. |
| Training data sources | Permissive licences recorded per source, uniqueness and attribution attributes present. | **PARTIALLY VERIFIED**; human review pending. |
| Nested non-permissive subtrees | GPL-2.0 and AGPL-3.0 texts found in MIT-labelled repos; conflicting subtrees quarantined. | **VERIFIED restriction/need for decision**. |
| External pretrained weights | None identified for the selected S2 checkpoint. | **VERIFIED absence** in reviewed evidence. |

No training data was downloaded for this assessment and no external repository was redistributed.

---

## 6. Verified facts, partial findings, and unknowns

### VERIFIED

- The selected checkpoint hash, size, architecture, vocabulary, and parameter count.
- `qm_code_bpe_v0.json` hash and 5,822 vocabulary size.
- The architecture is implemented in Taksh’s own `training/model.py`.
- No transformers-style pretrained initialization path was found in the training implementation.
- The corpus pipeline records pinned external commits and licence verification results.
- 7 of 9 external corpus sources are verified; 2 require human review due nested conflicting licences.

### PARTIALLY VERIFIED

- M3 dataset construction withheld unresolved-licence records before training.
- The S2 checkpoint is a frozen Taksh-trained checkpoint.
- Tokenizer generation code and report exist, but exact tokenizer input provenance is incomplete.
- Per-record corpus licenses and attribution fields exist, but they are not a project-level training-data license.

### UNKNOWN

- Exact run manifest, seed, config, and token stream for the selected `runE` checkpoint.
- Whether any real internal data, synthetic data, or unrecorded source entered the selected run.
- Published HumanEval/MBPP contamination status.
- Whether the tokenizer was trained from the exact recorded corpus input intended for the M3/v2.1 lineage.
- Whether outputs memorize copyrighted training code.

### OWNER APPROVAL REQUIRED

- Selection of a model-weights license.
- Human/nested-licence decision for `psf-black` and `pypa-setuptools` conflicts.
- Whether published-benchmark decontamination is required before any public capability claim.
- Whether the exact `runE` training record is sufficient for public provenance.
- Whether PII/secret controls in `data/filters.py` are sufficient for public training-data reuse.

---

## 7. Distribution blockers

1. **Model-weight license is absent.** No weights license is declared. Do not add or infer one.
2. **No training-data release license is recorded.** The repository docs explicitly state the project keeps no model-weights or training-data licence yet.
3. **Nested licence restrictions remain unresolved.** GPL-2.0 and AGPL-3.0 texts are quarantined pending human review.
4. **Published benchmark decontamination is not demonstrated.** Only the in-repo hand-transcribed subsets were checked.
5. **Selected checkpoint custody/provenance is incomplete.** The checkpoint is frozen and hashed, but the exact training-run bundle is not included in the artifact package.
6. **Tokenizer input provenance is incomplete.** The tokenizer file is verified, but its exact training input is not captured in the tokenizer report.
7. **No legal opinion is provided.** This report does not clear redistribution.

---

## 8. Specific owner decisions or records required

Before distribution, the owner must provide or approve:

1. A model-weights license decision and license text.
2. A training-data rights statement and redistribution decision.
3. A human nested-licence decision for `psf-black/src/blib2to3/` and `pypa/setuptools/config/_validate_pyproject/`.
4. A contamination decision: publish the known limitations or obtain a proper decontamination record.
5. The exact `runE` training manifest or an explicit finding that it cannot be recovered.
6. The tokenizer input corpus record and hash.
7. A governed release artifact directory with weights, tokenizer, config, hashes, model card, and license text.
8. Final legal review before any Hugging Face publication.

---

## 9. Final recommendation

**DISTRIBUTION BLOCKED — IDENTIFIED RESTRICTIONS**

This is not a finding that the weights are unsafe or incorrectly trained. It is a provenance and licensing finding: the existing repository evidence does not establish that the S2 weights and their training data are legally and factually cleared for distribution.

The artifact must remain **NOT APPROVED FOR DISTRIBUTION** until the owner decisions above are recorded.

This task did not upload, publish, license, train, or modify the checkpoint or tokenizer.
