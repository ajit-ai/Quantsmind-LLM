# Taksh v1.0.1-B — Training Corpus and License Traceability Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Checkpoint: `artifacts/taksh-s2-experimental/taksh-s2.pt`
Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
Tokenizer: `artifacts/taksh-s2-experimental/tokenizer/qm_code_bpe_v0.json`

Status: **TRACEABILITY INSUFFICIENT — DISTRIBUTION REMAINS BLOCKED**

This is a factual provenance assessment, not a legal opinion. No model license, data license, contact, contact, upload, publication, license file, or model-card change was made.

---

## 1. Repository baseline and preservation status

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a`
- Remote: `origin` = `https://github.com/ajit-ai/Quantsmind-LLM.git`
- Package identity: `taksh`, version `1.0.0` from `taksh/__init__.py`

At the start of the review, the working tree contained pre-existing modified and untracked research files. They were preserved. This report is the only file created by this task.

The original checkpoint and tokenizer were not moved, replaced, retrained, or modified.

---

## 2. Evidence sources inspected

Repository/local evidence only; no downloads or new training runs were used.

- `README.md`
- `taksh/__init__.py`
- `pyproject.toml`
- `LICENSE` and `NOTICE`
- `configs/model/code_m3_9m.yaml`
- `training/model.py`
- `training/pretrain.py`
- `training/checkpoint.py`
- `training/checkpoint_schema.py`
- `training/operation_tasks.py`
- `training/generalization_tasks.py`
- `tokenizer/train_tokenizer.py`
- `tokenizer/qm_code_bpe_v0.report.json`
- `configs/data/approved_code_sources.yaml`
- `data/DATA_CARD_code_v1.md`
- `data/code_corpus_v0.jsonl`
- `data/code_corpus_v0.stats.json`
- `data/code_corpus_v1.manifest.json`
- `data/code_corpus_v1.jsonl`
- `data/code_corpus_v2.manifest.json`
- `data/code_corpus_v2.jsonl`
- `data/m3_code_verified.manifest.json`
- `data/m3_code_verified.train.jsonl`
- `data/m3_code_verified.val.jsonl`
- `data/m7/pretrain.manifest.json`
- `data/m7/pretrain.train.jsonl`
- `data/m7/pretrain.val.jsonl`
- `data/m7/m7-code-supervised-v0.manifest.json`
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

Local run records inspected in the existing experiment directory:

- `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\manifest.json`
- `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\metrics\training.json`
- `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\integrity\hashes.json`
- `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt`

No unrelated drive scan was performed.

---

## 3. Training corpus inventory

### 3.1 Selected S2 training lineage

The selected checkpoint is the frozen v2.1.0 S2 checkpoint referenced by the v2.2 report as `runE/checkpoints/trained.pt`.

- Architecture in checkpoint config: `qm-code-m3-9m`
- Parameters: **9,359,104**
- Vocabulary: **5,822**
- Context: **512**
- Weight SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`

`configs/model/code_m3_9m.yaml` says the architecture is from scratch, Llama-style decoder-only, and intentionally corrects the `code_30m.yaml` vocabulary mismatch.

`docs/v2.1.0_INSTRUCTION_TO_OPERATION_LEARNING_AND_SUPERVISION_ALIGNMENT.md` records:

- Scratch-initialised S2 seed `20261021`
- 240 optimizer steps
- `6,421` supervised tokens
- Loss `8.617 → 0.088`, validation `0.100`

`training/operation_tasks.py` defines the v2.1 synthetic operation curriculum:

- 13 operations across arithmetic, conditioning, collection, text and aggregate families
- Partitions `T`, `V`, `H1`, `H2`, `H3`, `C`
- Training partition `T`: 22 examples

`runE/manifest.json` records `tokens_consumed: 6421` and the final checkpoint SHA, but the `metrics/training.json` record contains no dataset ID, dataset SHA, tokenizer ID, tokenizer SHA, or training-command record. Therefore the selected checkpoint’s exact curriculum is consistent with the v2.1 operation-task report, but the full run bundle is incomplete.

**Traceability status:** **PARTIALLY VERIFIED**.

### 3.2 External code corpora in the repository

| Corpus | Records | Tokens / notes | Role in selected S2 |
|---|---:|---|---|
| `code_corpus_v0.jsonl` | 77 | Self-sourced local Taksh chunks; source keys are `local:*`; no psf-black/setuptools source records. | Pipeline tokenizer input documented by README; not external S2 training corpus. |
| `code_corpus_v1.jsonl` | 447 | 2,277,473 tokens with `qm_code_bpe_v0.json`; 9 pinned external repositories. | Data-card/corpus lineage; not the verified M3/M7 S2 training subset. |
| `code_corpus_v2.jsonl` | 437 | Source corpus for M3 and M7 dataset preparation. | Parent corpus; verified subsets are selected by `prepare_m3_dataset` and `prepare_m7_pretrain`. |
| `data/m3_code_verified.train.jsonl` / `.val.jsonl` | 193 train, 63 val | Licence-gated M3 verified split; train tokens **1,165,217**, val tokens **340,077**; tokenizer SHA verified. | Verified training-input subset for M3 lineage. |
| `data/m7/pretrain.train.jsonl` / `.val.jsonl` | 193 train, 26 val | M7 pretraining verified subset; same licence gate and tokenizer. | M7 lineage; identical train records to M3 train. |
| `data/m7/m7-code-supervised-v0.*.jsonl` | 507 train, 58 val, 42 dev, 159 locked | Source-derived supervised dataset; only `verified_ids` sources enter partitions. | M7 supervised lineage, not selected S2 runE evidence. |
| Synthetic operation curriculum | 22 T examples in v2.1 | Defined in `training/operation_tasks.py`. | **Selected S2 v2.1 lineage.** |

---

## 4. Tokenizer input inventory

- Tokenizer implementation: byte-level BPE with NFKC normalization, byte-level pre-tokenization, byte-level decoding, and special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- `qm_code_bpe_v0.report.json` records vocabulary **5,822** and round-trip sanity samples.
- The report does **not** record the exact input corpus, corpus hash, or tokenizer-training command.

The repository’s README and `docs/STATUS.md` document the intended tokenizer command as using `data/code_corpus_v0.jsonl`. That is pipeline evidence, not a tokenizer training manifest.

- `code_corpus_v0.jsonl`: **77 self-sourced records**, sources are `local:*`, and a direct file-path scan found **0** `src/blib2to3/` or `setuptools/config/_validate_pyproject/` file paths.

However, because the tokenizer report does not record its exact input, tokenizer input provenance remains **PARTIALLY VERIFIED**, not complete.

---

## 5. GPL-2.0 exposure finding: `psf/black → src/blib2to3/`

### Raw corpus presence

- `code_corpus_v1.jsonl` contains **10** `src/blib2to3/` file paths.
- `code_corpus_v2.jsonl` contains **0** `src/blib2to3/` file paths.
- `code_corpus_v2.manifest.json` records `quarantined_licence_conflict: 12`.
- `licence_verification.json` records `psf-black` as `verified_with_quarantine`, with conflicting subtree `src/blib2to3: [gpl-2.0]`, `needs_human_review: true`.

### Verified training subsets

- `data/m3_code_verified.train.jsonl`: **0** `src/blib2to3/` file paths; **0** `psf-black` source records.
- `data/m3_code_verified.val.jsonl`: **0**; **0**.
- `data/m7/pretrain.train.jsonl`: **0**; **0**.
- `data/m7/pretrain.val.jsonl`: **0**; **0**.
- `data/m7/m7-code-supervised-v0*.jsonl`: **0**; **0**.

### Statuses

| Target | Status |
|---|---|
| Raw `code_corpus_v1` file presence | **CONFIRMED INCLUDED** in the corpus file. |
| Raw `code_corpus_v2` file presence | **CONFIRMED EXCLUDED** for `src/blib2to3/` file paths. |
| M3/M7 verified training subsets | **CONFIRMED EXCLUDED** by manifest and direct record scan. |
| Selected v2.1 S2 synthetic-operation training | **CONFIRMED EXCLUDED**; S2 runE trained on the repository-authored operation curriculum, not an external corpus. |
| Tokenizer input | **UNKNOWN**; exact input not recorded. |

Do not make a legal derivative-work conclusion. The GPL-2.0 text is a provenance and distribution question for owner/legal review.

---

## 6. AGPL-3.0 exposure finding: `pypa/setuptools → setuptools/config/_validate_pyproject/`

### Raw corpus presence

- `code_corpus_v1.jsonl` contains **5** `setuptools/config/_validate_pyproject/` file paths.
- `code_corpus_v2.jsonl` contains **5** such file paths:
  - `setuptools/config/_validate_pyproject/__init__.py`
  - `setuptools/config/_validate_pyproject/error_reporting.py`
  - `setuptools/config/_validate_pyproject/extra_validations.py`
  - `setuptools/config/_validate_pyproject/fastjsonschema_exceptions.py`
  - `setuptools/config/_validate_pyproject/formats.py`
- `licence_verification.json` records `pypa-setuptools` as `verified_with_quarantine`, with conflicting subtree `setuptools/config/_validate_pyproject: [agpl-3.0]`, `needs_human_review: true`.

### Verified training subsets

- `data/m3_code_verified.train.jsonl`: **0** `setuptools/config/_validate_pyproject/` file paths; **0** `pypa-setuptools` source records.
- `data/m3_code_verified.val.jsonl`: **0**; **0**.
- `data/m7/pretrain.train.jsonl`: **0**; **0**.
- `data/m7/pretrain.val.jsonl`: **0**; **0**.
- `data/m7/m7-code-supervised-v0*.jsonl`: **0**; **0**.

### Statuses

| Target | Status |
|---|---|
| Raw `code_corpus_v1`/`v2` file presence | **CONFIRMED INCLUDED** in corpus files. |
| M3/M7 verified training subsets | **CONFIRMED EXCLUDED** by manifest and direct record scan. |
| Selected v2.1 S2 synthetic-operation training | **CONFIRMED EXCLUDED**; S2 runE trained on the repository-authored operation curriculum, not an external corpus. |
| Tokenizer input | **UNKNOWN**; exact input not recorded. |

Do not make a legal derivative-work conclusion. The AGPL-3.0 text is a provenance and distribution question for owner/legal review.

---

## 7. `runE` manifest recovery result

Recovered locally:

- `runE/manifest.json`
  - Run ID: `runE`
  - Schema: `quantsmind.v210_experiment/1`
  - Stage: `eval`
  - Curriculum digest: `7e93613cafe13fedbcf0f58157873b34a763bce0a7ca1f8259fe17a39fa9694b`
  - Trained checkpoint SHA: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
  - Tokens consumed: `6421`
  - Final loss: `0.08847621083259583`
  - Validation loss: `0.09998818584110426`
- `runE/metrics/training.json`
  - Steps: `240`
  - Seed: `20261021`
  - Tokens consumed: `6421`
  - No dataset ID, dataset hash, tokenizer ID, tokenizer hash, training command, or configuration is stored in this metrics record.
- `runE/integrity/hashes.json`
  - Records only `evaluations/trained.json` and `manifest.json`.

Missing:

1. Full run manifest with config, command line, data manifest, and tokenizer identity.
2. Dataset input list and hash.
3. Tokenizer input list and hash.
4. Preprocessing configuration for the exact `runE` training data.
5. Training log tying scratch initialization to `training/operation_tasks.py`.
6. Tokenizer training manifest tying `qm_code_bpe_v0.json` to its exact input corpus.

Script `scripts/run_v220.py` later demonstrates the controlled continued-training pattern and records `base_checkpoint_sha256`, but the selected v2.1 `runE` record predates that richer provenance style.

---

## 8. Confirmed facts versus unknowns

### Confirmed facts

- The selected checkpoint hash, size, architecture, vocabulary, and parameter count.
- `training/model.py` is an original Taksh implementation; no `from_pretrained`/`AutoModel` initialization path was found in training.
- The v2.1 S2 training report describes a scratch-initialized S2 trained on the synthetic operation curriculum.
- M3/M7 verified training subsets exclude both `psf-black` and `pypa-setuptools` records.
- M3/M7 manifests record `withheld_sources: {psf-black: 27, pypa-setuptools: 135}` and `withheld_records: 162`.
- Only `verified_ids` sources may enter M3/M7 training/validation partitions.
- Raw `code_corpus_v1` contains both restricted subtrees.
- Raw `code_corpus_v2` contains the AGPL-3.0 setuptools subtree but not the GPL-2.0 `blib2to3` file paths found in v1.

### Unknowns

- Exact dataset/tokenizer bundle for the selected `runE` checkpoint.
- Whether any real internal or synthetic data entered the selected run beyond the documented operation curriculum.
- Exact tokenizer training input.
- Published HumanEval/MBPP contamination status.
- PII coverage outside the narrow India-specific patterns.
- Whether copyrighted text or output memorization occurred.

---

## 9. Minimum additional evidence required

1. Recover or reconstruct the exact `runE` training manifest, config, command, and dataset hash.
2. Recover the tokenizer training manifest: input corpus, input hash, vocab target, seed, and output hash.
3. Produce a per-source/per-record dataset inventory for the exact S2 lineage.
4. Obtain owner/legal decisions for the two nested-licence conflicts.
5. Decide whether published-benchmark decontamination is required.
6. Record a model-weights license and training-data redistribution decision.
7. Store the selected checkpoint, tokenizer, config, and provenance bundle in one governed artifact directory.

---

## 10. Recommendation for distribution eligibility

**TRACEABILITY INSUFFICIENT — DISTRIBUTION REMAINS BLOCKED**

The evidence supports a narrow conclusion:

- The restricted GPL-2.0 and AGPL-3.0 files that were present in raw external corpora were **confirmed excluded** from the licence-verified M3/M7 training subsets.
- The selected v2.1 S2 lineage is a repository-authored synthetic operation curriculum, not an external corpus.
- However, the selected checkpoint does not yet have a complete public training record, the tokenizer input is not proven, and no weights/data license is recorded.

Therefore, the S2 weights should remain **NOT APPROVED FOR DISTRIBUTION** until the missing provenance records and owner/legal decisions above are supplied.

This is not a legal conclusion. It is a traceability and eligibility finding.
