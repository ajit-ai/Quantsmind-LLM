# Taksh v1.0.1-C — Tokenizer and Training Run Evidence Recovery Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Checkpoint: `artifacts/taksh-s2-experimental/taksh-s2.pt`
Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
Tokenizer: `artifacts/taksh-s2-experimental/tokenizer/qm_code_bpe_v0.json`

Status: **PARTIAL RECOVERY — SPECIFIC EVIDENCE STILL REQUIRED**

This report uses existing local evidence only. No checkpoint, tokenizer, training data, manifest, experiment output, package version, branch, tag, or Hugging Face state was modified.

---

## 1. Repository baseline and preservation status

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a`
- Remote: `origin` = `https://github.com/ajit-ai/Quantsmind-LLM.git`
- Package identity: `taksh`, version `1.0.0` from `taksh/__init__.py`

At review time, the working tree contained pre-existing modified and untracked research files. They were preserved. This report is the only file created by this task.

The original checkpoint and tokenizer were not moved, replaced, retrained, downsampled, or modified.

---

## 2. Tokenizer provenance findings

### 2.1 What is verified

- `qm_code_bpe_v0.json` is a byte-level BPE tokenizer with NFKC normalization, byte-level pre-tokenization, byte-level decoding, and special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- `tokenizer/qm_code_bpe_v0.report.json` records vocabulary **5,822** and round-trip sanity samples.
- Artifact tokenizer SHA-256 is verified: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`.
- Git history shows `tokenizer/qm_code_bpe_v0.json` and `tokenizer/qm_code_bpe_v0.report.json` were first added in commit `fe771a3cb2a38dc4a241b87e508624c0542d690e` on `2026-09-27`, message `feat(m1): scale corpus to 144 files, 200-step pretrain (loss 10.44 to 4.87), SFT data 439 pairs`.
- That commit also added `data/code_corpus_v0.jsonl`, `data/code_corpus_v0.stats.json`, and `data/code_corpus_v0.tokens.meta.json`.
- `data/code_corpus_v0.tokens.meta.json` records:
  - vocab size **5,822**
  - tokens **100,244**
  - dtype `uint16`
  - corpus `data\code_corpus_v0.jsonl`
- `data/code_corpus_v0.stats.json` records:
  - seen **144**
  - kept **77**
  - by language: Python **68**, JavaScript **9**
  - `by_source: {}`
- A direct scan of `data/code_corpus_v0.jsonl` found **77** records with sources only under `local:*`, and **0** `src/blib2to3/` or `setuptools/config/_validate_pyproject/` file paths.

### 2.2 What is not recovered

No tokenizer training manifest was found. `qm_code_bpe_v0.report.json` does not record:

- exact input path
- input file hash
- full corpus inventory
- command line
- training date/time
- trainer version/seed
- output tokenizer SHA-256

Only pipeline documentation remains:

- README documents a tokenizer training command with `--input data/code_corpus_v0.jsonl --out tokenizer/qm_code_bpe_v0.json --vocab-size 32768`.
- `docs/STATUS.md` repeats that documented command.

This is **PARTIALLY VERIFIED**, not a tokenizer training record.

**Tokenizer provenance status:** **PARTIALLY VERIFIED**.

---

## 3. Recovered tokenizer input command and corpus details

The best-supported recovery is:

```powershell
python -m tokenizer.train_tokenizer \
  --input data/code_corpus_v0.jsonl \
  --out tokenizer/qm_code_bpe_v0.json \
  --vocab-size 32768
```

Evidence:

- README quick-start records this command.
- `docs/STATUS.md` records the same command.
- `data/code_corpus_v0.tokens.meta.json` ties the resulting **5,822** vocabulary and **100,244** token stream to `data/code_corpus_v0.jsonl`.
- Git history introduced the tokenizer, report, and code_corpus_v0 artifacts in the same commit.

However, no command log proves the exact execution. Therefore:

- Exact tokenizer input corpus: **PARTIALLY VERIFIED** as `data/code_corpus_v0.jsonl`.
- Input corpus hash: **NOT RECOVERED** as a tokenizer-generation record.
- Tokenizer training command: **PARTIALLY VERIFIED** from documented commands, not from a run manifest.
- Synthetic curriculum data as tokenizer input: **NOT APPLICABLE**; the tokenizer predates the v2.1 operation curriculum and is documented as trained from a text corpus.

---

## 4. `runE` training configuration and command findings

### 4.1 Recovered run record

`C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\manifest.json` contains:

- `run_id`: `runE`
- schema: `quantsmind.v210_experiment/1`
- stage: `eval`
- curriculum digest: `7e93613cafe13fedbcf0f58157873b34a763bce0a7ca1f8259fe17a39fa9694b`
- trained checkpoint SHA: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- tokens consumed: `6421`
- initial loss: `8.617237091064453`
- final loss: `0.08847621083259583`
- validation loss: `0.09998818584110426`
- elapsed seconds: `91.55`

`runE/metrics/training.json` additionally records:

- requested steps: `240`
- completed steps: `240`
- seed: `20261021`
- learning rate: `0.0003`
- train examples: `22`
- validation examples: `8`

`runE/integrity/hashes.json` records only:

- `evaluations/trained.json`: `046c218d885b55e416ba13c2cc1f26fe9ae51367d1216f91998eb52c11666eb6`
- `manifest.json`: `f503dbe6ccaf4563043e259b1f23f45d24bd2c9bab93d53e410a5832f37a71a2`

### 4.2 Training configuration reconstruction

The v2.1 report records the aligned-supervision run as:

- Model/tokenizer: S2 (`9,359,104`) / frozen 5,822-id tokenizer
- Context / micro / accumulation: `256 / 1 / 2`
- Optimizer: AdamW, learning rate `3e-4`, warmup `2`, cosine schedule, gradient clip `1.0`, weight decay `0.1`
- Steps / supervised tokens / runtime: `240 / 6,421 / 91.6 s`
- Loss: `8.617 → 0.088`, validation `0.100`

`training/operation_tasks.py` defines the synthetic v2.1 operation curriculum:

- 13 operations, five families
- Partitions `T`, `V`, `H1`, `H2`, `H3`, `C`
- Training partition `T`: **22 examples**
- The curriculum is repository-authored and synthetic.

### 4.3 Not recovered

No recovered record includes:

- full training configuration object
- command line
- dataset/curriculum file list or hash
- tokenizer path/hash inside the run manifest
- model initialization record
- full training log linked to scratch initialization

Therefore, the training mechanism is reconstructible, but the exact `runE` run bundle is incomplete.

**RunE training configuration status:** **PARTIALLY VERIFIED**.

---

## 5. Checkpoint-to-run traceability

The linkage is strong but not complete:

| Link | Evidence | Status |
|---|---|---|
| Checkpoint SHA in `runE/manifest.json` | `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` | **VERIFIED** |
| Artifact checkpoint SHA | same | **VERIFIED** |
| Model/tokenizer geometry | `qm-code-m3-9m`, 9,359,104 params, 5,822 vocab, 512 context | **VERIFIED** |
| Curriculum digest | `7e93613c...` in manifest and v2.1 report | **PARTIALLY VERIFIED** |
| Dataset input list/hash | absent from `runE/manifest.json` and `metrics/training.json` | **NOT RECOVERED** |
| Tokenizer path/hash in run record | absent from `runE/manifest.json` and `metrics/training.json` | **NOT RECOVERED** |
| Training command | absent from recovered run records | **NOT RECOVERED** |
| Model initialization | v2.1 report says scratch-initialised S2; training code has no pretrained initialization path | **PARTIALLY VERIFIED** |

The checkpoint can be traced to an existing frozen Taksh run, but the public provenance bundle is incomplete.

---

## 6. GPL-2.0 and AGPL-3.0 exposure implications

### GPL-2.0: `psf/black → src/blib2to3/`

- Raw `code_corpus_v1.jsonl`: **10** `src/blib2to3/` file paths.
- Raw `code_corpus_v2.jsonl`: **0** `src/blib2to3/` file paths.
- `code_corpus_v2.manifest.json` records `quarantined_licence_conflict: 12`.
- `licence_verification.json` records `psf-black` as `verified_with_quarantine`, conflicting subtree `src/blib2to3: [gpl-2.0]`, `needs_human_review: true`.
- M3 verified train/val: **0** `src/blib2to3/` paths and **0** `psf-black` records.
- M7 verified pretrain train/val: **0**; **0**.
- M7 supervised train/val/locked: **0**; **0**.
- Selected v2.1 S2 lineage used the synthetic operation curriculum, not an external corpus.

Statuses:

| Target | Status |
|---|---|
| Raw `code_corpus_v1` file presence | **CONFIRMED INCLUDED** |
| Raw `code_corpus_v2` file presence | **CONFIRMED EXCLUDED** for `src/blib2to3/` file paths |
| M3/M7 verified training subsets | **CONFIRMED EXCLUDED** |
| Tokenizer input | **UNKNOWN** because tokenizer input is not recorded |
| Selected v2.1 S2 training | **CONFIRMED EXCLUDED** |

### AGPL-3.0: `pypa/setuptools → setuptools/config/_validate_pyproject/`

- Raw `code_corpus_v1` and `code_corpus_v2`: **5** `setuptools/config/_validate_pyproject/` file paths in each file.
- `licence_verification.json` records `pypa-setuptools` as `verified_with_quarantine`, conflicting subtree `setuptools/config/_validate_pyproject: [agpl-3.0]`, `needs_human_review: true`.
- M3 verified train/val: **0** such file paths and **0** `pypa-setuptools` records.
- M7 verified pretrain train/val: **0**; **0**.
- M7 supervised train/val/locked: **0**; **0**.
- Selected v2.1 S2 lineage used the synthetic operation curriculum, not an external corpus.

Statuses:

| Target | Status |
|---|---|
| Raw `code_corpus_v1`/`v2` file presence | **CONFIRMED INCLUDED** |
| M3/M7 verified training subsets | **CONFIRMED EXCLUDED** |
| Tokenizer input | **UNKNOWN** because tokenizer input is not recorded |
| Selected v2.1 S2 training | **CONFIRMED EXCLUDED** |

No factual exposure in verified S2/M3/M7 training subsets was identified. This is not a legal conclusion about derivative works or redistribution.

---

## 7. HumanEval/MBPP decontamination evidence

- `data/contamination_audit.json` is for `code_corpus_v2`, not the selected v2.1 synthetic curriculum.
- It checks only (a) in-repo hand-transcribed HumanEval/MBPP subsets and (b) a lexical/near-duplicate audit for `code_corpus_v2`.
- It records:
  - exact overlap: **0**
  - 13-word shingle overlap: **0**
  - near-duplicate overlap: **0**
- It explicitly says published HumanEval (164 problems) and MBPP (500 problems) were **not available and were not checked**.
- It explicitly says lexical matching only; semantic/paraphrase overlap is not detectable.

For the selected S2/v2.1 lineage:

- The v2.1 report records `contamination_report` as **CLEAN**: no training text shared across partitions and the withheld family never appears in training.
- It does not document a published HumanEval/MBPP overlap check for the synthetic curriculum.

**Decontamination status:** **PARTIALLY VERIFIED** for split/partition isolation; **NOT RECOVERED** for published-suite contamination.

---

## 8. Remaining unknowns

1. Exact tokenizer generation input, input hash, command, and trainer version.
2. Full `runE` training command, config, dataset hash, curriculum file list, and tokenizer path/hash.
3. The selected S2 `runE` metrics do not name the dataset or tokenizer; the exact training bundle is incomplete.
4. Published HumanEval/MBPP contamination status for the selected lineage.
5. Whether the tokenizer’s exact training input included only `code_corpus_v0` or another corpus.
6. Whether copyrighted training text or memorized outputs affect distribution or release copy.
7. Owner decision on nested GPL/AGPL conflicts and any model/data license.

---

## 9. Exact owner/legal decisions still required

1. Confirm that `code_corpus_v0.jsonl` was the actual `qm_code_bpe_v0.json` tokenizer input, or recover the tokenizer manifest.
2. Recover the `runE` command, config, dataset manifest, tokenizer path/hash, and initialization record.
3. Decide whether published-benchmark decontamination is required before publication.
4. Resolve the `psf-black` and `pypa-setuptools` nested-license human-review items.
5. Select and record a model-weights license.
6. Record a training-data redistribution decision.
7. Approve the final artifact directory, hashes, model card, and Hugging Face target.

These are not resolved by this report.

---

## 10. Final distribution recommendation

**PARTIAL RECOVERY — SPECIFIC EVIDENCE STILL REQUIRED**

The recovery work established several important facts:

- The selected checkpoint is hash-verified and traceable to a frozen Taksh run.
- The architecture is original Taksh code.
- The v2.1 S2 lineage used a repository-authored synthetic operation curriculum.
- The licence-verified M3/M7 subsets contain no records from the two restricted-source repositories.
- Raw external corpora still contain the restricted subtrees, so repository presence alone was not treated as training exposure.

But exact distribution eligibility is not established because:

- the tokenizer input is not fully recorded,
- the `runE` full training bundle is missing,
- published benchmark decontamination is not documented for the selected lineage,
- nested-license human review is unresolved,
- and no model/data license is recorded.

The artifact must remain **NOT APPROVED FOR DISTRIBUTION** until the specific owner/legal evidence above is supplied.

This task did not upload, publish, license, train, or modify the checkpoint or tokenizer.
