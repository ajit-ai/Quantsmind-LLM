# Taksh - Execution Status Tracker

> **TAKSH roadmap status (2026-11-11).** v1.0.0 GA is **NOT established**. Qualified on
> `feature/l060-foundation-model-qualification` only: L0.6.x → L0.8.0-PRE (preparation),
> L1.0.0-ENG / ENG2 / ENG3 (engineering qualification; no capability claim). The canonical
> 111M artifact is MISSING; the frozen Python benchmark remains 0/30; Karkain remains
> DISCOVERED (KARK-BENCH 0 tasks, KCC `0xC0000135`); `v180_sft` is never substituted;
> A1.1-C / A2-E are owner-enforced; develop/main untouched. Live authority lives on
> `feature/l060-foundation-model-qualification` (milestone reports and
> `configs/roadmap/taksh-v3-roadmap.yaml`); this branch currently carries the CI/CD
> recovery port only. The M7-era banners below are the historical record, preserved
> verbatim.

> **M7 status (2026-10-21).** The M7 Sovereign Capability Experiment completed with a
> negative capability result: 0/40 locked syntax-valid generations, 0/8 functional
> probes, evidence level 0. The final packaging target is **v1.0.0** (Apache-2.0
> source release). The Git tag and GitHub release are published; the PyPI upload
> is blocked on registry credentials. See
> `docs/source/releases/M7_RELEASE_READINESS.rst` and
> `docs/source/roadmap/M7_EXPERIMENT_REPORT.rst`.

> **Roadmap authority (2026-10-18).** The single authoritative execution roadmap is
> [ROADMAP.md](ROADMAP.md). M2 acceptance governance and the four validation layers
> live in `docs/source/roadmap/M2_EXECUTION.rst`. This document tracks *status*
> only; where it disagrees with ROADMAP.md on milestone meaning, ROADMAP.md governs.
> Multimodal is **deferred**, not deleted.
>
> **Correction notice (2026-10-14).** Checkmarks below mean *code exists*, not *capability
> demonstrated*. Corrected values: `code_30m.yaml` is **16,257,280** params, not ~30M.
> The code tokenizer realises **5,822** vocab. The only code pretrain ran **200 steps**
> at `seq_len` 512. HumanEval/MBPP have never been run. The SFT corpus is **quarantined**.
> See [VERSIONING.md](VERSIONING.md) and [OPEN_ISSUES.md](OPEN_ISSUES.md).

> **Direction update (2026-09-27):** project pivoted to a **coding LLM** for corporate use.
> Python + JavaScript/TypeScript specialist, 15–100M params *(target)*, all use cases (completion, chat, review).
> See [CODE_LLM_STRATEGY](CODE_LLM_STRATEGY.md) for full strategy.

Living document. Update at every phase end; phase completion requires develop→main merge.
Legend: [x] done · [~] partially done / verify pending · [ ] not started

---

## CODE LLM PIVOT — Implementation Status

### Step 1: Code Tokenizer — ✅ CODE COMPLETE (realised vocab 5,822)

- [x] Code tokenizer config (`configs/tokenizer/code_bpe.yaml`)
- [ ] Code-specific special tokens (FIM, repo, file, language) — **NOT DONE.** Only `<pad> <bos> <eos> <unk>` exist. `<|system|>`, `<|assistant|>` and the FIM markers used by `training/sft.py:26` and `completion_server.py` are absent from the vocabulary
- [x] Tests (`tests/test_code_tokenizer.py`)

### Step 2: Code Model Config — ✅ CODE COMPLETE (16.3M, not 30M)

- [x] `configs/model/code_30m.yaml` — d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, seq_len=2048
- [x] **16,257,280 params**, not ~30M. The filename overstates; it was left unchanged to preserve run reproducibility

### Step 3: Code Corpus Pipeline — ⚠️ CODE COMPLETE, CORPUS NOT REAL

- [x] `data/build_code_corpus.py` — GitHub API download, license filter, language detect, quality filter, dedup, chunk
- [x] Supports Python, JavaScript, TypeScript
- [x] Permissive license filter (MIT, Apache-2.0, BSD, ISC)
- [x] Syntax validation for Python
- [ ] **The shipped corpus is not GitHub code.** `data/code_corpus_v0.jsonl` holds 77 chunks / 100,244 tokens, every one `source: local:*` — this repository's own files, produced by `scripts/scale_corpus.py`. 68 python, 9 javascript, **0 typescript**. `scripts/build_code_corpus.py` has never successfully populated it

### Step 4: Code Pretraining Config — ✅ CONFIG ONLY

- [x] `configs/train/code_local.yaml` — seq_len=2048, max_steps=10000, 65k tokens/step
- [ ] **Never executed at this setting.** The only code pretrain is `runs/code_pretrain`, **step 200**, `seq_len` 512, ≈1 epoch, from the 50-step smoke config with `--max-steps 200`

### Step 5: SFT Training Loop — ⚠️ CODE COMPLETE, DATA QUARANTINED

- [x] `training/sft.py` — instruction-response training with loss masking
- [ ] `data/build_code_sft.py` — generator emits boilerplate. **423 of 442 pairs are content-free templates**; the 19 real pairs come from a faulty regex extractor; refactoring responses return the original code while claiming edits were made. Corpus quarantined: `data/code_sft_v0.quarantine.json`
- [ ] SFT batching bug: only pairs `0..micro_bs-1` are ever sampled (Open Issue B1); `--init-ckpt` uses `strict=False`, which hid an architecture mismatch (B3)
- [x] Tests (`tests/test_sft.py`)

### Step 6: Code Evaluation — ⚠️ HARNESS ONLY, NOTHING MEASURED

- [x] `evaluation/eval_code.py` — runner + execution harness
- [ ] **HumanEval: UNMEASURED.** Table is a 5-problem subset of 164; reports stamp `status: "unmeasured"`, `promotion_eligible: false`
- [ ] **MBPP: UNMEASURED.** Table is a 5-problem subset of 500
- [ ] **Code perplexity: NOT MEASURABLE.** 1% tail of a 100,244-token corpus is 1,003 tokens, fewer than the 2,049 needed for one `seq_len=2048` window. Returns `perplexity: null, ok: false`
- [x] `evaluation/promotion.yaml` — declarative only; **no code enforces `rules:`** (Open Issue D4)
- [x] Tests (`tests/test_eval_code.py`) — assert table structure only; the runner is never called
- [x] Empty-split and unavailable-benchmark guards + regression tests

### Step 7: Code Serving — ⚠️ RUNS, NOT DEPLOYABLE

- [x] `serving/code_chat.py`, `serving/completion_server.py`, `serving/review_server.py` load real checkpoints and sample for real
- [ ] **Do not expose on an untrusted network.** `torch.load(weights_only=False)` on a request-supplied path, unauthenticated, bound to `0.0.0.0` (Open Issue A1)
- [ ] Streaming is character-at-a-time replay of a finished string (C1); EOS stop never fires (C2); `top_k` is used as a completion count (A6)

### Step 8: Corporate Fine-Tune Pipeline — ⚠️ CODE COMPLETE, NO REAL DATA

- [x] `data/ingest_internal.py` — internal repo ingestion with 4 secret regexes (Open Issue A7)
- [x] `configs/train/corporate_ft.yaml` — **its `dataset_tag: corporate-internal-v0` resolves to a file that does not exist**; `--tokens data/internal_corpus_v0.tokens.bin` is required
- [ ] Shipped internal corpus is **2,126 tokens from 5 synthetic demo files**. `runs/corporate_*` are 495,808-param models trained 3,088 epochs over those 5 files
- [x] Tests (`tests/test_internal_ingest.py`)

### Step 9: Documentation — ✅ REPAIRED 2026-10-14

- [x] `README.md` — verified-status section and correction log added
- [x] `docs/VERSIONING.md` — claims reconciled to artifacts
- [x] `docs/STATUS.md` — this file
- [x] `docs/OPEN_ISSUES.md` — created; records known unfixed defects

### Step 10: Dependencies — ✅ CODE COMPLETE

- [x] `requirements.txt` — includes `openai>=1.40`, which sits awkwardly against the "no external APIs" position in `docs/adr/0002`

---

## Parked: Multimodal (No Active Development)

| Component | Status |
|---|---|
| `training/vision.py` | Parked — maintain as-is |
| `training/multimodal.py` | Parked — maintain as-is |
| `training/train_mm.py` | Parked — maintain as-is |
| `evaluation/eval_caption.py` | Parked — maintain as-is |
| `evaluation/eval_vqa.py` | Parked — maintain as-is |
| `evaluation/india_finance/` | Parked — maintain as-is |

---

## Milestone Board

| Milestone | Target | Status |
|---|---|---|
| Code tokenizer + corpus pipeline code | Week 1 | ✅ CODE COMPLETE |
| Code model config + pretrain config | Week 1 | ✅ CODE COMPLETE |
| SFT training loop | Week 2 | ✅ CODE COMPLETE (data quarantined) |
| Code evaluation harness | Week 2 | ✅ HARNESS ONLY — nothing measured |
| Code serving code | Week 3 | ⚠️ RUNS, not deployable (A1) |
| Corporate fine-tune pipeline | Week 3 | ⚠️ CODE COMPLETE, 5 demo files |
| Documentation + tests | Week 3 | ✅ 183 tests, docs corrected |
| **Real code corpus (GitHub)** | — | 🟡 PIPELINE DONE, SCALE SHORT — 2,277,473 tokens / 447 records, 9 licence-verified repos, provenance + manifest. Target ≥100M |
| **Genuine SFT corpus** | — | 🔴 **THE BLOCKER** — 423/442 pairs quarantined |
| **First real benchmark measurement** | — | 🔴 BLOCKED on both of the above |
| Corporate deployment | Week 6 | 🔲 PENDING |

---

## Next Actions

Ordered by dependency. Items 1–3 are prerequisites for any meaningful measurement.

1. **Grow the code corpus to scale.** `python -m data.build_code_corpus_v1 --include-held-back --max-repos 9` currently yields 2,277,473 tokens. Reaching ≥100M needs many more approved sources in `configs/data/approved_code_sources.yaml` — the allowlist cannot be widened without explicit approval, and no licence requirement may be relaxed to hit the number.
2. **Rewrite `data/build_code_sft.py`** using Python's `ast` instead of regex, and delete the three boilerplate generators. Remove `data/code_sft_v0.quarantine.json` only after regenerating and re-verifying.
3. **Fix SFT batching** (`training/sft.py:185`) and add the missing `import random`, then make `--init-ckpt` load with `strict=True`.
4. **Run tests:** `python -m pytest tests -q` (183 tests collected)
5. **Train tokenizer:** `python -m tokenizer.train_tokenizer --input data/code_corpus_v0.jsonl --out tokenizer/qm_code_bpe_v0.json --vocab-size 32768` — realised vocab depends on corpus size; 32k needs a far larger corpus than the current one
6. **Pretrain:** `python -m training.pretrain --config configs/train/code_local.yaml --out-dir runs/code_30m`
7. **Generate SFT data:** `python -m data.build_code_sft --corpus data/code_corpus_v0.jsonl` *(after step 2)*
8. **SFT:** `python -m training.sft --config configs/train/code_local.yaml --sft-data data/code_sft_v0.jsonl --init-ckpt runs/code_30m/checkpoint_last.pt` *(after steps 2 and 3; refuses quarantined data)*
9. **Evaluate:** `python -m evaluation.eval_code --ckpt runs/code_sft/checkpoint_last.pt` — perplexity only becomes meaningful once the held-out tail exceeds one window
10. **Serve:** `python -m serving.code_chat --ckpt runs/code_sft/checkpoint_last.pt` *(localhost only, until A1 is fixed)*
