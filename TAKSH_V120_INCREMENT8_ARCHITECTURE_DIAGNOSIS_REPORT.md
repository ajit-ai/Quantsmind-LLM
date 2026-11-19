# TAKSH v1.2.0 — Increment 8 Architecture Diagnosis Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)

Verdict: **CONFIRMED DATA/CURRICULUM LIMITATION + CONFIRMED LEARNING-OBJECTIVE LIMITATION. No implementation defect found. POSSIBLE architecture/capacity contribution. Recommended next direction: B — redesign the learning curriculum and target construction (testable criterion defined in Section 8). No training was performed in this increment.**

Machine-readable evidence: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment8\diagnostics.json` (tokenizer stats, architecture config, param hashes, alignment results, copy-probe records for all three checkpoints).

**STOP after diagnosis. No Increment 9, training, promotion, or publication.**

---

## 1. Baseline and preservation

Verified at start and end: branch `develop`, HEAD `a654e5d`, package `1.0.0`; all five checkpoint hashes (`02cc86eb...`, `89c2d61c...`, `6ff86ddc...`, `3b3777af...`, `ea809e12...`), tokenizer `90fd1c8b...`, fixtures AD-1 `a5bf9406...` / C3 `1b72f538...`, supplemental digest `05c0c657...`. Tracked diff = four pre-existing modifications + reviewed Increment 4 fix. No invariant failed. No destructive Git command, training, weight/tokenizer modification, or publication occurred.

## 2. Tokenizer and identifier diagnosis

`qm_code_bpe_v0.json`: byte-level BPE, vocab **5,822**, NFKC, round-trip **perfect** on every probed identifier (familiar and unfamiliar).

| Identifier set | Mean fragments | Max | Round-trip |
|---|---:|---:|---|
| Familiar (13: `left_value`, `customer_total`, `raw_text`, ...) | **2.92** | 6 | ✓ all |
| Unfamiliar (14: `probe_00`, `zebra_count`, `quixotic_label`, ...) | **4.43** | 7 | ✓ all |

Examples: `left_value` → 3 (`left/_/value`); `customer_total` → 6 (`cu/st/om/er/_/total`); `text` → 1. Familiar morphemes survive as units; novel identifiers fragment further. Copying therefore requires assembling 3–7 fragments per name instead of 1–3, a real burden — but tokenization is **lossless and unambiguous**, and training/inference encode identically (verified in Increment 1). The tokenizer does not lose variable relationships; it makes exact copying harder for a tiny model. **Classification: no tokenizer defect; fragmentation is a contributing burden, not the root cause.**

## 3. Architecture and capacity diagnosis

`qm-code-m3-9m` (from checkpoint config, identical across all five checkpoints, 75 tensors, param hashes reproducible across loads):

`d_model=256`, `n_layers=8`, `n_heads=8`, `n_kv_heads=4` (head dim 32), `ffn_dim=1024`, `max_seq_len=512`, RoPE θ=10000, tied embeddings (5,822×256 = 1.49M of 9.36M total), AdamW.

No implementation defect was found: forward shapes are consistent, save/load round-trips are bit-stable (`load_consistent: true`), inference and training use the same encoding path, and generation is deterministic. Copying is in principle attention-friendly and causally available (prompt tokens precede target tokens), so the architecture is not *provably* insufficient — but no capacity ablation exists (all evidence is single-architecture). **Classification: POSSIBLE ARCHITECTURE/CAPACITY LIMITATION (unproven).** An ablation (same data, larger model) would be the test; this increment does not run it.

## 4. Training objective diagnosis

Traced end-to-end (`supervised.supervise` + `continued_training.train_from_checkpoint` + `build_batches`):

- Prompt text = `instruction + signature`; target text = body + EOS; ids concatenated; labels shifted and **prompt positions masked** (`IGNORE_INDEX`), target positions supervised (tested on 12 examples: `prompt_all_masked ✓`, `target_all_supervised ✓`, `eos_in_target ✓`).
- **Every example's target contains every parameter name** (`all_params_in_target ✓` for all probed examples) — i.e. the objective *does* reward exact identifier copying at token level; teacher forcing supplies the correct fragments as input.
- Optimizer/schedule/clipping/grad-accum are the standard existing path; checkpoint save/load consistent (Section 3).

So the objective is correctly implemented but **does not distinguish copying from templating**: token loss weights identifier fragments the same as syntax fragments, and with small/diverse-poor data the highest-probability (template) completion wins. Validation loss demonstrably fell (Increments 1/3/5/7) while binding stayed ~0 on novel names — a direct demonstration that target loss can improve while binding does not. **Classification: CONFIRMED LEARNING-OBJECTIVE LIMITATION (by omission, not by bug).**

## 5. Controlled learning diagnostics (inference only, no training)

13 synthetic copy/transform probes (novel + familiar identifiers, add/first/upper/sub/mul), clearly separated from frozen suites (overlap-tested), greedy decoding, scored by `exact_copy` (uses exactly the requested params, invents none):

| Checkpoint | Exact copy (of 13) | Novel (of 7) | Familiar (of 6) | Parses |
|---|---:|---:|---:|---:|
| Baseline | **0/13** | 0 | 0 | 3 |
| Increment 5 | **0/13** | 0 | 0 | 3 |
| Increment 7 | **0/13** | 0 | 0 | 3 |

Baseline outputs `(left/right)` templates; candidates output `(text...)` templates. Even **familiar** identifiers fail exact copy on these probes — the model cannot perform the elementary copy operation the entire curriculum presumes, before generalization is even reached. Combined with Increment 6 evidence (supplemental 0/48 both sides; Increment 7 targeted novel pools still 0/48), the failure order is:

1. Cannot copy trained-name → output relationship reliably (this diagnostic).
2. Therefore cannot generalize to novel names (supplemental 0/48).
3. Therefore frozen binding passes are template collisions, not learning (binding without function, Increments 3/5/7).

This distinguishes the four hypotheses: **not** a tokenizer defect (A refuted: lossless), **not** an objective implementation bug (B refuted as bug; masking/alignment correct), **data/curriculum + objective-limitation confirmed (D)**, architecture/capacity possible but unseparated (C inconclusive).

## 6. Comparison of experimental evidence (baseline / Inc5 / Inc7)

| Metric | Baseline | Inc5 | Inc7 |
|---|---:|---:|---:|
| AD-1 parse / binding / exact | 24 / 6 / 2 | 24 / 2 / 1 | 26 / 2 / 1 |
| C3 parse / binding / functional | 24 / 0 / 0 | 37 / 9 / 1 | 42 / 9 / 1 |
| Supplemental functional | 0/48 | 0/48 | 0/48 |
| Copy probes exact (this increment) | 0/13 | 0/13 | 0/13 |
| Collapse (candidate) | — | no | no |

Parseability improved across increments because syntax/topology templates are easy to fit; functional correctness did not because no run taught the prompt-conditioned identifier copy with sufficient, varied supervision to overcome template modes. The single recurring functional pass (`text.upper()`) is a memorized template, and the recurring AD-1 binding loss (4) shows the same tokens shifting between `left/right`, `text`, and comprehension attractors.

## 7. Diagnostic tests

`tests/test_diagnostics.py` **6 passed**: tokenizer round-trip/fragmentation, alignment/masking (prompt masked, target supervised, params in target, EOS present), checkpoint consistency + shared architecture, copy-helper correctness, frozen-suite non-overlap, entry-point presence. Regression: all Increment 1–7 focused suites **111 passed**. Diagnostic entry: `scripts/diagnose_v120.py` (loads checkpoints read-only, writes only the JSON report). Helpers: `evaluation/diagnostics.py`.

Two defects were caught during development (pre-final): prompt-inference indent handling in the probe scorer (fixed before results) and a variable shadowing in the diagnostic script (fixed). Neither affects frozen code or results.

## 8. Final engineering decision and next direction

- **CONFIRMED IMPLEMENTATION DEFECT: none found** (tokenizer lossless; masking correct; load consistent; train/inference encodings agree).
- **CONFIRMED LEARNING-OBJECTIVE LIMITATION:** token-level loss rewards copying and templating identically; no mechanism distinguishes the relationship, and validation loss is proven insensitive to binding.
- **CONFIRMED DATA/CURRICULUM LIMITATION:** template-dominated small data; novel-name generalization fails on every checkpoint (0/48, 0/13), targeted pools did not fix it in one 100-step run.
- **POSSIBLE ARCHITECTURE/CAPACITY LIMITATION:** 9.36M/d256 tied-embedding model plausibly strains under 4–7-fragment copying, but unproven without ablation.
- **INCONCLUSIVE** only on capacity vs curriculum proportions.

**Recommended next direction: B — Redesign the learning curriculum and target construction.** Testable criterion (define before any Increment 9 training): a candidate must score **≥10/13 exact copy** on these probes (baseline 0/13) *and* ≥3/48 supplemental functional passes across ≥3 behaviours, with AD-1 binding loss ≤1, before any functional claim is credited. Concretely: contrastive same-operation name-swap pairs, explicit copy-span supervision, and an early-stopping copy-probe gate — rather than more same-recipe steps or a premature capacity run. Direction C (capacity ablation) becomes warranted only if a redesigned objective still shows 0 copy at adequate data scale; D (retire the architecture) is not supported by current evidence.

**STOP after diagnosis. No Increment 9, training, promotion, or publication was performed. Awaiting owner review.**
