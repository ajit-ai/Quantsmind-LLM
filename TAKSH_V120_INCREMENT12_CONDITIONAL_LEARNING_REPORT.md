# TAKSH v1.2.0 — Increment 12 Conditional Learning Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 10: `e68495931da7bf4d0dde666615f16757a87a01304c0a2358ac4fa05bda634c9b` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Machine-readable: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment12\conditional_diagnosis.json`

Verdict: **PRIMARY FAILURE IN TARGET PREDICTION, NOT DECODING OR PROMPT REPRESENTATION.**

Given the full correct preceding context, the model still assigns the correct identifier first-token probability ~0.0002–0.003 (rank 33–5,625) and gets **zero identifier tokens correct on every side** — for familiar and unfamiliar names alike, baseline and Increment 10 alike. Prompt swaps move distributions slightly (TV 0.004–0.19) but in the wrong direction: template tokens dominate. Autoregressive generation agrees with teacher forcing (never contains the identifier), so this is not an error-accumulation or decoding artifact.

**STOP after diagnosis. No Increment 13 or training.**

---

## 1. Baseline Git status

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff = 4 pre-existing + Inc4 one-line fix + Inc10 integration (two files). All hashes re-verified after all work (Section 9). No invariant failed.

## 2. Prompt pairs (controlled)

Five paired prompts differing only in identifier names (full texts in `evaluation/conditional.py: PAIRED_CASES`):

| Pair | Operation | A (familiar) | B (unfamiliar) |
|---|---|---|---|
| P1-add | add | `left_value, right_value` | `source_amount, target_amount` |
| P2-sub | subtract | `left_value, right_value` | `source_amount, target_amount` |
| P3-first | first | `alpha_value, beta_value` | `probe_value, segment_value` |
| P4-upper | upper | `raw_text` | `message_body` |
| P5-mul | multiply | `left_value, right_value` | `customer_total, invoice_total` |

Target prefix (`    return `) and instruction identical within each pair. Tokenizer round-trip lossless (Inc8). Model: GQA causal SDPA (no attention weights exposed) — logit/distribution comparisons used; hidden-state internals were not separately recorded (limitation below).

## 3. Teacher-forced accuracy

Correct **first** identifier token at the return position (teacher-forced prefix):

| Pair | Baseline A prob / rank | Baseline B prob / rank | Inc10 A prob / rank | Inc10 B prob / rank |
|---|---|---|---|---|---|
| P1-add | 0.0003 / 35 | 0.0000 / 4662 | 0.0020 / 38 | 0.0001 / 841 |
| P2-sub | 0.0002 / 33 | 0.0000 / 4910 | 0.0028 / 35 | 0.0001 / 589 |
| P3-first | 0.0000 / 4078 | 0.0000 / 1309 | 0.0001 / 5625 | 0.0001 / 5310 |
| P4-upper | 0.0000 / 967 | 0.0000 / 1254 | 0.0000 / 3112 | 0.0001 / 1306 |
| P5-mul | 0.0002 / 33 | 0.0000 / 2008 | 0.0027 / 34 | 0.0001 / 1483 |

Teacher-forced continuation across the whole identifier (3–6 tokens): **0 correct tokens on every side** (both checkpoints, familiar and unfamiliar). Per-token cross-entropies 4.0–12.8 (familiar ~4–8, unfamiliar ~8–13). Even the best case (`left_value` first token) is ranked 33rd — 32 other tokens are more likely than the identifier the prompt literally requests.

## 4. Prompt sensitivity

Total-variation distance between A/B distributions: 0.004–0.19. The prompt identifier *does* move the distribution (weakly — attention carries some information), but the top-token rarely is and never tracks the expected identifier (e.g. P5: A-side top mass sits on `left`-template tokens; B-side correct token ranked >1000th). Sensitivity without correct direction: the model notices identifiers but its output head has learned template priors that overwhelm them.

## 5. Attention / activation evidence

Model inspected: `training/model.py` uses `F.scaled_dot_product_attention` with `is_causal=True` (or explicit mask) — **no attention probabilities are returned**, so layer-wise attention to prompt identifier positions cannot be read off directly. Reimplementing attention manually to extract weights would duplicate model math and risk misinterpretation; per the brief's fallback, controlled logit/activation comparisons were used instead: prompt swaps produce different logits (Section 4) at the exact position where the identifier is scored, proving causal flow from prompt to logits through the network, but with the wrong target direction. Hidden-state internals were computed in an exploratory pass but are not included in the machine-readable record (limitation acknowledged).

## 6. Autoregressive vs teacher-forced

Greedy generation (32 tokens, same prompts) **never contains the requested identifier** on either side (10/10 baseline, 10/10 Increment 10). Teacher forcing shows the correct sequence is also never produced. The two agree: failure is upstream of decoding (target prediction head + learned priors), not error accumulation — supply the correct context and the model still prefers template continuations.

## 7. Identifier-swap results

Swap tests are the paired cases themselves: swapping `left_value→source_amount` changes the expected target but neither checkpoint's output follows (Section 3–4). Two-identifier swaps (P1/P2/P5) and token-length variation (3-token familiar vs 5–6-token unfamiliar) both fail; longer unfamiliar identifiers score worst (ranks up to 5,625), consistent with the fragmentation burden from Increment 8 but now localized to the prediction head: longer identifiers need more low-probability fragment decisions in sequence.

## 8. Tests and verification

`tests/test_conditional_diagnosis.py`: paired-case construction (swap harness), probability/rank/CE math, divergence extremes, swap round-trip — **4 passed**. Audited-path regression: learning-signal (4), diagnostics (6), trainer-integration (7) — **17 passed**. Inference-only on original S2 + Increment 10 (4 CPUs threads, no weights written). Runtime: full paired probe ~minutes; memory observed via psutil during selection (~0.5–0.9 GiB range across runs; model loads are sequential and freed).

## 9. Hashes at the end

Baseline `02cc86eb...`, Increment 10 `e6849593...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...`, supplemental `05c0c657...` — all verified. HEAD `a654e5d`; no commits/pushes/tags/releases or fixture changes.

## 10. Classification of the primary failure mechanism

- **Prompt representation: NOT the failure** (lossless tokenizer; swaps move logits).
- **Decoding: NOT the failure** (TF and AR agree).
- **Target prediction head + learned template priors: PRIMARY FAILURE.** Given correct context, the head ranks template tokens above the requested identifier; identifiers carry <0.3% probability mass.
- **Architecture/capacity: contributing, still unproven.** The head (tied embeddings, d256) must realize a sharp prompt-conditioned distribution over 5,822 tokens from 256-dim states; that it cannot even do this under teacher forcing keeps capacity on the table, but the template-prior explanation requires no capacity claim: the same behavior appeared across four training recipes.
- **Supervision/labels: NOT the failure** (Increment 11 proved correct).

## 11. Recommended next technical decision

Do not run another continued-training round at these shapes — four candidates now share the identical 0-copy signature. Before any Increment 13, the owner should choose among:

1. **Copy-mechanism training** (pointer/auxiliary objective that forces prompt→output identifier alignment, with the ≥10/13 copy gate as the first criterion),
2. **Capacity ablation** (same protocol, larger model — the only experiment that can separately confirm or clear D),
3. **Objective/decoding redesign** (e.g. constrained/copy-aware decoding) — evaluated against the same frozen gates.

**STOP after diagnosis. No Increment 13 or training. Awaiting owner review.**
