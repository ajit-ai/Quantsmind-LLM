# TAKSH — I4 Capability Recovery (Aligned-format SFT on v180 checkpoint)

**Date:** 2026-11-19
**Branch context:** `feature/coding-genius-i4-capability`
**Authorized by:** owner decision (bounded CPU experiments + local corpora only; 2026-11-19)
**Conclusion:** **no demonstrated capability gain** — Native v0.1 remains 0/20 across all three evaluated states.

---

## 1. Objective

Determine whether a small, locally authored, full-function SFT run can move the immutable native v0.1 benchmark off zero — evidence for the product claim before any scale-up.

## 2. Scope and non-goals

- In scope: fine-tuning the existing 28.8M-parameter `v180_sft` checkpoint on the local CC0 `l02_sft` dataset, both partial- and full-coverage passes; direct re-evaluation on immutable `evaluation/native_v01` tasks.
- Out of scope: benchmark mutation, checkpoint reuse outside local `runs/`, external data, publication, capability claims, model re-evaluation for release, GPU-scale training, architecture changes.
- The frozen Python benchmark, tokenizer v1, and the immutable `native_v01` suite are unchanged.

## 3. Baseline (pre-I4)

Immutable v0.1 re-run on the untouched `runs/v180_sft/checkpoint_last.pt`:

- 0/20 syntax-valid, 0/20 executable, 0/20 correct, 0/20 instruction-followed
- 16/20 empty, 20/20 degenerate, 0/20 inference failures
- load: step=200, params=28,787,328; greedy decoding, max_new_tokens=96

## 4. Configurations evaluated

| State | Config | Steps | LR | Data | Coverage |
|---|---|---|---|---:|---|
| Baseline (v180_sft) | — | — | — | — | — |
| I4-partial | `configs/train/sft_i4_aligned.yaml` | 100 | 2.0e-5 cosine, 10 warmup | `data/l02_sft/sft.jsonl` | 76.9 % (200/260) |
| I4-full | same config | 130 | same | same | 100 % (260/260) |

Loss log summary (tail): final step 129, loss 4.861, EMA 4.899, grad_norm 2.944, coverage 260/260.

Checkpoints: `runs/i4_aligned_sft/checkpoint_last.pt`, `runs/i4_aligned_sft_full/checkpoint_last.pt`. Kept local; not committed.

## 5. Results — immutable v0.1

| Metric | I4-partial (step 100) | I4-full (step 130) |
|---|---:|---:|
| Syntax-valid | 0/20 | 0/20 |
| Executable | 0/20 | 0/20 |
| Correct | 0/20 | 0/20 |
| Instruction-following | 0/20 | 0/20 |
| Empty output | 16/20 | 14/20 |
| Degenerate output | 20/20 | 20/20 |
| Inference failure | 0/20 | 0/20 |

## 6. Observations

1. Aligned-format SFT did not produce any syntactically-valid code output on native v0.1. Improvement is limited to fewer empty outputs (16→14) with no syntax/correctness effect.
2. Degeneracy persists at 100%; raw generation remains token-fragment-heavy ("empty" fragments are non-code).
3. Loss decreases smoothly toward low-magnitude but does not produce behavioural change on the task signal; token loss is not a sufficient signal for code correctness here.
4. **No gain should be inferred from fewer empty outputs.**
5. This run is a single seed, single budget, single data variant — inconclusive for direction, negative for any capability claim.

## 7. Interpretation

Aligned-format exposure alone on 260 CC0 examples at CPU-scale fine-tuning from a 28.8M checkpoint does not move functional coding ability from zero. The aligned-instructional hypothesis remains unvalidated. Tasks in native v0.1 require generation of function bodies; training signal is insufficient both in diversity and in quantity to reach even the syntax gate.

Primary questions for the next step (owner-authorized experiment, not executed):

- Exact match between training task phrasing/output surface and the `native_v01` task phrasing is very tight (sum_evens, bounded ints) — generalisation gap.
- Capability may be limited by model scale, not just alignment.
- Aligned warmstart with higher-quality/complete-function data and/or code-dense pretraining would need explicit owner sign-off on scale (and ideally GPU).

## 8. Explicit admissions

- Empty-rate improvement (16→14) is noted strictly as an observational metric, **not** a capability signal, and receives no weight in qualification.
- No new benchmark, no frozen suite edit, no instruction-tuning claim — all non-goals.
- This experiment does not enable, authorize, or support release, publication, or any capability statement. Default answer on any interaction is: **capability NOT_EVALUATED; native v0.1 0/20.**

## 9. Recommended next steps

1. Do not pursue product integration on this evidence.
2. If progress is sought, execute one of (owner decision):
   a. **Scale experiment**: 1B-parameter dense model + l04 code-corpus "code-dense" warmstart, CPU infeasible; requires compute decision.
   b. **Targeted synthetic data synthesis** (owner-gated, no external dataset): generate thousands of synthetic full-function instruction->code pairs locally to test whether data alone at scale can move the syntax gate.
   c. **Architecture/tokenizer**: v2 with control tokens + aligned v1/v2 renderer; evaluate.
3. Keep the frozen benchmark at 0/30 and the native baseline at 0/20 visible in status documents.
