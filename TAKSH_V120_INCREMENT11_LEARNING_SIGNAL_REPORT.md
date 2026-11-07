# TAKSH v1.2.0 — Increment 11 Learning Signal Report: Minimal Copying Experiment

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 10: `e68495931da7bf4d0dde666615f16757a87a01304c0a2358ac4fa05bda634c9b` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Tiny candidate: `3f732c0c61c4151e50d425e4a28c6f189246c17e7897b8b3c020980692489de3` (isolated temp dir, unfinalized — no manifest was written; checkpoint + metrics only)

Verdict: **SUPERVISION CONFIRMED CORRECT; MODEL STILL CANNOT COPY — CLASSIFICATION: B (optimization/learning-signal defect), with D (capacity/representation) still possible. C refuted, A refuted.**

The audit proves the intended identifier positions receive correct labels, weights, and gradients. Yet a 30-step memorization run over 8 copy examples drops loss 5.41→1.72 while producing **zero exact copies even on the trained prompts**. Correct signal, no learning. **STOP. No Increment 12 or further training.**

---

## 1. Git baseline

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: four pre-existing + `training/program_analysis.py` (Inc4) + `training/supervised_train.py` & `training/continued_training.py` (Inc10 integration). All hashes re-verified at start and end (checkpoints, tokenizer, AD-1/C3, supplemental `05c0c657...`). No commit/push/merge/tag/version change.

## 2. Prediction-position trace (WP1)

Trace for `def add_values(left_value, right_value)` / `total = left_value + right_value; return total` (41 token positions, saved to `increment11/trace.json`):

- `prompt_len=24`, `target_len=17`, `eos_position=40` (last id). Labels = ids shifted by exactly one at every supervised position (test-asserted); masked labels are exactly indices `0..prompt_len-2`.
- EOS present in the supervised tail.
- Occurrences: `total` substituted(at store C1)+propagated, `left_value`/`right_value` copied (×2 each incl. store/load).
- Weights: mean-normalization makes syntax ≈0.708 and copy tokens up to **1.417** (ratio 2:1 as configured), 8 weighted tokens. No truncation, no off-by-one, no masking loss. **No silent removal of the copying signal exists.**

## 3. Gradient verification (WP2)

Two-example controlled batch through the real checkpoint (1 thread, CPU):

| Objective | Loss | Gradient Σ(diff²) ex.1 | ex.2 |
|---|---:|---:|---:|
| Unweighted | 5.0147 | 462.5 | 1672.1 |
| Identifier-weighted | 5.1620 | 446.7 | 1941.7 |

All gradients finite and non-zero; weighting changes gradient composition (example 2 +16%) without breaking finiteness or inflating scale (mean 1.0). No dilution beyond the intended reweighting, no cancellation anomalies, no missing contribution. **Labels and gradients are correct.**

## 4. Tiny memorization experiment (WP3)

Configuration: original S2 start (temp copy semantics — trainer reads baseline, writes only to `increment11/tiny-copy-30`), **8 examples** (2 familiar add, 2 familiar copy, 2 unfamiliar copy, 1 familiar subtract, 1 propagation), `lr=3e-4`, seed `20261111`, max length 128, identifier weights enabled, **30 steps**, CPU. Pre-run RAM 0.227 GiB at process start (model load OK); no pressure stop needed; runtime ~11 s training + seconds per evaluation load.

Results: steps 30/30, initial loss **5.4141** → final **1.7249** (validation 1.7137, same examples), tokens 589, `finite_loss` throughout.

Exact predictions (greedy, training prompts):

| Prompt | Baseline output | Tiny output | Exact copy |
|---|---|---|---|
| add (familiar) | `return left + return left + right` ✗ | `return left_value` ✗ | ✗ |
| add2 (familiar) | `return left left left` ✗ | `return left_value` ✗ | ✗ |
| copy (familiar `value`) | `return left - right` ✗ | `return value` ✓ | **✓** |
| copy (familiar `text`) | `return left - right` ✗ | `return value` ✗ | ✗ |
| copy (novel `probe_00`) | `return left left left - right` ✗ | `return value` ✗ | ✗ |
| copy (novel `segment_00`) | (same degenerate) ✗ | `return value` ✗ | ✗ |
| subtract (familiar) | `return left - right` ✗ | `return left_value - right_value - ...` | ~ (binds, semantically wrong) |
| propagation | `return left - right` ✗ | `return value` ✗ | ✗ |

The tiny model memorized the frequent surface fragments (`value`, `left_value`) and emits them regardless of the prompt. It overfits common tokens; per-example copying is not learned — and training-prompts exact copy is effectively **0/8 semantically correct** (the binding-only scorer credits `FAM-copy` and `FAM-sub`, but `FAM-sub` returns `left - right - right - right`).

## 5. Controlled interpretation (WP4)

- **A. Supervision implementation defect — REFUTED.** Labels, positions, weights, EOS, and gradients are verified correct (Sections 2–3, plus `tests/test_learning_signal.py`).
- **B. Optimization/learning-signal defect — CONFIRMED.** Correct signal exists, yet 30 steps over 8 examples × ~10 exposures each fails to produce per-example copying while training loss falls 68%. Token loss is dominated by high-frequency syntax/template fragments; the copy tokens are a minority even after 2× reweighting, and their gradient does not overcome template modes. The model learns token frequencies, not the prompt-conditioned relation.
- **C. Minimal memorization succeeds but doesn't generalize — REFUTED.** Memorization itself fails (0 semantically exact copies on trained prompts).
- **D. Capacity/representation — NOT EXCLUDED, not established.** A d256/8-layer/1.49M-embedding model may lack the precision to bind prompt fragments, but this experiment cannot separate that from B: both predict exactly what was observed. Declaring the architecture inadequate would require a capacity ablation, which is out of scope and was not run.

Multiple findings coexist: **B primary, D possible.**

## 6. Tests and verification

```powershell
python -m pytest tests\test_learning_signal.py -q -p no:cacheprovider
# 4 passed in 4.29s

python -m pytest tests\test_diagnostics.py tests\test_functional_eval.py tests\test_trainer_integration.py -q -p no:cacheprovider
# 28 passed
```

Coverage: label-shift alignment (every supervised label = next id, masked prompt prefix, EOS last), weighted-position containment, disabled-weights equivalence, finite/non-zero/different gradients, tiny-artifact loss/sha verification. Two test defects found and fixed pre-final: masked-label equality assertion and manifest-vs-checkpoint hash location (the tiny run was never finalized — no manifest exists; hash verified directly).

## 7. Hashes at the end

Baseline `02cc86eb...`, Increment 10 `e6849593...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...`, supplemental `05c0c657...`, tiny `3f732c0c...` — all verified. Available RAM at final check 0.544 GiB.

## 8. Recommended next technical decision

Do not run another continued-training round at these settings. Options for owner review:

1. **Learning-signal redesign (preferred):** supervise copying as a distinct phase (e.g. contrastive name-swap pairs: same prompt, swapped identifiers, forcing attention to prompt tokens), or a pointer/auxiliary copy objective, with the ≥10/13 copy gate as the *first* criterion.
2. **Capacity ablation (only if 1 is tried without movement):** same data/protocol at a larger shape to separate B from D — a controlled experiment, not a promotion path.
3. **Stop the continued-training family:** four candidates (Inc3/5/7/10) show the same signature (parse up, copy 0, AD-1 loss 4). Further runs without a mechanism change are not informative.

**STOP after this diagnosis and tiny experiment. No Increment 12, training, promotion, or publication. Awaiting owner review.**
