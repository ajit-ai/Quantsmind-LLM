# TAKSH v1.2.0 — Increment 7 Execution Report: Controlled Targeted Training

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Owner authorization: **GRANTED** (this execution only).
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Experiment identity: `88cf0126da0306999788ac95d21950fed25940bdebbb29c12014d64212578020`
Candidate checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7\increment7-targeted-100\checkpoints\trained.pt`
Candidate SHA-256: `ea809e12b148f7650f48057d63273b78012fcb13c38c5eab2b603bdf148c6cc4`

Verdict: **PROMOTION REJECTED (promote: false). Candidate preserved temp-only; no promotion, publication, or further training.**

Targeted novel-identifier training raised parse validity on frozen suites and repeated the familiar single functional pass, but produced **zero** functional success on novel-identifier supplemental cases and repeated the AD-1 binding loss of 4 cases. The predeclared generalization gate correctly rejects it.

**STOP for owner review.**

---

## 1. Preflight and memory recheck

- Branch `develop`, HEAD `a654e5d`, package `1.0.0` ✓
- Checkpoint hashes: baseline `02cc86eb...837f8ac`, Increment 1 `89c2d61c...b80`, Increment 3 `6ff86ddc...dfc6`, Increment 5 `3b3777af...5ad` ✓
- Digests: binding `bc533a18...07b7b`, targeted `86230100...cf67`, supplemental cases `05c0c657...` ✓
- Fixtures: AD-1 `a5bf9406...5e` (29), C3 `1b72f538...26e` (54) ✓
- Available RAM immediately before training: **0.924 GiB** (total 3.911 GiB) ≥ 0.50 GiB ✓; in-run preflight gate also passed at 0.76 GiB
- Exact command verified against `configs/v120_increment7.json` (identity `88cf0126...`): root `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7`, run ID `increment7-targeted-100`

## 2. Exact executed command

```powershell
python scripts/run_v120_increment7.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7 --run-id increment7-targeted-100
```

Configuration unchanged: `qm-code-m3-9m`, baseline-only start, combined 284/84 (binding 156/52 + targeted 128/32), **100 steps, lr `1e-05`, seed `20261108`, CPU**. No hyperparameters altered; no second run.

## 3. Training metrics

From `metrics/training.json` / `manifest.json`:

- Steps **100/100**, all `finite_loss: true`
- Initial loss **6.4443** → final **6.2188**; validation loss **5.0975**
- Tokens consumed **4,246** (42.46/example); elapsed **37.52 s** training (43.18 total)
- No non-finite loss, memory-pressure stop, interruption, or hash mismatch. Post-evaluation available RAM 0.906 GiB.
- Loss reported for record only, not used as capability evidence (heavily fluctuating 2.08–6.93, validation 5.10: the model barely moved at `1e-5`).

## 4. Candidate

- Path above; SHA-256 `ea809e12b148f7650f48057d63273b78012fcb13c38c5eab2b603bdf148c6cc4`; 9,359,104 parameters, step 100.
- Only in the Increment 7 temp directory. Not copied to `artifacts/`, not released.

## 5. Evaluation (frozen AD-1 + C3 + supplemental 48, identical decoding)

Baseline re-evaluation reproduced frozen counts exactly.

| Suite | Metric | Baseline | Increment 7 candidate |
|---|---|---:|---:|
| AD-1 (29) | Parse | 24/29 (82.8%) | **26/29 (89.7%)** |
| AD-1 | Exact / structural / binding | 2 / 12 / 6 | 1 / 12 / **2** (loss 4) |
| C3 (54) | Parse | 24/54 (44.4%) | **42/54 (77.8%)** |
| C3 | Exact / binding / functional | 0 / 0 / 0 | 1 / 9 / **1/45 (2.22%)** |
| C3 | Functional \| binding | — | 1/7 (14.29%) |
| Supplemental (48) | Parse | 8/48 (16.7%) | 13/48 (27.1%) |
| Supplemental | **Functional** | **0/48** | **0/48** |
| Supplemental | Binding / exact | 0 / 0 | 0 / 0 |
| Supplemental | Transitions | — | both_fail 48, improved 0, regressed 0 |

Supplemental per-behaviour: all eight behaviours 0/6 on both sides. Candidate parse gains on novel names do not convert into binding or function.

## 6. Promotion decision

`promote_v120_candidate7.py` verified baseline/candidate/fixture hashes, then returned **`promote: false`** with four reasons:

1. `AD-1 binding loss 4 exceeds the limit of 1`
2. `supplemental functional accuracy did not improve: candidate 0.0% vs baseline 0.0%`
3. `supplemental functional passes 0 below the minimum of 3`
4. `supplemental functional passes span 0 behaviours, below the minimum of 3`

No promotion action was taken. Loss decrease (6.44→6.22) was ignored as instructed.

## 7. Preserved state

Baseline, Increment 1/3/5 checkpoints still hash-identical; `git status`/`diff` unchanged (4 pre-existing + reviewed Increment 4 fix); HEAD `a654e5d`; package `1.0.0`. No commits, pushes, merges, tags, releases, package-version changes, or publication.

## 8. Interpretation (not executed)

Novel-identifier + prompt-variation training at `1e-5`/100 steps improved surface parseability (supplemental parse 17%→27%, frozen parses up) but not the name-copying relationship: supplemental binding remains 0/48 and AD-1 binding loss repeats at 4. Across Increments 3, 5, and 7 the same AD-1 loss (4) and same single C3 pass (`HELD-string-000`) recur, suggesting continued-training-from-S2 at conservative rates moves output distribution without teaching binding. Further same-recipe tuning is unlikely to succeed; architecture, objective (e.g. contrastive name supervision), or evaluation-decoding changes should be reviewed separately.

**STOP for owner review before promotion or further training.**
