# TAKSH v1.2.0 — Increment 10 Training Report: Identifier-Aware Integration

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Experiment identity: `f97030aba48b7b98cc94f0d55b6970025af5f3401919c19f90cae08f01fa8040`
Candidate checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment10\increment10-copy-100\checkpoints\trained.pt`
Candidate SHA-256: `e68495931da7bf4d0dde666615f16757a87a01304c0a2358ac4fa05bda634c9b`

Verdict: **PROMOTION REJECTED (promote: false). Candidate preserved temp-only.**

Identifier-aware reweighting plus a staged copy curriculum did not teach identifier copying: 0/13 frozen probes, 0/48 supplemental, AD-1 binding loss 4 (identical to Increments 5 and 7). The mechanism is correctly wired but insufficient at 100 steps / `1e-5`.

**STOP. No further training, promotion, or publication.**

---

## 1. Baseline Git status and HEAD

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: 4 pre-existing + `training/program_analysis.py` (Inc4) + `training/supervised_train.py` and `training/continued_training.py` (this increment's integration — see Section 2). All hashes re-verified after training: baseline `02cc86eb...`, Inc1 `89c2d61c...`, Inc3 `6ff86ddc...`, Inc5 `3b3777af...`, Inc7 `ea809e12...`, tokenizer `90fd1c8b...`, fixtures AD-1/C3.

## 2. Files changed

| File | Change |
|---|---|
| `training/supervised_train.py` | `build_batches(..., identifier_config=None)` sidecar weights + `weighted_mean_loss` dispatcher (None → previous objective exactly) |
| `training/continued_training.py` | `train_from_checkpoint(..., identifier_config=None)` threads weights to train **and** validation batches; weighted micro-loss + validation loss |
| `configs/v120_increment10.json` | Combined binding(156/52)+copy(112/24)=268/76, weights (2.0/1.5/4.0/0.85), 100 steps, lr `1e-5`, seed `20261110`, CPU |
| `training/v120_experiment10.py` | Gates (digests, quality with relaxed copy-drill thresholds — documented, functional execution of all 112, fixtures, memory, output, authorization) + `decide_promotion10` |
| `scripts/run_v120_increment10.py` | Runner: dry-run default, `--train` + `--authorize` required, isolated temp dir |
| `scripts/evaluate_v120_increment10.py` | Copy probes (13) + supplemental (48) + AD1/C3 frozen, identical greedy decoding |
| `scripts/promote_v120_candidate10.py` | Hash verification + predeclared gates |
| `tests/test_v120_increment10.py` | 13 focused tests (Section 4) |
| `TAKSH_V120_INCREMENT10_TRAINING_REPORT.md` | This report |

## 3. Trainer integration details

- `identifier_config=None` reproduces the previous objective bit-for-bit on labels/ids (tested: `torch.equal` labels, single-determination weights, uniform-weights equivalence).
- With config: weights computed **once per example** at batch build (no double application), mean-normalized to 1.0 across supervised positions, prompt-mask/target/EOS preserved, syntax tokens weight 1.0, copy tokens up to 4.0 with decay/floor.
- Applied identically to training and validation micro-batches (WP1 consistency requirement).

## 4. Tests and exact results

```powershell
python -m pytest tests\test_v120_increment10.py tests\test_trainer_integration.py tests\test_identifier_supervision.py -q -p no:cacheprovider
# 7 (integration) + 9 (supervision) + 13 (experiment) = 29 passed
python -m pytest tests\test_diagnostics.py tests\test_binding_curriculum.py tests\test_curriculum_quality.py tests\test_v120_increment3.py tests\test_v120_increment5.py tests\test_v120_increment7.py tests\test_functional_eval.py tests\test_stabilization.py -q -p no:cacheprovider
# 111 passed (Increments 1-8 regression, re-run pre-training)
```

Numerical verification: uniform weights reproduce token-mean exactly; enabled weights change only intended token contributions; disabling reproduces the previous objective (WP2 all satisfied before training).

## 5. Training configuration and seed

`configs/v120_increment10.json`: steps 100, lr `1e-05`, seed `20261110`, CPU, max length 256, grad accum 2, warmup 2, weight decay 0.1, grad clip 1.0. Preflight: all gates passed at 0.64 GiB available (dry-run), 0.62 GiB in-run.

## 6. Training results

Original command executed with both flags. 100/100 steps, all finite:
- Initial loss **6.2540** → final **5.3823**; validation loss **5.0805**
- Tokens consumed **3,916** (39.16/example); elapsed **46.38 s** training
- Loss reported for record only, not capability evidence (heavily fluctuating, cf. Increments 5/7)

## 7. Candidate SHA-256

`e68495931da7bf4d0dde666615f16757a87a01304c0a2358ac4fa05bda634c9b` (Section header). Temp directory only; never promoted/copied.

## 8. Evaluation comparison (baseline / Inc5 / Inc7 / Inc10)

Identical decoding; baselines reproduced frozen counts.

| Metric | Baseline | Inc5 | Inc7 | **Inc10** |
|---|---:|---:|---:|---:|
| Copy probes exact (13) | 0 | 0 | 0 | **0 (0 famil., 0 unfamil.)** |
| Supplemental functional (48) | 0 | 0 | 0 | **0** (parse 13/48) |
| AD-1 parse / binding | 24 / 6 | 24 / 2 | 26 / 2 | 24 / **2** |
| C3 parse / binding / functional | 24 / 0 / 0 | 37 / 9 / 1 | 42 / 9 / 1 | 37 / 9 / **1** |
| C3 functional \| binding | — | 1/10 | 1/9 | 1/9 |

Inc10 candidate copy completions are still `text`-template outputs (`return text.`, `return text = = = 0`). Machine-readable: `increment10-copy-100/evaluations/heldout_comparison.json`.

## 9. Promotion gates (all predeclared, evaluated by promoter)

| Gate | Required | Actual | Result |
|---|---|---:|---|
| Exact copy probes | ≥10/13 | **0/13** | **FAIL** |
| Supplemental functional | ≥3/48 | **0/48** | **FAIL** |
| Distinct behaviours | ≥3 | **0** | **FAIL** |
| AD-1 binding regression | ≤1 | **4** (6→2) | **FAIL** |
| AD-1 parse fall | none | held (24→24) | PASS |
| C3 functional regression | none | improved (0→1) | PASS |
| Collapse / missing / hashes | none/clean | none, all verified | PASS |

**Gate result: REJECTED.** Decision reasons: copy 0/13 < 10; supplemental 0/48 < 3 and unimproved; 0 behaviours < 3; AD-1 binding loss 4 > 1.

## 10. Explicit decisions

- **Promotion decision: NOT AUTHORIZED** (gates fail; candidate stays isolated in the temp directory).
- **Publication decision: NOT AUTHORIZED** (no HF action, no tag/release/commit/push/merge, no package-version change).
- Architecture, tokenizer, frozen fixtures: unchanged ($verified hashes$).

## 11. Interpretation

The mechanism works as built (weights applied, loss integrates, curriculum verified) but does not produce copying at this scale/schedule: 100 gentle steps with 2× copy weight are, like the previous three continued runs, enough to shift output distribution (C3 parse 37, binding 9) without moving the copy probe off zero. The bottleneck Increment 8 identified is not reweighting alone; a future redesign must either provide far more copy supervision per step (dedicated copy phase at higher effective weight/rate), a different objective interface, or capacity ablation. No further same-recipe run is recommended.

**STOP after this one bounded run and report. Awaiting owner review.**
