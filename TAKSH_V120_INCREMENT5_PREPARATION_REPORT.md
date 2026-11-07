# TAKSH v1.2.0 — Increment 5 Preparation Report: Functional Experiment Preparation

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Binding curriculum digest: `bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b`
Functional curriculum digest: `8c3f8d10c37a1f64add826bab759a4c9688ee4a7ec9661cfc65614ee01db0b60`
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Experiment identity: `a1d82b4b2d5e711e52e537d218c11d21c17a08354fb9a73b81af19d9228b20fe`

Verdict: **PREPARATION COMPLETE — NO TRAINING LAUNCHED. AWAITING EXPLICIT OWNER AUTHORIZATION.**

This increment prepared an isolated, reproducible experiment on the combined Increment 2 + Increment 4 curriculum with functional correctness as the primary promotion criterion. No training loop was initialized and no checkpoint was written or modified.

---

## 1. Verified repository state

- Branch `develop`, HEAD `a654e5d`, package version `1.0.0`.
- Git status: four pre-existing tracked modifications plus the intentional one-line `training/program_analysis.py` builtin fix from Increment 4; all pre-existing untracked work preserved. `git diff --name-only` at the end shows only those five files (four pre-existing + one reviewed Increment 4 fix).
- Baseline `02cc86eb...`, Increment 1 `89c2d61c...`, Increment 3 `6ff86ddc...` — all re-verified after all edits.
- Frozen fixtures: AD-1 `a5bf9406...5e` (29 cases), C3 `1b72f538...26e` (54 cases).
- Both curriculum digests verified above; combined counts 252 train / 84 validation.

Reviewed before editing: the four reports, `training/functional_curriculum.py`, `evaluation/functional_eval.py`, `training/curriculum_quality.py`, `training/v120_experiment.py`, the Increment 3 runner/promoter, `training/continued_training.py` (`train_from_checkpoint`), and `training/experiment_artifacts.py`. The Increment 5 runner reuses the existing trainer and immutable artifacts path; the Increment 5 promoter reuses the existing evaluation JSON format. Starting point is the original baseline only — neither rejected candidate is used for initialization.

---

## 2. Files changed

No existing tracked file was modified in this increment. Five new files:

| File | Purpose |
|---|---|
| `configs/v120_increment5.json` | Isolated config: architecture, baseline hash, both curriculum digests/counts, 100 steps, lr `1e-5`, seed `20261107`, CPU, memory floor, frozen fixtures, functional promotion rules. |
| `training/v120_experiment5.py` | Config validation, experiment identity, combined-curriculum builders, ten pre-training gates (including executing all 96 functional I/O relationships), functional-first `decide_promotion5`. |
| `scripts/run_v120_increment5.py` | Runner: dry run by default; `--train` + `--authorize` required; reuses `train_from_checkpoint`/`RunArtifacts`. |
| `scripts/promote_v120_candidate5.py` | Reads an evaluation JSON, verifies hashes, applies functional promotion rules. Writes nothing to checkpoints. |
| `tests/test_v120_increment5.py` | 23 focused tests (Section 6). |

---

## 3. Experiment configuration and identity

`configs/v120_increment5.json`, identity `a1d82b4b...`:

| Setting | Value |
|---|---|
| Architecture | `qm-code-m3-9m`, 9,359,104 parameters, vocab 5,822, context 512 (unchanged) |
| Starting checkpoint | Original baseline `02cc86eb...` only |
| Curriculum | Combined: binding 156/52 (digest `bc533a18...`) + functional 96/32 (digest `8c3f8d10...`), **252 train / 84 validation**; functional examples carry explicit I/O metadata and are distinguishable by `metadata["functional_focus"]` |
| Steps / lr / seed / device | **100 / 1e-05 / 20261107 / CPU** (enforced: steps 1–100, lr in (0, 1e-4]) |
| Output | `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5`, run `increment5-functional-100` (new, exclusive) |
| Memory floor | 0.5 GiB |

These are experimental starting values, not proven optimal.

---

## 4. Curriculum composition and statistics

Combined corpus through the unchanged Increment 2 checker (from Increment 4 verification): 252 train / 84 validation, 21 operations × 12, 20 template families, 87 normalised shapes, list-comprehension share 3.57%, max template share 15.5%, 0 duplicates, 0 leakage, 0 held-out overlap. Trainer compatibility was verified without training: the real tokenizer runs `build_batches` over combined examples (12 sampled train batches + 4 validation batches), and every example exposes `prompt_text`/`target_text`/`full_text` with metadata intact — no target format is altered and no metadata is discarded.

Functional gate detail: `check_functional_gates` executes **all 96 functional training examples** through the isolated safe gate against their stored inputs/outputs. A tampered expected output fails the gate (tested).

---

## 5. Functional evaluation criteria

`decide_promotion5` (used by `promote_v120_candidate5.py`) reports each metric independently — parse rate, exact-match, structural validity, binding, functional accuracy, functional-given-parse, functional-given-binding, failure-category distribution, template diversity, mode collapse — and rejects when:

1. AD-1 parse rate falls below baseline;
2. functional accuracy **does not improve** over baseline (strictly greater required);
3. functional accuracy conditional on binding **does not improve** (baseline `None` counts as 0.0);
4. mode collapse is detected in either suite;
5. required cases are missing (exactly 29 AD-1 / 54 C3 for both checkpoints);
6. any checkpoint or fixture hash fails verification.

Training loss is never consulted. End-to-end evidence: the promoter run on the real Increment 3 evaluation returns `promote: false` (AD-1 parse regression).

---

## 6. All gate results (dry-run evidence)

`python scripts/run_v120_increment5.py` (no flags) printed `dry_run` with preflight:

- configuration: **pass** (identity `a1d82b4b...`)
- baseline checkpoint: **pass** (`02cc86eb...`)
- curriculum digests / quality / separation: **pass** (both digests, 252/84, leakage 0)
- functional relationships: **pass** (96 examples verified through the safe gate)
- frozen fixtures: **pass** (AD-1 29, C3 54)
- memory: **fail** — available `0.23 GiB` below the required `0.50 GiB`. This only blocks an actual training start (same pattern as Increment 3 preparation at 0.16 GiB, later 0.71–1.06 GiB); the threshold was not lowered.
- output directory: **pass** (new run directory, not created)
- authorization: not authorized (dry run only)

Tests (23 passed): config validation (6 unsafe mutations), identity reproducibility, checkpoint/digest/fixture gates (pass + tamper), functional gate (pass + tampered-output rejection), trainer compatibility via `build_batches`, memory thresholds, authorization refusal (`run_preflight` raises; subprocess `--train` without `--authorize` exits 2), dry-run non-mutation (no run directory, baseline hash unchanged), all six promotion conditions plus acceptance, and the real Increment 3 rejection. Regression: the 51 Increment 1–4 tests still pass.

---

## 7. Exact complete launch command for future authorization

Only after explicit owner authorization and with memory at or above the floor:

```powershell
python scripts/run_v120_increment5.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5 --run-id increment5-functional-100
```

Followed by:

```powershell
python scripts/evaluate_v120_increment1.py --candidate <candidate-checkpoint> --baseline C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt --out <evaluation-json>
python scripts/promote_v120_candidate5.py --evaluation <evaluation-json>
```

Promotion requires the promoter to print `promote: true`.

---

## 8. Confirmation no training was launched

- The Increment 5 runner was executed only in dry-run mode. `--train` appeared solely in one refusal test (exit 2, no directory created) and in the recorded launch command.
- No `increment5` run directory exists; no checkpoint was written.
- All three checkpoints and both fixtures are byte-identical to their starting values (Section 1).
- No commits, pushes, merges, tags, releases, package-version changes, external weights/datasets, or paid services.

---

## 9. Recommendation for Increment 6

1. If Increment 5 trains and is rejected on functional grounds again, the bottleneck is no longer data diversity or dilute signalling — it is a model/expressive capacity or objective mismatch, and further same-architecture continued training is unlikely to be productive. At that point, revisit architecture or task formulation under separate review rather than adding a fifth tuning round.
2. If Increment 5 is promoted on functional grounds, replicate across seeds before any capability claim, and extend the functional curriculum toward AD-1-shaped arithmetic/paraphrase behaviours (the shapes that regressed in Increment 3).
3. Keep the functional-first promotion gate and conditional reporting for all future candidates; binding-only gains must never promote again.
4. Resolve the `while`/lambda safe-gate allowlist question before targeting those constructs as training objectives (Increment 4 limitation).

**STOP after Increment 5. Wait for explicit owner authorization before launching training.**
