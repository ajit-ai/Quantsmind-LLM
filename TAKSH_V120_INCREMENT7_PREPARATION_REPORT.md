# TAKSH v1.2.0 — Increment 7 Preparation Report: Targeted Generalization Improvement

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Binding curriculum digest: `bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b`
Targeted curriculum digest: `86230100b6a29e7c3beef98511b6f368ccc6b48307106770a5c8f8eb651cfc67`
Supplemental suite digest: `05c0c657a5ca40a743b0536a4698eed5f15272034f3cabd9c791ba0bc49cc32e`
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Experiment identity: `88cf0126da0306999788ac95d21950fed25940bdebbb29c12014d64212578020`

Verdict: **PREPARATION COMPLETE — NO TRAINING LAUNCHED. AWAITING EXPLICIT OWNER AUTHORIZATION.**

Increment 6 (decision B) showed both checkpoints score 0/48 on novel-identifier variants of the eight trained behaviours. This preparation builds a targeted curriculum that teaches novel identifiers and prompt wording for the same eight behaviours, retains binding/arithmetic coverage, and gates promotion on supplemental generalization + AD-1 retention.

---

## 1. Verified starting state

- Branch `develop`, HEAD `a654e5d`, package `1.0.0`.
- Checkpoint hashes verified: baseline `02cc86eb...`, Increment 1 `89c2d61c...`, Increment 3 `6ff86ddc...`, Increment 5 `3b3777af...` — all re-verified after all edits.
- Fixtures: AD-1 `a5bf9406...5e` (29), C3 `1b72f538...26e` (54); supplemental case digest `05c0c657...`.
- Tracked diff unchanged from Increment 6 state (4 pre-existing + reviewed Increment 4 one-line fix). All untracked work preserved. No invariant failed.

---

## 2. Files added/modified

No existing tracked file was modified. Six new files:

| File | Purpose |
|---|---|
| `training/targeted_curriculum.py` | Targeted curriculum: 8 behaviours × 16 train / × 4 validation, novel identifier pools, 6 paraphrases/behaviour, semantic train/validation separation. |
| `configs/v120_increment7.json` | Isolated config: combined 284/84 (binding 156/52 + targeted 128/32), 100 steps, lr `1e-5`, seed `20261108`, CPU, memory floor, all hashes, promotion rules. |
| `training/v120_experiment7.py` | Ten pre-training gates (incl. executing all 128 targeted I/O relationships + supplemental-overlap check) and generalization-first `decide_promotion7`. |
| `scripts/run_v120_increment7.py` | Runner: dry run default; `--train` + `--authorize` required; reuses existing trainer/artifacts. |
| `scripts/promote_v120_candidate7.py` | Reads evaluation JSON (frozen + supplemental blocks), verifies hashes, applies promotion rules. |
| `tests/test_v120_increment7.py` | 22 focused tests (Section 5). Plus `TAKSH_V120_INCREMENT7_PREPARATION_REPORT.md` (this report). |

---

## 3. Targeted curriculum design (from Increment 6 evidence)

**A. Parameter binding.** Novel identifier generator: 48 unique pairs + 48 singles (`probe_00`, `segment_23`, ...), mathematically disjoint from every Increment 2/4 pool and from the fixed names (`text`, `left`, `right`, `values`, `items`, ...). 32 distinct training param names; models can no longer succeed via fixed-name memorization.

**B. Prompt generalization.** Six paraphrases per behaviour (four Increment 4 + two new per behaviour). Training uses paraphrases 0–3, validation uses 4–5 — disjoint wording sets.

**C. Functional diversity.** All eight behaviours retained (scale, join, compose, compose_second, filter_count, long_words, is_even, sum_positive), four surface variants each (direct / intermediate / augmented / built-in-or-loop form), same safe-gate-compatible constructs.

**D. Capability retention.** The full Increment 2 binding curriculum is composed alongside: arithmetic (`add`/`subtract`/`multiply`), conditioning (`maximum`/`minimum`), binding probes, text/collection/aggregate. No behaviour was dropped.

**Semantic separation:** validation uses identifier half 24–47 (disjoint from training 0–23) and paraphrase half 4–5; `(instruction, signature)` disjointness verified. Duplicate/near-duplicate detection, template distribution, and functional execution all reuse the Increment 2/4 gates. Frozen AD-1/C3 and the Increment 6 supplemental suite are excluded from training (overlap check tested, Section 5).

Statistics: targeted digest `86230100...`; combined 284 train / 84 validation, 21 operations × 12+16/12+4 composition, list-comp share within cap, quality report `{train: 284, overlap: 0}`. Trainer compatibility verified via `build_batches` tokenization without training.

---

## 4. Pre-training gates (all verified in dry run)

1. Baseline checkpoint hash ✓
2. Curriculum digests (binding `bc533a18...`, targeted `86230100...`) ✓
3. AST + curriculum-quality + separation + zero held-out (frozen **and** supplemental) overlap ✓
4. Functional relationships: all **128 targeted examples executed** through the isolated safe gate ✓
5. Train/validation separation ✓ (included in 3)
6. Frozen fixtures AD-1/C3 hashes + supplemental case digest ✓
7. Memory: dry-run observed **0.74 GiB ≥ 0.50 GiB** ✓ (blocks training only; threshold not lowered)
8. Output directory new/isolated ✓ (run directory not created)
9. Configuration + seed recorded (identity `88cf0126...`) ✓
10. Authorization: dry-run only ✓ (`--train` without `--authorize` exits 2, tested)

---

## 5. Tests and results

```powershell
python -m pytest tests\test_v120_increment7.py -q -p no:cacheprovider
# 22 passed in 76.35s

python -m pytest tests\test_binding_curriculum.py tests\test_curriculum_quality.py tests\test_v120_increment3.py tests\test_v120_increment5.py tests\test_functional_eval.py tests\test_stabilization.py -q -p no:cacheprovider
# 83 passed in 96.61s (no regressions)
```

Coverage: determinism, novel-pool size/disjointness (≥64 names), prompt variation + semantic separation, full functional execution sweep, config/tamper rejection, checkpoint/fixture gates, combined quality + supplemental-overlap, trainer tokenization compatibility, memory thresholds, authorization refusal, dry-run non-mutation, promotion matrix (accept broad generalization; reject single-template pass, AD-1 binding regression, no improvement, collapse/hash failures), and the promoter CLI end-to-end (synthetic evaluation → `promote: true`).

Two defects found by the suite and fixed: a validation identifier-pool wraparound (validation names collided with training) and a supplemental gate that hashed the builder file instead of the case digest.

---

## 6. Evaluation and acceptance criteria (defined before training)

Future evaluation uses frozen AD-1 + C3 + the Increment 6 supplemental 48-case suite, identical prompts/decoding, reporting each metric independently (parse, exact, structural, binding, functional, functional|parse, functional|binding, categories, diversity, collapse).

`decide_promotion7` rejects unless **all** hold (thresholds stated in `training/v120_experiment7.py` with rationale):

- AD-1 parse ≥ baseline **and** AD-1 binding loss ≤ **1** case (retained capability; Increment 5 lost 4)
- Supplemental functional rate strictly improves over baseline **and** ≥ **3 passes** spanning ≥ **3 behaviours** (anti-single-template floor: one pass cannot distinguish memorization from learning)
- Frozen C3 functional does not regress; conditional functional not below baseline (None → 0)
- No mode collapse, exact case counts (29/54/48), all hashes verified
- Training loss never consulted

---

## 7. Exact verified training command (not executed)

```powershell
python scripts/run_v120_increment7.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7 --run-id increment7-targeted-100
```

Dry-run result: `dry_run`, identity `88cf0126...`, all static gates pass, memory 0.74 GiB, no run directory created.

---

## 8. Confirmation no training was launched

- Runner executed only in dry-run mode (plus one refusal test without `--authorize`, exit 2).
- No `increment7` run directory exists; no checkpoint written or altered. All four checkpoint hashes identical to Section 1 after all work.
- No commits, pushes, merges, tags, releases, package-version changes, external data/weights, or publication.

---

## 9. Recommendation for Increment 8

1. If Increment 7 trains and is promoted under the Section 6 criteria, replicate across seeds and extend identifier novelty further before any capability claim — one generalization probe suite remains small (48 cases).
2. If it is rejected on AD-1 binding again, binding retention needs its own gate *before* functional upside is credited (e.g. a binding-only warm phase), rather than diluting combined training.
3. If supplemental generalization still fails, the bottleneck is likely representational (9M model / fixed evaluation decoding), not data volume — revisit architecture or evaluation protocol under separate review instead of adding another tuning round.

**STOP after Increment 7 preparation. Wait for explicit owner authorization before training.**
