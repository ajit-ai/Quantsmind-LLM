# TAKSH v1.2.0 — Increment 3 Execution Report: Authorized Controlled Training

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Owner authorization: **GRANTED** (this execution only).
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Experiment identity: `47faeeaa94c0354280f29a305d6dec5f3c230ff95db6b3b0d0bfd1e65bde245b`
Candidate checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3\increment3-diverse-100\checkpoints\trained.pt`
Candidate SHA-256: `6ff86ddcec6678e2d1f5a42a0a9ef07f2ff6e3c771b5fb5b3c228d9a72bbdfc6`

Verdict: **CANDIDATE REJECTED — NOT PROMOTED.**

The authorized run completed all 100 configured steps stably, and it avoids the Increment 1 catastrophe (no mode-collapse verdict, much higher C3 parse rate). But it fails the pre-registered promotion gate: AD-1 parse rate fell below baseline. Mixed C3 gains do not override the gate. The candidate is preserved for inspection only.

---

## 1. Preflight gate results and actual available RAM

Verified before launch:

- Branch `develop`, HEAD `a654e5d`, package version `1.0.0`.
- Baseline checkpoint SHA-256 `02cc86eb...837f8ac` ✓, Increment 1 candidate `89c2d61c...b80` ✓ (both untouched).
- Curriculum digest `bc533a18...07b7b` ✓.
- Frozen fixtures: AD-1 `a5bf9406...5e` (29 cases), C3 `1b72f538...26e` (54 cases) ✓.
- Available RAM immediately before training: **1.06 GiB** (total 3.911 GiB, 72.9% used) — above the 0.5 GiB floor ✓.
- Increment 3 output directory `increment3-diverse-100`: confirmed new and isolated (did not exist) ✓.

All nine gates passed, as also recorded in the run's own preflight block (memory gate at launch: available `0.71 GiB >= 0.50 GiB`, authorization: authorized). No gate failed; training proceeded.

---

## 2. Exact executed command and configuration identity

```powershell
python scripts/run_v120_increment3.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3 --run-id increment3-diverse-100
```

Configuration (unchanged from the approved `configs/v120_increment3.json`): architecture `qm-code-m3-9m` (9,359,104 parameters, vocab 5,822, context 512), Increment 2 diversified curriculum (156 train / 52 validation, digest `bc533a18...`), **100 steps**, **lr 2e-05**, **seed 20261106**, **CPU**, max length 256, grad accum 2, warmup 2, weight decay 0.1, grad clip 1.0, memory floor 0.5 GiB. Experiment identity `47faeeaa...`. No hyperparameter was altered and no second run was launched.

---

## 3. Actual training steps and metrics

From `metrics/training.json` and `manifest.json` of the run:

- Steps completed: **100 / 100**. All 100 history entries have `finite_loss: true`.
- Initial loss (step 1): **5.8797**. Final loss (step 100): **2.8554**.
- Validation loss: **4.4682**.
- Tokens consumed: **3,960** (39.6 supervised tokens/example average).
- Elapsed training time: **41.26 s** (48.69 s total including startup/finalize).
- No instability was observed: losses stayed in the 2.2–6.2 range with no NaN/inf, and the process ran to completion. Memory after evaluation: available **1.046 GiB** (73.3% used), so no resource pressure event occurred.
- Important context: the validation loss (4.47) is far higher than Increment 1's (1.14). The diversified curriculum is a harder target and 100 steps at `2e-5` is conservative; per the promotion rules, loss is reported here for the record and was **not** used as evidence of capability.

Note the contrast with Increment 1 (240 steps at `1e-4`, loss 3.21→1.47, val 1.14): deeper optimization fit the narrow template distribution; this run was deliberately gentler against a harder, more diverse target.

---

## 4. Candidate checkpoint location and SHA-256

- Path: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3\increment3-diverse-100\checkpoints\trained.pt`
- SHA-256: `6ff86ddcec6678e2d1f5a42a0a9ef07f2ff6e3c771b5fb5b3c228d9a72bbdfc6` (also recorded in `manifest.json` under `files`)
- Size: 37,463,173 bytes pattern as Increment 1 (architecture unchanged; checkpoint payload includes step 100 metadata). Parameters 9,359,104, tensors 75.
- Evaluation output: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3\increment3-diverse-100\evaluations\heldout_comparison.json`
- The candidate was **not** copied into `artifacts/`, not registered as a release, and must not be promoted.

---

## 5. Baseline versus candidate evaluation

Same frozen suites, identical prompts and decoding as Increment 1 (greedy, `max_new_tokens=64`, seed `20261023`, raw output `binding_repair=False`). Full per-case records are in the evaluation JSON above.

| Suite | Metric | Baseline | Candidate | Change |
|---|---|---:|---:|---:|
| AD-1 (29) | Parses | 24/29 | 23/29 | **−1** |
| AD-1 | Exact | 2/29 | 1/29 | −1 |
| AD-1 | Structural | 12/29 | 12/29 | 0 |
| AD-1 | Binding | 6/29 | 2/29 | **−4** |
| C3 (54) | Parses | 24/54 | 35/54 | **+11** |
| C3 | Exact | 0/54 | 1/54 | +1 |
| C3 | Structural | 3/48 | 3/48 | 0 |
| C3 | Binding | 0/54 | 10/54 | **+10** |
| C3 | Functional | 0/45 | 1/45 | +1 |
| AD-1 | Mean latency | 0.343 s | 0.436 s | +0.09 s |
| C3 | Mean latency | 0.425 s | 0.588 s | +0.16 s |

The baseline re-evaluation again reproduced the frozen counts exactly (AD-1 2/12/6, C3 0/0/0), so the comparison is fair.

Template diversity and collapse:

- Candidate AD-1: 8 distinct templates, dominant `direct_listcomp_return` 6/29 (20.7%), **not collapsed**.
- Candidate C3: 8 distinct templates, dominant `direct_listcomp_return` 16/54 (29.6%, one point below the 30% collapse threshold), **not collapsed** by the formal rule — but with 15 further unparseable list-comp-shaped outputs, 31/54 C3 outputs are list-comprehension-shaped. The attractor is reduced versus Increment 1, not removed.
- Baseline itself is flagged collapsed by the same detector (AD-1 `direct_binop_return` 9/29 = 31.0%; C3 `unparseable` 29/54 = 53.7%). Promotion checks only the candidate, per the approved rules.

Illustrative generations (raw, identical prompts):

- `HELD-string-000` (upper-case a string): baseline `return left + right` (functional fail); candidate `return text.upper()` (exact, binding, **functional pass**). Same win as Increment 1.
- Candidate C3 binding wins are dominated by the comprehension template, e.g. `HELD-comprehension-000`: `return [value for value in values if value > 0]` (binding true, functional fail); `HELD-edge-001`: same template (binding true, functional fail). Binding credit here largely reflects emitting a well-formed template that uses the requested sequence name, not solving the requested task.
- AD-1 regression: the baseline's exact wins `E2-add-002` and `E2-subtract-001` are lost by the candidate (candidate AD-1 exact is only `E2-upper-001`); candidate AD-1 binding falls to 2/29.

---

## 6. Promotion decision and reasons

`python scripts/promote_v120_candidate.py --evaluation <heldout_comparison.json>` verified the baseline hash, candidate hash, and both fixture hashes, then returned:

- **`promote: false`**
- Reasons: `["parse rate fell in ad1: candidate 23/29 (79.3%) below baseline 24/29 (82.8%)"]`

No other gate fired: C3 parse did not fall (it rose), C3 functional did not fall (1/45 vs 0/45), no case-count mismatch (29/54 for both), no hash failure, and no candidate mode-collapse verdict. The single AD-1 parse regression is sufficient for rejection under the pre-registered rules, and the AD-1 binding collapse (6→2) reinforces that this candidate is not a uniform improvement. The C3 improvements are real but narrow (parse/binding via the comprehension template plus one string-case functional pass) and do not override the gate — exactly as the safeguard was designed to operate.

**Decision: do not promote. The frozen S2 baseline remains the only reference checkpoint.**

---

## 7. Resource usage and any failures

- Training: CPU, 4 threads, 100 steps, 41.26 s, 3,960 tokens. No memory-pressure stop was needed (pre-flight 1.06 GiB, in-run gate 0.71 GiB, post-evaluation 1.046 GiB available).
- Evaluation: 83 cases × 2 checkpoints at ~0.34–0.59 s per generation; models loaded sequentially.
- Failures: none during training or evaluation. The only "failure" is the intended safeguard outcome: promotion rejected on the AD-1 parse gate.
- No partial/aborted artifacts exist; the run finalized cleanly with manifest and hash inventory.

---

## 8. Confirmation of preserved baseline and repository state

- Baseline checkpoint still `02cc86eb...837f8ac`; Increment 1 candidate still `89c2d61c...b80`; artifact copy still `02cc86eb...837f8ac`. Verified after evaluation.
- `git status` / `git diff --name-only` unchanged: only the four pre-existing tracked modifications; HEAD `a654e5d`; package version `1.0.0`.
- No commits, pushes, merges, tags, releases, package-version changes, external weights, external datasets, paid services, or Hugging Face publication.
- New files from this execution: only this report (experiment artifacts live outside the repository under the approved temp root).

---

## 9. Interpretation (no new run authorized)

Increment 3 delivers a clean negative-plus-nuance result:

1. **Safer recipe, no catastrophe.** At `2e-5` / 100 steps the model no longer collapses into unparseable output (C3 parses 24→35 vs Increment 1's 24→12), and the formal collapse detector does not flag the candidate.
2. **But the attractor persists.** C3 outputs remain list-comp-dominated (31/54 shaped), and AD-1 capability (binding 6→2, exact 2→1) regressed. Diversity helped breadth of parseable output without teaching reliable parameter binding on the AD-1 task shapes.
3. **Likely undertrained for binding, overfit to one surface.** Validation loss 4.47 suggests the diversified target was not fit in 100 gentle steps, while the C3 "binding" gains come from template emission rather than task solutions (9 of 10 C3 binding wins fail functionally).

Plausible next levers (recorded for review, **not** executed): curriculum operations for string-join/scaling; weaker list-comp share or comprehension-specific counter-examples; slightly higher steps with the parse gate retained; seed replication. None of these is authorized by this execution.

**STOP after this experiment and report. Do not start Increment 4, retrain, or publish to Hugging Face without further authorization.**
