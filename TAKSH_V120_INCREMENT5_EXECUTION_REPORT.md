# TAKSH v1.2.0 — Increment 5 Execution Report: Authorized Functional Training

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Owner authorization: **EXPLICITLY GRANTED** (this execution only).
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Experiment identity: `a1d82b4b2d5e711e52e537d218c11d21c17a08354fb9a73b81af19d9228b20fe`
Candidate checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\checkpoints\trained.pt`
Candidate SHA-256: `3b3777af79f039fb7309e3d9d83ad8295f998543c767348bcbe3f2385344f5ad`

Verdict: **PROMOTION GATE PASSES (promote: true) — NO PROMOTION ACTION TAKEN. Candidate remains in the temporary run directory pending owner decision.**

The authorized run completed all 100 configured steps stably on the combined binding + functional curriculum. It clears every functional-first gate (AD-1 parse held, functional and conditional functional improved, no collapse), but the absolute capability gain is small (1 functional pass) and AD-1 binding still regressed (6→2). Per the execution constraints, this report stops before any promotion, publication, or further training.

---

## 1. Pre-execution verification and memory recheck

- Branch `develop`, HEAD `a654e5d`, package `1.0.0` ✓
- Baseline `02cc86eb...837f8ac` ✓, Increment 1 `89c2d61c...b80` ✓, Increment 3 `6ff86ddc...dfc6` ✓
- Digests: binding `bc533a18...07b7b`, functional `8c3f8d10...0b60` ✓
- Fixtures: AD-1 `a5bf9406...5e` (29), C3 `1b72f538...26e` (54) ✓
- Available RAM immediately before training: **0.971 GiB** (total 3.911 GiB, 75.2% used) ≥ 0.5 GiB floor ✓; run directory confirmed new ✓
- In-run preflight also passed all gates at **0.79 GiB** available.

---

## 2. Exact executed command and configuration

```powershell
python scripts/run_v120_increment5.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5 --run-id increment5-functional-100
```

Configuration (unchanged approved values): `qm-code-m3-9m` (9,359,104 parameters, vocab 5,822, context 512), baseline-only start, combined curriculum 252 train / 84 validation, **100 steps, lr `1e-05`, seed `20261107`, CPU**, max length 256, grad accum 2. Identity `a1d82b4b...`. No hyperparameters altered; no second run launched.

---

## 3. Training metrics

From `metrics/training.json` and `manifest.json`:

- Steps: **100 / 100**, all entries `finite_loss: true`
- Initial loss (step 1): **6.4140**. Final loss (step 100): **5.2101**
- Validation loss: **4.8530**
- Tokens consumed: **3,950** (39.5 supervised tokens/example)
- Elapsed: **42.24 s** training (49.57 s total)
- Stability: losses ranged 1.86–6.99 with no NaN/inf or interruption; no memory-pressure stop occurred (post-evaluation available RAM 0.896 GiB). Partial-candidate preservation was not needed.
- Loss is reported for the record only and was **not** used as capability evidence, per the functional-first rules. Note validation loss (4.85) remains high versus Increment 1 (1.14): at `1e-5`, 100 steps is still a gentle fit to a harder combined target.

---

## 4. Candidate checkpoint

- Path: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\checkpoints\trained.pt`
- SHA-256: `3b3777af79f039fb7309e3d9d83ad8295f998543c767348bcbe3f2385344f5ad`
- Parameters 9,359,104, tensors 75, step 100 metadata. Saved only in the designated Increment 5 directory; baseline, tokenizer, fixtures, and all pre-existing work untouched.
- Evaluation JSON: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\evaluations\heldout_comparison.json`

---

## 5. Baseline versus candidate evaluation

Identical frozen prompts and decoding (greedy, `max_new_tokens=64`, seed `20261023`, raw output). Baseline re-evaluation again reproduced the frozen counts exactly.

| Suite | Metric | Baseline | Candidate | Change |
|---|---|---:|---:|---:|
| AD-1 (29) | Parse rate | 24/29 (82.8%) | 24/29 (82.8%) | 0 (held) |
| AD-1 | Exact | 2/29 | 1/29 | −1 |
| AD-1 | Structural | 12/29 | 12/29 | 0 |
| AD-1 | Binding | 6/29 | 2/29 | **−4** |
| C3 (54) | Parse rate | 24/54 (44.4%) | 37/54 (68.5%) | **+13** |
| C3 | Exact | 0/54 | 1/54 | +1 |
| C3 | Structural | 3/48 | 3/48 | 0 |
| C3 | Binding | 0/54 | 9/54 | +9 |
| C3 | **Functional accuracy** | 0/45 (0%) | **1/45 (2.22%)** | **+1** |
| C3 | Functional \| parse | 0/22 (0%) | 1/32 (3.12%) | +1 pass |
| C3 | **Functional \| binding** | 0/0 (none) | **1/7 (14.29%)** | improved from none |
| Latency | AD-1 / C3 mean | 0.312 / 0.392 s | 0.305 / 0.419 s | ≈flat |

Failure categories (candidate): AD-1 — binding 22, syntax 5, operation 1, plausible-but-wrong 1. C3 — binding 28, syntax 17, operation 3, plausible-but-wrong 4, collection 1, pass 1.
Diversity/collapse: candidate AD-1 7 templates, dominant `direct_method_lower_return` 8/29 (27.6%), not collapsed. Candidate C3 12 templates, dominant `direct_listcomp_return` 16/54 (29.6%), not collapsed — but still list-comp-shaped on 22+ outputs including 6 degenerate-shaped ones.

Illustrative: `HELD-string-000` — candidate `return text.upper()` (exact, binding, functional pass). The other 8 C3 binding wins are overwhelmingly the same comprehension template failing functionally (e.g. `HELD-comprehension-000`: `return [value for value in values if value > 0]`, binding true, functional fail). AD-1 binding wins remain only `E2-upper-001` and `E4-filter_positive_count-0`.

---

## 6. Promotion decision

`python scripts/promote_v120_candidate5.py --evaluation <heldout_comparison.json>` verified baseline hash, candidate hash, and both fixture hashes, then returned:

- **`promote: true`**, `reasons: []`
  - AD-1 parse held: 24/29 vs 24/29 ✓
  - Functional improved: 2.22% vs 0% ✓
  - Functional-given-binding improved: 14.29% vs 0% (baseline had no binding passes) ✓
  - No collapse, no missing cases, no hash failures ✓

**Interpretation (no action taken):** the gate verdict is *eligible*, not *enacted*. No promotion, copy into `artifacts/`, release, commit, push, or publication was performed. The absolute gain is one functional pass of 45, and AD-1 binding/exact still regressed — the candidate is functionally better than baseline by the pre-registered definition but remains far from reliable Python generation (functional 2.22%, conditional 14.29%). Owner is asked to decide whether "eligible" is acted on; this execution takes no further step.

---

## 7. Resource usage, failures, preserved state

- Training 42.24 s CPU; evaluation ~83 generations × 2 checkpoints. No memory-pressure, non-finite-loss, hash-mismatch, or evaluation failure occurred. Nothing was interrupted, so no partial-candidate handling was needed.
- Baseline `02cc86eb...`, Increment 1 `89c2d61c...`, Increment 3 `6ff86ddc...` all still hash-identical; `git status`/`git diff` unchanged (4 pre-existing + 1 reviewed Increment 4 fix); HEAD `a654e5d`; package `1.0.0`.
- No commits, pushes, merges, tags, releases, package-version changes, or Hugging Face publication.

---

## 8. Stopping point

**STOP before any promotion or further training.** The candidate at `3b3777af...` awaits owner disposition. Suggested Increments 6+ directions (not executed): extend functional coverage to AD-1 arithmetic/paraphrase shapes where binding still regresses; replicate across seeds; investigate the persistent comprehension attractor (30%+ of C3 outputs) at the curriculum/objective level; resolve the `while`/lambda safe-gate question before targeting those constructs.
