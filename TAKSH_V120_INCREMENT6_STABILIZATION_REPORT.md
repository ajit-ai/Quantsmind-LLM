# TAKSH v1.2.0 — Increment 6 Stabilization Report: Capability Stabilization

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 1: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80` (unchanged)
Increment 3: `6ff86ddcec6678e2d1f5a42a0a9ef07f2ff6e3c771b5fb5b3c228d9a72bbdfc6` (unchanged)
Increment 5: `3b3777af79f039fb7309e3d9d83ad8295f998543c767348bcbe3f2385344f5ad` (unchanged)

Verdict: **B — Improvement limited to narrow behavior; further controlled work required. DO NOT PROMOTE Increment 5.**

The Increment 5 `promote: true` verdict is not sufficient evidence for acceptance (as instructed). Under the stronger pre-stated stabilization criteria, the candidate fails on AD-1 retention and shows zero functional success on novel surface forms of the very behaviours it was trained on. It remains a temp-directory artifact only.

**STOP after this report. No Increment 7, training, promotion, or publication was performed.**

---

## 1. Starting state verification

Verified before analysis: branch `develop`, HEAD `a654e5d`, package `1.0.0`; all four checkpoint hashes above; fixtures AD-1 `a5bf9406...5e`, C3 `1b72f538...26e`; tracked diff = four pre-existing modifications + the reviewed one-line Increment 4 `training/program_analysis.py` fix. All re-verified at the end with identical values.

---

## 2. Functional analysis by behavior

### 2.1 Supplemental behavior suite (new, separate from frozen suites)

`evaluation/behavior_suite.py`: 48 deterministic cases (6 per behavior × 8), novel wording/function names/parameter names/values, reference answers verified correct through the safe gate, zero `(instruction, signature)` overlap with frozen AD-1/C3 or any training builder (tested). Scored with identical greedy decoding for both checkpoints via `scripts/analyze_v120_stabilization.py`.

| Behavior | Baseline functional (n=6) | Increment 5 functional (n=6) | Baseline parse | Candidate parse | Binding (both) |
|---|---|---:|---:|---:|---:|
| scale | 0/6 | 0/6 | 1 | 1 | 0 |
| join | 0/6 | 0/6 | 0 | 1 | 0 |
| compose | 0/6 | 0/6 | 0 | 1 | 0 |
| compose_second | 0/6 | 0/6 | 0 | 0 | 0 |
| filter_count | 0/6 | 0/6 | 1 | 0 | 0 |
| long_words | 0/6 | 0/6 | 0 | 1 | 0 |
| is_even | 0/6 | 0/6 | 2 | 1 | 0 |
| sum_positive | 0/6 | 0/6 | 4 | 1 | 0 |
| **Aggregate (48)** | **0/48 (0%)** | **0/48 (0%)** | **8/48 (16.7%)** | **6/48 (12.5%)** | **0** |

Exact-match 0/48 both; structural 0/48 both. Case transitions: **both_fail 48**, improved 0, regressed 0, both_pass 0. Raw evidence: `increment5-functional-100/evaluations/stabilization_analysis.json`.

Interpretation: on novel identifiers and wording for the *same eight behaviours*, neither checkpoint binds a single parameter or produces a single functionally correct output. The baseline emits generic `left`/`right` (`return left * right:`), the candidate emits generic `text` (`return text.lower()`, `return text.`). There is no per-behaviour improvement to report because there is no functional success on either side to differentiate.

### 2.2 Frozen suites (existing evaluations, re-scored for conditional metrics)

| Suite | Metric | Baseline | Increment 5 |
|---|---|---:|---:|
| AD-1 (29) | Parse / exact / binding | 24 / 2 / 6 | 24 / 1 / **2** |
| C3 (54) | Parse / exact / binding / functional | 24 / 0 / 0 / 0 | 37 / 1 / 9 / **1** |
| C3 | Functional \| parse | 0/22 (0%) | 1/32 (3.12%) |
| C3 | Functional \| binding | — (0 binding) | **1/7 (14.29%)** |

Failure classification (Increment 5 candidate): AD-1 — incorrect binding 22/29; C3 — incorrect binding 28/54, syntax 17/54. Parse-but-fail-execution examples are the norm: e.g. `HELD-comprehension-000` (`[value for value in values if value > 0]`, binding credit, assertion failure), `HELD-edge-001` (same template, functional fail).

---

## 3. Investigation of the single C3 functional pass

The sole frozen functional pass is `HELD-string-000` ("returns the uppercase form"), candidate output `return text.upper()`.

- **Correct for the intended reason?** Yes, locally: the requested parameter `text` is used and `.upper()` implements the requested relationship.
- **Generalizes?** No. The supplemental suite contains no string-case behaviour (it covers the eight trained functional behaviours), but the same template fires on `E2-upper-001` (AD-1 exact) while the candidate fails **every** novel-name case in the supplemental suite — including arithmetic prompts where it emits `return text.lower()` / `return text.`. Small identifier changes break it completely.
- **Memorized/repeated template?** Yes, by all available evidence: `text.upper()` is the dominant upper template in both the binding and functional curricula; the Increment 1 candidate also passes this exact case and `E2-upper-001`; the baseline passes neither. One memorized template reproduced is not behaviour learning.
- **Boundary sensitivity:** the pass does not survive identifier changes (`text` appears in training-style signatures), and sibling string cases fail (`HELD-binding-004` upper vs lower confusion, `HELD-algorithm-002` reverse-string failure).

Conclusion: the single pass carries no generalization evidence.

---

## 4. AD-1 binding regression diagnosis

- Baseline binding wins: `E2-add-000, E2-add-002, E2-maximum-001, E2-subtract-001, E2-upper-001, E4-filter_positive_count-0`.
- Candidate binding wins: `E2-upper-001, E4-filter_positive_count-0` only. **Regressed: 4, gained: 0.**
- Concentration: all four regressions are arithmetic/conditional cases (`add`, `add`, `maximum`, `subtract`). On `E2-add-002` the baseline's correct `return left + right` becomes the candidate's `[value for value in values if value > 0]`; on `E2-maximum-001`/`E2-subtract-001` it becomes truncated `return text.`.
- Output-distribution shift (AD-1 templates): `direct_binop_return` 9→4, `if_name_return` 7→3, while `direct_method_lower_return` 3→8, `direct_listcomp_return` 2→6, `degenerate_listcomp_text` 0→3. The combined curriculum shifted mass from arithmetic/conditional templates toward string-method and comprehension templates, even though functional arithmetic behaviours were included — at `1e-5`, 100 steps moved the distribution without fitting the AD-1 name-copying relationship.
- Whether the functional portion caused this is not isolable from this single run (no ablation), but the direction is consistent with curriculum dilution: 96/252 functionally-explicit examples use type-specific names (`text`/`values`-style pools) that compete with AD-1's generic arithmetic names.

---

## 5. Evaluation and test quality

Supplemental suite limitations: 6 cases/behaviour gives ±~20pp uncertainty per behaviour; it is a direction/coverage probe, not a precise rate estimator. It uses novel identifiers precisely because the frozen suites are saturated with trained templates. No network, downloads, new licensing-relevant data, or frozen-suite changes. The functional evaluator's fail-closed subprocess behaviour is unchanged (one-line builtin fix from Increment 4 only).

---

## 6. Stabilization decision

Pre-stated thresholds (`evaluation/stabilization.py`, rationale in module docstring): AD-1 binding loss ≤1 (retained capability; 4 is not noise); per-behaviour functional regressions ≤1 of 8; strictly greater aggregate functional AND functional-given-binding; no collapse; no AD-1 parse fall.

Applied evidence:

- Functional improved on frozen C3: 2.22% vs 0%, conditional 14.29% vs 0% — but from a single memorized template, with **0/48 both checkpoints** on the supplemental suite.
- AD-1 binding loss **4**, exceeding the limit of 1.
- Behaviour regressions 0 (floor effect: both at zero) — no coverage loss, but also no coverage gain.
- No candidate collapse; AD-1 parse held.

**Decision: B — Improvement limited to narrow behavior; further controlled work required.**

Why not A: AD-1 binding retention fails and novel-surface functional success is zero. Why not C: the frozen functional improvement (0→1 pass, 0→14.29% conditional, C3 parses 24→37) is real and repeatable (deterministic decoding), so "no meaningful improvement" overstates the null. Why not D: functional accuracy did not regress and parse did not fall; the candidate is not worse overall, it is narrowly better on trained templates and worse on AD-1 binding.

The Increment 5 candidate must therefore be **retained as a temp artifact only** — not promoted, copied, or published.

---

## 7. Deliverables, tests, and preserved state

Added: `evaluation/stabilization.py`, `evaluation/behavior_suite.py`, `scripts/analyze_v120_stabilization.py`, `tests/test_stabilization.py` (9 tests: suite determinism/completeness, no frozen/training overlap, reference-answer correctness via safe gate, per-behaviour aggregation, decision matrix A–D), plus this report and `stabilization_analysis.json` under the Increment 5 temp directory. No tracked file was modified.

Test results: `tests/test_stabilization.py` **9 passed**; regression run of Increment 1–5 focused suites **74 passed**. All hashes in Section 1 re-verified identical after all work. No training, checkpoint modification, commit, push, merge, tag, release, package-version change, or publication occurred.

---

## 8. Recommended next action

Do not promote Increment 5. Any Increment 7 must be separately reviewed and should:

1. Restore AD-1 binding explicitly (arithmetic/paraphrase shapes with generic names) with a binding-loss acceptance floor of ≤1 case, tested *before* functional gains are credited.
2. Extend functional behaviours with identifier-novelty training (same behaviour, many unrelated names) since both checkpoints score 0 on novel names — the current curriculum teaches templates, not name-copying.
3. Reduce the comprehension/lower-method attractor's share and add counter-examples (string prompts must not yield comprehensions; arithmetic prompts must not yield `.lower()`).
4. Retain the stronger stabilization criteria (Section 6) with seed replication before any capability claim.

**STOP after reporting. No Increment 7 was started.**
