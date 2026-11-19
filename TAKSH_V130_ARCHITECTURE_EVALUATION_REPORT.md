# TAKSH v1.3.0 — Architecture Evaluation & Product Direction Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (46 cases)
Machine-readable: `Temp\opencode\v120\integrated\qualification_results.json`

**Campaign status: PARTIALLY COMPLETE.** Full evaluation, prototype, and regression executed; primary architecture recommendation is made, but independent-generalization and neural-qualification bars failed honestly.

**STOP after this campaign. No training, promotion, or publication.**

---

## 1. Baseline Git status (start and end identical)

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: 4 pre-existing + Inc4 fix + Inc10 integration; no new tracked file was modified this campaign (all work is in untracked files + planner prototype). Hashes verified at end: S2, tokenizer, AD-1, C3, supplemental-48, generalization-24, independent-40+6 — all match v1.2.4 evidence.

## 2. Failure architecture analysis (22 failures from v1.2.4 independent functional suite)

Classified by case-level evidence (Sec. 3-4 synthesis debugging):

| Root-cause class | Examples | Count |
|---|---|---:|
| Operation-family/selection misfires (rules) | product-modulo, compose-ordering, sum-vs-count, relative-threshold, both-positives, conjunction collisions | ~10 |
| Abstention overreach (false-positive syntax-safe refusals) | 1 abstain miss (hybrid guessed a conjunction) | 1 |
| Undefined/other syntax | model garbage on uncovered structures | ~4 rest |
| Vocabulary gaps (synth abstains) | 19/40 independent coverage gaps → model path inherits failure | ~7 orchards |

Shared structural cause: **keyword-rule precedence and coverage debt**, not evaluation/scoring error (references 46/46 verified through the gate). Rule blocks shadow each other (single-value rules fire on two-parameter conjunctions, `is_positive` vs both/conjunctions); the file is at a maintainability limit where each new rule changes global precedence.

## 3. Architecture evaluation

**A. Rule-based hybrid engine** (current): strongest measured option (v1.2.4 numbers: supp 33/48, gen 21/24, new suite 18/40 after v1.2.4's improvements, C3 functional 8, copy 10/13). Strengths: single `synthesize()` with audit; weaknesses: precedence conflicts, keyword collisions, abstains on thin vocabulary, maintenance complexity grows linearly with vocabulary.

**B. Structured operation planner — prototype implemented** (`evaluation/operation_planner.py`, typed `Plan`/`PlanStep` with dependency validation + codegen): evaluated head-to-head on the same suites:
- Supplemental: **18/48** (vs hybrid 33/48) — plan coverage smaller (35 planned, 13 unsupported)
- Generalization: planned 19/24, passed 12
- New independent: planned 21/40, passed 8
- Audit advantages real (explicit dependencies, unsupported-vs-invalid separation, literal-aware validation) but *functional correctness does not follow from explicitness* — coverage, not architecture, is the bottleneck. Prototype verdict: useful as an **audit layer**, not a replacement backend.

**C. Neural-assisted planner — evidence rejects**: Inc12 teacher-forced probes show correct-token probability 0.0002–0.003, rank 33+ everywhere; greedy generation is template-dominated; the head has *no* measurable operation-selection signal to propose. Architecture C is technically infeasible on this checkpoint without training; fabricating a neural path is prohibited.

## 4. v1.2.4 improvements (implemented, re-qualified)

General corrections (no fixture keying): propagation chains (3-local identity, store-scale, accumulate-tail), conjunction ordering by *mentioned order* (add-then-multiply vs multiply-then-add, subtract-then-double), filter-then-haggregate boundaries, halving (//2), product-modulo, un-covered string ops (stripUpper, wordsplit first/last), counting thresholds incl. English numerals, sum-positive-before-sum, keep-even-before-parity, lower-beats-shout hint priority, sum-all for n-ary `plus`, translate English percent factor, count-long-words default 3. Also: diagnostics scorer bug fixed prior campaign (attribute names excluded), both-positive precedence fix, parallel-range (mul-add optional constant) de-confliction.

Four-mode final numbers (actual outputs):

| Suite | Model | Adapter | v1.2.0 hybrid | **v1.2.4 hybrid** |
|---|---:|---:|---:|---:|
| Copy exact (13) | 0/13 | 2/13 | 8/13 | **10/13 ✅** |
| Supplemental functional (48) | 0/48 | 0/48 | 33/48 | **33/48** |
| Generalization24 functional | 0/24 | 0/24 | 21/24 | **21/24** |
| **New independent (40+6)** | 0/40, 0/6 abstain | 0/40, 6/6 | (via v1.2.2: 6/40, 5/6) | **18/40 (45%), 5/6 abstain** |
| AD-1 binding | 6 | 18 | 21 | **23** |
| C3 binding / functional | 0 / 0 | 7 / 0 | 26-28 / 6-8 | **28 / 8** |

New-suite ≥80% target **missed** (45%). Residual failures cluster on phrasing templates outside rule vocabulary + tree depths the VOCAB grammar lacks.

## 5. Supported Python task boundary

| Family | Classification | Evidence |
|---|---|---|
| Arithmetic (binary/trinary, div/mod) | BOUNDED SUPPORTED | 6/6 scale+arithmetic cases; copy/adapt hybrids consistent |
| String transforms | BOUNDED SUPPORTED for covered methods | upper/lower/strip-join; replace excluded (gate) |
| Sequence filtering/aggregation | SUPPORTED | keep/sum/count, 60–90% region |
| Conditional logic | LIMITED | max/min/even/positive + both-positives boolean conjunctions |
| Multi-step function generation | LIMITED–EXPERIMENTAL | 2-step locals track; depth-3 chains need rule growth |
| Loops/accumulators | LIMITED | enumeration/for with safe-gate constructs only |
| Nested data structures | EXPERIMENTAL | word-list ops only |
| Class/OOP generation | UNSUPPORTED | no implementation evidence |
| File ops / external libraries | ABSTAIN | safe gate forbids; no safe execution basis |
| General Python | ABSTAIN | outside covered grammar; fuzzy semantics fail closed |

## 6. Decisions

- **Primary architecture: Bounded hybrid with planner.** Keep the rule engine as generation backend; adopt the structured Plan IR as an *audit/validation layer* (dependency resolution, unsupported-vs-invalid separation, provenance) — it weakens to coarser coverage but sharpens transparency. Neither a pure rule org (maintenance already at limit) nor a neural planner (no measurable model proposals) is defensible alone.
- **Neural model: NOT QUALIFIED** (model-only functional: 0 in all suites; operation-selection probes never right).
- **Continued training: NOT JUSTIFIED** (five candidates share the 0-copy signature; Inc11 labels/gradients correct; head template-bound).
- **Hybrid deployment: CONTROLLED PILOT ONLY.**
- **Evaluation gates: copy 10/13 MET; supplemental-behaviour 8/8 met; generalization-40: 18/40 (45%, target 32/40) MISSED.**

## 7. Verification & resources

APC additions pass: planner unit tests (6), regression slices (30+11) green. Planner/head-to-head numbers Section 3. Machine-readable evidence (case-level outcomes, op-selection top-tokens, hashes, commands) in `qualification_results.json`. CPU-only, single small-model loads; all checkpoint/tokenizer/fixture hashes intact; HEAD `a654e5d`; no commits/tags/releases/publication.

**STOP after this evaluation. Awaiting owner review.**
