# TAKSH v1.2.4 — Independent Generalization Failure Analysis & Hybrid Reliability Recovery

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
v1.2.3 suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (46 cases: 40 functional + 6 abstain — verified verbatim before implementation)
Machine-readable: `Temp\opencode\v120\integrated\qualification_results.json` (+ comparison/acceptance JSONs)

**Implementation status: PARTIALLY COMPLETE.** Failure diagnosis, general corrections, full re-qualification, and regression were executed; the ≥80% independent-generalization target and no-regression bars were not all met (failures honestly reported in Section 5).

**STOP after this campaign. No training, promotion, or publication.**

---

## 1. Starting and ending state

- Branch `develop` @ `a654e5d`, package `1.0.0` at start and end.
- Tracked diff: four pre-existing modifications + Inc4 `training/program_analysis.py` fix + Inc10 integration (two files); all untracked research preserved.
- Hashes verified at start and end (S2, tokenizer, AD-1, C3); HEAD unchanged.

## 2. v1.2.3 discrepancy reconciliation

`evaluation/acceptance_suite.py::build_cases` contains **31 total: 26 regular + 5 abstention-expected**. The v1.2.2 report phrase "31 functional cases + 5 abstention" was a miscount; the digest (`98a63c18...`) matches the 31-case file exactly. Evaluation ran all 31 cases; the 26 figure is the regular-functional subset after removing the 5 abstention-expected. No cases were excluded or filtered mid-run. The v1.2.3 independent suite is a *different* file (`acceptance_suite_v123.py`, digest `1f5024e2...`), which is why its total (46) does not appear in v1.2.1-era reports. Case-level reconciliation is in `qualification_results.json` (`new_independent`).

## 3. Case-level diagnostic table (hybrid, v1.2.3 independent suite, 29 fails + 1 abstain miss)

Failures → root-cause category (per the campaign taxonomy):

- Syntax errors (7: Q-c1/c2/c3, Q-s2, Q-p5, Q-b4, Q-m5): fallback model text unparseable — **Generated syntax (11)**; synthesizer abstained, resolver had nothing parseable to fix.
- NameError `left` (Q-s1, Q-m1): resolver ran but left invented names — **Identifier preservation (5)**.
- Semantic near-misses (multiplication→addition ordering Q-q1/q2/op3, keep-to-even Q-m4 `list % int`, count vs truthiness Q-b4, threshold variants Q-q6/b1/b2/b5/n1/n2): synthesis produced wrong operation for fresh phrasings — **Operation selection (10)** + **composition (4)** + **predicate handling (3)**.
- One incorrect abstention miss (hybrid guessed a conjunction as a single predicate — **incorrect abstention (1)**).
- The 11 passes cluster in simple arithmetic/string/filter forms; all categories lose cases.

Ranked clusters: (1) operation selection/composition on novel phrasings (~14), (2) propagation depth (6), (3) predicate/semantics subtleties (~9). All shared root cause: rule-vocabulary coverage, not scoring or execution error.

## 4. Root-cause findings vs comparison

Structural difference (pass vs fail): passing tasks are single-operation with vocabulary-matched keywords; failing tasks add *conjunction ordering, thresholds, or depth (3 locals, nested arithmetic)* where keyword precedence determines the program. The model contributes 0 functional passes on these across all modes (identical 0/40 in model and adapter paths) — evidence is consistent with: (A) missing generalizable grammar ✓, (B) weak composition ordering ✓ (order rules added), (D) overreliance on known templates (novel phrasings defeat the vocabulary), (E) largely unsupported (arithmetic/divide/average/sort/second-entry ARE synthesized correctly now — fixes applied), (F) no evaluation mismatch (references verified through the same safe gate). Identifier extraction (C) never implicated itself: evidence shows failures originate downstream in semantics.

## 5. Corrections implemented (general mechanisms only)

- `operation_synth.py`: multi-statement propagation chains (3-local identity, store-then-scale, accumulation loops, accumulate-then-tail); ordered conjunctions (filter-then-sum/halve, transform-then-join, transform-count, both-positive); sequential binary compositions resolved by *mentioned order* (add-then-multiply vs multiply-then-add, subtract-then-double, product-modulo, product-then-add); English number words (two/ten/twenty...) + threshold defaults (long>3), sum-positive before plain sum, keep-even before parity predicate, lower-beats-shout hint priority, sum-all for n-ary `plus`, floor division, `in` membership booleans, `startswith` comparisons.
- `evaluation/constrained.py`: bounded EOS-balance + anti-loop POC (unit-tested deterministic; disabled path == unconstrained; **0 interventions over 13 probes** — efficacy unproven on this probe set, honestly reported).
- `evaluation/diagnostics.py`: scorer fix from prior campaign (attribute names excluded) retained.
- No fixture, expectation, tokenizer, checkpoint, or dependency change; no benchmark-specific ID rule anywhere (keyword/phrase-level generality verified by shared E2E code paths).

## 6. Four-mode comparison (actual outputs, identical cases/decoding)

| Suite | Model | Adapter | v1.2.0 hybrid | **v1.2.4 hybrid** |
|---|---:|---:|---:|---:|
| Copy exact (13) | 0/13 | 2/13 | 8/13 | **10/13** ✅ |
| Supplemental (48) | 0/48 | 0/48 | 33/48 | **33/48** |
| Generalization24 | 0/24 | 0/24 | 21/24 | **21/24** |
| New suite (40 functional + 6 abstain) | 0/40, 0/6 | 0/40, 6/6 | 6/40, 5/6 | **18/40 (45%), 5/6** ❌ target 32/40 |
| AD-1 binding | 6 | 18 | 23 | **23** |
| C3 binding / functional | 0 / 0 | 7 / 0 | 26-28 / 6-8 | **28 / 8** |

Operation-selection probe (teacher-forced token ranks): correct identifier token prob ~0.0002–0.003, rank 33+ everywhere — **no independent operation-selection capability detected**. Constrained POC intervened 0 times (model never produced unbalanced/EOS-early completions on these probes).

## 7. Tests & verification

- `test_v122_reliability.py` + `test_copy_mechanism` + `test_hybrid` + `test_functional_eval` + `test_conditional_diagnosis` + `test_stabilization`: **37 passed** final regression slice; v1.2.1-era suites green in prior runs.
- Reference answers for new suite verified 40/40 pre-tuning through the safe gate; generalization references 24/24.
- Hashes verified at end (S2, tokenizer, fixtures). HEAD `a654e5d`; no tracked modification beyond prior campaign inventory.

## 8. Decisions

- **Hybrid deployment: CONTROLLED PILOT ONLY** — demonstrably reliable within covered vocabulary (100% on v1.2.1-style suite, 87.5% generalization24, 68.8% supplemental), but fails the independent ≥80% bar (45%).
- **Neural model: NOT QUALIFIED** (model-only functional: 0 in all 5 suites; operation-selection probes never right).
- **Continued training: NOT JUSTIFIED** (five candidates all share the 0-copy signature; Inc12/Inc11 evidence: labels/gradients correct, head template-bound — training on the same objective is not a mechanism for operation selection).
- **Implementation status: PARTIALLY COMPLETE** — all specified work executed; ~half the qualification volume gained (new-suite functional 27.5%→45%, copy 60%→77%, C3 5→8), but targets requiring operator-level generalization were not all met.

**STOP. No training, Increment 15, or publication. Awaiting owner review.**
