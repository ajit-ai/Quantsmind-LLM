# TAKSH v1.2.3 — Independent Qualification & Neural Operation-Selection Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
New suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (46 cases: 40 functional + 6 abstain; frozen before implementation tuning)
Machine-readable: `Temp\opencode\v120\integrated\qualification_results.json` (+ comparison/acceptance JSONs from v1.2.x runs)

**Implementation status: PARTIALLY COMPLETE.** All implementation, integration, testing and qualification work in this assignment executed; definition-based targets were largely met, but the ≥80% new-suite bar failed badly (11/40, 27.5%), so independent generalization is NOT demonstrated at the required level.

**STOP after this campaign. No training, promotion, or publication.**

---

## 1-2. Starting state & v1.2.2 discrepancy reconciliation

Verified branch `develop` @ `a654e5d`, package `1.0.0`, all checkpoint/tokenizer/fixture hashes, tracked diff = four pre-existing + Inc4 fix + Inc10 integration; all untracked research preserved.

Discrepancy root cause: `evaluation/acceptance_suite.py::build_cases` contains **31 cases total: 26 regular + 5 abstention-expected**. The v1.2.2 report's "31 functional + 5 abstention" phrasing was wrong; the digest (`98a63c18...`) matches the 31-case data exactly, the evaluator executed all 31 (26 regular filtered for functional scoring, 5 for abstention), and the 21→26 pass progression within this campaign was re-verified (now 26/26). No cases were excluded or filtered mid-run.

## 3. New independent suite

`evaluation/acceptance_suite_v123.py` — 46 cases (40 functional + 6 abstain), novel identifiers/structures/depths across copy, substitution, propagation, composition, operation, filtering, aggregation, booleans, string, collection. References verified 46/46 through the safe gate before any rule fitting. Separate from AD-1/C3/supp-48/gen-24/accept-31/copy-13. Full case table is embedded in `qualification_results.json` (`new_independent`).

## 4. Neural operation-selection (model-only, logits inspected)

Teacher-forced operation probes (top-token inspection): prompts for sum/difference/product/upper/lower/count/sum/even. Top predicted tokens after `return` never constitute the correct operation token sequence across all probes (e.g. sum prompt's top token is `␣return`/`␣right`, difference's is `␣left`); correct-operation token probability mass remains negligible under both familiar and unfamiliar identifier conditioning. Verdict: **no measurable independent operation-selection capability in the neural path.** Constrained POC logged **0 interventions over 13 probes** (model never tried unbalanced/EOS-early), so syntax-control efficacy remains unproven.

## 5. Comparison across modes (frozen + supplemental + generalization + new)

| Suite | Model | Adapter | v1.2.0 hybrid | **v1.2.3 hybrid (corrected)** |
|---|---:|---:|---:|---:|
| Copy exact (13) | 0 | 2 | 8 (8→10 after scorer fix) | **10** ✅ |
| Supplemental functional (48) | 0 | 0 | 33 | **33** |
| Generalization24 functional | 0 | 0 | 21 | **21** |
| New suite (46) | 0/40, 0/6 abstain | 0/40, 6/6 | (via v1.2.2: 21/26 v1.2.1-suite) | **11/40 (27.5%), 5/6 abstain** |
| AD-1 parse/exact/binding | 24/2/6 | 24/11/18 | 26/10/21 | **26/10/23** |
| C3 parse/exact/binding/structural/functional | 24/0/0/3/0 | 24/1/7/3/0 | 39/4/26/11/6 | **39/7/28/11/8** |

New-suite failures (hybrid) span 10 behaviours: aggregation, boolean, collection, composition, conjunction, copy, operation, propagation, string, substitution — the synthesizer's rule vocabulary does not cover this suite's fresh phrasings, and the model covers none of them. Abstention errors: hyphenated conjunctions ("Return True only when...") were still being guessed as single predicates in one non-abstain case.

## 6. Corrections implemented & tests

Added/changed (no tracked file modified): `evaluation/acceptance_suite_v123.py` (new), `evaluation/constrained.py` (bounded POC), `evaluation/operation_synth.py` (general expansions: propagation depth, conjunction ordering, halve/product-modulo precedence, English numerals, threshold defaults, sum-positive-before-sum), `evaluation/hybrid.py` (provenance fix earlier), `evaluation/diagnostics.py` (scorer attribute bug fix from prior campaign), `scripts/qualify_v123.py` (evaluator), `scripts/acceptance_v121.py` (alignment fixes), `tests/test_reliability_v122.py` (6).
Run: hybrid/copy/reliability suites **37 passed**; consolidated regression slice **52 passed** earlier.

## 7. Promotion decisions

| Capability | Decision |
|---|---|
| Hybrid deployment | **CONTROLLED PILOT ONLY** (binding/copy gains real; generalization bar missed) |
| Neural model | **NOT QUALIFIED** (model-only functional: 0 in all 5 suites; operation-selection probes never right) |
| Continued training | **NOT JUSTIFIED** (five candidates share the 0-copy signature; Inc12/Inc11 evidence: labels/gradients correct, head template-bound) |

## 8. Hash verification & resources

S2 `02cc86eb...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...`, supplemental digest unchanged, gen24 unchanged — all verified. CPU-only, single small-model loads. All checkpoint/tokenizer/fixture hashes intact; no training, commits, or publication.

## 9. Limitations

1. New suite is small (46) and design-informed by prior failures; it measures template novelty, not real-world task coverage.
2. Synthesis remains keyword-rule-based (no parser/grammar); uncovered en-US phrasings fail safe-side (abstain or model error).
3. Constrained decoding demonstrated zero interventions — syntax-error reduction is unproven.
4. The assessor's judgements are mechanical; no human semantic review was performed.
5. `left_value → source_amount`-family identifier familiarity still dominates probe classes; long-tail name generalization untested at scale.

**STOP after this campaign. Awaiting owner review.**
