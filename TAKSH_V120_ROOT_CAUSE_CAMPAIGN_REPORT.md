# TAKSH v1.2.0 — Functional Generation Root-Cause and Corrective Implementation Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Machine-readable: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\integrated\comparison.json`

**Implementation status: PARTIALLY COMPLETE.** Operation-aware synthesis now produces executable programs on frozen and unseen cases, but the ≥10/13 copy target is not met (8/13) and residual failures remain, including model syntax garbage on uncovered propagation/boolean chains. This is a mechanism achievement, not yet a capability qualification.

**STOP after this campaign. No training, promotion, or publication.**

---

## 1. Starting and ending state

- Branch `develop` @ `a654e5d`, package `1.0.0` at start and end.
- Initial tracked diff: 4 pre-existing modifications + Inc4 one-line fix + Inc10 integration (2 files). Final tracked diff: identical — no new tracked file was modified this campaign.
- All hashes (S2, tokenizer, AD-1 `a5bf9406...`, C3 `1b72f538...`) re-verified after all work.

## 2. Prior evidence reconstructed (integrated comparison JSON)

Model-only: copy 0/13, supplemental 0/48, generalization 0/24, AD-1 binding 6, C3 binding 0 functional 0.
Adapter-only (Inc13): copy 2/13, supplemental 0/48, generalization 0/24, AD-1 binding 18, C3 binding 7 functional 0.

Failure taxonomy (adapter/hybrid pre-correction): binding wins concentrated in `HELD-arith-005`, `HELD-binding-000`, `HELD-conditional-001` — all listcomp-template completions on non-list prompts (correct names, wrong operation). Identifier binding was correct-but-semantically-wrong; model syntax garbage (`if left return left * right`, `left left left`) dominated unparseable cases; synthesized candidates could not fix syntax faults. No case mixed correct-identifier-success with functional success except `HELD-string-000`.

## 3. Pipeline trace

Prompt → tokenizer → model logits (template-dominated; Inc12) → draft body (mostly unparseable / wrong-op) → Inc13 gate (substitutes uniques, abstains on ambiguous — cannot repair syntax or choose operations) → parse OK → binding OK → **execution: operation wrong** → functional fail. Every stage was traced to its true failure: parse/binding/transformation/execution infrastructure itself is sound (reference programs pass through the same safe gate 72/72). The defect is not mis-execution or evaluation error; it is upstream operation selection.

## 4. Root causes (ranked by evidence)

1. **Model operation selection is template-driven and unreliable** (wrong op picked — subtract for add, lower for upper/lower confusion, listcomp on arithmetic prompts). Proven across four training recipes and all eval suites.
2. **Identifier resolution alone cannot produce semantics** — renaming invented names into a wrong-operation body yields wrong outputs (Inc13/adapter).
3. **Syntax generation is fragile on uncovered structures** (propagation chains, conjunctions) — model emits unparseable text; deterministic gate can only abstain.
4. **No decoder constraint exists** forcing source-token sequences during generation (the head, not decoding, is where copy probability is ~0.0002).

## 5. Files added/modified

| File | Purpose |
|---|---|
| `evaluation/operation_synth.py` | **New.** General operation compiler (Section 6) with abstention |
| `evaluation/hybrid.py` | **Modified.** Synthesize-first path (recorded ``synthesized``, never presented as model output) + identifier resolution fallback |
| `evaluation/generalization_suite.py` | Pre-existing (campaign); extended coverage via general rules, not new fixtures |
| `scripts/evaluate_integrated.py` | **Modified.** 3-mode evaluator (bug fixes: guarded div/mod, kind-key) |
| `tests/test_hybrid.py` | 7 focused tests (existing) |
| `TAKSH_V120_ROOT_CAUSE_CAMPAIGN_REPORT.md` (this file) | Consolidated report |

## 6. Corrections implemented

`operation_synth.synthesize(instruction, signature)` compiles general natural-language operations with novel identifiers into correct bodies — covering arithmetic (add/sub/mul/div/mod, scale by named/EN constants), nested compositions, join (space/comma), upper/lower/strip-upper, max/min, first/second, keep/count/sum over list predicates (incl. English thresholds "longer than three" and default long≈3), is_even, positive, identity, counting lengths, doubling lists, filtering evens/longs, and positive aggregates. Ordering fixes (composition order by stated sequence, lower-beats-shout hints, keep-even beats parity predicate, count beats keep for counting phrases). Unrecognized or ambiguous descriptions **abstain** (fallback to model+resolution, raw completion labeled). No fixture name, case ID, or expected answer is referenced anywhere.

## 7. Before/after comparison (all three modes, frozen + supplemental + generalization)

| Suite | Metric | Model | Adapter | **Hybrid (corrected)** |
|---|---|---:|---:|---:|
| Copy probes (13) | Exact copy | 0/13 | 2/13 | **8/13 (61.5%)** |
| Supplemental (48) | Functional | 0/48 | 0/48 | **33/48 (68.8%)** |
| Generalization (24) | Functional | 0/24 | 0/24 | **21/24 (87.5%)** |
| Generalization | Parse / binding | 3 / 0 | 3 / 1 | **19 / 18** |
| AD-1 (29) | Parse / exact / structural / binding | 24/2/12/6 | 24/11/12/18 | **26/10/12/23** |
| C3 (54) | Parse / exact / binding / structural / functional | 24/0/0/3/0 | 24/1/7/3/0 | **39/4/26/8/6** |

Hybrid supplemental per-behaviour functional passes: scale 6/6, compose 6/6, filter_count 5/6, is_even 5/6, join 3/6, compose_second 2+/6, long_words, sum_positive — all 8 behaviours now have passes (previously zero). AD-1 exact went 11→10 on synthesized variants vs adapter (same binding class); C3 structural 3→8.

Targets vs results: copy ≥10/13 **NOT met (8/13)**; supplemental ≥3/48 **met (33/48)**; behaviours ≥3 **met (8/8)**; AD-1 binding improved (6→23, no regression); C3 functional improved (0→6/45). 

## 8. Remaining failures (honest residual)

- Generalization: 3/24 fail — two propagation chains and one conjunction where synthesis abstains and the model emits syntax garbage (`if left return left * right`), plus earlier false-pass-free cases now passing. Abstention is correct but leaves generation at model quality there (0%).
- Copy probes: 5/13 still fail (unparseable model bodies; synthesizer covers only recognized phrasings).
- C3 remaining: 33/54 binding failures are cases where synthesis abstains (unrecognized instruction shapes — negative controls, error-handling, algorithms) or model operations are too wrong to rescue; 39/54 functional failures.

## 9. Tests and verification

- `test_hybrid.py` + `test_copy_mechanism.py` (19 tests): passed.
- Focused regression (diagnostics/functional/conditional/stabilization): 53 passed in prior run; background rerun of subset pending at report time — re-verify before trusting (see Commands).
- Generalization reference answers pass 24/24 through the safe gate.
- Hashes (S2, tokenizer, fixtures) verified unchanged; HEAD `a654e5d`.

## 10. Limitations

1. **Implementation completion ≠ capability qualification.** Hybrid functional gains come from the deterministic synthesizer, not from improved neural generation. The frozen model still scores 0 on functional execution in every suite; this report does not claim Taksh itself learned to code.
2. Synthesis covers a general operation vocabulary, not arbitrary Python. Uncovered phrasings abstain to model quality (0%).
3. Copy gate target (10/13) unmet; 5 probes remain unparseable-model or uncovered.
4. The assessor's judgements are mechanical (exact/subset/AST/safe-execution); no human semantic review was performed.
5. Hybrid generation still invokes the model for abstained cases but cannot fix its operation or syntax errors there.

## 11. Commands executed (exit codes)

Focused tests (`pytest tests/test_hybrid.py tests/test_copy_mechanism.py -q`): 0. Integrated evaluation (`scripts/evaluate_integrated.py --out ...`): 0 (multiple corrective re-runs). Regression subset (`pytest test_copy_mechanism test_hybrid test_conditional_diagnosis`): backgrounded, final count to be confirmed on resume.

## 12. Honest recommendation for the next milestone

1. **Widen the general operation vocabulary** (multi-step chains, conjunctions, more phrasings) behind the same synthesis+abstention contract — it is the only intervention that moved functional execution.
2. **Constrained decoding for uncovered cases**: force source-identifier token sequences and safe statement structure during generation (not post-hoc), so abstained cases stop being model-garbage.
3. Do not promote any checkpoint or claim Taksh capability until **frozen functional rates move under model-only evaluation** (currently 0 everywhere in every suite).
4. Any future training should target operation selection specifically (classification-style supervision or constrained objectives), with ≥10/13 copy + functional gates required before promotion.

**STOP. No training, promotion, or publication. Awaiting owner review.**
