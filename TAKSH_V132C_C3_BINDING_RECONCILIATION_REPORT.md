# TAKSH v1.3.2-C — C3 Binding Gate Reconciliation

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a` (verified), package version `1.0.0`
Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (verified unchanged)
Tokenizer SHA-256: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (verified unchanged)
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (verified unchanged)

Verdict: evidence-first diagnostic. No source code, fixture, checkpoint, tokenizer, dependency, or version change was made in this campaign. The C3 binding discrepancy is reconciled as a **reporting discrepancy (E)** in the v1.3.1 report text, **not** a v1.3.2-B regression (A). The 28-point gate is **not genuinely satisfied**: both the v1.3.1 and v1.3.2-B machine-readable runs record 27 binding passes on identical C3 completions. No speculative fix was implemented.

## 1. Preflight preservation evidence

- `develop` @ `a654e5d`, package `1.0.0`; working tree contains pre-existing tracked/untracked changes plus prior campaign deliverables, all preserved. No reset, clean, stash, checkout, revert, or discard was used. No commits, pushes, merges, tags, releases, or publication occurred.
- The only artifacts created by this diagnostic are this report and `Temp\opencode\v132c\c3_binding_comparison.json`.

## 2. Historical v1.3.1 C3 evidence

The v1.3.1 report (`TAKSH_V131_COMPLETION_REPORT.md` §5) tabulates the v1.2.4/v1.3.1 hybrid C3 row as **39 parse / 28 binding / 8 functional**, and cites machine-readable evidence at `Temp\opencode\v120\integrated\qualification_results.json`. That artifact was inspected directly:

- Hybrid C3: 39 parses, **27 binding**, **9 functional** (54 cases) — the artifact does not contain 28 binding.
- The same report's C3 functional figure (8) also differs from its cited artifact (9), indicating the report's C3 row was transcribed, not read from the artifact. The report's own v1.2.0-hybrid cell was also left as an ambiguous literal "39/26-28/6".

## 3. v1.3.2-B C3 evidence

The v1.3.2-B run (`Temp\opencode\v132b\qualification_results.json`, hybrid mode, same `qualify_v123.py` harness and seed) records C3: 39 parse / **27 binding** / 9 functional. Case-level comparison against the historical artifact (below) shows C3 completions and binding decisions are **identical** — zero changed cases.

Reproduction (bounded; C3 binding only, no full suite re-run): the v1.3.2-B hybrid C3 completions were re-scored through the current `score_c3_case` path (`scripts/evaluate_v120_increment1.py` → `score_static`) against the frozen `python_benchmark_heldout_v1.json` fixture. Re-scored binding decisions match the recorded decisions for all 54 cases; re-scored binding count = **27**. The evaluator and scoring rules are unchanged between the two runs.

## 4. Case-by-case comparison (v1.3.1 artifact vs v1.3.2-B)

C = completion identical, B = binding pass, F = functional pass. Every case matches.

| Case | B (v131) | B (v132b) | C | F (v131) | F (v132b) |
|---|---|---|---|---|---|
| HELD-arith-000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-arith-001 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-arith-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-arith-003 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-arith-004 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-arith-005 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-binding-000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-binding-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-binding-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-binding-003 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-binding-004 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-conditional-000 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-conditional-001 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-conditional-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-conditional-003 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-loop-000 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-loop-001 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-loop-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-loop-003 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-loop-004 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-mapping-000 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-mapping-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-mapping-002 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-mapping-003 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-mapping-004 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-string-000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-string-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-string-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-string-003 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-comprehension-000 | ✗ | ✗ | ✓ | ✓ | ✓ |
| HELD-comprehension-001 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-comprehension-002 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-nested-000 | ✓ | ✓ | ✓ | — | — |
| HELD-nested-001 | ✓ | ✓ | ✓ | — | — |
| HELD-nested-002 | ✓ | ✓ | ✓ | — | — |
| HELD-composition-000 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-composition-001 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-composition-002 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-algorithm-000 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-algorithm-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-algorithm-002 | ✓ | ✓ | ✓ | ✓ | ✓ |
| HELD-error-000 | ✓ | ✓ | ✓ | ✗ | ✗ |
| HELD-error-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-edge-000 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-edge-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-negative-000 | ✗ | ✗ | ✓ | — | — |
| HELD-negative-001 | ✗ | ✗ | ✓ | — | — |
| HELD-negative-002 | ✓ | ✓ | ✓ | — | — |
| HELD-negative-003 | ✗ | ✗ | ✓ | — | — |
| HELD-negative-004 | ✗ | ✗ | ✓ | — | — |
| HELD-negative-005 | ✗ | ✗ | ✓ | — | — |
| HELD-ambiguous-000 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-ambiguous-001 | ✗ | ✗ | ✓ | ✗ | ✗ |
| HELD-ambiguous-002 | ✓ | ✓ | ✓ | ✓ | ✓ |

Full machine-readable rows (54 cases with prompts' completions, binding, functional) are in `Temp\opencode\v132c\c3_binding_comparison.json`.

## 5. Reproduction methodology and results

1. Loaded the historical hybrid C3 records and the v1.3.2-B hybrid C3 records; compared case IDs in order (identical), `completion` strings (0 differences), `binding` decisions (0 differences), and functional outcomes (identical, 9 passes).
2. Re-scored v1.3.2-B completions through the current `score_c3_case` against the frozen fixture: 27 binding, all decisions matching — confirming evaluator consistency.
3. Checked the v1.3.2-B synthesizer operations actually fired on C3 prompts: none of the new v1.3.2-B operations (join_length, count_even, slice_without_last, product_all, is_absent, upper_equals, difference_at_least, duplicate_value, precedence_add_mul, propagation_strip, ordered compose rules) appear in any C3 synthesis, so the implementation cannot have altered C3 outputs.
4. The independent functional result is untouched: the same comparison shows old 18/40 → new 39/40 functional; C3 stability coexists with the independent-suite improvement, as expected since the new rules target independent-suite mechanism families absent from C3.

## 6. Root-cause classification

- **A (genuine v1.3.2-B regression): ruled out.** Completions are byte-identical for all 54 C3 cases; v1.3.2-B does not touch C3 behavior.
- **B (pre-existing baseline drift): ruled out as the cause of the 27-vs-28 gap.** There is no drift between the historical and v1.3.2-B artifacts — both runs match exactly. The differing figure is the v1.3.1 report text, which disagrees with its own cited artifact.
- **C (evaluator/scoring change): ruled out.** Re-scoring through the current scorer reproduces both runs' decisions exactly.
- **D (different inputs/configuration): ruled out.** Same fixture, same harness (`qualify_v123.py` hybrid mode), same seed, same case order.
- **E (reporting/aggregation discrepancy): supported.** The v1.3.1 report's C3 row (39/28/8) does not match the artifact it cites (39/27/9). The probable mechanism is manual transcription (binding read as 28; functional recorded as 8 while the artifact shows 9). The v1.3.2-B report's "baseline drift" characterization from the previous campaign is superseded by this direct artifact comparison — the correct prior value is **27**, and the v1.3.2-B result of 27 is a faithful reproduction of the v1.3.1 hybrid.

**Is the 28-point gate genuinely satisfied?** No. On the verified artifacts the system scores 27/54 binding in both campaigns; no comparable evidence supports 28 for the v1.3.1 hybrid under this harness. The gate threshold is left unchanged; the claim "v1.3.1 scored 28" is corrected to "v1.3.1 scored 27 per its cited artifact; the report's 28 was a transcription error." Genuinely satisfying a 28 gate would require one additional binding pass, which current outputs do not contain.

## 7. Impact on the 39/40 independent functional qualification

**None.** Impact assessment: the reconciliation concerns only the C3 binding metric and its historical citation. The independent functional 39/40 result, its case-level delta (18→39, zero regressions, one remaining Q-p1 failure), and its provenance are unaffected and remain qualified as reported in the v1.3.2-B completion report. The sole change this diagnostic makes to prior interpretations is the correction of the C3 baseline figure from 28 (report text) to 27 (cited artifact).

## 8. Recommended next action

**Request authorization for the smallest correction, stop before implementing:** if the owner wishes to genuinely satisfy the 28 gate, the smallest candidate is a single general C3 binding improvement that converts one of the 27 currently non-binding cases by emitting its parameters without invented names — e.g., the binding case HELD-conditional-002 (keep_long_words synthesis) or HELD-binding-001 (plausible but currently abstained-multiply). Such a rule must be a general mechanism (not a case keyed to the fixture), covered by unit tests, and re-qualified across all suites before any gate claim. No such change is authorized by this diagnostic.

## 9. Unavailable evidence

- **Unavailable:** any artifact supporting the v1.3.1 report's literal "39/28/8" C3 row. No machine-readable run cited in the repository records 28 for the v1.3.1 hybrid; the cited artifact records 27/9. If a different harness or mode produced 28 (for example the `hybrid122` leg of the qualification script, which the v1.3.1 table also references ambiguously), that artifact was not preserved in the locations examined and cannot be reconstructed without re-running a full suite (prohibited in this diagnostic). The reconciliation therefore rests on the preserved comparable artifacts, which are mutually consistent.
- **Unavailable:** a v1.2.0-hybrid C3 artifact resolving the report's "26-28" range; not needed for this gate comparison (v1.3.1 hybrid is the baseline).

## 10. Final verdict

**PASS** (diagnostic objective met: the discrepancy has an evidence-backed classification — reporting discrepancy (E), no v1.3.2-B regression — with unavailable evidence explicitly identified; preservation and freeze constraints held; the 39/40 qualification is unaffected; confirmed facts are distinguished from hypotheses throughout).

This verdict concerns only the diagnostic reconciliation. It does **not** describe the overall TAKSH product as GA-ready.
