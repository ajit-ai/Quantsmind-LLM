# TAKSH v1.2.0 — Increment 2 Report: Curriculum Diversity and Mode-Collapse Prevention

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Curriculum digest before: `6a531633ba62f76806bef8722c49bdde33140d3a2a584177b8b0e4f2cde77c9e`
Curriculum digest after: `bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b`
Baseline checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 1 candidate SHA-256: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80` (unchanged)

Verdict: **IMPLEMENTATION COMPLETE — NO TRAINING RUN. READY FOR REVIEW BEFORE ANY INCREMENT 3 EXPERIMENT.**

This increment changed curriculum and evaluation code only. No model training was launched, no checkpoint was modified, and no frozen evaluation case was altered or used as training data.

---

## 1. Verified starting state

Verified before any edit:

- Branch: `develop`
- HEAD: `a654e5d` (`a654e5dc13dc77e925b3981d09792bade43a970a`)
- Package version: `1.0.0` from `taksh/__init__.py`
- Git status: four pre-existing tracked modifications, unchanged by this increment:
  - `serving/code_chat.py`
  - `serving/completion_server.py`
  - `tests/test_v200_training_stability.py`
  - `tests/test_v210_supervision_learning.py`
- Baseline checkpoint hash: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- Increment 1 candidate hash: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80`
- Artifact checkpoint copy hash: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`

`TAKSH_V120_INCREMENT1_REPORT.md` (especially §10), `training/binding_curriculum.py`, `scripts/evaluate_v120_increment1.py`, and `tests/test_binding_curriculum.py` were read before editing.

---

## 2. Files changed and implementation details

No existing tracked file was modified. `git diff --name-only` at the end still shows only the four pre-existing files above. Four new or rewritten untracked files contain all Increment 2 work:

| File | Change |
|---|---|
| `training/binding_curriculum.py` | Rewritten (Increment 2 revision). Four surface variants per operation, type-appropriate name pools, four instruction paraphrases per operation, rotated validation split, per-example variant metadata. Public API preserved (`TRAINED_OPERATIONS`, `build_training_examples`, `build_validation_examples`, `curriculum_digest`). |
| `training/curriculum_quality.py` | New. AST template classification, normalised shapes, `check_curriculum` safeguards, `diversity_report`, `detect_mode_collapse`, configurable `QualityThresholds`. |
| `scripts/evaluate_v120_increment1.py` | Extended. `aggregate()` now also reports `ad1_diversity`, `c3_diversity`, and `mode_collapse`; original parse/exact/structural/binding/functional keys unchanged. Same frozen AD-1/C3 cases, same prompts and decoding. |
| `tests/test_curriculum_quality.py` | New. Ten focused tests (Section 7). |

### 2.1 Curriculum diversity (`training/binding_curriculum.py`)

Each of the 13 operations now has **four semantically equivalent, structurally distinct bodies**, cycled deterministically (`variant = index % 4`). Examples:

- Arithmetic (`add`/`subtract`/`multiply`): direct `return a + b`; named intermediate (`total = a + b`); augmented assignment (`result = a; result += b`); operand-swapped or unaried form (`return b + a`, `return a + (-b)`).
- Conditioning (`maximum`/`minimum`): if/else; ternary; assign-and-update (`chosen = a; if b > chosen: chosen = b`); non-strict ternary (`a if a >= b else b`).
- Collection (`keep_positive`/`keep_negative`): list comprehension; explicit `for` + `append`; `result.extend(generator)`; indexed `while` loop.
- Text (`upper`/`lower`): direct method call; intermediate assignment; empty-string guard returning the input unchanged; f-string method call (`f"{text}".upper()`).
- Aggregate (`count_values`): `for` counter; `len(...)`; `sum(1 for ...)`; `while` counter. (`sum_values`): `for` total; `sum(...)`; `sum(item for ...)`; `while` total.
- Binding (`choose_first`/`choose_second`): direct return; intermediate assignment; docstring + return; assign-and-conditional return.

Further variation:

- **Type-appropriate parameters:** numeric pairs for binary ops, 20 string names for text ops, 20 sequence names for collection/aggregate ops (previously one pool was reused across types).
- **Prompt wording:** four paraphrases per operation (was three).
- **Boundary/constants in code:** empty-string guard for text ops; `0`/`1` counter constants; indexed loops bounded by `len(...)`.
- **Validation separation:** validation now also rotates function names (`function_offset=2`) and instructions (`instruction_offset=1`) in addition to a different name offset, and uses a distinct `V1-` ID prefix, so train/validation `(instruction, signature)` overlap is impossible by construction and verified by check.
- **Reproducibility:** fixed seeds preserved; `curriculum_digest()` covers IDs, instructions, signatures, bodies, and operations.

### 2.2 Safeguards (`training/curriculum_quality.py`)

- `template_family(signature, body)`: coarse AST family ignoring identifiers (e.g. `direct_binop_return`, `for_loop_name_return`, `while_loop_name_return`, `direct_ternary_return`, `assign_if_name_return`), with `unparseable` for invalid code.
- `normalised_shape(signature, body)`: fine AST shape with parameters masked by position, locals masked, function name masked, constants masked by type. Renamings share a shape; restructurings do not.
- `check_curriculum(train, val, thresholds, heldout_pairs)`: raises `CurriculumQualityError` naming the failed check and observed value. Checks: operation balance, global template balance, per-operation template diversity, per-operation variant dominance, list-comprehension dominance, exact duplicates and ID uniqueness, AST validity, parameter binding (including the `choose_first`/`choose_second` single-parameter rule and generic-identifier rejection via undefined names), train/validation leakage on both `(instruction, signature)` and `(signature, body)`, and optional frozen held-out overlap.
- Thresholds are configurable (`QualityThresholds`) and justified in the module docstring: `max_operation_share=0.12` (≈55% over the 1/13 uniform share), `min_templates_per_operation=3` (Increment 1 §10), `max_variant_share_within_operation=0.5`, `max_template_share=0.20`, `max_listcomp_share=0.10` (Increment 1 scored 0.154).
- `diversity_report(records)` and `detect_mode_collapse(records)`: evaluation-side template counting. Collapse requires one template on ≥30% of outputs, ≥5 cases, and ≥3 categories, or degenerate `for item in in ...` text on ≥3 records (never legitimate).

### 2.3 Evaluation tooling (`scripts/evaluate_v120_increment1.py`)

`aggregate()` additionally returns per-suite template-diversity statistics and a mode-collapse verdict, computed from the same scored records. No decoding, prompt, suite, or execution change was made. The `sum`/`list` builtins used by new variants were already in the existing analyzer's builtin set, so no analyzer change was needed (an earlier draft used `filter`, which was replaced by `extend` for exactly this reason).

---

## 3. Before/after curriculum distribution statistics

Training set size and balance preserved: 156 train / 52 validation, 13 operations × 12 train.

| Statistic | Before (Increment 1) | After (Increment 2) |
|---|---:|---:|
| Curriculum digest | `6a531633...` | `bc533a18...` |
| Train / validation examples | 156 / 52 | 156 / 52 |
| Examples per operation | 12 (all 13) | 12 (all 13) |
| Instruction paraphrases per operation | 3 | 4 |
| Distinct function names per operation | 6 | 6 |
| Parameter name pools | 1 shared pool | 3 type-appropriate pools |
| Distinct normalised shapes (global) | 13 (1 per operation) | 52 (4 per operation) |
| Distinct global template families | 9 | 15 |
| List-comprehension examples | 24/156 (15.4%) | 6/156 (3.85%) |
| `for item in` token occurrences | 48 | 30 |
| Max single-template global share | 15.4% (`direct_listcomp_return` plus `direct_name_return` families at 12 each) | 17.3% (`assign_name_return`, 27/156) |
| Single shape within one operation | 100% (12/12) | 25% (3/12) |

Family distribution is unchanged (arithmetic 36, conditioning 24, collection 24, text 24, aggregate 24, binding 24), which is intentional: diversity was added *within* each operation rather than by reweighting operations.

---

## 4. Examples per operation and output-template family

After (training set, 156 examples):

| Operation | n | Template families |
|---|---:|---|
| `add` | 12 | `direct_binop_return`, `assign_name_return`, `augassign_name_return` |
| `subtract` | 12 | `direct_binop_return`, `assign_name_return`, `augassign_name_return` |
| `multiply` | 12 | `direct_binop_return`, `assign_name_return`, `augassign_name_return` |
| `maximum` | 12 | `if_name_return`, `direct_ternary_return`, `assign_if_name_return` |
| `minimum` | 12 | `if_name_return`, `direct_ternary_return`, `assign_if_name_return` |
| `keep_positive` | 12 | `direct_listcomp_return`, `for_loop_name_return`, `assign_name_return`, `while_loop_name_return` |
| `keep_negative` | 12 | `direct_listcomp_return`, `for_loop_name_return`, `assign_name_return`, `while_loop_name_return` |
| `upper` | 12 | `direct_method_upper_return`, `assign_name_return`, `if_name_return` |
| `lower` | 12 | `direct_method_lower_return`, `assign_name_return`, `if_name_return` |
| `count_values` | 12 | `for_loop_name_return`, `direct_len_call_return`, `direct_sum_call_return`, `while_loop_name_return` |
| `sum_values` | 12 | `for_loop_name_return`, `direct_sum_call_return`, `while_loop_name_return` |
| `choose_first` | 12 | `direct_name_return`, `assign_name_return`, `docstring_direct_name_return`, `assign_if_name_return` |
| `choose_second` | 12 | `direct_name_return`, `assign_name_return`, `docstring_direct_name_return`, `assign_if_name_return` |

Global template counts: `assign_name_return` 27, `direct_binop_return` 18, `direct_sum_call_return` 9, `augassign_name_return` 9, `direct_ternary_return` 12, `assign_if_name_return` 12, `if_name_return` 12, `for_loop_name_return` 12, `while_loop_name_return` 12, `direct_listcomp_return` 6, `direct_method_upper_return` 6, `direct_method_lower_return` 6, `direct_name_return` 6, `docstring_direct_name_return` 6, `direct_len_call_return` 3. Total 156.

Every operation has 3–4 template families and exactly 4 normalised shapes. The 17.3% `assign_name_return` share is below the 20% cap but is the largest family (it collects the "named intermediate" variant from eight operations); it is a watch item for Increment 3 (Section 10).

---

## 5. Duplicate, leakage, and AST-validity results

From `check_curriculum(train, val)` on the new curriculum:

- Exact duplicates: **0**
- Train/validation leakage on `(instruction, signature)`: **0**
- Train/validation leakage on `(signature, body)`: **0**
- Invalid Python examples: **0** (all 208 train+validation examples parse as one function)
- Binding failures (undefined names or missing/replaced parameters): **0**
- All 208 examples were additionally **executed** against fixed sample inputs per operation in the tests, and every surface variant returned the expected value.

The checks were also proven to fire: a single-template corpus is rejected by the template-diversity check, a comprehension-only corpus by the list-comprehension check, an appended duplicate by the duplicate check, and a train example passed as validation by the leakage check (Section 7).

---

## 6. Mode-collapse detection test results

- **Increment 1 failure shape detected:** six synthetic records reproducing the observed degenerate completions (`return [value for item in in values if item > 0]`, etc.) across six categories → `collapsed: true`, degenerate count 6, with reasons naming the degenerate template. The evaluator's `mode_collapse` verdict on the same records is `true` for both suites.
- **Diverse outputs not flagged:** six records with six different templates across six categories → `collapsed: false`, 6 distinct templates, 0 degenerate.
- **Single-category safeguard:** five legitimate list-comp outputs in one category → `collapsed: false` (category-spanning rule prevents false positives).

---

## 7. Test commands and actual results

```powershell
python -m pytest tests\test_binding_curriculum.py tests\test_curriculum_quality.py -q -p no:cacheprovider
# 13 passed in 10.22s (3 pre-existing Increment 1 tests + 10 new Increment 2 tests)

python -m py_compile training\binding_curriculum.py training\curriculum_quality.py scripts\evaluate_v120_increment1.py tests\test_curriculum_quality.py
# COMPILE OK

python scripts\run_v120_increment1.py --root <increment1-root> --run-id increment1-binding-240 --steps 240 --lr 0.0001 --seed 20261024 --min-available-gib 0.0
# dry_run only (no --train): prints 156 train / 52 validation and digest bc533a18...; writes no checkpoint
```

During development, four defects were found by the tests and fixed before final verification: the builder's validator now allows `filter`-family builtins (unused after the `extend` rewrite), the template classifier distinguishes assign-plus-conditional from plain conditional, the list-comprehension counter summed template names instead of counts, and a `choose_*` variant that mentioned the ignored parameter was rewritten to use only the requested parameter. The final run above is the passing state. The full repository test suite was not run, per the focused-tests instruction.

---

## 8. Confirmation that no training was run

- No training command with `--train` was executed in this increment. The only run of `run_v120_increment1.py` was a dry run (prints the plan, writes nothing).
- No new checkpoint file was created. The experiment root still contains only the Increment 1 run.
- Both checkpoint hashes were re-verified after all edits and match the starting values in Section 1.

---

## 9. Confirmation that frozen data and pre-existing work are preserved

- Frozen C3 suite `evaluation/python_benchmark_heldout_v1.json` SHA-256: `1b72f5388d86372582f03ae8cdf78ca031bfe34d27c0e3d9bbdc7cac4532a26e` (read-only use; never written).
- Frozen AD-1 suite SHA-256: `a5bf9406f49335fe0c2d8e1bff4be1c6cdbb4a88e8bd4a866f262457420ca45e` (read-only use; never trained on).
- `git diff --name-only` still lists only the four pre-existing tracked modifications. No reset, clean, stash, revert, or discard was performed.
- No commit, push, merge, tag, release, package-version change, external weights, external dataset, or paid service.

---

## 10. Recommendation for a separate Increment 3 experiment

Increment 3 (training) requires explicit authorization and must not reuse the Increment 1 recipe. Recommended protocol:

1. **Pre-flight gate:** run `check_curriculum` on the exact train/validation builders and record the digest. Refuse to train if any check fails.
2. **Conservative settings:** continued training from the frozen S2 checkpoint at `lr=3e-5` for at most 120 steps (Increment 1 used `1e-4` / 240 and collapsed).
3. **During/after training:** gate promotion first on parse rate (candidate ≥ baseline on both AD-1 and C3), then on exact/binding/functional. Run `detect_mode_collapse` on the held-out outputs; any `collapsed: true` verdict rejects the candidate regardless of other metrics.
4. **Watch item:** `assign_name_return` is 17.3% of the new curriculum. If Increment 3 still drifts toward named intermediates, split the intermediate variant (e.g. different local-name roles) or lower its share.
5. **Coverage gaps worth adding in Increment 3 data (not in this increment):** string-join and simple scaling operations, which `HELD-binding-000` and `HELD-arith-000` expose as missing. Adding operations changes the 13-operation counts asserted by the Increment 1 test, so it should be a deliberate, reviewed change.
6. **Keep the comparison pre-registered:** reuse `scripts/evaluate_v120_increment1.py` with the same prompts, decoding, and frozen suites so Increment 3 numbers are directly comparable to Increment 1 Section 6.

**Stop after Increment 2. Do not begin Increment 3 or launch training without review and explicit authorization.**
