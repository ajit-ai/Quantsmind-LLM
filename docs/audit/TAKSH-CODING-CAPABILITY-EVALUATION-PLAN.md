# TAKSH — Coding Capability Evaluation Plan (L1.0.0, 2026-11-12)

**Authorization:** A1.1-C / A2-E. **Nothing in this plan is executed today.** This document defines how a future, owner-authorized evaluation would be run. It records the facts that a qualification would need, the facts that exist, and the historical 0/30 result — which is preserved verbatim.

---

## 1. Current evaluation state (verified)

| Item | State |
|---|---|
| Frozen Python benchmark | **SHA-256 `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6`** — exists on the feature branch only; **absent from develop and main** |
| Benchmark content | 30 self-contained Python problems (`evaluation/l02_python_bench_v1.json`); immutable; no near-duplicates in training corpus (probed in L0.8.0-PRE: **zero collisions**) |
| Multilingual benchmark | **SHA-256 `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e`** — 64 tasks; feature-branch only; same absence on develop |
| Harness | `evaluation/eval_code.py` — imports `HUMANEVAL_PROBLEMS`/`MBPP_PROBLEMS`; distributed but never used to qualify a model here |
| Historical result | **0/30 correct on 2026-10-21** (M7 milestone), never re-run; preserved as historical record |
| TAKSH-KARK-BENCH | 0 instantiated tasks; KCC blocked (`0xC0000135`); Karkain DISCOVERED |
| Model availability | **MODEL_UNAVAILABLE** (no canonical artifact; v180_sft GEOMETRY_INCOMPATIBLE) |
| v180_sft capability measurement | **never performed** (blocked by checkpoint/preflight at L0.6.2; never re-attempted) |

## 2. Non-negotiable boundaries

1. This plan never mutates the frozen benchmark. If an artifact disagrees, report, never fix.
2. Contributors never edit tasks, prompts, expected outputs, or scoring code to make a model pass.
3. No capability claim without 3 reproducible seeds, fixed config, and full artifact lineage (checkpoint sha, tokenizer sha, generation seed, decoding temperature, prompt, benchmark sha).
4. No claim appears on a GA decision, release note, or marketing artifact from this plan alone.
5. Model-only and compiler-assisted/tool-assisted metrics are always reported separately — never merged.

## 3. Comparative protocol (when an artifact exists)

### 3.1 Baseline models

| Arm | Purpose | Authorization |
|---|---|---|
| `v180_sft` (28.8M, legacy lineage) | Historical in-repo baseline | owner sign-off required for any execution |
| `taksh-foundation-111m` (canonical) | Primary candidate | requires canonical artifact (currently missing) |
| External control (if owner authorizes) | Calibration anchor only, clearly labeled; no public comparisons | owner authorization required |

### 3.2 Configuration per run

```
prompt:        exact benchmark task text, no system/guided hints, seed=<n>
sampling:      temperature=0 (greedy) primarily; {temperature=0.2, top_p=0.95} as a secondary view
max_new_tokens: 512 (benchmark default), never exceeding the frozen harness limit
context:       2048 (canonical)
seeds:         n ∈ {1, 42, 999} — three independent runs minimum
hardware:      CPU for final deterministic runs; GPU accepted for intermediate diagnostics with identical seeds
runtime:       at least 3 full repetitions; pass-rate range recorded, not "best run"
```

### 3.3 Scoring rules (must not be tuned)

- A generation passes IFF: (a) it parses as Python (`ast.parse`), (b) it executes the task in the harness's sandbox without exceptions, (c) all assertions pass, (d) the compiled/returned value matches the expected contract per the frozen scoring logic.
- Partial credit: no. Binary per task.
- Syntax-valid generations: tracked separately (informational).
- Time-to-first-correct / repair iterations: tracked separately.
- Baseline comparison: v180_sft vs canonical on identical prompts/seeds; deltas with confidence intervals, never single-run claims.

### 3.4 Metrics

| Metric | Definition |
|---|---|
| pass@1 (greedy) | fraction of 30 tasks passed, exact greedy decode |
| pass@1 (sampled) | fraction passed under each seed; mean ± spread |
| syntax-validity rate | fraction of generations that parse |
| execution-success rate | fraction that run to completion |
| assertion pass rate | tests passed per task / total |
| repair success | after feedback loop, fraction recovered (DLCAE mode only) |
| latency/task (s) | measured under identical clock config |
| memory peak (MiB) | tracemalloc + process RSS |
| tool-assisted | same metrics with KCC/toolchain assistance (KARK-BENCH only; scope pinned) |

### 3.5 Evidence retention

Each evaluation run must produce: `reports/capability_eval/<date>/` with `{config, prompts, raw_generations.jsonl, scoring.json, environment.json, traces.jsonl}`; every file hashed (SHA-256) with a manifest; immutable after write. Retention: ≥ 3 years (owner rule).

## 4. Task inclusion/exclusion

| Class | Treatment |
|---|---|
| Frozen benchmark tasks | Included only; never edited, never regenerated |
| Near-duplicates found in any corpus in this program | **Excluded** from any training set (contamination guard, verified zero overlaps) |
| Adversarial/safety-flagged tasks | Included — safety capability is not a benchmark bypass |
| Non-Python tasks (Karkain) | Out of scope until KARK-BENCH has instantiated tasks and owner authorizes |
| Multi-file repository tasks | Out of scope for this milestone; defined for v2 track |

## 5. Karkain capability evaluation (separate gated protocol)

Preconditions (all required):

1. KCC reproducible on host/CI (`0xC0000135` resolved or documented workaround agreed).
2. TAKSH-KARK-BENCH v1.0.0 instantiated with **≥ 3 real Karkain tasks** with agreed bars.
3. KARK-BENCH harness green on synthetic fixtures.
4. Owner authorization for the Karkain track.
5. Karkain frozen corpus licensed (owner decision).

Metrics would mirror Python's but are never reported as "multilingual-ready" until all KARK-BENCH bars are met.

## 6. Minimum criteria for a coding-capability claim

A release claim "TAKSH can code" requires ALL of:

1. Canonical artifact exists and is traceable (fingerprint equality canonical config + tokenizer).
2. ≥ 3 reproducible seeds; no hidden seed selection.
3. Python benchmark **≥ 5/30 correct, greedy, no tool assistance**, documented as the measurement, not as product marketing.
4. Tool-assisted scores reported separately and always labeled.
5. v180_sft treated as prior art only, never as the GA candidate (incompatible).
6. Owner sign-off on every release ack; no self-promotion (DLCAE rule).
7. Capability statements in docs/changelogs carry exact hashes and run IDs.

## 7. What else must exist before any evaluation can run

- A canonical 111M artifact (currently missing — required).
- A develop-branch copy of the frozen benchmarks (absent; would need a dedicated authorized port) or an agreed location.
- Evaluator wiring for KCC if the Karkain track proceeds.
- `dev_tasks.json`-style synthetic development harness exists already (`FILE-ref: configs/l100eng3/dev_tasks.json`) — usable for pipeline, **never** for capability claims.
- No model is ever evaluated on a contaminated split.

## 8. Honest current position

No code has demonstrated Python coding capability beyond milestone-protected synthetic fixtures. The historical 0/30 result stands. This plan exists so that the moment an artifact does exist, a claim is either earned or refused — never assumed.
