# DLCAE Research Plan — A/B/C compared experiments

**Status:** experimental design specification (L0.6.0-R1). **None of these experiments has been executed.** DLCAE superiority over RAG is a hypothesis to be tested, never asserted.

Goal: identical tasks, identical source material, identical evaluation criteria, identical resource constraints — three arms:

- **A — Retrieval-based baseline (RAG):** external doc/spec retrieval + context injection into the base model. Retrieval accuracy and task performance are both measured.
- **B — DLCAE internal knowledge formation:** progressive comprehension → structured, versioned language knowledge → verification-gated controlled updates → held-out evaluation.
- **C — Hybrid:** DLCAE internal knowledge + *optional* external retrieval + foundation-model learning + compiler/execution feedback.

---

## 1. Common experimental protocol

1. Freeze the base model (`taksh-foundation-111m` v1, sha256-pinned) for all three arms.
2. Same held-out task suite (L0.2 Python benchmark + multilingual starter + TAKSH-KARK-BENCH stubs once instantiated).
3. Same prompt rendering (tokenizer contract v1, `render_for`); same decoding (greedy unless stated).
4. Same token/time budget for retrieval vs no-retrieval arms.
5. Identical scoring (AST gate → sandbox → unit-test pass rate; syntax-valid rate; category breakdown).
6. Report model-only and compiler-assisted results separately. Never mix.

## 2. Metrics (all nine, all measured)

| # | Metric | Definition | Instrument |
|---|---|---|---|
| 1 | Comprehension accuracy | % of L1–L5 checks passing on held-out probes | per-level probes |
| 2 | Knowledge retention | pass@1 on suite-A at t=0, t=1week, after new data | scheduled re-eval |
| 3 | New-language adaptation time | steps/hours from REGISTERED → first FUNCTIONALLY_QUALIFIED verdict | adaptation ledger |
| 4 | Generalization to unseen tasks | held-out (same style, unseen names/ids) | reserved split |
| 5 | Retrieval dependence | model-only vs retrieval-augmented delta | identical prompts ± context |
| 6 | Model-only performance | pass@1 with zero tooling | frozen benchmark, repair off |
| 7 | Full-system performance | pass@1 with DLCAE + KCC + execution | same benchmark, repair on |
| 8 | Regression / forgetting | delta on previously qualified language suite | re-run after every adaptation |
| 9 | Resource consumption | wall-time, CPU seconds, peak RAM, tokens | OS-level sampling |

## 3. Arm A — RAG baseline

- Corpus: Karkain SPEC.md (§-referenced), a fixed slice of approved docs (stdlib.md, memory-model.md), checksum-pinned set S.
- Retrieval: BM25 over heading-chunked text (≤ 512-token chunks); k=4; seeded cutoff.
- Metrics: retrieval hit@k, context utilization (via ablation), all nine baselines above.
- **Deliverable:** A-report with per-task outcomes; no conclusion beyond the measured data.

## 4. Arm B — DLCAE internal knowledge formation

- Same source material S; no retrieval permitted at eval; all knowledge via the comprehension pipeline (L1→L7) into versioned knowledge records.
- Evaluation: identical suite; plus A/B/C ALSO evaluated with retrieval *allowed* for upper bound reference (reported separately, not as B's config).
- **Deliverable:** B-report; explicitly report whether B exceeds A, ties A, or loses to A on each metric.

## 5. Arm C — Hybrid

- B's internal knowledge + optional external retrieval + full DLCAE feedback loop.
- Learns from verified execution verdicts only (no ingestion of unverified outputs).
- **Deliverable:** C-report; C is expected to dominate on part of the suite — but that must be shown, not asserted.

## 6. Controls

- Same held-out suites hidden from all arms (frozen hashes recorded).
- Same random seed; variance across seeds reported (n ≥ 3).
- Contamination audit: S must not contain benchmark solutions; verified against benchmark hashes before the run.
- Regression suite from each arm frozen before the experiment (no moving goalposts).

## 7. Deliverables for any L0.6.1 decision gate

1. A/B/C reports with the nine metrics tabulated per arm per language;
2. Statistical spread (n≥3) reported for every headline number;
3. Explicit statement of which approach wins each metric or "no winner";
4. Explicit statement of what data/license/compute each arm required;
5. Explicit recommendation or "no recommendation" — **no recommendation will be made from n < 3 or from cherry-picked tasks**.

## 8. What this milestone authorizes

**Nothing.** This document is a plan. Zero experiments have been executed. Zero data has been downloaded. No benchmark has been extended. No GPU has been allocated. Every step above is a specification for work that would follow explicit owner authorization (data acquisition + compute budget + benchmark execution license).
