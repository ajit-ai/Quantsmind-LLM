# Quantsmind Coder — Versioning & Phase Matrix

---

## Version Scheme

**Format:** `qm-code-{size}-v{major}.{patch}`

| Tag | Meaning |
|---|---|
| `qm-code-tiny-v0.1` | Tiny model (~500K params), local smoke/CI only |
| `qm-code-30m-v0.1` | 30M params, local prototype |
| `qm-code-30m-v1.0` | First promoted 30M model (passes promotion gate) |
| `qm-code-100m-v1.0` | 100M params, single-GPU |
| `qm-code-pilot-300m-v1.0` | 300M params, 8×A100 |

---

## Phase Matrix

| Phase | Name | Version Target | Status | Gate |
|---|---|---|---|---|
| **M0** | Pipeline proven | `qm-code-tiny-v0.1` | COMPLETE | 165 tests green; smoke train + CI |
| **M1** | Code base model | `qm-code-30m-v0.1` | IN PROGRESS | Pretrain loss sane; PPL < 15; tokenizer round-trip OK |
| **M2** | SFT-aligned coder | `qm-code-30m-v0.2` | PENDING | HumanEval pass@1 >= 10%; code review precision > 40% |
| **M3** | Corporate fine-tune | `qm-code-30m-v1.0` | PENDING | Internal eval passed; no secret leakage; no regression vs base |
| **M4** | Serving E2E | `qm-code-30m-v1.0` | PENDING | Chat UI live; completion API < 200ms; review API < 2s |
| **M5** | Scale to 100M | `qm-code-100m-v1.0` | PENDING | Beats 30M on HumanEval + MBPP; single-GPU deployable |
| **M6** | 300M pilot | `qm-code-pilot-300m-v1.0` | PENDING | 8×A100 training; MFU >= 40%; beats open rivals on code tasks |

---

## Completed Milestones

### M0 — Pipeline Proven (2026-09-27)

| Component | Evidence |
|---|---|
| Code corpus pipeline | 77 local files, ~100k tokens, dedup + quality filter |
| Code BPE tokenizer | 5,817 vocab, round-trip OK |
| Pretrain smoke test | 50 steps, loss 10.39 -> 7.42, checkpoint save/resume |
| SFT pipeline | 439 pairs generated, 30 steps, loss 8.67 -> 8.55 |
| Evaluation harness | HumanEval + MBPP + perplexity; promotion gate configured |
| Serving modules | Code chat UI, completion API, review API all import-verified |
| CI/CD | 165 tests green; code LLM checks added to workflow |
| Git | All commits under Ajit Kumar; develop + main synced |

---

## In-Progress Milestones

### M1 — Code Base Model

| Task | Status | Blocker |
|---|---|---|
| Scale corpus to 5B+ tokens | PENDING | Need GPU or more local Python files |
| Full pretrain (10k steps) | PENDING | Memory constraint on current machine |
| Eval on full model | PENDING | Need larger corpus for meaningful PPL |
| Promotion gate pass | PENDING | Requires full training run |

---

## Future Milestones

### M2 — SFT-Aligned Coder

| Task | Target |
|---|---|
| SFT on 50k+ code pairs | HumanEval pass@1 >= 10% |
| Code review training | Precision > 40% on internal eval |
| Refactoring training | Qualitative improvement |
| Bug fixing training | HumanEval pass@1 improvement |

### M3 — Corporate Fine-Tune

| Task | Target |
|---|---|
| Internal repo ingestion | All internal Python/JS/TS repos |
| Continued pretrain on internal | 1-2B tokens, loss decrease |
| Internal SFT | Code review/QA pairs from internal patterns |
| Security audit | Zero secrets in training data |
| Promotion gate | All rules green |

### M4 — Serving E2E

| Task | Target |
|---|---|
| Code chat UI deploy | localhost:7860 live |
| Completion API deploy | < 200ms latency |
| Review API deploy | < 2s latency |
| IDE plugin | VS Code extension |
| CI/CD integration | GitHub Action for PR review |

### M5 — Scale to 100M

| Task | Target |
|---|---|
| Model config | 100M params, single-GPU trainable |
| Corpus scale | 50B+ tokens |
| Training | Single GPU, < 1 week |
| Eval | Beats 30M on all benchmarks |
| Deploy | Single A100/consumer GPU |

### M6 — 300M Pilot

| Task | Target |
|---|---|
| Hardware | 8×A100 rental |
| Corpus | 100B+ tokens |
| Training | 1-2 days, MFU >= 40% |
| Rivals | Score vs DeepSeek-Coder, CodeLlama |
| Decision | Go/no-go for 1.5B |

---

## Version History

| Version | Date | Description |
|---|---|---|
| `v0.1-tiny` | 2026-09-27 | Tiny model; pipeline proven; 165 tests; smoke train + SFT |
