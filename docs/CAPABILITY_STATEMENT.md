# Taksh — Capability Statement

**Date:** 2026-11-14
**Package:** `taksh 1.0.0`
**Audience:** product owners, customers, auditors and release reviewers
**Status:** engineering-qualified platform; coding capability NOT DEMONSTRATED

---

## 1. What this document claims

Taksh is an independently developed coding-LLM platform with modern transformer architecture, its own tokenizer, data pipeline, training loops, evaluation harnesses, serving stack, audit controls and six-engine DLCAE runtime.

Taksh does **not** currently have a product checkpoint that can reliably write code. No capability is claimed from loss curves, architecture, imported libraries or fluent outputs.

---

## 2. Measurable capability

| Capability | State | Evidence |
|---|---|---|
| Code generation | Not demonstrated | Frozen Python benchmark 0/30 |
| Code completion | Not demonstrated | Locked syntax-valid 0/40; functional 0/8 |
| Functional correctness | Not demonstrated | HumanEval/MBPP not measured; 5-case smoke subsets only |
| Instruction following | Not demonstrated | SFT corpus quarantined historically; no product instruction model |
| Debugging/repair | Not demonstrated | v2.2.0 generalization result: learning improved, transfer not demonstrated |
| Test generation | Not demonstrated | No qualification result |
| Multilingual code | Not demonstrated | Frozen Python benchmark only; other languages have zero qualified corpus |
| Karkain | DISCOVERED | `kcc.exe` `0xC0000135`; KARK-BENCH zero tasks |
| Repository engineering | Not demonstrated | Engineering infrastructure only |
| Security hardening code | Not demonstrated | Static guards exist; model-level secret safety unmeasured |
| Autonomous coding agent | Not demonstrated | Orchestrator exists; model-dependent paths use labelled deterministic doubles |

---

## 3. What works today

- From-scratch Llama-family transformer implementation (RMSNorm, RoPE, GQA, SwiGLU).
- Deterministic training loops, loss logging, checkpoint lifecycle and manifests.
- Strict checkpoint compatibility checks.
- Frozen benchmark scaffolding and sandboxed execution.
- OpenAI-subset loopback API, completion API, review API and OpenAI-compatible server.
- KV-cache integration and context extensions (application-level, not model capability).
- DLCAE comprehension/validation/knowledge/feedback runtime foundation.
- Evaluation, provenance, contamination and release-governance tooling.
- Immutable evidence records and four-axis qualification discipline.

---

## 4. What does not work yet

- A canonical product model does not exist.
- Frozen coding benchmarks are not passed.
- There is no measured pass@1 for HumanEval, MBPP, BigCodeBench or SWE-bench.
- No language other than Python has a qualified corpus.
- No deployment latency or throughput claim is supported against a real product checkpoint.
- No GA or publication claim is authorized.

---

## 5. Model lineage and artifacts

| Artifact | Parameters | Measured outcome | Role |
|---|---:|---|---|
| code_tiny / S2 experiments | 9,359,104 | 0/40 syntax-valid; 0/8 functional | historical negative evidence |
| v160 | approximately 15M | pipeline runs; no pass | demonstration only |
| v180_sft | approximately 28.8M | overfit seen tasks; 0/30 unseen | experimental artifact |
| canonical taksh-foundation-111m | 110,658,816 | missing | planned release foundation |
| v190/v200/v210/v220/v230/v300 cards | proposed | no trained product weights | planned variants, not current products |

Historical negative results are retained as evidence and must not be reinterpreted as positive capability.

---

## 6. Safe use today

Taksh may be used for:

- research and engineering of the training/evaluation platform;
- governance, provenance and security workflow development;
- infrastructure qualification;
- benchmark and harness development;
- internal review of repository engineering pipelines with human oversight.

Taksh must not be represented as:

- a reliable coding assistant;
- a production code-review product;
- a security or compliance oracle;
- a general assistant;
- a code-generation API with assured correctness;
- a multilingual coding product;
- an autonomous software engineer.

---

## 7. Required evidence before capability is claimed

1. Canonical checkpoint selected and identified by hash.
2. Frozen benchmark and suite hashes recorded before evaluation.
3. pass@1 / syntax / execution results with raw outputs and seeds.
4. Repeat runs and variance report.
5. Regression checks against prior qualified artifacts.
6. Independent review of the evidence.
7. Owner release decision, if publication is pursued.
