# TAKSH Architecture — Independent Adaptive Coding Intelligence

**Status:** strategic architecture specification (L0.6.0-R1 alignment milestone).
**Branch:** `feature/l060-foundation-model-qualification` @ `c601a737…`.
**Baseline preserved:** 53-language registry, 16-language L1.0 target, tokenizer, frozen Python benchmark, L0.6.0 foundation config, all checkpoints.

---

## 1. Product identity

> **TAKSH is an independently trained, locally deployable, adaptive coding
> intelligence specializing in programming, software development, debugging,
> testing, repository engineering and controlled autonomous software development.**

Two identities are preserved together and are not in tension:

1. **Independent foundation model.** TAKSH's own weights learn to program.
   Capability lives in the model, not in its tooling.
2. **Complete adaptive coding intelligence platform.** Around the foundation sit
   cooperating subsystems that retrieve, execute, verify, remember and act —
   they multiply the model's capability but never substitute for missing capability.

A tool-wrapped non-programmer is not this product. A brilliant model with no
verification pipeline is not this product either.

## 2. The cooperating subsystems

These are **logical subsystems**. They are **not** eight models, eight services,
eight processes or eight training runs. They are responsibilities that can be
realized inside one deployment, or distributed where evidence and operations justify it.

| # | Subsystem | Responsibility (logical) |
|---|---|---|
| 1 | Learning Engine | Next-token prediction; instruction fine-tuning; continual-learning with provenance, replay and rollback |
| 2 | Knowledge Engine | Structured internal knowledge: concepts, relationships, versioned language state, verification |
| 3 | Code Intelligence Engine | Comprehension and generation: lexical→syntax→semantic→task understanding; code emission |
| 4 | Engineering Engine | Repository-scale work: multi-file edits, tests, build integration, code review, debugging |
| 5 | **DLCAE — Dynamic Language Comprehension and Adaptation Engine** | Language-independent progressive comprehension + adaptation (see DLCAE-ARCHITECTURE.md) |
| 6 | Memory Engine | Session memory, task state, checkpoint/resume of sequences, adaptation history |
| 7 | Evaluation Engine | Benchmarks, gates, regression detection, promotion/demotion rules |
| 8 | Agent Engine | Controlled autonomy: plans, executes, observes, recovers; human approval where required |

```
                ┌─────────────────────────────────────────────┐
                │                 TAKSH PLATFORM               │
                │                                              │
   prompts ──▶  │  Agent Engine ──▶ Engineering Engine          │
                │       ▲                 │                     │
                │  Memory Engine   ┌──────▼───────┐              │
                │       ▲         │ Code          │              │
                │       │         │ Intelligence  │              │
                │  DLCAE▼         └──────┬───────┘              │
                │  ──▶ Knowledge Engine   │                      │
                │       ▲                 │                     │
                │  Learning Engine ◀──────▼────── Evaluation     │
                └─────────────────────────────────────────────┘
```

## 3. What must remain true (non-negotiables)

- The foundation model is **independently trained**; no external LLM API in the
  critical path; no telemetry; no phoning home.
- Local deployability: a user can run TAKSH on their own machine with their own
  checkpoint and tokenizer.
- The model's capability must be **measured by held-out executable tests** —
  never claimed from loss curves, perplexity, imports, or fluent text.
- Tooling (retrieval, compilers, execution, agents) can **enhance but not conceal**
  missing capability; raw-model and system performance are reported separately.
- No silent capability claims. An unexecuted language is not a qualified language.

## 4. Current measured state (honest)

| Layer | Status |
|---|---|
| Foundation model (28.8M) | trains; 0/30 pass@1 on the frozen Python benchmark |
| Tokenizer | 5,822-vocab BPE; adequate per audit; roundtrip exact |
| SFT pipeline | works (overfit 24/24; 0/30 unseen) |
| Continued pretraining | works; val 6.87→5.41; still 0/30 |
| Code-dense warmstart (L0.4) | code-val 5.06 (best); still 0/30 |
| DLCAE | specified, not implemented (this milestone) |
| Karkain integration | contracts defined; toolchain broken on this host (see KARKAIN-INTEGRATION.md) |
| 53-language catalogue | preserved intact; Karkain added as 54th |
| Multilingual benchmark | 64 tasks, 16 languages; denominator fixed (≥16 tasks, ≥50% pass) |

## 5. Scale and staging

- **Stage 1 (now):** engineering qualification of the 28.8M baseline; tokenizer,
  pipelines, benchmarks, guards — all verified, 0/30 honest.
- **Stage 2 (blocked on owner authorizations):** data acquisition + GPU compute
  for a ≥ 110M model; strategy C (pretrain → SFT); goal ≥ 1 language with
  measurable pass@1 > 0 on a frozen benchmark.
- **Stage 3+:** multilingual expansion per DLCAE; Karkain first-class support;
  repository-engineering qualification.

## 6. Invariants that survive every revision

- 53 languages stay; 16 stay L1.0; supporting technologies stay separate.
- Python benchmark stays frozen; multilingual bar stays ≥ 50% of ≥ 16 executed tasks per language.
- Package version `taksh==1.0.0` until an owner-authorized release process says otherwise.
- `.kark` is the Karkain extension; Karkain is a first-class language, id `karkain`.
- No architectural claim is made without measured evidence.
