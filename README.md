# Quantsmind-LLM

A production-grade large language model, **trained from scratch**, with a
quantitative-finance focus and strong general reasoning.

- **Status**: Phase 0 — Foundations (see [ROADMAP](docs/ROADMAP.md))
- **Design**: full architecture in [DESIGN_ARCHITECTURE.md](docs/DESIGN_ARCHITECTURE.md)
- **License**: see [LICENSE](LICENSE)

## What this repo contains

| Path | Purpose |
|---|---|
| `configs/` | Model, training-stage, and corpus-mix YAML configs |
| `tokenizer/` | BPE tokenizer training + CI tests |
| `data/` | Acquire → dedup → filter → PII scrub → decontaminate → mix → tokenize |
| `training/` | Pretrain / SFT / DPO entry points |
| `evaluation/` | Eval harness configs + model promotion gate |
| `safety/` | Red-team suites, guardrail policies |
| `serving/` | vLLM engine workers + OpenAI-compatible API gateway |
| `infra/` | Cluster / cloud provisioning |
| `docs/` | Architecture, roadmap, ADRs, model cards |

## Branching

- `main` — protected; tagged releases only (merge via release PR from `develop`)
- `develop` — integration branch; all work lands here first
- Feature branches: `feat/<topic>`

## Quick start (Phase 0)

```bash
git clone https://github.com/ajit-ai/Quantsmind-LLM.git
cd Quantsmind-LLM
# environment setup lands with the first training code (Phase 0 deliverable)
```
