# Quantsmind-LLM

A finance & trading vertical LLM platform for **B2B2C** distribution — a white-label
"Investor Copilot" (grounded Q&A, doc intelligence, tool-verified arithmetic) built by
adapting an Apache-2.0 open base model for Indian-first, world-ready finance.

- **Status**: Phase 0 — Foundations (see [ROADMAP](docs/ROADMAP.md))
- **Design**: [DESIGN_ARCHITECTURE.md](docs/DESIGN_ARCHITECTURE.md) (ADR-000: open-base extension strategy)
- **License**: see [LICENSE](LICENSE)

## What this repo contains

| Path | Purpose |
|---|---|
| `configs/` | Model / training-stage / corpus-mix YAML configs (+ tiny CI smoke configs) |
| `tokenizer/` | Byte-BPE trainer + sanity tests (draft-model & data tooling) |
| `data/` | Sample-corpus generator, encode-to-tokens pipeline |
| `training/` | Transformer implementation (`model.py`) + pretrain loop with checkpoint/resume & watchdogs — powers the draft model and future from-scratch option; CPT/SFT/DPO land in Phase 2 |
| `evaluation/` | Eval harness config skeleton + `promotion.yaml` release gate |
| `safety/` | Red-team suites, guardrail policies (Phase 2+) |
| `serving/` | Gateway, RAG service, vLLM workers (Phase 3) |
| `infra/` | Cluster provisioning |
| `scripts/` | End-to-end CPU smoke pipeline used by CI |
| `tests/` | pytest unit tests |

## Branching & releases

- `main` — protected; tagged releases only (merge via release PR from `develop`)
- `develop` — integration branch; all work lands here first
- Feature branches: `feat/<topic>` → PR to `develop`
- Project rule: **every phase ends with a develop → main merge**

## Quick start (local)

```bash
git clone https://github.com/ajit-ai/Quantsmind-LLM.git
cd Quantsmind-LLM
python -m pip install -r requirements.txt   # torch required for training tests
python -m pytest tests -q                    # unit tests
bash scripts/smoke_pipeline.sh               # end-to-end CPU smoke run (Linux/macOS)
# Windows PowerShell:
powershell -File scripts/smoke_pipeline.ps1
```
