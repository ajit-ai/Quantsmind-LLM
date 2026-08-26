# Quantsmind-LLM

> **⏸️ STATUS: PAUSED** — fully preserved at Phase 1 (85%), ₹0 spent, zero recurring costs.
> See [docs/PAUSED.md](docs/PAUSED.md) for the exact resume playbook.

A from-scratch, sovereign-weights Indian LLM project (finance & trading vertical,
B2B2C distribution) — own architecture, own tokenizer, own training engine, own data
pipeline. Prototype-first: local training on founder hardware, scale when ready
(ADR-0003).

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

## Chat UI (type a prompt → financial knowledge)

```bash
python -m serving.app          # opens http://localhost:7860
```

- **Default: demo mode** — canned finance explainers, no external model, nothing leaves your machine.
- **Phase 3: your own model.** Quantsmind's trained model will be served via vLLM
  (OpenAI-compatible API) and the UI plugs into it directly:

```powershell
$env:QM_API_BASE = "http://localhost:8000/v1"   # YOUR Quantsmind vLLM server
$env:QM_API_KEY  = "local-dev"
$env:QM_MODEL    = "quantsmind-1.5b"
python -m serving.app
```

> **Note:** Quantsmind does NOT depend on any third-party model. The variables above are
> a generic socket — for quick UI testing you *may* point it at any OpenAI-compatible
> backend, but the product target is Quantsmind's own from-scratch model (see
> `docs/DESIGN_ARCHITECTURE.md`). Nothing in training/data/tokenizer uses external models.

Every answer carries the mandatory *"not investment advice"* disclaimer.

### India data residency (default ON)

Prompts are blocked from leaving your controlled network unless you explicitly override:

```powershell
$env:QM_STRICT_RESIDENCY = "0"   # conscious override; default is "1" (private/India-only)
```

Policy: localhost / private-network / India-hosted endpoints only (see `docs/adr/0002-india-data-residency.md`).
