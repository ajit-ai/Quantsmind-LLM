# Quantsmind-LLM

> **India's own from-scratch LLM — trained on your machine, owned by you, ₹0 forever.**

A sovereign-weights Indian LLM project: own architecture, own tokenizer, own training
engine, own data pipeline. Same Transformer family as Llama (RMSNorm/RoPE/GQA/SwiGLU),
trained from scratch on Indian finance data.

**Status:** Local prototype ready — train your own LLM on your own hardware tonight.

---

## Quick start — train your own LLM (₹0)

```bash
git clone https://github.com/ajit-ai/Quantsmind-LLM.git
cd Quantsmind-LLM
pip install -r requirements.txt

# Train your own LLM (~11M params, 2-5 hours on CPU, ₹0)
python -m training.pretrain --config configs/train/local_15m.yaml --out-dir runs/local_15m

# Chat with your own trained model
python -m serving.local_inference --ckpt runs/local_15m/checkpoint_last.pt
# Open http://localhost:7860
```

## What this repo contains

| Path | Purpose |
|---|---|
| `training/model.py` | From-scratch Transformer: RMSNorm / RoPE / GQA / SwiGLU — same family as Llama |
| `training/pretrain.py` | Training engine: cosine LR, checkpoint/resume, NaN watchdog |
| `training/data.py` | Token dataset loader (memmap, uint16) |
| `tokenizer/` | Own BPE trainer + 49k tokenizer (no HuggingFace dependency) |
| `data/` | Corpus pipeline: downloaders, chunking, PII scrub, MinHash fuzzy dedup, eval decontamination |
| `evaluation/` | Eval harness: held-out PPL + India-finance seed scoring + promotion gates |
| `serving/` | Local inference server (own weights → text → Gradio UI) |
| `configs/model/` | Model configs: local_15m, pilot_300m, prod_1p5b, scale_7b |
| `configs/train/` | Training configs: local_15m, pilot, smoke CI |
| `tests/` | 46 passing unit tests |
| `.github/workflows/` | CI on every push |

## Architecture

```
Input tokens → [Token Embedding + RoPE] → 6× Transformer Blocks → [RMSNorm] → [LM Head] → Output logits

Each Transformer Block:
  ├── RMSNorm → Multi-Head Attention (GQA) → Residual
  └── RMSNorm → SwiGLU FFN → Residual
```

| Config | Params | Hardware | Time |
|---|---|---|---|
| `local_15m` | ~11M | CPU / MX150 | 2-5 hours |
| `pilot_300m` | ~300M | 8×A100 | 1-2 days |
| `prod_1p5b` | 1.5B | 32×A100 | 1-2 weeks |

## Branching & releases

- `main` — protected; tagged releases only (merge via release PR from `develop`)
- `develop` — integration branch; all work lands here first
- Feature branches: `feat/<topic>` → PR to `develop`
- Project rule: **every phase ends with a develop → main merge**

## Running tests

```bash
python -m pytest tests -q   # should pass 46 tests
```

## Data residency (India-first)

Prompts are blocked from leaving your controlled network by default:

```powershell
$env:QM_STRICT_RESIDENCY = "0"   # conscious override; default is "1" (private/India-only)
```

See [docs/adr/0002-india-data-residency.md](docs/adr/0002-india-data-residency.md).

## Docs

- [ROADMAP](docs/ROADMAP.md) — development roadmap (Phase 0–6)
- [STATUS](docs/STATUS.md) — execution status tracker
- [DESIGN_ARCHITECTURE](docs/DESIGN_ARCHITECTURE.md) — full design doc
- [COMPETITIVE_ANALYSIS](docs/COMPETITIVE_ANALYSIS.md) — India market evaluation
- [ADR-0003](docs/adr/0003-from-scratch-sovereign.md) — from-scratch sovereign pretrain decision

## License

See [LICENSE](LICENSE).
