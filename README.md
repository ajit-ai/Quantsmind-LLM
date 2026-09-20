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
| `tests/` | Unit tests (46 M0 + 19 M1.1 + 20 M1.2) |
| `.github/workflows/` | CI on every push (incl. hermetic M1.1 MM smoke) |
| `training/vision.py` | Multimodal M1.1: from-scratch ViT encoder (patch 16, 112×112) |
| `training/multimodal.py` | Multimodal M1.1: ViT → projector → QM decoder, MM attention mask |
| `training/train_mm.py` | Multimodal fit-check loop (synthetic M1.1 / real-data M1.2 via manifest) |
| `training/mm_manifest.py` | Multimodal M1.2: provenance-aware MANIFEST.json parser |
| `training/image_data.py` | Image datasets: synthetic `ImageTextDataset` + real `RealImageDataset` |
| `scripts/synth_images.py` | Multimodal M1.1: deterministic synthetic image/caption generator |
| `data/mm_m1/` | Multimodal M1.2: real-data structure (manifest schema, data card; no images committed) |

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
python -m pytest tests -q   # 46 M0 + 19 M1.1 + 20 M1.2 tests
```

## Experimental: M1.1 multimodal fit-check (CPU, synthetic)

Architecture evidence only — no real image corpus, no image generation yet. Runs on
fully synthetic, hermetic image/caption pairs:

```bash
python scripts/synth_images.py --out runs/mm_synth/data.npz --n 256
python -m training.train_mm --config configs/train/mm_fit_check.yaml \
  --out-dir runs/mm_fit_check --init-ckpt runs/local_15m/checkpoint_last.pt
```

The loop verifies forward/mask shapes, ViT+projector gradients flow, synthetic data
memorization, image-conditioning changes next-token distributions, and that the M0
text model remains bit-identical after warm-start. See [STATUS](docs/STATUS.md) and
[ROADMAP](docs/ROADMAP.md) phase A3.

## Experimental: M1.2 real-data pipeline (no training run)

Infrastructure for real image/caption data. **M1 is not promoted** — this is
preparation only, no large training run and no claim of M1 completion.

```bash
# Real-data drive is manifest -> image -> RGB -> resize/crop -> M1 tensor -> caption
python scripts/synth_images.py --out runs/mm_synth/data.npz --n 256   # synthetic path unchanged
# To exercise the real-data pipeline with real images, populate data/mm_m1/
# (train|val|test + MANIFEST.json per data/mm_m1/DATA_CARD.rst), then:
python -m training.train_mm --config configs/train/mm_m1_local.yaml \
  --out-dir runs/mm_m1_local --manifest data/mm_m1/MANIFEST.json --max-steps 3
```

Manifest/loader/schema: `training/mm_manifest.py`, `training/image_data.py`,
`data/mm_m1/MANIFEST.json`, `data/mm_m1/DATA_CARD.rst`. Images are never
committed and never auto-downloaded. RST docs: `docs/source/mm_m1.rst`.

## Data residency (India-first)

Prompts are blocked from leaving your controlled network by default:

```powershell
$env:QM_STRICT_RESIDENCY = "0"   # conscious override; default is "1" (private/India-only)
```

See [docs/adr/0002-india-data-residency.md](docs/adr/0002-india-data-residency.md).

## Docs

- [ROADMAP](docs/ROADMAP.md) — multimodal development roadmap (Phases A–G, milestones M0–M7)
- [SCOPE_AND_PRINCIPLES](docs/SCOPE_AND_PRINCIPLES.md) — what we build and the rules it follows
- [STATUS](docs/STATUS.md) — execution status tracker
- [DESIGN_ARCHITECTURE](docs/DESIGN_ARCHITECTURE.md) — full design doc
- [COMPETITIVE_ANALYSIS](docs/COMPETITIVE_ANALYSIS.md) — India market evaluation
- [ADR-0003](docs/adr/0003-from-scratch-sovereign.md) — from-scratch sovereign pretrain decision
- Sphinx/RST docs: `docs/source/` (`index.rst`, `mm_m1.rst`)

## License

See [LICENSE](LICENSE).
