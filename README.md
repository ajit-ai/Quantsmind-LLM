# Quantsmind Coder — Corporate Coding LLM

> **Your own from-scratch coding LLM — trained on your code, owned by you.**

A sovereign-weights coding LLM project: own architecture, own tokenizer, own training
engine, own data pipeline. Same Transformer family as Llama (RMSNorm/RoPE/GQA/SwiGLU),
trained from scratch on Python + JavaScript/TypeScript code.

**Status:** Local prototype ready — train your own coding LLM on your own hardware tonight.

---

## Quick start — train your own coding LLM

```bash
git clone https://github.com/ajit-ai/Quantsmind-LLM.git
cd Quantsmind-LLM
pip install -r requirements.txt

# 1. Build code corpus (public GitHub, permissive licenses)
python -m data.build_code_corpus --out data/code_corpus_v0.jsonl

# 2. Train code tokenizer
python -m tokenizer.train_tokenizer --input data/code_corpus_v0.jsonl \
  --out tokenizer/qm_code_bpe_v0.json --vocab-size 32768

# 3. Encode corpus to tokens
python -m data.encode_dataset --corpus data/code_corpus_v0.jsonl \
  --tokenizer tokenizer/qm_code_bpe_v0.json --out data/code_corpus_v0.tokens.bin

# 4. Pretrain (~30M params, 3-6 hours on CPU, ₹0)
python -m training.pretrain --config configs/train/code_local.yaml --out-dir runs/code_30m

# 5. SFT on code tasks
python -m training.sft --config configs/train/code_local.yaml \
  --sft-data data/code_sft_v0.jsonl --out-dir runs/code_sft \
  --init-ckpt runs/code_30m/checkpoint_last.pt

# 6. Chat with your coding assistant
python -m serving.code_chat --ckpt runs/code_sft/checkpoint_last.pt
# Open http://localhost:7860
```

## What this repo contains

| Path | Purpose |
|---|---|
| `training/model.py` | From-scratch Transformer: RMSNorm / RoPE / GQA / SwiGLU |
| `training/pretrain.py` | Training engine: cosine LR, checkpoint/resume, NaN watchdog |
| `training/sft.py` | SFT training loop with loss masking on assistant tokens |
| `training/data.py` | Token dataset loader (memmap, uint16) |
| `tokenizer/` | Own BPE trainer + code tokenizer (32k vocab) |
| `data/build_code_corpus.py` | Code corpus: GitHub download, license filter, dedup, encode |
| `data/build_code_sft.py` | SFT pair generator (code gen, explanation, review, refactor) |
| `data/ingest_internal.py` | Internal corporate repo ingestion with secret scanning |
| `evaluation/eval_code.py` | HumanEval + MBPP + code perplexity evaluation |
| `evaluation/promotion.yaml` | Promotion gate rules for code benchmarks |
| `serving/code_chat.py` | Code chat UI (Gradio, syntax highlighting) |
| `serving/completion_server.py` | REST API for IDE code completion (FastAPI) |
| `serving/review_server.py` | REST API for automated code review (FastAPI) |
| `serving/local_inference.py` | Local model loading + text generation |
| `configs/model/` | Model configs: code_30m, local_15m, pilot_300m |
| `configs/train/` | Training configs: code_local, corporate_ft, local_15m |
| `tests/` | Unit tests (model, tokenizer, SFT, eval, serving) |
| `.github/workflows/` | CI on every push |

## Architecture

```
Input tokens → [Token Embedding + RoPE] → 8× Transformer Blocks → [RMSNorm] → [LM Head] → Output logits

Each Transformer Block:
  ├── RMSNorm → Multi-Head Attention (GQA) → Residual
  └── RMSNorm → SwiGLU FFN → Residual
```

| Config | Params | Vocab | Seq Len | Use Case |
|---|---|---|---|---|
| `code_30m` | ~30M | 32,768 | 2048 | Python + JS/TS code |
| `local_15m` | ~11M | 49,152 | 512 | General text (legacy) |
| `pilot_300m` | ~300M | 49,152 | 4096 | Scale-up target |

## Serving

### Code Chat UI
```bash
python -m serving.code_chat --ckpt runs/code_sft/checkpoint_last.pt --port 7860
```
- Conversational coding assistant
- Python + JavaScript/TypeScript specialist
- Syntax highlighting, code context upload

### Completion API (IDE integration)
```bash
python -m serving.completion_server --port 8000
# POST /complete {prefix, suffix, language, max_tokens}
```
- Inline code suggestions
- Fill-in-the-middle support
- Top-k completions with scores

### Review API (CI/CD integration)
```bash
python -m serving.review_server --port 8001
# POST /review {code, language, context}
```
- Bug detection, security scanning
- Performance suggestions
- Structured JSON output

## Corporate Fine-Tuning

```bash
# Ingest internal codebase
python -m data.ingest_internal --repo /path/to/internal/repo \
  --repo-name myproject --out data/internal_corpus_v0.jsonl --scan-secrets

# Fine-tune on internal code
python -m training.pretrain --config configs/train/corporate_ft.yaml \
  --out-dir runs/corporate_ft --tokens data/internal_corpus_v0.tokens.bin
```

## Data Sources

| Source | License | Languages |
|---|---|---|
| GitHub public repos | MIT, Apache-2.0, BSD | Python, JS, TS |
| Python stdlib | PSF | Python |
| npm packages | MIT, ISC | JS, TS |
| Internal repos | Corporate | All |

## Docs

- [CODE_LLM_STRATEGY](docs/CODE_LLM_STRATEGY.md) — corporate coding LLM strategy
- [ROADMAP](docs/ROADMAP.md) — development roadmap
- [STATUS](docs/STATUS.md) — execution status tracker
- [DESIGN_ARCHITECTURE](docs/DESIGN_ARCHITECTURE.md) — full design doc

## License

See [LICENSE](LICENSE).
