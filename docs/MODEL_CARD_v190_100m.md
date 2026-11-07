# Taksh v1.9.0 — Model Card: qm-code-v170-100m

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v170-100m |
| Version | v1.9.0 |
| Parameters | ~100M |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 5,822 |
| Context length | 4,096 tokens |
| Training data | ~450K tokens (project-owned Python source) |
| Training steps | 200 (demonstration scale) |
| License | Apache-2.0 (source code only) |
| Release date | 2026-11-07 |
| Status | Demonstration / Research |

## Intended Use

This is a **demonstration-scale model** trained to validate the Taksh training pipeline.
It is **not** intended for production use. Use cases:

- Pipeline validation (pretrain → SFT → eval)
- Architecture testing at 100M scale
- Research on small code models
- Baseline for larger models

## Training Data

| Source | Tokens | License |
|---|---|---|
| Project-owned Python source | ~450K | Cleared (Apache-2.0) |

The tokenizer vocabulary (`tokenizer/taksh_code_bpe_v1.json`, 5,822 tokens) was trained on this same corpus.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3.0e-3 (cosine decay) |
| Warmup steps | 20 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch size (tokens/step) | 2,048 |
| Total steps | 200 |
| Total tokens | ~410K |
| Mixed precision | float32 (CPU) / bfloat16 (GPU) |

## Evaluation Results

| Metric | Value |
|---|---|
| Final training loss | ~6.0 |
| Validation loss | ~5.9 |
| Uniform random baseline | 8.67 (ln(5,822)) |
| HumanEval pass@1 | Not evaluated (demonstration scale) |
| Syntax valid rate | Not evaluated |

**Note:** This model was trained for demonstration purposes only. It does not produce functionally correct code. See TAKSH roadmap for scale-up path.

## Limitations

- **Not production-ready**: 100M parameters is too small for useful code generation
- **Limited training data**: ~450K tokens is insufficient for learning code patterns
- **Short training**: 200 steps is far below convergence
- **Small context**: 4K tokens limits file-size understanding
- **No alignment**: No SFT or RLHF applied

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1× A100 (40GB) | 8× A100 (80GB) |
| System RAM | 32GB | 128GB |
| Storage | 50GB NVMe | 500GB NVMe |
| Training time | ~1 hour (CPU) | ~10 minutes (8× A100) |

## How to Train

```bash
# From repository root
export PYTHONPATH=/path/to/Quantsmind-LLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python training/pretrain.py \
  --config configs/train/pretrain_v170_100m.yaml \
  --out-dir runs/v190_100m_pretrain
```

## How to Evaluate

```bash
python scripts/eval_v160_simple_humaneval.py \
  --ckpt runs/v190_100m_pretrain/checkpoint_last.pt \
  --model-config configs/model/code_v170_100m.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v1.json
```

## Citation

```
@misc{taksh-v190-100m,
  title = {Taksh v1.9.0: 100M Parameter Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {1.9.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

This is research software. The model is not suitable for production use. Generated code should not be executed without review. The authors make no claims about correctness, safety, or suitability for any purpose.
