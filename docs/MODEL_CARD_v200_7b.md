# Taksh v2.0.0 — Model Card: qm-code-v200-7b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v200-7b |
| Version | v2.0.0 |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Training data | 5B–50B tokens (cleared code corpus) |
| Training stages | Pretrain → SFT → DPO |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Product release candidate |

## Intended Use

This is a **product-class coding LLM** trained for code generation, debugging,
refactoring, and explanation tasks. Intended uses:

- Code generation from natural language instructions
- Code debugging and bug fixing
- Code refactoring and optimization
- Code explanation and documentation
- Multi-turn coding conversation

Not intended for: Malicious code generation, security vulnerability creation,
or any harmful use.

## Training Data

| Source | Tokens | License |
|---|---|---|
| Cleared code corpora | 5B–50B | Permissive licenses (MIT, Apache-2.0, BSD) |
| Synthetic instruction pairs | 10K–100K | Cleared |
| Preference pairs | 50K–500K | Cleared |

Tokenizer: `tokenizer/taksh_code_bpe_v2.json` (32,768 vocab, BPE).

## Training Configuration

| Stage | Hyperparameter | Value |
|---|---|---|
| Pretrain | Learning rate | 3.0e-4 (cosine) |
| Pretrain | Warmup steps | 2,000 |
| Pretrain | Weight decay | 0.1 |
| Pretrain | Batch size | 262,144 tokens/step |
| Pretrain | Total steps | 100,000 |
| Pretrain | Total tokens | ~26B |
| SFT | Learning rate | 2.0e-5 (cosine) |
| SFT | Warmup steps | 100 |
| SFT | Batch size | 32,768 tokens/step |
| SFT | Total steps | 10,000 |
| DPO | Learning rate | 5.0e-7 (cosine) |
| DPO | Beta | 0.1 |
| DPO | Total steps | 5,000 |
| Mixed precision | bfloat16 | All stages |

## Evaluation Results

| Benchmark | Target | Result |
|---|---|---|
| HumanEval pass@1 | > 45% | TBD (requires training) |
| MBPP pass@1 | > 50% | TBD (requires training) |
| Syntax valid rate | > 90% | TBD |
| Context window | 32K | TBD |

**Note:** Results pending GPU training. See TAKSH roadmap for scale-up path.

## Limitations

- Requires GPU cluster for training (64–256× A100/H100)
- Requires 5B–50B cleared training tokens
- Generated code should be reviewed before execution
- Not suitable for security-critical code without audit

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (training) | 64× A100 (80GB) | 256× H100 (80GB) |
| GPUs (inference) | 1× A100 (80GB) | 4× A100 (80GB) |
| System RAM | 512GB | 2TB+ |
| Storage | 10TB NVMe | 50TB NVMe |
| Network | 10GbE | 100GbE InfiniBand |

## How to Train

```bash
# From repository root
export PYTHONPATH=/path/to/Quantsmind-LLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 1. Train tokenizer
python tokenizer/train_tokenizer.py \
  --input data/v200_corpus/taksh_code_v2.txt \
  --out tokenizer/taksh_code_bpe_v2.json \
  --vocab-size 32768

# 2. Encode corpus
python data/encode_dataset.py \
  --corpus data/v200_corpus/taksh_code_v2.txt \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --out data/v200_tokens/train.tokens.bin

# 3. Pretrain
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v200_7b.yaml \
  --out-dir runs/v200_7b_pretrain

# 4. SFT
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v200_7b.yaml \
  --out-dir runs/v200_7b_sft \
  --base-checkpoint runs/v200_7b_pretrain/checkpoint_last.pt

# 5. DPO
python training/dpo_train.py \
  --config configs/train/dpo_v200_7b.yaml \
  --out-dir runs/v200_7b_dpo \
  --base-checkpoint runs/v200_7b_sft/checkpoint_last.pt
```

## How to Evaluate

```bash
python scripts/eval_v200_humaneval.py \
  --ckpt runs/v200_7b_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v200_7b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --benchmark data/v200_eval/humaneval.jsonl \
  --device cuda
```

## Citation

```
@misc{taksh-v200-7b,
  title = {Taksh v2.0.0: 7B Parameter Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.0.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

This is research software. The model is not suitable for production use without
further testing and security review. Generated code should not be executed
without review. The authors make no claims about correctness, safety, or
suitability for any purpose.
