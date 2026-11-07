# Taksh v2.3.0 — Model Card: qm-code-v230-70b

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v230-70b |
| Version | v2.3.0 |
| Parameters | ~70B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 131,072 tokens |
| Training stages | Pretrain → SFT → DPO |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Enterprise release candidate |

## Intended Use

This is an **enterprise-grade coding LLM** trained for high-complexity code generation,
debugging, refactoring, and explanation tasks. Intended uses:

- Complex code generation from natural language instructions
- Complex debugging and bug fixing across large codebases
- Complex code refactoring and optimization
- Code explanation and documentation
- Multi-turn coding conversation with long context (131K tokens)
- Enterprise deployment (on-premises, VPC, SLA support)

Not intended for: Malicious code generation, security vulnerability creation,
or any harmful use.

## Training Data

| Source | Tokens | License |
|---|---|---|
| Cleared code corpora | 50B+ | Permissive licenses (MIT, Apache-2.0, BSD) |
| Synthetic instruction pairs | 100K–1M | Cleared |
| Preference pairs | 50K–500K | Cleared |

Tokenizer: `tokenizer/taksh_code_bpe_v3.json` (32,768 vocab, BPE).

## Training Configuration

| Stage | Hyperparameter | Value |
|---|---|---|
| Pretrain | Learning rate | 3.0e-4 (cosine) |
| Pretrain | Warmup steps | 2,000 |
| Pretrain | Weight decay | 0.1 |
| Pretrain | Batch size | 262,144 tokens/step |
| Pretrain | Total steps | 100,000 |
| Pretrain | Total tokens | ~26B |
| SFT | Learning rate | 1.0e-5 (cosine) |
| SFT | Warmup steps | 100 |
| SFT | Batch size | 131,072 tokens/step |
| SFT | Total steps | 20,000 |
| DPO | Learning rate | 5.0e-7 (cosine) |
| DPO | Beta | 0.1 |
| DPO | Total steps | 10,000 |
| Mixed precision | bfloat16 | All stages |

## Evaluation Results

| Benchmark | Target | Result |
|---|---|---|
| HumanEval pass@1 | > 60% | TBD (requires training) |
| MBPP pass@1 | > 65% | TBD |
| BigCodeBench pass@1 | > 45% | TBD |
| Syntax valid rate | > 90% | TBD |
| Context window | 131K | TBD |

**Note:** Results pending GPU training.

## Limitations

- Requires GPU cluster for training (64–256× A100/H100)
- Requires 50B+ cleared training tokens
- Generated code should be reviewed before execution
- Not suitable for security-critical code without audit

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (training) | 64× A100 (80GB) | 256× H100 (80GB) |
| GPUs (inference) | 4× A100 (80GB) | 8× H100 (80GB) |
| System RAM | 1TB | 2TB+ |
| Storage | 50TB NVMe | 100TB NVMe |
| Network | 10GbE | 100GbE InfiniBand |

## How to Train

```bash
# From repository root
export PYTHONPATH=/path/to/Quantsmind-LLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 1. Train tokenizer
python tokenizer/train_tokenizer.py \
  --input data/v230_corpus/taksh_code_v3.txt \
  --out tokenizer/taksh_code_bpe_v3.json \
  --vocab-size 32768

# 2. Encode corpus
python data/encode_dataset.py \
  --corpus data/v230_corpus/taksh_code_v3.txt \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --out data/v230_tokens/train.tokens.bin

# 3. Pretrain
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v230_70b.yaml \
  --out-dir runs/v230_70b_pretrain

# 4. SFT
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v230_70b.yaml \
  --out-dir runs/v230_70b_sft \
  --base-checkpoint runs/v230_70b_pretrain/checkpoint_last.pt

# 5. DPO
python training/dpo_train.py \
  --config configs/train/dpo_v230_70b.yaml \
  --out-dir runs/v230_70b_dpo \
  --base-checkpoint runs/v230_70b_sft/checkpoint_last.pt
```

## How to Evaluate

```bash
python scripts/eval_v230_multimodal.py \
  --ckpt runs/v230_70b_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v230_70b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --benchmark data/v230_eval/humaneval.jsonl \
  --device cuda
```

## Citation

```
@misc{taksh-v230-70b,
  title = {Taksh v2.3.0: 70B Parameter Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.3.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

This is research software. The model is not suitable for production use without
further testing and security review. Generated code should not be executed
without review. The authors make no claims about correctness, safety, or
suitability for any purpose.
