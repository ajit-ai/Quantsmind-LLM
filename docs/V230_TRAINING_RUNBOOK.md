# TAKSH v2.3.0 — Training Runbook

## Overview

This runbook covers the complete training pipeline for qm-code-v230-70b:
1. Preflight validation
2. Tokenizer training
3. Corpus encoding
4. Pretraining (GPU cluster required)
5. Supervised fine-tuning (SFT)
6. Direct Preference Optimization (DPO)
7. Evaluation

All commands assume you are in the repository root (`F:\Codes\Git\Quantsmind-LLM`).

---

## Prerequisites

### Hardware (GPU Cluster Required)

| Component | Minimum | Recommended |
|---|---|---|
| GPUs | 64× A100 (80GB) | 256× H100 (80GB) |
| System RAM | 1TB | 2TB+ |
| Storage | 50TB NVMe | 100TB NVMe |
| Network | 10GbE | 100GbE InfiniBand |

### Software

```bash
pip install torch>=2.2 tokenizers>=0.15 numpy>=1.26 pyyaml>=6.0 \
    fastapi>=0.104 uvicorn>=0.24 pydantic>=2.0 gradio>=4.44 \
    wandb tqdm accelerate pytest>=8.0
```

### Environment Variables

```bash
export PYTHONPATH=/path/to/Quantsmind-LLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=<your-wandb-key>
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
```

---

## Step 1: Preflight Validation

Before training, verify everything is ready:

```bash
python scripts/prepare_v190_training.py --check-all
python scripts/validate_v190_configs.py
```

## Step 2: Tokenizer Training

```bash
python tokenizer/train_tokenizer.py \
  --input data/v230_corpus/taksh_code_v3.txt \
  --out tokenizer/taksh_code_bpe_v3.json \
  --vocab-size 32768
```

## Step 3: Corpus Encoding

```bash
python data/encode_dataset.py \
  --corpus data/v230_corpus/taksh_code_v3.txt \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --out data/v230_tokens/train.tokens.bin
```

## Step 4: Pretraining (GPU Cluster)

```bash
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v230_70b.yaml \
  --out-dir runs/v230_70b_pretrain
```

## Step 5: SFT

```bash
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v230_70b.yaml \
  --out-dir runs/v230_70b_sft \
  --base-checkpoint runs/v230_70b_pretrain/checkpoint_last.pt
```

## Step 6: DPO

```bash
python training/dpo_train.py \
  --config configs/train/dpo_v230_70b.yaml \
  --out-dir runs/v230_70b_dpo \
  --base-checkpoint runs/v230_70b_sft/checkpoint_last.pt
```

## Step 7: Evaluation

```bash
python scripts/eval_v230_multimodal.py \
  --ckpt runs/v230_70b_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v230_70b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --benchmark data/v230_eval/humaneval.jsonl \
  --device cuda
```

---

## Success Criteria

v2.3.0 is complete when:

- [ ] Pretraining completes 100K steps with loss < 2.5
- [ ] Validation loss tracks training loss
- [ ] HumanEval pass@1 > 60%
- [ ] MBPP pass@1 > 65%
- [ ] BigCodeBench pass@1 > 45%
- [ ] Syntax valid rate > 90%
- [ ] 131K context window works without quality degradation
- [ ] Checkpoint saves, verifies, reloads in fresh process

---

## References

- [TAKSH v2.2.0 Training Runbook](V220_TRAINING_RUNBOOK.md)
- [Model Card](docs/MODEL_CARD_v230_70b.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
