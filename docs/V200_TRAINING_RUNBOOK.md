# TAKSH v2.0.0 — Training Runbook

## Overview

This runbook covers the complete training pipeline for qm-code-v200-7b:
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
| System RAM | 512GB | 2TB+ |
| Storage | 10TB NVMe | 50TB NVMe |
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

Expected output: All checks pass, GPU available.

## Step 2: Tokenizer Training

Train the BPE tokenizer on the cleared code corpus:

```bash
python tokenizer/train_tokenizer.py \
  --input data/v200_corpus/taksh_code_v2.txt \
  --out tokenizer/taksh_code_bpe_v2.json \
  --vocab-size 32768
```

## Step 3: Corpus Encoding

Encode the corpus to token binaries:

```bash
python data/encode_dataset.py \
  --corpus data/v200_corpus/taksh_code_v2.txt \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --out data/v200_tokens/train.tokens.bin
```

Split into train/val:

```bash
python -c "
import numpy as np
from pathlib import Path
train = np.memmap('data/v200_tokens/train.tokens.bin', dtype=np.uint16, mode='r')
split = int(len(train) * 0.9)
train[:split].tofile('data/v200_tokens/train.tokens.bin')
train[split:].tofile('data/v200_tokens/val.tokens.bin')
"
```

## Step 4: Pretraining (GPU Cluster)

Launch pretraining:

```bash
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v200_7b.yaml \
  --out-dir runs/v200_7b_pretrain
```

### Expected Output

```
device=cuda
validation stream: data/v200_tokens/val.tokens.bin
loss log: runs/v200_7b_pretrain/loss_log.jsonl
step 0/100000 loss=8.78 ema=8.78 lr=3.00e-04 tok/s=50000
step 1000/100000 loss=4.50 ema=4.55 lr=2.99e-04 tok/s=55000
step 10000/100000 loss=3.20 ema=3.22 lr=2.70e-04 tok/s=58000
step 50000/100000 loss=2.80 ema=2.81 lr=1.50e-04 tok/s=60000
step 100000/100000 loss=2.50 ema=2.51 lr=3.00e-05 tok/s=62000
checkpoint saved: runs/v200_7b_pretrain/checkpoint_last.pt (step 100000)
run manifest: runs/v200_7b_pretrain/run_manifest.json
status=complete steps=100000/100000 tokens=26214400000
```

### Monitoring

```bash
# Watch loss in real-time
tail -f runs/v200_7b_pretrain/loss_log.jsonl

# Or use Weights & Biases
wandb init --project quantsmind-llm --name v200-7b-pretrain
```

## Step 5: Supervised Fine-Tuning (SFT)

After pretraining completes:

```bash
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v200_7b.yaml \
  --out-dir runs/v200_7b_sft \
  --base-checkpoint runs/v200_7b_pretrain/checkpoint_last.pt
```

## Step 6: Direct Preference Optimization (DPO)

After SFT completes:

```bash
python training/dpo_train.py \
  --config configs/train/dpo_v200_7b.yaml \
  --out-dir runs/v200_7b_dpo \
  --base-checkpoint runs/v200_7b_sft/checkpoint_last.pt
```

Note: DPO training can run on fewer GPUs (8–16) since it's smaller scale.

## Step 7: Evaluation

Evaluate the trained model:

```bash
python scripts/eval_v200_humaneval.py \
  --ckpt runs/v200_7b_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v200_7b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --benchmark data/v200_eval/humaneval.jsonl \
  --device cuda
```

### Expected Output

```
Loading model from runs/v200_7b_dpo/checkpoint_last.pt
Loaded 7B tensors from runs/v200_7b_dpo/checkpoint_last.pt
Model: 7B params, vocab=32768
Loading tokenizer from tokenizer/taksh_code_bpe_v2.json
Tokenizer vocab: 32768
Evaluating on 164 HumanEval problems...

  [PASS] HumanEval/0: syntax=True, exec=True
  [PASS] HumanEval/1: syntax=True, exec=True
  ...

============================================================
TAKSH v2.0.0 Evaluation Results
============================================================
Total problems:     164
Syntax valid:       150/164 (91.5%)
Execution passed:   75/164 (45.7%)
```

---

## Troubleshooting

### Out of Memory (OOM)

| Symptom | Solution |
|---|---|
| CUDA out of memory | Reduce `micro_batch_size` or increase `grad_accum_steps` |
| Runtime error: CUDA error | Check GPU memory with `nvidia-smi` |
| NaN loss | Reduce learning rate or increase warmup steps |

### Slow Training

| Symptom | Solution |
|---|---|
| Low tokens/sec | Increase `micro_batch_size` or use more GPUs |
| CPU bottleneck | Ensure `dtype: bfloat16` and GPU is available |
| Multi-GPU slow | Check InfiniBand/network connectivity |

### Checkpoint Corruption

| Symptom | Solution |
|---|---|
| Checkpoint load fails | Use `checkpoint_best.pt` instead |
| Missing optimizer state | Restart from last valid checkpoint |

---

## Rollback Plan

If training fails at any point:

1. **Pretraining fails**: Use last valid checkpoint from `runs/v200_7b_pretrain/checkpoint_*.pt`
2. **SFT fails**: Use pretrained checkpoint from `runs/v200_7b_pretrain/checkpoint_last.pt`
3. **DPO fails**: Use SFT checkpoint from `runs/v200_7b_sft/checkpoint_last.pt`
4. **Evaluation fails**: Check model format and tokenizer compatibility

---

## Completion Criteria

v2.0.0 is complete when:

- [ ] Pretraining completes 100K steps with loss < 2.5
- [ ] Validation loss tracks training loss
- [ ] HumanEval pass@1 > 30%
- [ ] MBPP pass@1 > 40%
- [ ] Syntax valid rate > 90%
- [ ] 32K context window works without quality degradation
- [ ] OpenAI-compatible API serves requests correctly
- [ ] Checkpoint saves, verifies, reloads in fresh process

---

## References

- [TAKSH v1.9.0 Roadmap](TAKSH_V170_ROADMAP_FINAL.md)
- [Model Card](docs/MODEL_CARD_v200_7b.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
