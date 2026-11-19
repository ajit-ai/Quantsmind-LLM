# TAKSH v1.9.0 — Training Runbook

## Overview

This runbook covers the complete training pipeline for qm-code-v170-100m:
1. Preflight validation
2. Pretraining
3. Supervised fine-tuning (SFT)
4. Evaluation

All commands assume you are in the repository root (`F:\Codes\Git\Quantsmind-LLM`).

---

## Prerequisites

### Hardware (GPU Required for Training)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1× A100 (40GB) | 8× A100 (80GB) |
| System RAM | 32GB | 128GB |
| Storage | 50GB NVMe | 500GB NVMe |
| Network | 1GbE | 10GbE |

### Software

```bash
pip install torch>=2.2 tokenizers>=0.15 numpy>=1.26 pyyaml>=6.0 \
    fastapi>=0.104 uvicorn>=0.24 pydantic>=2.0 gradio>=4.44 pytest>=8.0
```

### Environment Variables

```bash
export PYTHONPATH=/path/to/Quantsmind-LLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust for your GPUs
export WANDB_API_KEY=<your-wandb-key>  # Optional, for logging
```

---

## Step 1: Preflight Validation

Before training, verify everything is ready:

```bash
python scripts/prepare_v190_training.py --check-all
```

Expected output:
```
FILES: 20/20
CONFIGS: ALL OK
DATA: ALL CONSISTENT
ENVIRONMENT: CUDA AVAILABLE, 8 GPU(S)
READY FOR GPU TRAINING
```

If any checks fail, fix them before proceeding.

## Step 2: Pretraining

Launch pretraining on GPU cluster:

```bash
# Single GPU
python training/pretrain.py \
  --config configs/train/pretrain_v170_100m.yaml \
  --out-dir runs/v190_100m_pretrain

# Multi-GPU (recommended)
torchrun --nproc_per_node=8 training/pretrain.py \
  --config configs/train/pretrain_v170_100m.yaml \
  --out-dir runs/v190_100m_pretrain
```

### Expected Output

```
device=cuda
validation stream: data/v170_tokens/val.tokens.bin (47,816 tokens, evaluation only)
loss log: runs/v190_100m_pretrain/loss_log.jsonl
step 0/200 loss=8.78 ema=8.78 lr=1.50e-04 tok/s=5000
step 10/200 loss=6.50 ema=6.55 lr=1.65e-03 tok/s=8000
step 20/200 loss=6.10 ema=6.15 lr=3.00e-03 tok/s=8200
...
checkpoint saved: runs/v190_100m_pretrain/checkpoint_last.pt (step 200)
run manifest: runs/v190_100m_pretrain/run_manifest.json
status=complete steps=200/200 tokens=410000
```

### Monitoring

```bash
# Watch loss in real-time
tail -f runs/v190_100m_pretrain/loss_log.jsonl

# Or use Weights & Biases (if configured)
wandb init --project quantsmind-llm --name v190-100m-pretrain
```

## Step 3: Supervised Fine-Tuning (SFT)

After pretraining completes, run SFT:

```bash
python training/sft_train.py \
  --config configs/train/sft_v170.yaml \
  --out-dir runs/v190_100m_sft \
  --base-checkpoint runs/v190_100m_pretrain/checkpoint_last.pt
```

### Expected Output

```
base checkpoint : runs/v190_100m_pretrain/checkpoint_last.pt (step 200, 111 tensors)
freeze policy   : None (28,787,328 trainable / 28,787,328 total)
train examples  : 20 encoded, 0 rejected, 83 supervised tokens
val examples    : 20 encoded
step 0/100 loss=6.88 lr=1.00e-05 tok/s=312
step 10/100 loss=5.54 lr=4.97e-05 tok/s=986
...
step 99/100 loss=4.89 lr=5.01e-06 tok/s=769
checkpoint saved: runs/v190_100m_sft/checkpoint_last.pt (step 100)
run manifest: runs/v190_100m_sft/run_manifest.json
status=complete steps=100/100 tokens=102400
```

## Step 4: Evaluation

Evaluate the trained model:

```bash
python scripts/eval_v160_simple_humaneval.py \
  --ckpt runs/v190_100m_sft/checkpoint_last.pt \
  --model-config configs/model/code_v170_100m.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v1.json \
  --device cuda
```

### Expected Output

```
Loading model from runs/v190_100m_sft/checkpoint_last.pt
Loaded 111 tensors from runs/v190_100m_sft/checkpoint_last.pt
Model: 28,787,328 params, vocab=5822
Loading tokenizer from tokenizer/taksh_code_bpe_v1.json
Tokenizer vocab: 5822
Evaluating on 20 simple_humaneval problems...

  [PASS] p01_add (arithmetic): syntax=True, expected=True
  [FAIL] p02_mul (arithmetic): syntax=False, expected=False
  ...

v1.6.0 simple_humaneval Results
============================================================
Total:        20
Passed:       X/20 (X.X%)
Syntax valid: X/20 (X.X%)
Expected tail: X/20 (X.X%)
```

**Note:** At 100M parameters with demonstration-scale data, pass@1 is expected to be low. This is normal.

---

## Step 5: Model Card Generation

After training, update the model card:

```bash
# Copy template and fill in actual values
cp docs/MODEL_CARD_v190_100m.md docs/MODEL_CARD_v190_100m_actual.md

# Edit with actual training results
# - Final loss values
# - Qualification results
# - Training duration
# - Hardware used
```

---

## Troubleshooting

### Out of Memory (OOM)

| Symptom | Solution |
|---|---|
| CUDA out of memory | Reduce `micro_batch_size` or `grad_accum_steps` |
| Runtime error: CUDA error | Check GPU memory with `nvidia-smi` |
| NaN loss | Reduce learning rate or increase warmup steps |

### Slow Training

| Symptom | Solution |
|---|---|
| Low tokens/sec | Increase `micro_batch_size` or use more GPUs |
| CPU bottleneck | Ensure `dtype: bfloat16` and GPU is available |

### Checkpoint Corruption

| Symptom | Solution |
|---|---|
| Checkpoint load fails | Use `checkpoint_best.pt` instead |
| Missing optimizer state | Restart from last valid checkpoint |

---

## Rollback Plan

If training fails at any point:

1. **Pretraining fails**: Use last valid checkpoint from `runs/v190_100m_pretrain/checkpoint_*.pt`
2. **SFT fails**: Use pretrained checkpoint from `runs/v190_100m_pretrain/checkpoint_last.pt`
3. **Evaluation fails**: Check model format and tokenizer compatibility

---

## Completion Criteria

v1.9.0 is complete when:

- [ ] Pretraining completes all steps without NaN loss
- [ ] Validation loss tracks training loss
- [ ] Checkpoint saves and reloads correctly
- [ ] Evaluation runs without errors
- [ ] Model card is updated with actual results
- [ ] Training manifest is complete

---

## References

- [TAKSH v1.7.0 Roadmap](TAKSH_V170_ROADMAP_FINAL.md)
- [Model Card](docs/MODEL_CARD_v190_100m.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
