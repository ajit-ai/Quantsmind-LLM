# TAKSH v3.0.0 — Training Runbook

## Overview

This runbook covers the complete training pipeline for qm-code-v300-70b:
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

Train the BPE tokenizer on the cleared code corpus:

```bash
python tokenizer/train_tokenizer.py \
  --input data/v300_corpus/taksh_code_v3.txt \
  --out tokenizer/taksh_code_bpe_v3.json \
  --vocab-size 65536
```

## Step 3: Corpus Encoding

Encode the corpus to token binaries:

```bash
python data/encode_dataset.py \
  --corpus data/v300_corpus/taksh_code_v3.txt \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --out data/v300_tokens/train.tokens.bin
```

## Step 4: Pretraining (GPU Cluster)

Launch pretraining:

```bash
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v300_70b.yaml \
  --out-dir runs/v300_70b_pretrain
```

### Expected Output

```
device=cuda
validation stream: data/v300_tokens/val.tokens.bin
loss log: runs/v300_70b_pretrain/loss_log.jsonl
step 0/100000 loss=8.78 ema=8.78 lr=1.50e-04 tok/s=50000
step 1000/100000 loss=4.50 ema=4.55 lr=2.99e-04 tok/s=55000
step 10000/100000 loss=3.20 ema=3.22 lr=2.70e-04 tok/s=58000
step 50000/100000 loss=2.80 ema=2.81 lr=1.50e-04 tok/s=60000
step 100000/100000 loss=2.50 ema=2.51 lr=3.00e-05 tok/s=62000
checkpoint saved: runs/v300_70b_pretrain/checkpoint_last.pt (step 100000)
run manifest: runs/v300_70b_pretrain/run_manifest.json
status=complete steps=100000/100000 tokens=26214400000
```

### Monitoring

```bash
# Watch loss in real-time
tail -f runs/v300_70b_pretrain/loss_log.jsonl

# Or use Weights & Biases
wandb init --project quantsmind-llm --name v300-70b-pretrain
```

## Step 5: Supervised Fine-Tuning (SFT)

After pretraining completes:

```bash
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v300_70b.yaml \
  --out-dir runs/v300_70b_sft \
  --base-checkpoint runs/v300_70b_pretrain/checkpoint_last.pt
```

## Step 6: Direct Preference Optimization (DPO)

```bash
python training/dpo_train.py \
  --config configs/train/dpo_v300_70b.yaml \
  --out-dir runs/v300_70b_dpo \
  --base-checkpoint runs/v300_70b_sft/checkpoint_last.pt
```

## Step 7: Evaluation

Evaluate the trained model:

```bash
python scripts/eval_v300_multimodal.py \
  --ckpt runs/v300_70b_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v300_70b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v3.json \
  --benchmark data/v300_eval/humaneval.jsonl \
  --device cuda
```

### Expected Output

```
Loading model from runs/v300_70b_dpo/checkpoint_last.pt
Loaded 70B tensors from runs/v300_70b_dpo/checkpoint_last.pt
Model: 70B params, vocab=32768
Loading tokenizer from tokenizer/taksh_code_bpe_v3.json
Tokenizer vocab: 65536
Evaluating on 164 HumanEval problems...

  [PASS] HumanEval/0: syntax=True, exec=True
  [PASS] HumanEval/1: syntax=True, exec=True
  ...

============================================================
TAKSH v3.0.0 Evaluation Results
============================================================
Total problems:     164
Syntax valid:       150/164 (91.5%)
Execution passed:   100/164 (61.0%)
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

1. **Pretraining fails**: Use last valid checkpoint from `runs/v300_70b_pretrain/checkpoint_*.pt`
2. **SFT fails**: Use pretrained checkpoint from `runs/v300_70b_pretrain/checkpoint_last.pt`
3. **DPO fails**: Use SFT checkpoint from `runs/v300_70b_sft/checkpoint_last.pt`
4. **Evaluation fails**: Check model format and tokenizer compatibility

---

## Completion Criteria

v3.0.0 is complete when:

- [ ] Pretraining completes 100K steps with loss < 2.5
- [ ] Validation loss tracks training loss
- [ ] HumanEval pass@1 > 60%
- [ ] MBPP pass@1 > 65%
- [ ] BigCodeBench pass@1 > 45%
- [ ] Image-to-code match > 70%
- [ ] Document-to-code match > 70%
- [ ] Syntax valid rate > 90%
- [ ] 131K context window works without quality degradation
- [ ] Checkpoint saves, verifies, reloads in fresh process

---

## References

- [TAKSH v2.2.0 Training Runbook](V220_TRAINING_RUNBOOK.md)
- [TAKSH v2.3.0 Training Runbook](V230_TRAINING_RUNBOOK.md)
- [Model Card](docs/MODEL_CARD_v300_70b.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
