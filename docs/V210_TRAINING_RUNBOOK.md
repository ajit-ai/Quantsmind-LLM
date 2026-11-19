# TAKSH v2.1.0 — Training Runbook

## Overview

This runbook covers the complete training pipeline for qm-code-v210-7b-agent:
1. Pretrain 7B base model (v2.0.0)
2. SFT (v2.0.0)
3. DPO (v2.0.0)
4. Agent tool use fine-tuning (v2.1.0)
5. Agent trajectory fine-tuning (v2.1.0)
6. Agent DPO (v2.1.0)
7. Agent evaluation

All commands assume you are in the repository root (`F:\Codes\Git\Quantsmind-LLM`).

---

## Prerequisites

### Hardware (GPU Cluster Required)

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (training) | 64× A100 (80GB) | 256× H100 (80GB) |
| GPUs (inference) | 1× A100 (80GB) | 4× A100 (80GB) |
| Sandbox containers | 1 concurrent | 10 concurrent |
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

## Step 1: Pretrain 7B Base Model

Follow the v2.0.0 pretraining procedure:

```bash
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v200_7b.yaml \
  --out-dir runs/v200_7b_pretrain
```

## Step 2: SFT

```bash
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v200_7b.yaml \
  --out-dir runs/v200_7b_sft \
  --base-checkpoint runs/v200_7b_pretrain/checkpoint_last.pt
```

## Step 3: DPO

```bash
python training/dpo_train.py \
  --config configs/train/dpo_v200_7b.yaml \
  --out-dir runs/v200_7b_dpo \
  --base-checkpoint runs/v200_7b_sft/checkpoint_last.pt
```

## Step 4: Agent Tool Use Fine-Tuning

```bash
torchrun --nproc_per_node=64 training/agent_train.py \
  --config configs/train/agent_v210_tool_use.yaml \
  --out-dir runs/v210_agent_tool_use \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt
```

## Step 5: Agent Trajectory Fine-Tuning

```bash
torchrun --nproc_per_node=64 training/agent_train.py \
  --config configs/train/agent_v210_trajectory.yaml \
  --out-dir runs/v210_agent_trajectory \
  --base-checkpoint runs/v210_agent_tool_use/checkpoint_last.pt
```

## Step 6: Agent DPO

```bash
python training/dpo_train.py \
  --config configs/train/agent_v210_dpo.yaml \
  --out-dir runs/v210_agent_dpo \
  --base-checkpoint runs/v210_agent_trajectory/checkpoint_last.pt
```

## Step 7: Evaluate Agent

```bash
python scripts/eval_v210_agent.py \
  --ckpt runs/v210_agent_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v210_7b_agent.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --benchmark data/v210_agent/benchmark.jsonl \
  --device cuda
```

### Expected Output

```
Loading model from runs/v210_agent_dpo/checkpoint_last.pt
Loaded 7B tensors from runs/v210_agent_dpo/checkpoint_last.pt
Model: 7B params, vocab=32768
Loading tokenizer from tokenizer/taksh_code_bpe_v2.json
Tokenizer vocab: 32768
Evaluating agent on 50 tasks...

  [SOLVED] task-001: solved=True, steps=5, tools=3, interventions=0
  [FAILED] task-002: solved=False, steps=8, tools=7, interventions=2
  ...

============================================================
TAKSH v2.1.0 Agent Evaluation Results
============================================================
Total tasks:     50
Tasks solved:    32/50 (64.0%)
Avg tool calls:  4.2
Avg interventions: 0.8
```

---

## Troubleshooting

### Agent Loop Issues

| Symptom | Solution |
|---|---|
| Agent loops forever | Increase max_steps limit, add step counter |
| Agent doesn't use tools | More SFT on tool-use data |
| Agent makes destructive changes | Stricter approval policy |
| Sandbox escape | Tighten container restrictions |

### Training Issues

| Symptom | Solution |
|---|---|
| OOM | Reduce micro_batch_size, increase grad_accum_steps |
| NaN loss | Reduce learning rate |
| Slow convergence | More training data, tune learning rate |

---

## Completion Criteria

v2.1.0 is complete when:

- [ ] Agent resolves > 40% of SWE-bench issues
- [ ] Agent completes > 60% of TAKE-1 tasks autonomously
- [ ] Average tool calls per task < 10
- [ ] Human interventions per task < 2
- [ ] No security incidents in sandbox
- [ ] Approval workflow functions correctly

---

## References

- [TAKSH v2.0.0 Training Runbook](V200_TRAINING_RUNBOOK.md)
- [Model Card](docs/MODEL_CARD_v210_7b_agent.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
