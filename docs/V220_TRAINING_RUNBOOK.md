# TAKSH v2.2.0 — Training Runbook

## Overview

This runbook covers the fine-tuning pipeline for 4 domain-specific models:
1. Frontend domain model
2. Backend domain model
3. Data science domain model
4. Embedded/IoT domain model

All fine-tuning starts from the v2.0.0 7B base model (runs/v200_7b_dpo/checkpoint_last.pt).

---

## Prerequisites

### Hardware (GPU Cluster Required)

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (per domain) | 32× A100 | 64× H100 |
| GPUs (all 4 domains) | 128× A100 | 256× H100 |
| System RAM | 512GB per node | 1TB+ per node |
| Storage | 20TB NVMe | 100TB NVMe |
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

## Step 1: Prepare Domain Training Data

Prepare domain-specific instruction→code pairs:

```bash
# Example: Frontend domain
python scripts/build_v220_domain_data.py \
  --domain frontend \
  --sources /path/to/cleared/frontend/code \
  --output data/v220_frontend/

# Repeat for backend, datascience, embedded
```

Each domain needs 50K–200K instruction→code pairs.

## Step 2: Fine-Tune Each Domain Model (Parallel)

### Frontend
```bash
torchrun --nproc_per_node=32 training/domain_train.py \
  --config configs/train/domain_v220_frontend_7b.yaml \
  --out-dir runs/v220_frontend_7b \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt
```

### Backend
```bash
torchrun --nproc_per_node=32 training/domain_train.py \
  --config configs/train/domain_v220_backend_7b.yaml \
  --out-dir runs/v220_backend_7b \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt
```

### Data Science
```bash
torchrun --nproc_per_node=32 training/domain_train.py \
  --config configs/train/domain_v220_datascience_7b.yaml \
  --out-dir runs/v220_datascience_7b \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt
```

### Embedded
```bash
torchrun --nproc_per_node=32 training/domain_train.py \
  --config configs/train/domain_v220_embedded_7b.yaml \
  --out-dir runs/v220_embedded_7b \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt
```

## Step 3: Evaluate Each Domain Model

```bash
python scripts/eval_v220_domain.py \
  --model runs/v220_frontend_7b/checkpoint_last.pt \
  --model-config configs/model/code_v220_frontend_7b.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --domain frontend \
  --benchmark data/v220_frontend/benchmark.jsonl \
  --device cuda
```

Repeat for backend, datascience, embedded domains.

---

## Success Criteria

v2.2.0 is complete when:

- [ ] Frontend model > 80% pass@1 on frontend tasks
- [ ] Backend model > 75% pass@1 on backend tasks
- [ ] Data science model > 75% pass@1 on data science tasks
- [ ] Embedded model > 70% pass@1 on embedded tasks
- [ ] Domain router correctly selects model ≥ 90% of the time
- [ ] No domain model regresses on general tasks (> 5% degradation)

---

## Domain Router

The `serving/domain_router.py` module routes requests to the appropriate domain model based on keyword matching and context:

```python
from serving.domain_router import route_request

decision = route_request("Create a React button component")
print(decision.model_name)  # qm-code-v220-frontend-7b
print(decision.confidence)  # 0.85
print(decision.reason)      # Matched 5 keyword(s) for frontend
```

---

## References

- [TAKSH v2.0.0 Training Runbook](V200_TRAINING_RUNBOOK.md)
- [TAKSH v2.1.0 Training Runbook](V210_TRAINING_RUNBOOK.md)
- [Model Cards](docs/MODEL_CARD_v220_*_7b.md)
- [Repository](https://github.com/ajit-ai/Quantsmind-LLM)
