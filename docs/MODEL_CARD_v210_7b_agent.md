# Taksh v2.1.0 — Model Card: qm-code-v210-7b-agent

## Model Details

| Attribute | Value |
|---|---|
| Model name | qm-code-v210-7b-agent |
| Version | v2.1.0 |
| Base model | qm-code-v200-7b |
| Parameters | ~7B |
| Architecture | Llama-family decoder-only (RMSNorm, RoPE, GQA, SwiGLU) |
| Vocabulary size | 32,768 |
| Context length | 32,768 tokens |
| Training stages | Pretrain → SFT → DPO → Agent SFT → Agent DPO |
| License | Apache-2.0 (source code only) |
| Release date | 2026 |
| Status | Agentic coding release candidate |

## Intended Use

This is an **agentic coding LLM** capable of autonomous coding tasks with tool use.
Intended uses:

- Autonomous bug fixing
- Feature implementation
- Code refactoring
- Test generation
- Multi-step coding workflows with tool use

Not intended for: Malicious code generation, security vulnerability creation,
unauthorized system access, or any harmful use.

## Training Data

| Stage | Data | Count | License |
|---|---|---|---|
| Pretrain | Code corpus | 5B–50B tokens | Cleared |
| SFT | Instruction→code | 10K–100K | Cleared |
| DPO | Preference pairs | 50K–500K | Cleared |
| Agent SFT | Tool-use trajectories | 50K–500K | Cleared |
| Agent DPO | Agent preference pairs | 50K–500K | Cleared |

Tokenizer: `tokenizer/taksh_code_bpe_v2.json` (32,768 vocab, BPE).

## Training Configuration

| Stage | Learning Rate | Steps |
|---|---|---|
| Pretrain | 3.0e-4 | 100,000 |
| SFT | 2.0e-5 | 10,000 |
| DPO | 5.0e-7 | 5,000 |
| Agent SFT | 1.0e-5 | 20,000 |
| Agent Trajectory | 5.0e-6 | 10,000 |
| Agent DPO | 1.0e-6 | 5,000 |

All stages use cosine decay, bfloat16 precision.

## Evaluation Results

| Benchmark | Target | Result |
|---|---|---|
| SWE-bench issues resolved | > 15% | TBD (requires training) |
| TAKE-1 tasks solved | > 60% | TBD |
| Tool calls per task | < 10 | TBD |
| Human interventions per task | < 2 | TBD |

**Note:** Results pending GPU training.

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPUs (training) | 64× A100 | 256× H100 |
| GPUs (inference) | 1× A100 | 4× A100 |
| Sandbox containers | 1 concurrent | 10 concurrent |
| System RAM | 512GB | 2TB+ |

## How to Train

```bash
# 1. Pretrain (7B base)
torchrun --nproc_per_node=64 training/pretrain.py \
  --config configs/train/pretrain_v200_7b.yaml \
  --out-dir runs/v200_7b_pretrain

# 2. SFT
torchrun --nproc_per_node=64 training/sft_train.py \
  --config configs/train/sft_v200_7b.yaml \
  --out-dir runs/v200_7b_sft \
  --base-checkpoint runs/v200_7b_pretrain/checkpoint_last.pt

# 3. DPO
python training/dpo_train.py \
  --config configs/train/dpo_v200_7b.yaml \
  --out-dir runs/v200_7b_dpo \
  --base-checkpoint runs/v200_7b_sft/checkpoint_last.pt

# 4. Agent tool use
torchrun --nproc_per_node=64 training/agent_train.py \
  --config configs/train/agent_v210_tool_use.yaml \
  --out-dir runs/v210_agent_tool_use \
  --base-checkpoint runs/v200_7b_dpo/checkpoint_last.pt

# 5. Agent trajectory
torchrun --nproc_per_node=64 training/agent_train.py \
  --config configs/train/agent_v210_trajectory.yaml \
  --out-dir runs/v210_agent_trajectory \
  --base-checkpoint runs/v210_agent_tool_use/checkpoint_last.pt

# 6. Agent DPO
python training/dpo_train.py \
  --config configs/train/agent_v210_dpo.yaml \
  --out-dir runs/v210_agent_dpo \
  --base-checkpoint runs/v210_agent_trajectory/checkpoint_last.pt
```

## How to Evaluate Agent

```bash
python scripts/eval_v210_agent.py \
  --ckpt runs/v210_agent_dpo/checkpoint_last.pt \
  --model-config configs/model/code_v210_7b_agent.yaml \
  --tokenizer tokenizer/taksh_code_bpe_v2.json \
  --benchmark data/v210_agent/benchmark.jsonl \
  --device cuda
```

## Citation

```
@misc{taksh-v210-7b-agent,
  title = {Taksh v2.1.0: Agentic Coding LLM},
  author = {Ajit Kumar},
  year = {2026},
  version = {2.1.0},
  url = {https://github.com/ajit-ai/Quantsmind-LLM}
}
```

## Disclaimer

This is research software. The agent can execute code in a sandbox, but
generated code should be reviewed before execution. The authors make no
claims about correctness, safety, or suitability for any purpose.
