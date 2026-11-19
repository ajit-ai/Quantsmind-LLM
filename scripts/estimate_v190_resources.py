"""Taksh v1.9.0 — Resource Estimation Tool.

Estimates GPU, memory, storage, and time requirements for v1.9.0 training.

Usage:
    python scripts/estimate_v190_resources.py --model 100m --steps 200 --seq-len 1024
    python scripts/estimate_v190_resources.py --model 1b --steps 100000 --seq-len 16384
    python scripts/estimate_v190_resources.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parent.parent

MODEL_SPECS = {
    "100m": {
        "params": 100_000_000,
        "d_model": 512,
        "n_layers": 16,
        "vocab_size": 5822,
        "description": "CPU-friendly demonstration model",
    },
    "300m": {
        "params": 300_000_000,
        "d_model": 768,
        "n_layers": 24,
        "vocab_size": 5822,
        "description": "Mid-scale demonstration model",
    },
    "1b": {
        "params": 1_000_000_000,
        "d_model": 1024,
        "n_layers": 24,
        "vocab_size": 5822,
        "description": "Production-scale model (GPU required)",
    },
    "3b": {
        "params": 3_000_000_000,
        "d_model": 2048,
        "n_layers": 32,
        "vocab_size": 5822,
        "description": "Large production model (GPU cluster required)",
    },
}


def estimate_model_memory(params: int, dtype_bytes: int = 2) -> Dict[str, float]:
    """Estimate memory requirements for model weights and gradients."""
    # bfloat16 = 2 bytes, fp32 = 4 bytes
    weights_gb = params * dtype_bytes / (1024**3)
    gradients_gb = params * dtype_bytes / (1024**3)  # Same size as weights
    optimizer_states_gb = params * 8 / (1024**3)  # Adam: 2 * fp32 (4+4 bytes)
    total_training_gb = weights_gb + gradients_gb + optimizer_states_gb
    return {
        "weights_gb": round(weights_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "optimizer_states_gb": round(optimizer_states_gb, 2),
        "total_training_gb": round(total_training_gb, 2),
    }


def estimate_activation_memory(d_model: int, n_layers: int, seq_len: int,
                               micro_batch: int, dtype_bytes: int = 2) -> float:
    """Estimate activation memory per GPU."""
    # Rough estimate: activations ~ 2 * d_model * seq_len * micro_batch * n_layers * dtype_bytes
    activations_gb = (2 * d_model * seq_len * micro_batch * n_layers * dtype_bytes) / (1024**3)
    # Add attention scores: n_heads * seq_len * seq_len * micro_batch * dtype_bytes
    n_heads = d_model // 64  # Approximate
    attention_gb = (n_heads * seq_len * seq_len * micro_batch * dtype_bytes) / (1024**3)
    return round(activations_gb + attention_gb, 2)


def estimate_training_time(params: int, steps: int, seq_len: int,
                           micro_batch: int, grad_accum: int,
                           gpus: int, tokens_per_sec_per_gpu: float = 1000) -> Dict[str, Any]:
    """Estimate training time."""
    tokens_per_step = micro_batch * seq_len * grad_accum
    total_tokens = tokens_per_step * steps
    tokens_per_sec_total = tokens_per_sec_per_gpu * gpus
    seconds = total_tokens / tokens_per_sec_total
    return {
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
        "total_tokens_per_sec": round(tokens_per_sec_total, 0),
        "estimated_seconds": round(seconds, 0),
        "estimated_hours": round(seconds / 3600, 1),
        "estimated_days": round(seconds / 86400, 1),
    }


def estimate_storage(params: int, steps: int, checkpoint_every: int,
                     vocab_size: int = 5822) -> Dict[str, float]:
    """Estimate storage requirements."""
    weights_gb = params * 2 / (1024**3)  # bfloat16
    num_checkpoints = steps // checkpoint_every
    checkpoint_storage_gb = num_checkpoints * weights_gb * 1.2  # + optimizer states
    logs_gb = 5  # Estimated log size
    total_gb = checkpoint_storage_gb + logs_gb + weights_gb  # + final model
    return {
        "weights_gb": round(weights_gb, 2),
        "num_checkpoints": num_checkpoints,
        "checkpoint_storage_gb": round(checkpoint_storage_gb, 2),
        "logs_gb": logs_gb,
        "total_storage_gb": round(total_gb, 2),
    }


def print_model_spec(name: str, spec: Dict[str, Any]) -> None:
    """Print model specification."""
    print(f"  Name: qm-code-v170-{name}")
    print(f"  Parameters: {spec['params']:,} ({spec['params'] / 1e6:.0f}M)")
    print(f"  d_model: {spec['d_model']}")
    print(f"  n_layers: {spec['n_layers']}")
    print(f"  vocab_size: {spec['vocab_size']}")
    print(f"  Description: {spec['description']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.9.0 Resource Estimator")
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="100m",
                        help="Model size to estimate")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of training steps")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--all", action="store_true",
                        help="Estimate for all model sizes")
    args = parser.parse_args()

    print("=" * 60)
    print("TAKSH v1.9.0 Resource Estimator")
    print("=" * 60)
    print()

    if args.all:
        for name, spec in MODEL_SPECS.items():
            print(f"--- {name.upper()} ({spec['description']}) ---")
            print_model_spec(name, spec)
            mem = estimate_model_memory(spec["params"])
            print(f"  Memory (bfloat16):")
            print(f"    Weights: {mem['weights_gb']} GB")
            print(f"    Gradients: {mem['gradients_gb']} GB")
            print(f"    Optimizer states: {mem['optimizer_states_gb']} GB")
            print(f"    Total training: {mem['total_training_gb']} GB")
            print()
        return

    spec = MODEL_SPECS[args.model]
    steps = args.steps or (200 if args.model == "100m" else 100000)
    seq_len = args.seq_len or spec.get("max_seq_len", 1024)
    micro_batch = 1 if args.model == "100m" else 4
    grad_accum = 2 if args.model == "100m" else 16

    print(f"Model: {args.model.upper()} ({spec['description']})")
    print_model_spec(args.model, spec)
    print()

    # Memory estimation
    print("--- Memory Requirements (per GPU) ---")
    mem = estimate_model_memory(spec["params"])
    act_mem = estimate_activation_memory(spec["d_model"], spec["n_layers"],
                                         seq_len, micro_batch)
    print(f"  Model weights: {mem['weights_gb']} GB")
    print(f"  Gradients: {mem['gradients_gb']} GB")
    print(f"  Optimizer states: {mem['optimizer_states_gb']} GB")
    print(f"  Activations (approx): {act_mem} GB")
    print(f"  Total per GPU: {round(mem['total_training_gb'] + act_mem, 2)} GB")
    print()

    # Training time
    print("--- Training Time Estimate ---")
    time_est = estimate_training_time(spec["params"], steps, seq_len,
                                      micro_batch, grad_accum, args.gpus)
    print(f"  Tokens per step: {time_est['tokens_per_step']:,}")
    print(f"  Total tokens: {time_est['total_tokens']:,}")
    print(f"  Tokens/sec/GPU: {time_est['tokens_per_sec_per_gpu']:,}")
    print(f"  Total tokens/sec: {time_est['total_tokens_per_sec']:,}")
    print(f"  Estimated time: {time_est['estimated_hours']} hours ({time_est['estimated_days']} days)")
    print()

    # Storage
    print("--- Storage Estimate ---")
    storage = estimate_storage(spec["params"], steps, 500)
    print(f"  Model weights: {storage['weights_gb']} GB")
    print(f"  Checkpoints: {storage['num_checkpoints']} x ~{storage['weights_gb']} GB")
    print(f"  Checkpoint storage: {storage['checkpoint_storage_gb']} GB")
    print(f"  Logs: {storage['logs_gb']} GB")
    print(f"  Total storage: {storage['total_storage_gb']} GB")
    print()

    # GPU recommendation
    total_gpu_mem = mem['total_training_gb'] + act_mem
    gpus_needed = max(1, int(total_gpu_mem / 70) + 1)  # 70GB usable per 80GB GPU
    print("--- GPU Recommendation ---")
    print(f"  Memory per GPU needed: {round(total_gpu_mem, 2)} GB")
    print(f"  Recommended GPUs: {gpus_needed} (80GB each)")
    print(f"  Recommended GPU type: A100 80GB or H100 80GB")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model.upper()} ({spec['params'] / 1e6:.0f}M params)")
    print(f"Training steps: {steps:,}")
    print(f"Sequence length: {seq_len}")
    print(f"Estimated training time: {time_est['estimated_hours']} hours on {args.gpus} GPU(s)")
    print(f"Total storage: {storage['total_storage_gb']} GB")
    print(f"GPU memory per GPU: {round(total_gpu_mem, 2)} GB")
    if args.gpus < gpus_needed:
        print(f"WARNING: {args.gpus} GPU(s) may not have enough memory. Recommend {gpus_needed} GPU(s).")
    else:
        print(f"GPU count ({args.gpus}) is sufficient.")


if __name__ == "__main__":
    main()
