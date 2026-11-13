"""TAKSH L0.5 — WP1: architecture feasibility audit.

Reproducible parameter-count and architecture analysis of the current TAKSH
model and the candidate scenarios A/B/C from configs/feasibility/l05_scenarios.yaml.

Analytic counting (no model instantiation required): embedding + per-layer
attention (wq/wk/wv/wo with GQA) + per-layer SwiGLU MLP + norms. The analytic
count is verified against QMModel.num_params() in the test suite for the
baseline configuration (exact match: 28,787,328).

Writes reports/l05_architecture_audit.json (consumed by the feasibility chain).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCENARIOS_PATH = REPO / "configs" / "feasibility" / "l05_scenarios.yaml"


def _yaml():
    import yaml

    return yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))


def count_parameters(cfg: dict) -> dict:
    """Analytic parameter count for a Taksh-family config."""
    vocab = int(cfg["vocab_size"])
    d = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])
    n_kv = int(cfg["n_kv_heads"])
    ffn = int(cfg["ffn_dim"])
    tie = bool(cfg.get("tie_embeddings", True))
    head_dim = d // n_heads

    embedding = vocab * d
    lm_head = 0 if tie else vocab * d
    attn_wq = d * n_heads * head_dim
    attn_wk = d * n_kv * head_dim
    attn_wv = d * n_kv * head_dim
    attn_wo = d * d  # d -> d regardless of head layout
    attn_per_layer = attn_wq + attn_wk + attn_wv + attn_wo
    mlp_per_layer = 3 * d * ffn
    norms_per_layer = 2 * d
    layer = attn_per_layer + mlp_per_layer + norms_per_layer
    normalization = n_layers * norms_per_layer + d
    total = embedding + lm_head + n_layers * layer + d  # final norm
    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "tie_embeddings": tie,
        "per_layer": {
            "attention": attn_per_layer,
            "mlp": mlp_per_layer,
            "norms": norms_per_layer,
            "total": layer,
        },
        "layers_total": n_layers * layer,
        "final_norm": d,
        "total_params": total,
        "head_dim": head_dim,
        "checkpoint_bytes_fp32": total * 4,
        "checkpoint_bytes_bf16": total * 2,
        "optimizer_bytes_adamw_fp32_state": total * 8,   # m + v slots, fp32
        "training_bytes_fp32_weights_grads_adam": total * 16,  # weights + grads + m + v
    }


def describe_family(cfg: dict) -> dict:
    vocab = int(cfg["vocab_size"])
    d = int(cfg["d_model"])
    n_heads = int(cfg["n_heads"])
    n_layers = int(cfg["n_layers"])
    return {
        "architecture": "decoder-only causal transformer (RMSNorm + RoPE + GQA + SwiGLU)",
        "vocab_size": vocab,
        "d_model": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": int(cfg["n_kv_heads"]),
        "gqa": "n_heads:n_kv_heads = %d:%d" % (int(cfg["n_heads"]), int(cfg["n_kv_heads"])),
        "ffn_dim": int(cfg["ffn_dim"]),
        "ffn_activation": "SwiGLU (gate * SiLU(up))",
        "max_seq_len": int(cfg["max_seq_len"]),
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "norm": "RMSNorm (eps=%g)" % float(cfg.get("norm_eps", 1e-05)),
        "attention_impl": "F.scaled_dot_product_attention (is_causal=True)",
        "positional": "RoPE (half-dim interleaved), no learned positional embeddings",
        "residuals": "pre-norm transformer with residual connections",
        "loss": "next-token cross-entropy, ignore_index=-100",
        "initialization": "nn.init.normal_(std=0.02) for Linear/Embedding",
    }


def audit_current(cfg: dict, measured: int) -> dict:
    counts = count_parameters(cfg)
    return {
        "name": "existing_baseline",
        "config": cfg,
        "family": describe_family(cfg),
        "params": counts,
        "measured_params": measured,
        "analytic_vs_measured": (
            "EXACT MATCH" if counts["total_params"] == measured else
            "MISMATCH: analytic=%d measured=%d" % (counts["total_params"], measured)
        ),
        "checkpoint_bytes_measured": 345596357,
        "compatible_checkpoints": [
            "runs/v160_pretrain/checkpoint_last.pt",
            "runs/v180_sft/checkpoint_last.pt",
            "runs/l02_sft_hp/checkpoint_last.pt",
            "runs/l03_pretrain/checkpoint_last.pt",
            "runs/l04_e1/checkpoint_last.pt",
            "runs/l04_e2/checkpoint_last.pt",
            "runs/l04_e3/checkpoint_last.pt",
        ],
        "notes": [
            "Tied embeddings: lm_head reuses tok_emb; only one vocab x d matrix is counted/stored.",
            "GQA: 8 query heads, 4 KV heads, head_dim 48.",
            "12 layers x (attn 442,368 + mlp 1,769,472 + norms 768) = 26,551,296 + tied embedding 2,235,552 + final norm 384 = 28,787,328.",
        ],
    }


def audit_scenario(key: str, cfg: dict, baseline_params: int) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("vocab_size", 5822)
    cfg.setdefault("tie_embeddings", True)
    cfg.setdefault("rope_theta", 10000.0)
    cfg.setdefault("norm_eps", 1e-05)
    cfg.setdefault("max_seq_len", cfg.get("context", 2048))
    counts = count_parameters(cfg)
    return {
        "scenario": key,
        "name": cfg["name"],
        "config": {
            "d_model": cfg["d_model"],
            "n_layers": cfg["n_layers"],
            "n_heads": cfg["n_heads"],
            "n_kv_heads": cfg["n_kv_heads"],
            "ffn_dim": cfg["ffn_dim"],
            "max_seq_len": cfg["max_seq_len"],
            "vocab_size": cfg["vocab_size"],
            "tie_embeddings": True,
            "rope_theta": 10000.0,
        },
        "params": counts,
        "params_ratio_vs_baseline": round(counts["total_params"] / baseline_params, 3),
        "headroom_note": cfg.get("notes", ""),
        "architecture_changes_required": [
            "none — same module family (RMSNorm + RoPE + GQA + SwiGLU, tied embeddings)",
            "tokenizer unchanged (vocab 5,822; same specials; no embedding resize)",
        ],
        "checkpoint_compatibility": {
            "warmstart_from_existing": False,
            "reason": "different d_model/n_layers; all Linear/Embedding weights change shape; tokenizer reusable (vocabulary is unchanged)",
            "partial_init_possible": "tied embedding rows could be re-used only if d_model matched; it does not",
        },
        "risks": [
            "RoPE extrapolation beyond 2048 context is unvalidated (affects B and C)",
            "same recipe family; gains come from data and compute, not architecture",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l05_architecture_audit.json")
    args = parser.parse_args(argv)

    scenarios = _yaml()
    baseline_cfg = dict(scenarios["baseline"])
    current = audit_current(baseline_cfg, scenarios["baseline"]["measured_params"])
    baseline_params = current["params"]["total_params"]

    out = {
        "schema": "taksh.l05_architecture_audit/1",
        "generated_by": "evaluation/l05_architecture_audit.py",
        "current_baseline": current,
        "scenarios": [
            audit_scenario(key, cfg, baseline_params)
            for key, cfg in scenarios["architecture_scenarios"].items()
            if key != "current"
        ],
        "feasibility_verdict": {
            "architecture_blocks": False,
            "reason": (
                "The architecture family (RMSNorm + RoPE + GQA + SwiGLU, tied embeddings) "
                "scales to 100M/300M/1B by changing width/depth/ffn only. No module family "
                "change is required. Existing checkpoints cannot warmstart the scaled models "
                "(all weights change shape) but the tokenizer is fully reusable."
            ),
            "architecture_risk_items": [
                "RoPE beyond 2048 context unvalidated — validated only to 2048",
                "GQA ratio 2:1 retained at all three scales; nothing else changes",
                "vocab 5,822 is small; output projection stays tied and cheap",
            ],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("current baseline params:", current["params"]["total_params"], current["analytic_vs_measured"])
    for s in out["scenarios"]:
        print(f"{s['name']:24s} params={s['params']['total_params']:>12,}  ratio={s['params_ratio_vs_baseline']}x")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
