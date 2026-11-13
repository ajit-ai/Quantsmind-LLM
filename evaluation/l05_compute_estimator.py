"""TAKSH L0.5 — WP4: reproducible compute and memory estimator.

Dense-transformer training FLOPs estimated as  6 x parameters x tokens  (the
standard forward+backward approximation). Assumptions and limitations are
listed in the output and never hidden. No GPU prices are invented; compute-
hour estimates are reported with an uncertainty range, and USD amounts are
only produced if the owner supplies a --gpu-price-per-hour value.

Writes reports/l05_compute_estimate.json.
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


# Measured anchor from this repository: L0.3 run, ~2048 tokens per ~9.3 s
# (see runs/l03_pretrain/loss_log.jsonl and run_manifest.json).
CPU_ANCHOR_TOKS_PER_SEC = 215
CPU_ANCHOR_RANGE = (180, 260)


def flops(params: int, tokens: int) -> float:
    """Training FLOPs, forward+backward, dense multiplication. Exact within its
    assumption set; attention-quadratic terms are neglected (standard practice
    for dense models with d_model ~<= 2048; |d| << T_max attention share ignored)."""
    return 6.0 * params * tokens


def memory_breakdown(params: int, d_model: int, n_layers: int, seq_len: int, micro_batch: int) -> dict:
    """Byte-level breakdown. Two precision classes are shown.

    fp32_naive:  weights 4P + grads 4P + Adam(m,v) 8P = 16P bytes.
    bf16_mixed:  weights 2P + grads 2P + master weights 4P + Adam(m,v) 8P = 16P bytes.
    Both are ~16 bytes/parameter; the bf16 class saves bandwidth, not capacity.
    """
    weights_fp32 = params * 4
    weights_bf16 = params * 2
    grads_fp32 = params * 4
    adam_state = params * 8
    fp32_total = weights_fp32 + grads_fp32 + adam_state
    bf16_total = weights_bf16 * 1 + params * 2 + params * 4 + adam_state  # bf16 grads + fp32 master + Adam fp32
    # Activation rule-of-thumb for a pre-norm transformer, bytes (fp32 storages):
    # ~10 tensors x bsz x seq x d x layers + attention workspace (seq^2 terms small at 512-2048)
    activation_est = 34 * micro_batch * seq_len * d_model * n_layers
    return {
        "weights_fp32_bytes": weights_fp32,
        "weights_bf16_bytes": weights_bf16,
        "grads_fp32_bytes": grads_fp32,
        "adam_m_v_bytes": adam_state,
        "training_working_set_fp32_bytes": fp32_total,
        "training_working_set_bf16_bytes": bf16_total,
        "activation_estimate_bytes": activation_est,
        "activation_estimate_note": "34 x micro_batch x seq x d_model x layers (approx); implementation-dependent; includes norm/residual/MLP buffers, excludes attention scores blow-up at seq > 2048",
        "checkpoint_model_only_bytes": params * 4,
        "checkpoint_with_optimizer_bytes": fp32_total + params * 2,
        "checkpoint_model_bf16_bytes": params * 2,
    }


def scenario_estimates(name: str, cfg: dict, params: int, tokens: int) -> dict:
    return {
        "tokens": tokens,
        "flops_6xPxT": flops(params, tokens),
        "memory": memory_breakdown(params, cfg["d_model"], cfg["n_layers"], 2048, 1),
    }


def hardware_feasibility(fp32_set: int, activation_est: int, hw: dict, params: int) -> dict:
    vram = hw["vram_gb"] * (1 << 30)
    budget = int(vram * 0.75)  # 25% reserve for fragmentation/workspace
    fits_train = budget >= fp32_set + activation_est
    fits_inference = (hw["vram_gb"] << 30) >= params * 4 * 1.2
    if vram == 0:
        verdict = "CPU-only: trains slowly but predictably; measured ~215 tok/s for 28.8M model"
        usd = None
    elif fits_train:
        verdict = "FITS training working set + activations with >= 25% headroom"
        usd = None
    elif fits_inference:
        verdict = "Inference-only; full training working set does not fit"
        usd = None
    else:
        verdict = "DOES NOT FIT for from-scratch training (weights alone exceed budget)"
        usd = None
    return {
        "hardware": hw["label"],
        "vram_gb": hw["vram_gb"],
        "usable_vram_bytes_for_training": budget if vram else 0,
        "verdict": verdict,
        "fits_training": bool(fits_train),
        "fits_inference": bool(fits_inference),
    }


def duration_estimates(params: int, tokens: int) -> dict:
    flops_total = flops(params, tokens)
    cpu_seconds = tokens / CPU_ANCHOR_TOKS_PER_SEC
    cpu_seconds_low = tokens / CPU_ANCHOR_RANGE[1]
    cpu_seconds_high = tokens / CPU_ANCHOR_RANGE[0]
    return {
        "cpu_anchor_tokens_per_second": CPU_ANCHOR_TOKS_PER_SEC,
        "cpu_range_tokens_per_second": CPU_ANCHOR_RANGE,
        "cpu_seconds_central": round(cpu_seconds),
        "cpu_seconds_fast": round(cpu_seconds_low),
        "cpu_seconds_slow": round(cpu_seconds_high),
        "flops": flops_total,
        "assumption": "CPU duration measured from this repository's L0.3 run (~9.3 s/step, 2048 tokens/step); GPU durations require utilization assumption which is deliberately NOT invented — see uncertainty note",
        "gpu_duration_uncertainty": (
            "GPU runtime cannot be estimated honestly without a measured benchmark on the exact stack. "
            "Order-of-magnitude only: 100M-300M-class, 2-10B tokens, A100-class card: hours-to-days range. "
            "Owner supplies --gpu-price-per-hour to price it; no price is invented here."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l05_compute_estimate.json")
    parser.add_argument("--gpu-price-per-hour", type=float, default=None,
                        help="optional owner-supplied USD/GPU-hour; unpriced by default")
    args = parser.parse_args(argv)

    scenarios = _yaml()
    hw_classes = scenarios["hardware_classes"]

    model_table = {
        "existing_28.8M": (scenarios["baseline"], 28_787_328),
        "candidate_A_110M": (scenarios["architecture_scenarios"]["A"], 110_658_816),
        "candidate_B_289M": (scenarios["architecture_scenarios"]["B"], 289_115_136),
        "candidate_C_1019M": (scenarios["architecture_scenarios"]["C"], 1_018_624_000),
    }
    token_scenarios = scenarios["token_ratios"]

    report = {
        "schema": "taksh.l05_compute_estimate/1",
        "generated_by": "evaluation/l05_compute_estimator.py",
        "method": {
            "flops": "training FLOPs = 6 x parameters x training tokens (standard forward+backward approximation)",
            "limitations": [
                "attention-quadratic term ignored (valid for d_model <= 2048, seq <= 2048-4096)",
                "activation memory is a rule-of-thumb (34 x bsz x seq x d x layers), not a profiler measurement",
                "no comm/checkpoint I/O overhead included",
                "GPU durations carry no utilization claim; only CPU duration is anchored to measurement",
            ],
            "assumptions": [
                "dense (non-MoE) transformer; all params trained",
                "fp32 or bf16 mixed precision; ~16 bytes/parameter working set either way",
                "Adam(W) optimizer with fp32 state",
            ],
        },
        "cpu_anchor": {
            "measured_from": "runs/l03_pretrain/loss_log.jsonl + run_manifest.json (~9.3 s/step, 2048 tokens/step)",
            "tokens_per_second": CPU_ANCHOR_TOKS_PER_SEC,
            "range": CPU_ANCHOR_RANGE,
        },
        "gpu_price_per_hour": args.gpu_price_per_hour,
        "matrix": {},
        "hardware_feasibility": {},
        "cpu_duration_estimates": {},
    }

    for name, (cfg, params) in model_table.items():
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        rows = []
        for ratio in token_scenarios:
            tokens = params * ratio
            rows.append({
                "tokens_per_param": ratio,
                "tokens": tokens,
                **scenario_estimates(name, cfg, params, tokens),
                "checkpoint_model_only_gb": round(params * 4 / 1e9, 3) if isinstance(params, int) else None,
            })
        report["matrix"][name] = {
            "params": params,
            "by_token_ratio": rows,
            "for_reference_full_l04_epoch": scenario_estimates(name, cfg, params, 3_182_359),
        }

    for name, (cfg, params) in model_table.items():
        mem = report["matrix"][name]["by_token_ratio"][0]["memory"]
        fp32_set = mem["training_working_set_fp32_bytes"]
        act = mem["activation_estimate_bytes"]
        report["hardware_feasibility"][name] = {
            "params": params,
            "training_working_set_fp32_bytes": fp32_set,
            "activation_estimate_bytes": act,
            "verdicts": [
                hardware_feasibility(fp32_set, act, hw, params)
                for hw in hw_classes.values()
            ],
        }

    # CPU durations for the current model and the practical next step (A @ 100:1)
    for name, (cfg, params) in model_table.items():
        ratio = 100
        report["cpu_duration_estimates"][name] = {
            **duration_estimates(params, params * ratio),
            "note": f"duration at {ratio} tokens/parameter (CPU anchor); GPU duration requires owner benchmark; not invented",
        }

    if args.gpu_price_per_hour is not None:
        report["gpu_pricing"] = {
            "gpu_hours": "see matrix + gpu_duration_uncertainty; owner-supplied anchor required per workload",
            "note": "GPU hour counts still require a measured throughput anchor on the target stack; pricing cannot be invented honestly without it.",
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, entry in report["hardware_feasibility"].items():
        fits = sum(1 for v in entry["verdicts"] if v["fits_training"])
        print(f"{name:22s} params={entry['params']:>12,}  training_fits={fits}/{len(entry['verdicts'])}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
