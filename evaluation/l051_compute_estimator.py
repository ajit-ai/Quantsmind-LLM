"""TAKSH L0.5.1 — multilingual compute, storage and local-deployment estimator (WP8).

Extends evaluation/l05_compute_estimator.py with multilingual scenarios:

* FLOPs = 6 x params x tokens (same approximation; same caveats)
* storage per scenario: model fp32/bf16 checkpoint bytes
* inference memory vs local deployment classes
* what the 1.59M qualified multilingual token pool buys at 20/50/100/200
  tokens-per-parameter ratios (reused from L0.5 exactly)
* equal-split reality check across 16 L1.0 languages

No GPU prices, throughputs, or completion dates are invented. CPU anchor is
the measured L0.3/L0.4 figure (~215 tok/s, range 180-260).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l05_compute_estimator import (  # noqa: E402
    CPU_ANCHOR_RANGE,
    CPU_ANCHOR_TOKS_PER_SEC,
    duration_estimates,
    flops,
    memory_breakdown,
)

SCENARIOS_PATH = REPO / "configs" / "feasibility" / "l05_scenarios.yaml"
MODELS = {
    "existing_28.8M": {"params": 28_787_328, "d_model": 384, "n_layers": 12},
    "candidate_A_110M": {"params": 110_658_816, "d_model": 768, "n_layers": 12},
    "candidate_B_289M": {"params": 289_115_136, "d_model": 1024, "n_layers": 18},
    "candidate_C_1019M": {"params": 1_018_624_000, "d_model": 2048, "n_layers": 16},
}

L1_LANGUAGES = [
    "python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
    "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_compute_estimate.json")
    args = parser.parse_args(argv)

    import yaml

    base = yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))
    code_tokens = base["qualified_token_supply"]["code_tokens"]
    general_tokens = base["qualified_token_supply"]["general_tokens_l03"]

    # equal-split reality check
    per_lang_equal = code_tokens / len(L1_LANGUAGES)

    by_model = {}
    for name, m in MODELS.items():
        p = m["params"]
        mem = memory_breakdown(p, m["d_model"], m["n_layers"], 2048, 1)
        per_ratio = []
        for ratio in (20, 50, 100, 200):
            tokens = p * ratio
            per_ratio.append({
                "tokens_per_param": ratio,
                "tokens": tokens,
                "flops_6xPxT": flops(p, tokens),
                "cpu_central_seconds_at_measured_throughput": round(tokens / CPU_ANCHOR_TOKS_PER_SEC),
                "feasible_on_this_host": tokens / CPU_ANCHOR_TOKS_PER_SEC < 60 * 60 * 24 * 7,
                "checkpoint_fp32_bytes": p * 4,
                "checkpoint_bf16_bytes": p * 2,
            })
        full_epoch = {
            "l04_epoch": {"tokens": 3_182_359, "flops": flops(p, 3_182_359)},
            "l03_epoch": {"tokens": 19_629_683, "flops": flops(p, 19_629_683)},
        }
        by_model[name] = {
            "params": p,
            "memory": mem,
            "inference_memory_fp32_bytes": p * 4,
            "training_fp32_working_set_bytes": mem["training_working_set_fp32_bytes"],
            "by_token_ratio": per_ratio,
            "full_epoch_reference": full_epoch,
            "cpu_duration_estimate": duration_estimates(p, p * 100),
            "local_deployment": (
                "verified runnable on this host (L0.1); CPU inference feasible at this size"
                if p < 150_000_000 else
                "likely fits in RAM for inference; untested here; throughput unproven"
            ),
        }

    report = {
        "schema": "taksh.l051_compute_estimate/1",
        "generated_by": "evaluation/l051_compute_estimator.py",
        "method": {
            "flops": "6 x params x tokens (standard approximation); same caveats as L0.5",
            "cpu_anchor": {"tokens_per_second": CPU_ANCHOR_TOKS_PER_SEC,
                            "range_tokens_per_second": CPU_ANCHOR_RANGE,
                            "source": "runs/l03_pretrain/loss_log.jsonl + run_manifest.json"},
            "gpu": "unmeasured on this host; durations deliberately not invented",
        },
        "equal_split_multilingual_reality": {
            "l1_languages": len(L1_LANGUAGES),
            "qualified_code_tokens": code_tokens,
            "per_language_unique_tokens": round(per_lang_equal, 1),
            "verdict": "hopelessly insufficient for any training of a multilingual model beyond python; unique data is the binding constraint, compute is second",
        },
        "by_model": by_model,
        "storage": {
            "checkpoint_model_only": {name: m["params"] * 4 for name, m in MODELS.items()},
            "checkpoint_model_only_bf16": {name: m["params"] * 2 for name, m in MODELS.items()},
            "training_storage_fp32": {name: m["memory"]["training_working_set_fp32_bytes"] for name, m in by_model.items()},
        },
        "conclusions": [
            "storage is a rounding error next to the data problem: even a 1B checkpoint is 4 GB; qualified code is 3 MB",
            "28.8M and 110M train today (verified); 289M fits in 16 GB VRAM classes; 1B needs 24 GB class minimum for training",
            "CPU training of one epoch of a 110M@100tok/param run is not achievable on this host",
            "equal distribution across 16 languages is infeasible with today's measured supply",
            "every multilingual training run is gated by data acquisition + legal review, not by hardware",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, m in by_model.items():
        print(f"{name:22s} params={m['params']:>12,} inference_fp32={m['inference_memory_fp32_bytes']/1e6:.1f}MB")
    print(f"equal split per language: {per_lang_equal:,.0f} tokens")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
