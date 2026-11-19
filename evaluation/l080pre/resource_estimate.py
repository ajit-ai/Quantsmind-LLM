"""TAKSH L0.8.0-PRE — E: resource and hardware preflight (read-only).

Measures current host state (psutil-safe read-only), then produces
transparent estimates for the canonical 111M training workload. Never
estimates from purchased/cloud sources; never stores provisioned anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PARAMS = 110_658_816


def verify() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        free = None
        try:
            import shutil

            free = shutil.disk_usage(REPO).free
        except OSError:
            free = None
        measured = {
            "available_bytes": int(vm.available),
            "total_bytes": int(vm.total),
            "percent_used": float(vm.percent),
            "cpu_count": psutil.cpu_count(logical=True),
            "disk_free_bytes": int(free) if free else None,
            "tool": "psutil (+shutil.disk_usage)",
            "labels": "measured point-in-time",
        }
    except Exception:
        measured = {"available_bytes": None, "measured": False, "labels": "unavailable"}

    # estimates for canonical 111M (AdamW fp32 standard)
    fp32_weights = PARAMS * 4
    fp32_grads = PARAMS * 4
    fp32_adam_m = PARAMS * 4
    fp32_adam_v = PARAMS * 4
    fp32_total = fp32_weights + fp32_grads + fp32_adam_m + fp32_adam_v
    estimates = {
        "fp32_weights_bytes": fp32_weights,
        "fp32_grads_bytes": fp32_grads,
        "fp32_adam_m_bytes": fp32_adam_m,
        "fp32_adam_v_bytes": fp32_adam_v,
        "fp32_optimizer_state_total_bytes": fp32_total,
        "bf16_weights_bytes": PARAMS * 2,
        "bf16_grads_bytes": PARAMS * 2,
        "bf16_adam_total_bytes": PARAMS * 6 + PARAMS * 8,   # weights+grads bf16 + fp32 m,v
        "activation_note": "several GiB at context 2048 with fp32 attention unless checkpointed",
        "labels": "estimates — algorithmic derivation from parameter count; no run has measured them",
    }
    gpu = {
        "model": "NVIDIA GeForce MX150",
        "vram_mib": 4096,
        "driver": "457.49",
        "cuda_driver_capability": "11.1",
        "torch_cuda_available": False,
        "viability": "NOT VIABLE — MX150 unusable with installed CPU torch build; insufficient for canonical training",
        "labels": "measured (nvidia-smi header + torch.cuda.is_available()==False)",
    }
    return {
        "schema": "taksh.l080pre_resource_estimate/1",
        "generated_by": "evaluation/l080pre/resource_estimate.py",
        "measured": measured,
        "estimates": estimates,
        "gpu": gpu,
        "verdict": "NOT FEASIBLE HERE — host cannot hold the training state; minimum viable = GPU ≥16 GiB VRAM, ≥16 GiB RAM, ≥80 GiB disk, owner-authorized",
        "clearly_separated": True,
        "note": "no cloud pricing consulted; no provisioning possible under this milestone",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_resource_estimate.json")
    args = parser.parse_args(argv)

    result = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"available={result['measured'].get('available_bytes')} B; estimated optimizer state={result['estimates']['fp32_optimizer_state_total_bytes']:,} B")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
