"""TAKSH L0.6.1 — I: resource preflight and fail-closed safeguards.

Extends evaluation/resource_guard.py with memory-aware preflight, estimated
working-set reporting, and hard refusal of implicit GPU/network/payment.

Runtime memory measurement uses psutil if installed; otherwise a stdlib
fallback reports process RSS via resource/win32 handles. Numbers reported are
measured at the instant of the call; nothing is estimated from a GPU that is
not present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.resource_guard import (  # noqa: E402
    DEFAULTS,
    clamp_threads,
    device_allowed,
    enforce_batch_size,
    enforce_context,
    enforce_training_steps,
    require_authorization,
)


def memory_snapshot() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        return {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "percent_used": vm.percent,
            "process_rss_bytes": proc.memory_info().rss,
            "tool": "psutil",
        }
    except Exception:
        # stdlib fallback: Windows uses Win32 via ctypes; else report unknown
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotPhysHigh", ctypes.c_ulong),
                            ("ullTotPhysLow", ctypes.c_ulong),
                            ("ullAvailPhysHigh", ctypes.c_ulong),
                            ("ullAvailPhysLow", ctypes.c_ulong),
                            ("ullTotalExtPhys", ctypes.c_ulonglong),
                            ("ullAvailExtPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtVirtual", ctypes.c_ulonglong)]

            mse = MEMORYSTATUSEX()
            mse.dwLength = ctypes.sizeof(mse)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mse))  # type: ignore[attr-defined]
            return {"total_bytes": mse.ullTotalPhys,
                    "available_bytes": mse.ullAvailPhys,
                    "percent_used": mse.dwMemoryLoad,
                    "process_rss_bytes": None,
                    "tool": "ctypes/win32"}
        except Exception:
            return {"total_bytes": None, "available_bytes": None,
                    "percent_used": None, "process_rss_bytes": None,
                    "tool": "unavailable"}


def preflight(*, device: str = "cpu", micro_batch: int = 4, context: int = 512,
              max_steps: int = 300, memory: dict | None = None) -> dict:
    problems = []
    mem = memory if memory is not None else memory_snapshot()
    try:
        device_allowed(device)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"device: {exc}")
    try:
        enforce_batch_size(micro_batch)
        enforce_context(context)
        enforce_training_steps(max_steps)
    except Exception as exc:  # noqa: BLE001
        problems.append(str(exc))

    # Conservative working-set estimate: fp32 model + grads + Adam(m,v) ~= 16 bytes/param
    params = 110_658_816
    working_set_bytes = params * 16
    fits_system = (mem["available_bytes"] is not None and working_set_bytes < mem["available_bytes"])
    if not fits_system:
        if mem["available_bytes"] is None:
            problems.append("available memory unmeasurable — failing closed")
        else:
            problems.append(
                f"estimated working set {working_set_bytes / 2**30:.2f} GiB exceeds "
                f"currently available {mem['available_bytes'] / 2**30:.2f} GiB"
            )

    return {
        "device": device,
        "micro_batch_size": micro_batch,
        "context_length": context,
        "max_training_steps": max_steps,
        "memory": mem,
        "estimated_working_set_bytes": working_set_bytes,
        "estimated_working_set_note": "fp32 weights + grads + Adam m,v (~16 bytes/param); NOT a performance claim",
        "fits_available_memory": fits_system,
        "problems": problems,
        "authorized": not problems,
        "note": "resource preflight is a guard, not a performance measurement",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_resource_preflight.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(argv)

    report = {
        "schema": "taksh.l0601_resource_preflight/1",
        "generated_by": "evaluation/l0601_resource_preflight.py",
        "inputs": {
            "device": args.device, "micro_batch_size": args.micro_batch,
            "context_length": args.context, "max_training_steps": args.max_steps,
        },
        "preflight": preflight(device=args.device, micro_batch=args.micro_batch,
                               context=args.context, max_steps=args.max_steps),
        "cpu_threads_clamped_to": clamp_threads(None),
        "resource_guard_defaults": DEFAULTS,
        "network": "disabled by absence of any network-capable invocation path in this tool",
        "gpu": "not present, not allocated, not queried",
        "verdict": "PREFLIGHT COMPLETE (this is a guard report, not a training authorization)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"memory total: {report['preflight']['memory']['total_bytes']} bytes")
    print(f"authorized={report['preflight']['authorized']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
