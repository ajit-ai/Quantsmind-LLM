"""Local resource inventory and feasibility (Taksh v1.7.0).

Read-only inspection of what this host can actually do, plus a transparent
resource model for the first authorized experiment. Nothing is installed,
provisioned or configured.

Machine-identifying details are reduced: the processor string is hashed, and
no serial numbers, host names or user paths are recorded.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil

GIB = 1024 ** 3


def _processor_label() -> str:
    label = platform.processor() or platform.machine() or "unknown"
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


def hardware_inventory() -> dict:
    """What this host offers, read-only."""
    inventory = {
        "processor": _processor_label(),
        "logical_cores": os.cpu_count(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "disk_free_gib": None,
        "total_ram_gib": None,
        "accelerator": "none",
        "torch_version": None,
        "torch_threads": None,
    }
    try:
        usage = shutil.disk_usage(".")
        inventory["disk_free_gib"] = round(usage.free / GIB, 2)
    except OSError:
        pass
    try:
        import torch

        inventory["torch_version"] = torch.__version__
        inventory["torch_threads"] = int(torch.get_num_threads())
        if torch.cuda.is_available():
            inventory["accelerator"] = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            inventory["accelerator"] = "mps"
    except Exception:  # noqa: BLE001 - torch is optional for this report
        pass
    try:
        import psutil  # type: ignore

        inventory["total_ram_gib"] = round(psutil.virtual_memory().total / GIB, 2)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    return inventory


def memory_model(*, parameters: int, vocab_size: int, d_model: int, layers: int,
                  context_length: int, batch_size: int = 1,
                  bytes_per_parameter: int = 4) -> dict:
    """Weights, optimizer and activation memory for a geometry."""
    weights = parameters * bytes_per_parameter
    optimizer = weights * 2  # AdamW first and second moments
    activation = layers * batch_size * context_length * d_model * bytes_per_parameter
    return {
        "weight_gib": weights / GIB,
        "optimizer_gib": optimizer / GIB,
        "activation_gib": activation / GIB,
        "training_total_gib": (weights + optimizer + activation) / GIB,
    }


def runtime_model(*, tokens: int, measured_tokens_per_second: float | None) -> dict:
    """Runtime for a token budget, from measured throughput when available."""
    if not measured_tokens_per_second:
        return {
            "tokens": tokens,
            "seconds": None,
            "hours": None,
            "basis": "unknown: no measured throughput is available for this configuration",
        }
    seconds = tokens / measured_tokens_per_second
    return {
        "tokens": tokens,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600, 3),
        "basis": f"measured {measured_tokens_per_second:.1f} tokens/s on this host",
    }


def feasibility_verdict(inventory: dict, training_gib: float) -> str:
    """The practical decision for a CPU-only first experiment."""
    accelerator = inventory.get("accelerator")
    free_disk = inventory.get("disk_free_gib")
    if accelerator == "none" and (free_disk is None or free_disk < 5):
        return "INSUFFICIENT_RESOURCE_EVIDENCE"
    if accelerator == "none" and training_gib > 4:
        return "S2_RESOURCE_CONSTRAINED"
    return "S2_DIAGNOSTIC_FEASIBLE"
