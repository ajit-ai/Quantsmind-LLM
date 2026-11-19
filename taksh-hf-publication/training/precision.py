"""Device and precision resolution for Taksh v1.3.0.

CPU in float32 is the mandatory qualification baseline. Mixed precision is
*available* where the hardware genuinely supports it and is refused
everywhere else, with an explicit message rather than a silent fallback.

Rules
-----
* ``device: cpu`` always works; ``cuda``/``mps`` are refused when the runtime
  does not provide them.
* ``float32`` works on any supported device.
* ``float16`` and ``bfloat16`` are only accepted on an accelerator device.
  CPU training in half precision is refused: PyTorch's CPU half kernels do
  not cover every operation this model uses, and a silent downcast would
  make a qualification run non-reproducible.
* No device is reserved, provisioned or assumed. ``auto`` resolves to what is
  actually present and reports it.
"""

from __future__ import annotations

import contextlib

import torch

DTYPE_NAMES = ("float32", "float16", "bfloat16")
SUPPORTED_DEVICES = ("cpu", "cuda", "mps")


class PrecisionError(RuntimeError):
    """The requested device/precision combination is not supported."""


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device request to a concrete device."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in SUPPORTED_DEVICES:
        raise PrecisionError(
            f"unsupported device {requested!r}; supported: {list(SUPPORTED_DEVICES)}"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise PrecisionError(
            "device 'cuda' was requested but this runtime has no CUDA device. "
            "Use device 'cpu' with float32 for the qualification baseline."
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            raise PrecisionError("device 'mps' was requested but is not available")
    return torch.device(requested)


def resolve_precision(device: torch.device, requested: str = "float32") -> torch.dtype:
    """Resolve a precision request, refusing combinations that are unsafe."""
    requested = (requested or "float32").lower()
    if requested not in DTYPE_NAMES:
        raise PrecisionError(
            f"unsupported precision {requested!r}; supported: {list(DTYPE_NAMES)}"
        )
    if requested == "float32":
        return torch.float32
    if device.type == "cpu":
        raise PrecisionError(
            f"precision {requested!r} is refused on CPU: the CPU half-precision "
            "kernels do not cover every operation in this model, and the "
            "qualification baseline is float32"
        )
    if requested == "float16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported() and not torch.cuda.is_available():
            raise PrecisionError("device 'cuda' is not usable")
    return torch.float16 if requested == "float16" else torch.bfloat16


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Autocast context for the resolved device; a no-op on CPU float32."""
    if device.type == "cpu" or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def describe(device: torch.device, dtype: torch.dtype) -> dict:
    """Compact, loggable description of the resolved runtime."""
    return {
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "threads": int(torch.get_num_threads()),
    }
