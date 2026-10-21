"""Deterministic execution helpers (v1.1.0).

The M7 run could not answer "would this command produce the same run again?"
because seeds were not plumbed into torch or numpy, and deterministic kernels were
not requested. This module is the single place that answers the question.

Limits are stated rather than hidden: deterministic algorithms are requested, not
guaranteed. Some CUDA kernels have no deterministic variant; when that happens
PyTorch raises, the caller records the deviation in the run manifest, and the run
is reported as "nondeterministic operation: <name>" instead of silently varying.
"""

from __future__ import annotations

import os
import random


def set_determinism(seed: int, *, strict: bool = False) -> dict:
    """Seed every RNG the training loop touches and request deterministic kernels.

    Returns the settings applied so a caller can record them. ``strict=True``
    turns a missing deterministic kernel into an error instead of a warning.
    """
    import numpy as np
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(bool(strict), warn_only=not strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": int(seed),
        "torch_deterministic": bool(strict),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
