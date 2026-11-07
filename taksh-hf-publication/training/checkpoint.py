"""Checkpoint compatibility checking (A2 / ``OPEN_ISSUES`` B3).

``training/sft.py`` loaded ``--init-ckpt`` with ``strict=False`` and then printed
"initialized from ..." regardless of what actually happened. That is how
``runs/code_sft`` ended up as a 495,808-parameter ``code_tiny`` model while the
operator believed it was continuing from the 16.3M pretrain: every parameter
failed to load, every parameter stayed random, and the run reported success.

This module makes the outcome explicit and checked:

* missing / unexpected / shape-mismatched parameters are all reported by name;
* shape mismatches are **never** tolerated, even in partial mode, because a
  mismatched tensor means the architecture differs;
* missing parameters are fatal by default;
* partial initialization exists, but only as an explicit opt-in that reports
  exactly what stayed random;
* there is no silent fallback to random weights: if loading fails, the caller is
  expected to raise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CheckpointIncompatible(RuntimeError):
    """The checkpoint does not match the target model."""


class LoadReport:
    """Outcome of a weight load: what matched, what did not."""

    def __init__(self, loaded, missing, unexpected, mismatched, partial=False):
        self.loaded = list(loaded)
        self.missing = list(missing)
        self.unexpected = list(unexpected)
        self.mismatched = list(mismatched)  # list of (name, ckpt_shape, model_shape)
        self.partial = partial

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and not self.mismatched

    def describe(self) -> str:
        lines = [
            f"loaded parameters   : {len(self.loaded)}",
            f"missing parameters  : {len(self.missing)}",
            f"unexpected          : {len(self.unexpected)}",
            f"shape mismatches    : {len(self.mismatched)}",
            f"partial load        : {self.partial}",
        ]
        for name in self.missing[:10]:
            lines.append(f"  MISSING    {name}")
        for name in self.unexpected[:10]:
            lines.append(f"  UNEXPECTED {name}")
        for name, ck, mk in self.mismatched[:10]:
            lines.append(f"  MISMATCH   {name}: checkpoint {tuple(ck)} vs model {tuple(mk)}")
        if len(self.missing) > 10 or len(self.unexpected) > 10 or len(self.mismatched) > 10:
            lines.append("  ... (truncated)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "loaded": len(self.loaded),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "mismatched": [
                {"name": n, "checkpoint_shape": list(c), "model_shape": list(m)}
                for n, c, m in self.mismatched
            ],
            "partial": self.partial,
            "ok": self.ok,
        }


def load_model_weights(
    model: nn.Module,
    state_dict: dict,
    *,
    allow_partial: bool = False,
    source: str = "checkpoint",
) -> LoadReport:
    """Load ``state_dict`` into ``model``, refusing silent partial results.

    ``allow_partial=True`` permits *missing* parameters only. Shape mismatches and
    unexpected keys remain fatal: both indicate the checkpoint was produced for a
    different architecture, which is exactly the mistake this exists to prevent.
    """
    model_state = model.state_dict()
    ckpt_state = dict(state_dict)

    missing = [k for k in model_state if k not in ckpt_state]
    unexpected = [k for k in ckpt_state if k not in model_state]
    mismatched = []
    compatible = {}
    for name, tensor in ckpt_state.items():
        if name not in model_state:
            continue
        target = model_state[name]
        if tuple(tensor.shape) != tuple(target.shape):
            mismatched.append((name, tuple(tensor.shape), tuple(target.shape)))
        else:
            compatible[name] = tensor

    report = LoadReport(
        loaded=sorted(compatible),
        missing=sorted(missing),
        unexpected=sorted(unexpected),
        mismatched=sorted(mismatched),
        partial=allow_partial,
    )

    if mismatched:
        raise CheckpointIncompatible(
            f"{source} does not match the model architecture.\n{report.describe()}\n"
            "Shape mismatches are never tolerated: the checkpoint was produced "
            "for a different model. Refusing to continue."
        )
    if unexpected:
        raise CheckpointIncompatible(
            f"{source} contains parameters the model does not define.\n"
            f"{report.describe()}\n"
            "Refusing to continue: this checkpoint is for a different "
            "architecture."
        )
    if missing and not allow_partial:
        raise CheckpointIncompatible(
            f"{source} is missing {len(missing)} required parameter(s).\n"
            f"{report.describe()}\n"
            "Refusing to silently start from random weights. If partial "
            "initialization is intended, pass allow_partial=True and treat the "
            "listed parameters as untrained."
        )

    model.load_state_dict(compatible, strict=False)
    return report


def init_from_checkpoint(
    model: nn.Module,
    path: str,
    *,
    allow_partial: bool = False,
    source_label: str | None = None,
) -> LoadReport:
    """Load a checkpoint file and validate it against ``model``."""
    label = source_label or path
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError as exc:
        raise CheckpointIncompatible(f"checkpoint not found: {path}") from exc

    if not isinstance(ckpt, dict):
        raise CheckpointIncompatible(
            f"{label} is not a checkpoint dictionary (got {type(ckpt).__name__}); "
            "expected a dict with a 'model' entry"
        )
    state = ckpt.get("model")
    if state is None:
        # Tolerate a bare state_dict, but say so.
        if all(hasattr(v, "shape") for v in ckpt.values()):
            state = ckpt
        else:
            raise CheckpointIncompatible(
                f"{label} has no 'model' entry and is not a bare state_dict"
            )
    return load_model_weights(model, state, allow_partial=allow_partial, source=label)