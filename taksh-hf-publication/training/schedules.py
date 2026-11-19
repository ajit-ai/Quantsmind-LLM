"""Learning-rate schedules for Taksh v1.3.0.

A thin, stateful wrapper around the warmup-plus-cosine schedule that already
exists in :func:`training.pretrain.lr_at`, so the v1.3.0 trainer and the
historical pretraining path cannot drift apart numerically.

The schedule is an engineering default, not a claim about optimal
hyperparameters. What it does guarantee is that it is *reproducible*: the
learning rate for a given optimizer step is a pure function of the
configuration, and the schedule state can be checkpointed and restored so a
resumed run continues on the same curve.
"""

from __future__ import annotations

from training.pretrain import lr_at


class WarmupCosineSchedule:
    """Linear warmup into a cosine decay, with a documented floor."""

    def __init__(
        self,
        *,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_ratio: float = 0.1,
    ):
        if peak_lr <= 0:
            raise ValueError(f"peak_lr must be positive, got {peak_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError(f"min_ratio must be within [0, 1], got {min_ratio}")
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_ratio = float(min_ratio)

    def lr_at(self, step: int) -> float:
        """Learning rate for the optimizer step about to be taken.

        ``step`` is zero-based: step 0 is the first optimizer step. During
        warmup the rate rises linearly from ``peak_lr / warmup_steps``; at
        ``step == warmup_steps`` the schedule enters the cosine decay at the
        peak.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        return lr_at(
            step, self.total_steps, self.peak_lr, self.warmup_steps, self.min_ratio
        )

    def state_dict(self) -> dict:
        return {
            "type": "warmup_cosine",
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_ratio": self.min_ratio,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        if state.get("type") not in (None, "warmup_cosine"):
            raise ValueError(f"unsupported schedule type: {state.get('type')!r}")
        self.peak_lr = float(state.get("peak_lr", self.peak_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.min_ratio = float(state.get("min_ratio", self.min_ratio))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WarmupCosineSchedule(peak_lr={self.peak_lr}, "
            f"warmup_steps={self.warmup_steps}, total_steps={self.total_steps}, "
            f"min_ratio={self.min_ratio})"
        )
