"""Validation without touching weights (M3).

Two properties this module exists to guarantee:

1. **Validation never updates weights.** Every pass runs under
   ``torch.no_grad()`` and ``model.eval()``, and the caller restores training
   mode afterwards.
2. **The same number is comparable across steps.** Validation windows are drawn
   from a fixed seed, reset before every pass, so a change in validation loss
   reflects the model rather than a different sample.

Perplexity is only reported when it is meaningful. ``exp(loss)`` on a loss far
above the uniform-random baseline ``ln(vocab_size)`` is a very large number that
tells the reader nothing, and it overflows readily, so it is reported as ``None``
with an explicit reason rather than as a fabricated figure.

Validation data is loaded from a separate token artifact, so validation examples
cannot appear in an optimizer step. That separation is enforced at load time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


@dataclass
class ValidationResult:
    """Outcome of one validation pass."""

    loss: float | None = None
    perplexity: float | None = None
    perplexity_note: str | None = None
    tokens: int = 0
    batches: int = 0
    windows: list[int] = field(default_factory=list)
    finite: bool = True
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def perplexity_from_loss(loss: float | None, vocab_size: int) -> tuple[float | None, str | None]:
    """Perplexity, or ``(None, reason)`` when it would not be interpretable."""
    if loss is None or not math.isfinite(loss):
        return None, "loss is not finite"
    try:
        ppl = math.exp(loss)
    except OverflowError:
        return None, f"exp({loss:.4f}) overflows; perplexity is not interpretable"
    if not math.isfinite(ppl):
        return None, "exp(loss) is not finite"
    uniform = math.log(vocab_size) if vocab_size > 0 else None
    if uniform is not None and loss > uniform:
        return None, (
            f"loss {loss:.4f} exceeds the uniform-random baseline "
            f"ln({vocab_size}) = {uniform:.4f}; perplexity would imply a better "
            "model than chance and is reported as null instead"
        )
    return ppl, None


def assert_separate(train_tokens, val_tokens, val_name="validation"):
    """Refuse to evaluate on data that is also being trained on.

    Comparing the two artifact paths is a cheap, strong check: the same file
    passed as both train and validation data is the classic way a reported
    metric stops meaning anything.
    """
    if train_tokens is None or val_tokens is None:
        return
    if str(train_tokens) == str(val_tokens):
        raise ValueError(
            f"the {val_name} token artifact is the same file as the training "
            f"artifact ({val_tokens}). Validation on training data produces a "
            "meaningless number. Point data.val_tokens_path at separate data."
        )


def evaluate(model, dataset, seq_len: int, batches: int, seed: int,
             device: str = "cpu", batch_size: int = 1,
             vocab_size: int | None = None) -> ValidationResult:
    """Run ``batches`` fixed windows through the model and average the loss.

    The dataset's RNG is re-seeded immediately before drawing, so repeated calls
    with the same ``seed`` score exactly the same windows.
    """
    import numpy as np
    import torch

    result = ValidationResult(batches=batches)
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for index in range(batches):
                dataset.rng = np.random.default_rng(seed)
                try:
                    x, y = dataset.batch(batch_size, seq_len)
                except ValueError as exc:
                    result.errors.append(str(exc))
                    break
                result.windows.append(int(x[0, 0].item()))
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, targets=y)
                if logits.shape[-1] != (vocab_size or logits.shape[-1]):
                    result.errors.append(
                        f"batch {index}: logits width {logits.shape[-1]} does not "
                        f"match expected vocab {vocab_size}"
                    )
                    break
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]).float(), y.reshape(-1)
                )
                value = float(loss.item())
                if not math.isfinite(value):
                    result.finite = False
                    result.errors.append(f"batch {index}: non-finite loss {value}")
                    break
                total_loss += value
                total_tokens += int(y.numel())
        result.tokens = total_tokens
        if total_tokens and result.finite:
            result.loss = total_loss / batches
            result.perplexity, result.perplexity_note = perplexity_from_loss(
                result.loss, vocab_size or 0
            )
    finally:
        if was_training:
            model.train()
    return result