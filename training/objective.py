"""Causal language-modelling objective for Taksh v1.3.0.

The loss is computed here rather than through ``QMModel.decode`` because the
model's built-in loss does not accept an ``ignore_index``: pretraining needs
padding and masked positions excluded from the objective, and the model must
not be modified for that.

Normalization is **token-mean over the accumulation window**: the summed
per-token loss of the whole gradient-accumulation window is divided by the
number of supervised tokens in that window. Computing the window's token
count before the forward pass makes gradient accumulation numerically
equivalent to a single larger batch, which is a testable property rather
than an approximation.

Shifted labels are explicit. A sequence of tokens ``t0..tn`` is trained as
inputs ``t0..t(n-1)`` and labels ``t1..tn``; the final input has no target
and the first token has no context, so neither is invented.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def shift_for_causal(tokens, pad_id: int):
    """Split a token sequence into inputs and next-token labels.

    ``tokens`` is a 1-D tensor of ids. The returned labels have ``IGNORE_INDEX``
    wherever the target is padding, so padded positions are never predicted.
    """
    if tokens.dim() != 1:
        raise ValueError(f"expected a 1-D token sequence, got shape {tuple(tokens.shape)}")
    if tokens.numel() < 2:
        raise ValueError("a sequence needs at least two tokens to form one prediction")
    inputs = tokens[:-1]
    labels = tokens[1:].clone()
    labels[labels == pad_id] = IGNORE_INDEX
    return inputs, labels


def token_mean_loss(logits, labels, ignore_index: int = IGNORE_INDEX):
    """Return ``(summed_loss, supervised_token_count)``.

    The summed loss is over the supervised tokens only. The caller divides by
    the count to obtain a mean.
    """
    if logits.dim() != 3:
        raise ValueError(f"expected [batch, time, vocab] logits, got {tuple(logits.shape)}")
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_labels = labels.reshape(-1)
    if flat_logits.size(0) != flat_labels.size(0):
        raise ValueError(
            f"logits and labels disagree on length: {flat_logits.size(0)} vs "
            f"{flat_labels.size(0)}"
        )
    count = int((flat_labels != ignore_index).sum().item())
    if count == 0:
        return torch.zeros((), dtype=torch.float32, device=logits.device), 0
    summed = F.cross_entropy(
        flat_logits, flat_labels, ignore_index=ignore_index, reduction="sum"
    )
    return summed, count


def accumulation_weights(counts) -> list:
    """Scale factors that make an accumulation window equal one large batch.

    Each microbatch's summed loss is multiplied by its own supervised-token
    count divided by the window total, so the window's total gradient equals
    the gradient of a single batch containing all of the window's tokens.
    """
    counts = [int(c) for c in counts]
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def mean_loss(summed_loss, count: int) -> float:
    """Token-mean loss as a float. Zero count yields 0.0, never a division error.

    Accepts a tensor or an already-summed float, so a caller can accumulate
    across microbatches without keeping the graph alive.
    """
    if count <= 0:
        return 0.0
    value = (
        summed_loss.item() if hasattr(summed_loss, "item") else float(summed_loss)
    )
    return value / count
