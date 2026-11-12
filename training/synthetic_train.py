"""Bounded synthetic training cells with stability instrumentation (v2.0.0).

A cell trains the S2 model on the *contrastive* synthetic curriculum and
records, per optimizer step: the loss, the gradient norm before clipping, the
parameter-update magnitude, and the ratio of the update to the parameter norm.
Artifacts are written through :mod:`training.experiment_artifacts`, so a run
directory can never be reused or overwritten.

This is a diagnostic harness. It trains only on locally generated synthetic
Python, it never touches the production corpus registry, and it never promotes
a checkpoint.
"""

from __future__ import annotations

import math
import time

import torch

from training.determinism import set_determinism
from training.objective import accumulation_weights, token_mean_loss
from training.schedules import WarmupCosineSchedule
from training.sequence import SequenceBuilder


def build_corpus(partitions: dict, split: str = "train") -> list:
    """Training records for a partition: the reference solution of each case."""
    return [
        {
            "text": case.reference,
            "case_id": case.case_id,
            "family": case.family,
            "operation": (case.params or {}).get("operation"),
        }
        for case in partitions[split]
    ]


def _windows(builder: SequenceBuilder, records: list) -> list:
    return builder.build_windows(
        [(r["case_id"], r["case_id"], r["text"]) for r in records]
    )


def train_cell(
    *,
    model_config: dict,
    tokenizer,
    train_records: list,
    val_records: list,
    lr: float,
    steps: int,
    context_length: int,
    grad_accum: int = 2,
    seed: int = 20261021,
    warmup: int = 2,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    init_checkpoint: str | None = None,
    artifacts=None,
    device: str = "cpu",
) -> dict:
    """Run one training cell and return its metrics."""
    from training.model import QMModel

    set_determinism(seed)
    model = QMModel(model_config).to(device)
    if init_checkpoint:
        from serving.local_inference import torch_load_checkpoint

        model.load_state_dict(torch_load_checkpoint(init_checkpoint)["model"])

    builder = SequenceBuilder(
        tokenizer,
        context_length=context_length,
        model_vocab=model_config["vocab_size"],
        policy="document",
        append_eos=True,
    )
    train_windows = _windows(builder, train_records)
    val_windows = _windows(builder, val_records)
    if not train_windows:
        raise ValueError("the training partition produced no windows")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    schedule = WarmupCosineSchedule(
        peak_lr=lr, warmup_steps=warmup, total_steps=steps, min_ratio=0.1
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(train_windows), generator=generator).tolist()

    history = []
    consumed_tokens = 0
    started = time.time()
    for step in range(steps):
        batches = []
        for micro in range(grad_accum):
            index = order[(step * grad_accum + micro) % len(order)]
            batches.append(builder.to_batch([train_windows[index]]))
        counts = [batch.supervised_tokens for batch in batches]
        weights = accumulation_weights(counts)
        optimizer.zero_grad(set_to_none=True)
        window_loss = 0.0
        for batch, weight in zip(batches, weights):
            logits, _ = model(batch.input_ids.to(device))
            summed, supervised = token_mean_loss(logits, batch.labels.to(device))
            if supervised == 0:
                continue
            micro_mean = summed / supervised
            (micro_mean * weight).backward()
            window_loss += float(micro_mean.item()) * weight
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )
        )
        param_norm = float(
            torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))
        )
        before = [p.detach().clone() for p in model.parameters()]
        learning_rate = schedule.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        delta = float(
            torch.sqrt(
                sum(
                    ((p.detach() - b) ** 2).sum()
                    for p, b in zip(model.parameters(), before)
                )
            )
        )
        consumed_tokens += sum(counts)
        history.append(
            {
                "step": step + 1,
                "loss": window_loss,
                "lr": learning_rate,
                "grad_norm_before_clip": grad_norm,
                "param_norm": param_norm,
                "update_norm": delta,
                "update_to_param_ratio": (delta / param_norm) if param_norm else None,
                "supervised_tokens": sum(counts),
                "finite_loss": math.isfinite(window_loss),
            }
        )
        if not math.isfinite(window_loss):
            break

    elapsed = time.time() - started
    val_loss = _validate(model, builder, val_windows, device)
    result = {
        "lr": lr,
        "steps": len(history),
        "requested_steps": steps,
        "tokens_consumed": consumed_tokens,
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "validation_loss": val_loss,
        "elapsed_seconds": round(elapsed, 2),
        "history": history,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "init_checkpoint": init_checkpoint,
        "seed": seed,
        "context_length": context_length,
        "grad_accum": grad_accum,
    }
    if artifacts is not None:
        artifacts.write_metric("training.json", result)
    return result, model


def _validate(model, builder, windows, device) -> float | None:
    """Mean token loss over the validation windows (no gradient)."""
    if not windows:
        return None
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for window in windows:
            batch = builder.to_batch([window])
            logits, _ = model(batch.input_ids.to(device))
            summed, supervised = token_mean_loss(logits, batch.labels.to(device))
            total += float(summed.item())
            count += supervised
    model.train()
    return (total / count) if count else None


def save_model_checkpoint(model, path) -> None:
    """Write a model state dict for later evaluation."""
    torch.save({"model": model.state_dict()}, path)
