"""Continued training from the frozen v2.1.0 checkpoint for v2.2.0 cells."""

from __future__ import annotations

import math
import time

import torch

from training.determinism import set_determinism
from training.objective import accumulation_weights, token_mean_loss
from training.schedules import WarmupCosineSchedule
from training.supervised_train import (_batch_tensor, _validation_loss,
                                       build_batches, weighted_mean_loss)

DEFAULT_SEED = 20261022


def save_checkpoint(path, model, model_config, **metadata):
    payload = {"config": {"model": dict(model_config)},
               "model": model.state_dict()}
    payload.update(metadata)
    torch.save(payload, path)
    return path


def train_from_checkpoint(
    *,
    checkpoint_path,
    checkpoint_hash,
    tokenizer,
    examples,
    val_examples,
    lr: float,
    steps: int,
    max_length: int = 256,
    grad_accum: int = 2,
    seed: int = DEFAULT_SEED,
    identifier_config=None,
    warmup: int = 2,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    artifacts=None,
    device: str = "cpu",
):
    """Fine-tune the frozen checkpoint and return metrics plus the model."""
    from serving.local_inference import load_model

    set_determinism(seed)
    model, model_config = load_model(checkpoint_path, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=weight_decay)
    schedule = WarmupCosineSchedule(
        peak_lr=lr, warmup_steps=warmup, total_steps=steps, min_ratio=0.1)
    train_batches = build_batches(tokenizer, examples, max_length=max_length, identifier_config=identifier_config)
    val_batches = build_batches(tokenizer, val_examples, max_length=max_length, identifier_config=identifier_config)
    if not train_batches:
        raise ValueError("no trainable examples were produced")

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(train_batches), generator=generator).tolist()
    history = []
    consumed = 0
    started = time.time()
    for step in range(steps):
        window = []
        for micro in range(grad_accum):
            index = order[(step * grad_accum + micro) % len(order)]
            window.append(train_batches[index])
        counts = [item["supervised"] for item in window]
        weights = accumulation_weights(counts)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for item, weight in zip(window, weights):
            batch = _batch_tensor([item])
            logits, _ = model(batch.input_ids.to(device))
            micro_mean, supervised = weighted_mean_loss(
                logits, batch.labels.to(device), item.get("weights"))
            if supervised == 0 or micro_mean is None:
                continue
            (micro_mean * weight).backward()
            loss_value += float(micro_mean.item()) * weight
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=False))
        learning_rate = schedule.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        consumed += sum(counts)
        finite = math.isfinite(loss_value)
        history.append({
            "step": step + 1, "loss": loss_value, "lr": learning_rate,
            "grad_norm_before_clip": grad_norm,
            "supervised_tokens": sum(counts), "finite_loss": finite,
        })
        if not finite:
            break

    validation_loss = _validation_loss(model, val_batches, device)
    model.eval()
    result = {
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": checkpoint_hash,
        "lr": lr, "steps": len(history), "requested_steps": steps,
        "grad_accum": grad_accum, "max_length": max_length,
        "tokens_consumed": consumed,
        "supervised_tokens_per_example": round(
            consumed / max(len(history), 1), 2),
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "validation_loss": validation_loss,
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history, "train_examples": len(examples),
        "val_examples": len(val_examples), "seed": seed,
        "parameters": int(model.num_params()),
    }
    if artifacts is not None:
        artifacts.write_metric("training.json", result)
        payload = {"config": {"model": dict(model_config)},
                   "model": model.state_dict(), "step": len(history),
                   "seed": seed, "base_checkpoint_sha256": checkpoint_hash}
        path = artifacts.write_checkpoint("trained.pt", payload)
        from training.experiment_artifacts import file_sha256
        result["checkpoint_sha256"] = file_sha256(path)
    return result, model, model_config
