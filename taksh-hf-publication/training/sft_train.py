"""M4 instruction-tuning engine.

Reuses the M3 engine rather than duplicating it:

* ``training.checkpoint.load_model_weights`` -- base-model compatibility checking
* ``training.pretrain.save_checkpoint`` / ``capture_rng_state`` /
  ``restore_rng_state`` / ``lr_at``
* ``training.manifest.build_run_manifest`` -- the run record
* ``training.instruction_format`` -- formatting and loss masking
* ``training.sft_dataset`` -- the encoded instruction dataset

What is M4-specific: loading the M3 base checkpoint rather than random init,
recording the base-checkpoint identity in every artifact, and the freeze policy.

The base checkpoint is opened read-only and never written back.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from training.checkpoint import CheckpointIncompatible, load_model_weights
from training.instruction_format import IGNORE_INDEX, collate
from training.manifest import build_run_manifest, write_run_manifest
from training.pretrain import (
    capture_rng_state,
    lr_at,
    restore_rng_state,
    save_checkpoint,
)
from training.sft_dataset import EpochShuffler, InstructionDataset

FORMAT_NAME = "qm.plain_instruction/1"


def load_base_model(base_checkpoint, model_config_path, device="cpu"):
    """Load the M3 base. Raises rather than silently starting from random weights."""
    from training.model import QMModel

    path = Path(base_checkpoint)
    if not path.exists():
        raise CheckpointIncompatible(f"base checkpoint not found: {path}")

    cfg = yaml.safe_load(Path(model_config_path).read_text(encoding="utf-8"))["model"]
    model = QMModel(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if state is None:
        raise CheckpointIncompatible(
            f"{path} has no 'model' entry; refusing to fine-tune it"
        )
    report = load_model_weights(model, state, source=str(path))
    if not report.ok:
        raise CheckpointIncompatible(
            f"{path} did not fully load:\n{report.describe()}"
        )
    return model.to(device), cfg, {
        "path": str(path),
        "step": payload.get("step"),
        "tokens_processed": payload.get("tokens_processed"),
        "val_loss": payload.get("val_loss"),
        "parameters_loaded": len(report.loaded),
        "dataset_identity": payload.get("dataset_identity"),
        "tokenizer_identity": payload.get("tokenizer_identity"),
    }


def apply_freeze_policy(model, policy):
    """``none`` (full fine-tune) or a parameter-name substring to freeze."""
    if policy in (None, "none", "", []):
        frozen = 0
    else:
        terms = [policy] if isinstance(policy, str) else list(policy)
        frozen = 0
        for name, param in model.named_parameters():
            if any(term in name for term in terms):
                param.requires_grad_(False)
                frozen += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "policy": policy,
        "frozen_tensors": frozen,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def masked_loss(logits, labels):
    """Cross-entropy over supervised positions only.

    Guards the all-``-100`` batch, which would otherwise produce NaN and be
    quietly averaged into the loss.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    if int((flat_labels != IGNORE_INDEX).sum()) == 0:
        raise ValueError(
            "batch has no supervised tokens; masking left nothing to learn from"
        )
    return torch.nn.functional.cross_entropy(
        flat_logits.float(), flat_labels, ignore_index=IGNORE_INDEX
    )


def _checkpoint_extra(step, dataset_report, base_info, tokens_per_step,
                      resumed_tokens, val_loss, cfg, start_step=0):
    """Everything a resume needs, plus the M4 identity chain (M4 §18)."""
    return {
        "dataset_tokens": dataset_report["total_tokens"],
        "seq_len": cfg["data"]["max_length"],
        "tokens_per_step": tokens_per_step,
        "optimizer_steps_completed": step,
        "tokens_processed": resumed_tokens
        + max(0, step - start_step) * tokens_per_step,
        "rng_state": capture_rng_state(),
        "val_loss": val_loss,
        "instruction_format": FORMAT_NAME,
        "m4": {
            "base_checkpoint": base_info["path"],
            "base_step": base_info["step"],
            "training_strategy": "full fine-tune (assistant-only loss)"
            if (cfg.get("adaptation") or cfg.get("model") or {}).get("freeze_policy") in (None, "none")
            else f"freeze={(cfg.get('adaptation') or cfg.get('model') or {}).get('freeze_policy')}",
            "instruction_format": FORMAT_NAME,
            "dataset": cfg["data"]["train_jsonl"],
            "dataset_manifest": cfg["data"].get("dataset_manifest"),
            "tokenizer": cfg["data"]["tokenizer"],
            "seed": cfg.get("run", {}).get("seed"),
        },
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _validate(model, val_set, micro_bs, batches, seed, max_length, device,
              pad_id):
    """Validation without weight updates, on a fixed window stream."""
    previous = model.training
    model.eval()
    total, counted = 0.0, 0
    try:
        with torch.no_grad():
            stream = EpochShuffler(len(val_set), micro_bs, seed=seed)
            for _ in range(batches):
                items = [val_set[i] for i in stream.next_indices()]
                batch = collate(items, pad_id=pad_id)
                _, loss = model(
                    batch["input_ids"].to(device),
                    targets=batch["labels"].to(device),
                )
                total += float(loss.item())
                counted += 1
    finally:
        if previous:
            model.train()
    return total / counted if counted else None


def run(config_path, out_dir, max_steps=None, resume=False, device=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg, run_cfg = cfg["data"], cfg["run"]
    train_cfg, opt_cfg = cfg["training"], cfg["optimizer"]
    batch_cfg = cfg["batch"]

    if device is None:
        requested = (cfg.get("precision") or {}).get("device", "auto")
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if requested == "auto" else requested

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(data_cfg["tokenizer"])
    model, model_cfg, base_info = load_base_model(
        run_cfg["base_checkpoint"], run_cfg["base_model_config"], device
    )
    print(f"base checkpoint : {base_info['path']} "
          f"(step {base_info['step']}, {base_info['parameters_loaded']} tensors)")

    freeze_report = apply_freeze_policy(
        model, (cfg.get("adaptation") or cfg.get("model") or {}).get("freeze_policy")
    )
    print(f"freeze policy   : {freeze_report['policy']} "
          f"({freeze_report['trainable_parameters']:,} trainable / "
          f"{freeze_report['total_parameters']:,} total)")

    supervise_prompt = bool(data_cfg.get("supervise_prompt", False))
    train_set = InstructionDataset(
        data_cfg["train_jsonl"], tokenizer,
        max_length=data_cfg["max_length"],
        supervise_prompt=supervise_prompt, seed=run_cfg.get("seed", 42),
    )
    val_set = InstructionDataset(
        data_cfg["val_jsonl"], tokenizer,
        max_length=data_cfg["max_length"],
        supervise_prompt=supervise_prompt, seed=1234,
    )
    report_train = train_set.report()
    print(f"train examples  : {len(train_set)} encoded, "
          f"{len(train_set.rejected)} rejected, "
          f"{report_train['supervised_tokens']:,} supervised tokens")
    print(f"val examples    : {len(val_set)} encoded")

    micro_bs = batch_cfg["micro_batch_size"]
    accum = batch_cfg.get("grad_accum_steps", 1)
    max_length, pad_id = data_cfg["max_length"], 0
    tokens_per_step = micro_bs * accum * max_length
    shuffle = EpochShuffler(len(train_set), micro_bs, seed=run_cfg.get("seed", 42))
    val_seed = data_cfg.get("val_seed", 1234)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg["lr_peak"],
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    last_ckpt = out_path / "checkpoint_last.pt"
    best_ckpt = out_path / "checkpoint_best.pt"

    total_steps = max_steps or train_cfg["max_steps"]
    start_step, resumed_tokens, warnings = 0, 0, []
    if resume:
        if not last_ckpt.exists():
            raise SystemExit(f"--resume requested but no checkpoint at {last_ckpt}")
        payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        load_model_weights(model, payload["model"], source=str(last_ckpt))
        optim.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        resumed_tokens = int(payload.get("tokens_processed") or 0)
        if restore_rng_state(payload.get("rng_state")):
            print(f"resumed at step {start_step} (RNG state restored)")
        else:
            warnings.append("resumed without RNG state; best-effort continuity")
            print(f"resumed at step {step if False else start_step} (best-effort)")

    eval_every = train_cfg.get("eval_every_steps", 0)
    val_batches = train_cfg.get("val_batches", 8)
    ckpt_every = train_cfg.get("checkpoint_every_steps", 40)
    log_every = train_cfg.get("log_every_steps", 10)
    nan_every = (cfg.get("watchdog") or {}).get("nan_check_every_steps", 10)

    best_val = None
    history: list[dict] = []
    events: list[dict] = []
    model.train()
    started = time.perf_counter()
    start_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, opt_cfg["lr_peak"],
                   opt_cfg.get("warmup_steps", 0), opt_cfg.get("min_ratio", 0.1))
        for group in optim.param_groups:
            group["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(accum):
            items = [train_set[i] for i in shuffle.next_indices()]
            batch = collate(items, pad_id=pad_id)
            _, loss = model(batch["input_ids"].to(device),
                            targets=batch["labels"].to(device))
            (loss / accum).backward()
            loss_val += float(loss.item()) / accum

        if nan_every and step % nan_every == 0:
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                raise RuntimeError(
                    f"non-finite gradients in {len(bad)} parameter(s) at step "
                    f"{step} (first: {bad[0]}); aborting per watchdog policy"
                )
        if opt_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
        optim.step()

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}")

        val_loss = None
        if eval_every and ((step + 1) % eval_every == 0
                           or (step + 1) == total_steps):
            val_loss = _validate(model, val_set, micro_bs, val_batches,
                                 val_seed, max_length, device, pad_id)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_ckpt, model, optim, step + 1, cfg,
                    extra=_checkpoint_extra(
                        step + 1, report_train, base_info, tokens_per_step,
                        resumed_tokens, val_loss, cfg, start_step),
                )
                events.append({"event": "best_validation", "step": step + 1,
                               "val_loss": val_loss, "path": str(best_ckpt)})
                print(f"  val_loss={val_loss:.4f} -> new best")
            else:
                print(f"  val_loss={val_loss:.4f} (best {best_val:.4f})")

        elapsed = time.perf_counter() - started
        if step % log_every == 0 or (step + 1) == total_steps:
            rate = tokens_per_step * (step - start_step + 1) / (elapsed or 1)
            print(f"step {step}/{total_steps} loss={loss_val:.4f} "
                  f"lr={lr:.2e} tok/s={rate:.0f}")
            history.append({"step": step, "train_loss": loss_val,
                            "val_loss": val_loss, "lr": lr,
                            "tokens_per_second": rate,
                            "elapsed_seconds": elapsed})

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            save_checkpoint(
                last_ckpt, model, optim, step + 1, cfg,
                extra=_checkpoint_extra(
                    step + 1, report_train, base_info, tokens_per_step,
                    resumed_tokens, val_loss, cfg, start_step),
            )
            events.append({"event": "periodic", "step": step + 1,
                           "path": str(last_ckpt)})
            print(f"checkpoint saved: {last_ckpt} (step {step + 1})")

    steps_run = total_steps - start_step
    tokens_processed = resumed_tokens + steps_run * tokens_per_step
    finished = start_step + steps_run >= total_steps
    status = "complete" if finished else "stopped_early"
    elapsed = time.perf_counter() - started

    manifest = build_run_manifest(
        run_id=run_cfg.get("name", "m4-sft"),
        status=status, config=cfg, config_path=config_path,
        model_cfg=model_cfg, model_config_path=run_cfg.get("base_model_config"),
        params=sum(p.numel() for p in model.parameters()),
        tokenizer_path=data_cfg["tokenizer"],
        tokenizer_vocab=tokenizer.get_vocab_size(),
        dataset_manifest=data_cfg.get("dataset_manifest"),
        dataset_paths=[data_cfg["train_jsonl"], data_cfg["val_jsonl"]],
        seed=run_cfg.get("seed"), checkpoint_path=str(last_ckpt),
        steps_completed=steps_run, planned_steps=total_steps,
        tokens_processed=tokens_processed,
        metrics={
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "best_validation_loss": best_val,
            "uniform_random_baseline_loss": math.log(model_cfg["vocab_size"]),
            "tokens_per_step": tokens_per_step,
        },
        start_time=start_iso,
        end_time=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        elapsed_seconds=elapsed, warnings=warnings,
        stopping_criteria={"planned_max_steps": total_steps,
                           "stop_reason": None if finished else "step limit"},
        history=history,
        notes=["Base checkpoint is opened read-only and never modified."],
    )
    manifest["adaptation"] = {
        "base_checkpoint": base_info,
        "freeze_policy": freeze_report,
        "instruction_format": FORMAT_NAME,
        "assistant_only_loss": not supervise_prompt,
        "dataset_report_train": train_set.report(),
        "dataset_report_val": val_set.report(),
        "coverage": shuffle.coverage_report(),
        "checkpoint_events": events,
        "validation": {
            "batches": val_batches, "seed": val_seed,
            "note": ("fixed seed and fixed window order, so validation loss is "
                     "comparable across steps"),
        },
    }
    manifest["holdout_policy"] = (
        "The M2 holdout (test split) was not used for training, validation, or "
        "any threshold choice. It remains unspent."
    )
    manifest_path = write_run_manifest(
        (cfg.get("logging") or {}).get("run_manifest_path")
        or (out_path / "run_manifest.json"),
        manifest,
    )

    print(f"run manifest: {manifest_path}")
    print(f"status={status} steps={start_step + steps_run}/{total_steps} "
          f"tokens={tokens_processed}")
    if best_val is not None:
        print(f"best validation loss: {best_val:.4f}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M4 instruction tuning on the M3 base checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs/m4_sft")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    try:
        run(args.config, args.out_dir, max_steps=args.max_steps,
            resume=args.resume, device=args.device)
    except (CheckpointIncompatible, ValueError) as exc:
        print(f"instruction tuning refused: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())