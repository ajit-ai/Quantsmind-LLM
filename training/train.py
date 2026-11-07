"""Taksh v1.3.0 pretraining entry point.

The loop is deliberately small and explicit, because the properties that
matter here are the ones a reader can check: what data was authorized, how
the text was tokenized, what the loss counted, when the optimizer stepped,
and what was written when the run stopped.

Flow
----
``training/authorization``  -> dataset manifest proven usable in this mode
``training/precision``      -> device and dtype resolved or refused
``training/sequence``       -> documents tokenized into windows (reserved
                               tokens refused, boundaries explicit)
``training/data_loader``    -> deterministic epoch order, resume-aware
``training/objective``      -> token-mean causal loss over the window
``training/schedules``      -> warmup + cosine, checkpointed
``training/pretrain``       -> checkpoint writer (schema /2) and RNG capture
``training/manifest``       -> run manifest
``training/loss_log``       -> append-only telemetry

Pretraining here is **plain code**, not chat. The canonical chat renderer is
not applied to Python source; only an explicit end-of-document token is
added. Instruction tuning is a different objective and a different pipeline
(v1.5.0), and nothing here may be read as its prerequisite being satisfied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from training.authorization import AuthorizationError, authorize_dataset
from training.data_loader import DeterministicSampler, ManifestDataset, WindowStream
from training.determinism import set_determinism
from training.manifest import build_run_manifest, write_run_manifest
from training.objective import accumulation_weights, mean_loss, token_mean_loss
from training.precision import PrecisionError, describe, resolve_device, resolve_precision
from training.pretrain import (
    capture_rng_state,
    check_vocab_alignment,
    load_yaml,
    restore_rng_state,
    save_checkpoint,
)
from training.schedules import WarmupCosineSchedule
from training.sequence import SequenceBuilder, SequenceError
from training.tokenizer_contract import detect_contract
from training.train_config import (
    TrainConfigError,
    config_identity,
    load_train_config,
    resolve_path,
)

from training.checkpoint_schema import build_provenance


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _resolve_input(cfg: dict, value) -> Path:
    """Resolve an input path: configuration-relative first, then working-dir.

    A configuration that lives next to its data uses a path relative to
    itself; a reusable configuration in ``configs/`` names repository files
    relative to the repository root. Absolute paths are rejected by the
    configuration validator, so a configuration stays portable.
    """
    candidate = resolve_path(cfg, value)
    if candidate.exists():
        return candidate
    alternative = Path(value)
    if alternative.exists():
        return alternative.resolve()
    return candidate


def _build_windows(dataset, builder) -> list:
    """Tokenize every record of a split into training windows."""
    documents = [
        (record["source_id"], record["file_path"], record["text"])
        for record in dataset
    ]
    return builder.build_windows(documents)


def _validate_on(model, batches, device) -> float:
    """Mean token loss over a bounded, deterministic set of validation batches."""
    if not batches:
        return float("nan")
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            logits, _ = model(batch.input_ids.to(device))
            summed, supervised = token_mean_loss(logits, batch.labels.to(device))
            total += float(summed.item())
            count += supervised
    return mean_loss(total, count)


def _provenance(*, model_cfg, model, tokenizer, contract_id, cfg, seed,
                manifest_path, dataset_sha, tokenizer_file_sha, mode, sequence_policy):
    provenance = build_provenance(
        model_cfg=model_cfg,
        parameter_count=sum(p.numel() for p in model.parameters()),
        contract_id=contract_id,
        config=cfg,
        seed=seed,
        tokenizer=tokenizer,
        corpus_manifest=str(manifest_path),
    )
    provenance["tokenizer"]["file_sha256"] = tokenizer_file_sha
    provenance["mode"] = mode
    provenance["synthetic"] = mode == "synthetic"
    provenance["dataset_manifest_sha256"] = dataset_sha
    provenance["train_config_sha256"] = config_identity(cfg)
    provenance["sequence_policy"] = sequence_policy
    return provenance


def run_training(config_path, *, resume_from=None, max_steps_override=None) -> dict:
    """Run a bounded pretraining job. Returns the run manifest."""
    from training.loss_log import LossLogger

    cfg = load_train_config(config_path)
    mode = cfg["mode"]
    training_cfg = cfg["training"]
    limits = cfg["limits"]
    data_cfg = cfg["data"]
    optimizer_cfg = cfg["optimizer"]
    batch_cfg = cfg["batch"]
    run_cfg = cfg["run"]

    out_dir = resolve_path(cfg, training_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolve_input(cfg, data_cfg["manifest"])
    tokenizer_path = _resolve_input(cfg, data_cfg["tokenizer"])
    model_config_path = _resolve_input(cfg, run_cfg["model_config"])

    authorization = authorize_dataset(
        manifest_path, mode=mode, tokenizer_path=tokenizer_path
    )

    device = resolve_device(cfg["precision"].get("device", "auto"))
    dtype = resolve_precision(device, cfg["precision"].get("dtype", "float32"))
    runtime = describe(device, dtype)

    seed = int(run_cfg["seed"])
    determinism = set_determinism(seed)

    model_cfg = load_yaml(model_config_path)["model"]
    override = (cfg.get("model") or {}).get("vocab_size_override")
    if override:
        model_cfg["vocab_size"] = int(override)
    tokenizer = _load_tokenizer(tokenizer_path)
    problems = check_vocab_alignment(model_cfg["vocab_size"], tokenizer.get_vocab_size())
    if problems:
        raise TrainConfigError("; ".join(problems))

    control_floor = (
        model_cfg["vocab_size"]
        if tokenizer.get_vocab_size() > model_cfg["vocab_size"]
        else None
    )
    builder = SequenceBuilder(
        tokenizer,
        context_length=int(data_cfg["context_length"]),
        model_vocab=int(model_cfg["vocab_size"]),
        policy=data_cfg.get("sequence_policy", "document"),
        append_eos=bool(data_cfg.get("append_eos", True)),
        control_token_floor=control_floor,
        allow_cross_document=bool(data_cfg.get("allow_cross_document", False)),
    )
    train_dataset = ManifestDataset(
        manifest_path, "train", require_ready=False
    )
    windows = _build_windows(train_dataset, builder)
    if not windows:
        raise TrainConfigError("the training split produced no windows")

    val_dataset = ManifestDataset(manifest_path, "val", require_ready=False)
    val_builder = SequenceBuilder(
        tokenizer,
        context_length=int(data_cfg["context_length"]),
        model_vocab=int(model_cfg["vocab_size"]),
        policy=data_cfg.get("sequence_policy", "document"),
        append_eos=bool(data_cfg.get("append_eos", True)),
        control_token_floor=control_floor,
        allow_cross_document=bool(data_cfg.get("allow_cross_document", False)),
    )
    val_windows = _build_windows(val_dataset, val_builder)
    val_batches = [
        val_builder.to_batch(val_windows[start : start + int(batch_cfg["micro_batch_size"])])
        for start in range(
            0,
            min(len(val_windows), int(training_cfg.get("val_batches", 8)) * int(batch_cfg["micro_batch_size"])),
            int(batch_cfg["micro_batch_size"]),
        )
    ]

    from training.model import QMModel

    model = QMModel(model_cfg).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    sampler = DeterministicSampler(len(windows), seed, shuffle=bool(batch_cfg.get("shuffle", True)))
    stream = WindowStream(
        windows,
        sampler=sampler,
        batch_builder=builder.to_batch,
        micro_batch_size=int(batch_cfg["micro_batch_size"]),
        grad_accum_steps=int(batch_cfg["grad_accum_steps"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr_peak"]),
        betas=(float(optimizer_cfg.get("beta1", 0.9)), float(optimizer_cfg.get("beta2", 0.95))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.1)),
    )
    max_steps = int(max_steps_override or training_cfg["max_steps"])
    schedule = WarmupCosineSchedule(
        peak_lr=float(optimizer_cfg["lr_peak"]),
        warmup_steps=int(optimizer_cfg.get("warmup_steps", 0)),
        # The schedule follows the *planned* run, not a short override, so a
        # resumed run continues on the same learning-rate curve.
        total_steps=int(training_cfg["max_steps"]),
        min_ratio=float(optimizer_cfg.get("min_lr_ratio", 0.1)),
    )

    start_step = 0
    consumed_tokens = 0
    epoch = 0
    position = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "quantsmind.checkpoint/2":
            raise TrainConfigError(
                f"resume requires a /2 checkpoint, found {checkpoint.get('schema')!r}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint.get("scheduler") or {})
        restore_rng_state(checkpoint.get("rng_state"))
        sampler.load_state_dict(checkpoint.get("sampler_state") or {})
        start_step = int(checkpoint.get("step", 0))
        consumed_tokens = int(checkpoint.get("consumed_tokens", 0))
        epoch = int((checkpoint.get("sampler_state") or {}).get("epoch", 0))
        position = int((checkpoint.get("sampler_state") or {}).get("position", 0))
        if (checkpoint.get("dataset_identity") or {}).get("manifest_sha256") != (
            authorization.manifest_sha256
        ):
            raise TrainConfigError(
                "the checkpoint was written against a different dataset; "
                "refusing to resume"
            )
        if (checkpoint.get("tokenizer_identity") or {}).get("sha256") != (
            authorization.tokenizer_sha256
        ):
            raise TrainConfigError(
                "the checkpoint was written with a different tokenizer; "
                "refusing to resume"
            )
        if checkpoint.get("train_config_sha256") != config_identity(cfg):
            raise TrainConfigError(
                "the checkpoint was written under a different training "
                "configuration; refusing to resume"
            )

    contract_id = authorization.contract_id or detect_contract(tokenizer)
    provenance = _provenance(
        model_cfg=model_cfg,
        model=model,
        tokenizer=tokenizer,
        contract_id=contract_id,
        cfg=cfg,
        seed=seed,
        manifest_path=manifest_path,
        dataset_sha=authorization.manifest_sha256,
        tokenizer_file_sha=authorization.tokenizer_sha256,
        mode=mode,
        sequence_policy=builder.policy,
    )

    logger = LossLogger(
        out_dir / "loss_log.jsonl",
        run_id=f"{run_cfg['name']}-{mode}",
        run_kind="pretrain_v13",
        model_config=str(model_config_path),
        dataset=str(manifest_path),
        config=cfg,
    )
    logger.log_header(
        mode=mode,
        synthetic=(mode == "synthetic"),
        dataset_manifest_sha256=authorization.manifest_sha256,
        tokenizer_sha256=authorization.tokenizer_sha256,
        device=runtime,
        determinism=determinism,
        sequence=builder.statistics(),
        windows=len(windows),
        planned_steps=max_steps,
        resource_limits={
            "max_steps": max_steps,
            "max_tokens": int(training_cfg["max_tokens"]),
            "max_wall_seconds": float(training_cfg["max_wall_seconds"]),
            "max_checkpoints": int(limits["max_checkpoints"]),
            "max_output_bytes": int(limits["max_output_bytes"]),
        },
        evidence_scope=(
            "engineering qualification only: a controlled run on this data "
            "says nothing about general Python ability"
        ),
    )

    start_time = time.time()
    history: list = []
    stop_reason = "max_steps"
    step = start_step
    checkpoints_written: list = []
    initial_loss = None
    initial_val_loss = None
    best_val_loss = None
    last_loss = None

    try:
        while step < max_steps:
            for window in stream.accumulation_windows(epoch, position):
                if step >= max_steps:
                    stop_reason = "max_steps"
                    break
                counts = [batch.supervised_tokens for batch in window]
                weights = accumulation_weights(counts)
                optimizer.zero_grad(set_to_none=True)
                window_loss = 0.0
                for batch, weight in zip(window, weights):
                    logits, _ = model(batch.input_ids.to(device))
                    summed, supervised = token_mean_loss(
                        logits, batch.labels.to(device)
                    )
                    if supervised == 0:
                        continue
                    # ``weight`` is this microbatch's share of the window's
                    # supervised tokens, so ``mean * weight`` is the
                    # microbatch's contribution to the window's token-mean
                    # loss and to its gradient.
                    micro_mean = summed / supervised
                    (micro_mean * weight).backward()
                    window_loss += float(micro_mean.item()) * weight
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimizer_cfg.get("grad_clip", 1.0))
                )
                learning_rate = schedule.lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                step += 1
                position += len(window)
                consumed_tokens += sum(counts)
                last_loss = window_loss
                if initial_loss is None:
                    initial_loss = window_loss
                elapsed = time.time() - start_time
                logger.log_step(
                    step,
                    window_loss,
                    lr=learning_rate,
                    grad_norm=float(grad_norm),
                    tokens_per_second=(
                        consumed_tokens / elapsed if elapsed > 0 else None
                    ),
                    consumed_examples=consumed_tokens,
                    extra={
                        "epoch": epoch,
                        "supervised_tokens": sum(counts),
                        "accumulation_steps": len(window),
                    },
                )
                history.append(
                    {
                        "step": step,
                        "loss": window_loss,
                        "lr": learning_rate,
                        "grad_norm": float(grad_norm),
                        "tokens": consumed_tokens,
                        "epoch": epoch,
                    }
                )
                if step % int(training_cfg.get("eval_every_steps", 10)) == 0 or step == max_steps:
                    val_loss = _validate_on(model, val_batches, device)
                    if initial_val_loss is None:
                        initial_val_loss = val_loss
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                    logger.log_validation(step, val_loss)
                if step % int(training_cfg.get("checkpoint_every_steps", 25)) == 0:
                    path = out_dir / "checkpoint_last.pt"
                    save_checkpoint(
                        path,
                        model,
                        optimizer,
                        step,
                        {"model": model_cfg, "train_config": cfg},
                        extra={
                            "scheduler": schedule.state_dict(),
                            "rng_state": capture_rng_state(),
                            "sampler_state": stream.state_dict(epoch, position),
                            "consumed_tokens": consumed_tokens,
                            "dataset_identity": {
                                "manifest_sha256": authorization.manifest_sha256,
                                "manifest_path": str(manifest_path),
                                "split": "train",
                                "corpus_class": authorization.corpus_class,
                            },
                            "tokenizer_identity": {
                                "path": str(tokenizer_path),
                                "sha256": authorization.tokenizer_sha256,
                                "vocab_size": tokenizer.get_vocab_size(),
                            },
                            "train_config_sha256": config_identity(cfg),
                            "mode": mode,
                            "synthetic": mode == "synthetic",
                            "created_at": _now_iso(),
                        },
                        provenance=provenance,
                    )
                    checkpoints_written.append(path)
                    while len(checkpoints_written) > int(limits["max_checkpoints"]):
                        oldest = checkpoints_written.pop(0)
                        if oldest.exists():
                            oldest.unlink()
                if consumed_tokens >= int(training_cfg["max_tokens"]):
                    stop_reason = "max_tokens"
                    break
                if (time.time() - start_time) >= float(training_cfg["max_wall_seconds"]):
                    stop_reason = "max_wall_seconds"
                    break
            if stop_reason != "max_steps" or step >= max_steps:
                break
            epoch += 1
            position = 0
    finally:
        elapsed = time.time() - start_time
        final_val_loss = _validate_on(model, val_batches, device)
        if best_val_loss is None or final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
        status = "complete" if step >= max_steps else f"stopped:{stop_reason}"
        logger.close(status=status, steps=step, tokens=consumed_tokens)

    # Final artifacts -------------------------------------------------------
    final_checkpoint = out_dir / "checkpoint_last.pt"
    save_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        step,
        {"model": model_cfg, "train_config": cfg},
        extra={
            "scheduler": schedule.state_dict(),
            "rng_state": capture_rng_state(),
            "sampler_state": stream.state_dict(epoch, position),
            "consumed_tokens": consumed_tokens,
            "dataset_identity": {
                "manifest_sha256": authorization.manifest_sha256,
                "manifest_path": str(manifest_path),
                "split": "train",
                "corpus_class": authorization.corpus_class,
            },
            "tokenizer_identity": {
                "path": str(tokenizer_path),
                "sha256": authorization.tokenizer_sha256,
                "vocab_size": tokenizer.get_vocab_size(),
            },
            "train_config_sha256": config_identity(cfg),
            "mode": mode,
            "synthetic": mode == "synthetic",
            "created_at": _now_iso(),
        },
        provenance=provenance,
    )

    evaluation = {
        "schema": "quantsmind.pretrain_evaluation/1",
        "mode": mode,
        "synthetic": mode == "synthetic",
        "corpus_class": authorization.corpus_class,
        "corpus_ready": authorization.corpus_ready,
        "dataset_manifest_sha256": authorization.manifest_sha256,
        "tokenizer_sha256": authorization.tokenizer_sha256,
        "tokenizer_contract_id": contract_id,
        "steps": step,
        "tokens_consumed": consumed_tokens,
        "windows": len(windows),
        "sequence": builder.statistics(),
        "initial_train_loss": initial_loss,
        "final_train_loss": last_loss,
        "initial_validation_loss": initial_val_loss,
        "final_validation_loss": final_val_loss,
        "best_validation_loss": best_val_loss,
        "final_validation_perplexity": (
            math.exp(final_val_loss) if final_val_loss == final_val_loss else None
        ),
        "uniform_baseline_loss": math.log(model_cfg["vocab_size"]),
        "device": runtime,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "evidence_scope": (
            "Engineering qualification only. A decreasing loss on this data "
            "shows the pipeline learns a controlled distribution. It does not "
            "establish general Python proficiency, functional code generation, "
            "HumanEval or MBPP performance, or any production capability."
        ),
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = build_run_manifest(
        run_id=f"{run_cfg['name']}-{mode}",
        status=status,
        config=cfg,
        config_path=cfg.get("__config_path__"),
        model_cfg=model_cfg,
        model_config_path=run_cfg["model_config"],
        params=sum(p.numel() for p in model.parameters()),
        tokenizer_path=str(tokenizer_path),
        tokenizer_vocab=tokenizer.get_vocab_size(),
        dataset_manifest=str(manifest_path),
        dataset_paths=[
            str(manifest_path.parent / artifact["name"])
            for artifact in authorization.manifest.get("artifacts", [])
        ],
        seed=seed,
        checkpoint_path=str(final_checkpoint),
        steps_completed=step,
        planned_steps=max_steps,
        tokens_processed=consumed_tokens,
        metrics={
            "initial_train_loss": initial_loss,
            "final_train_loss": last_loss,
            "initial_validation_loss": initial_val_loss,
            "final_validation_loss": final_val_loss,
            "best_validation_loss": best_val_loss,
            "uniform_baseline_loss": math.log(model_cfg["vocab_size"]),
        },
        start_time=_now_iso(),
        end_time=_now_iso(),
        elapsed_seconds=elapsed,
        warnings=[],
        deviations=[],
        interruptions=[],
        stopping_criteria={
            "planned_max_steps": max_steps,
            "max_tokens": int(training_cfg["max_tokens"]),
            "max_wall_seconds": float(training_cfg["max_wall_seconds"]),
            "stop_reason": stop_reason,
        },
        history=history,
        notes=[
            f"mode: {mode}",
            f"corpus_class: {authorization.corpus_class}",
            f"corpus_ready: {authorization.corpus_ready}",
            "checkpoint schema: quantsmind.checkpoint/2",
            "engineering qualification only; no capability claim",
        ],
    )
    manifest["checkpoint"]["format"] = "quantsmind.checkpoint/2"
    manifest["dataset"]["manifest_sha256"] = authorization.manifest_sha256
    manifest["dataset"]["corpus_class"] = authorization.corpus_class
    manifest["training"]["mode"] = mode
    write_run_manifest(out_dir / "run_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Taksh v1.3.0 bounded Python pretraining"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        manifest = run_training(
            args.config,
            resume_from=args.resume_from,
            max_steps_override=args.max_steps,
        )
    except (
        AuthorizationError,
        PrecisionError,
        SequenceError,
        TrainConfigError,
    ) as exc:
        print(f"REFUSED: {exc}")
        return 2
    training = manifest.get("training", {})
    print(f"status          : {manifest.get('status')}")
    print(f"steps           : {training.get('steps_completed')}")
    print(f"tokens          : {training.get('tokens_processed')}")
    print(f"initial loss    : {training.get('metrics', {}).get('initial_train_loss')}")
    print(f"final loss      : {training.get('metrics', {}).get('final_train_loss')}")
    print(
        "final val loss  : "
        f"{training.get('metrics', {}).get('final_validation_loss')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
