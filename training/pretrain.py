import argparse
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from training.checkpoint import CheckpointIncompatible, load_model_weights
from training.data import TokenDataset
from training.loss_log import LossLogger
from training.model import QMModel

# Per-run context that the checkpoint/manifest helpers need. Module-level rather
# than threaded through every helper so `checkpoint_extra` stays callable from
# inside the training loop.
run_context: dict = {}


def validate_train_config(cfg: dict) -> list[str]:
    """Return actionable problems with a train config.

    B4 recorded that ``configs/train/pretrain_pilot.yaml`` could not run:
    ``training.max_steps`` was missing (the key was ``max_tokens``), and
    ``batch.micro_batch_size`` was the string ``auto``. Both failed late, or not
    at all. This check runs before any model is built.

    It only reports keys this training loop actually consumes. Declaration-only
    keys are listed separately by :func:`unused_config_keys` rather than being
    treated as errors.
    """
    problems: list[str] = []

    if "run" not in cfg:
        problems.append("missing top-level 'run' section")
    elif "model_config" not in cfg["run"]:
        problems.append("missing 'run.model_config' (path to the model YAML)")

    data_cfg = cfg.get("data", {})
    if "seq_len" not in data_cfg:
        problems.append("missing 'data.seq_len'")
    elif not isinstance(data_cfg["seq_len"], int) or data_cfg["seq_len"] <= 0:
        problems.append(f"'data.seq_len' must be a positive int, got {data_cfg['seq_len']!r}")
    if not (data_cfg.get("tokens_path") or data_cfg.get("dataset_tag")):
        problems.append(
            "cannot determine the token file: set 'data.tokens_path' or "
            "'data.dataset_tag' (or pass --tokens)"
        )

    train_cfg = cfg.get("training", {})
    if "max_steps" not in train_cfg:
        present = sorted(k for k in train_cfg if "step" in k or "token" in k)
        problems.append(
            "missing 'training.max_steps' (integer optimizer steps). "
            f"Keys present that look related: {present or 'none'}. Note that "
            "'training.max_tokens' is a budget note and is not read by this loop."
        )
    elif not isinstance(train_cfg["max_steps"], int) or train_cfg["max_steps"] <= 0:
        problems.append(
            f"'training.max_steps' must be a positive int, got {train_cfg['max_steps']!r}"
        )

    batch_cfg = cfg.get("batch", {})
    if "micro_batch_size" not in batch_cfg:
        problems.append("missing 'batch.micro_batch_size'")
    elif not isinstance(batch_cfg["micro_batch_size"], int) or batch_cfg["micro_batch_size"] < 1:
        problems.append(
            "'batch.micro_batch_size' must be a concrete positive int; 'auto' is "
            f"not resolved by this loop (got {batch_cfg['micro_batch_size']!r})"
        )
    accum = batch_cfg.get("grad_accum_steps", 1)
    if not isinstance(accum, int) or accum < 1:
        problems.append(
            f"'batch.grad_accum_steps' must be a positive int, got {accum!r}. "
            "Note: it is read from 'batch', not from 'precision'."
        )

    opt_cfg = cfg.get("optimizer", {})
    if "lr_peak" not in opt_cfg:
        problems.append("missing 'optimizer.lr_peak'")
    return problems


# Keys that shipped in configs but that no code path reads. Reported so a config
# cannot quietly advertise behaviour it does not get.
_KNOWN_UNREAD_KEYS = {
    "data.pack_documents",
    "data.cross_doc_attention_mask",
    # M3 implemented training.eval_every_steps, training.val_batches and
    # data.val_tokens_path; they were removed from this list when the validation
    # loop landed. `keep_checkpoints` remains declared-but-unread: the loop
    # writes checkpoint_last.pt / checkpoint_best.pt and does not prune.
    "training.keep_checkpoints",
    "watchdog.loss_spike_sigma",
    "watchdog.spike_action",
    "batch.global_tokens_per_step",
    "precision.grad_accum_steps",
}


def unused_config_keys(cfg: dict) -> list[str]:
    """Config keys that are present but never read by this training loop."""
    found = []
    for dotted in _KNOWN_UNREAD_KEYS:
        section, _, key = dotted.partition(".")
        if key in cfg.get(section, {}):
            found.append(dotted)
    return found


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_model_cfg(train_cfg: dict) -> dict:
    model_cfg = load_yaml(train_cfg["run"]["model_config"])["model"]
    override = train_cfg.get("model", {}).get("vocab_size_override")
    if override:
        model_cfg["vocab_size"] = int(override)
    return model_cfg


def resolve_tokens_path(train_cfg: dict, override: str | None) -> Path:
    """Resolve which token file a run trains on.

    Precedence: explicit ``--tokens`` > ``data.tokens_path`` in the config >
    ``data/<dataset_tag>.tokens.bin`` (``-`` normalised to ``_``).

    The previous hardcoded default of ``data/corpus_v0.tokens.bin`` meant every
    config -- including the code ones -- silently trained on the general-text
    corpus, because ``data.dataset_tag`` was never consulted.
    """
    if override:
        return Path(override)

    data_cfg = train_cfg.get("data", {})
    if data_cfg.get("tokens_path"):
        return Path(data_cfg["tokens_path"])

    tag = data_cfg.get("dataset_tag")
    if not tag:
        raise SystemExit(
            "cannot resolve token file: pass --tokens, or set data.tokens_path "
            "or data.dataset_tag in the train config"
        )
    return Path("data") / f"{tag.replace('-', '_')}.tokens.bin"


def check_vocab_alignment(model_vocab: int, tokenizer_vocab: int | None) -> list[str]:
    """Compare model embedding width against the tokenizer that produced the data.

    Returns human-readable problems. A tokenizer wider than the model would index
    out of bounds; a much wider model wastes parameters on ids that never occur.
    """
    if tokenizer_vocab is None:
        return []
    if tokenizer_vocab > model_vocab:
        return [
            f"tokenizer vocab {tokenizer_vocab} exceeds model vocab_size {model_vocab}; "
            f"set model.vocab_size_override in the train config"
        ]
    if model_vocab > tokenizer_vocab * 2:
        return [
            f"model vocab_size {model_vocab} is {model_vocab / tokenizer_vocab:.1f}x the "
            f"tokenizer vocab {tokenizer_vocab}; embedding is mostly dead weight"
        ]
    return []


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: dict,
                    extra: dict | None = None, provenance: dict | None = None):
    """Write a checkpoint. Supplying ``provenance`` upgrades the file to schema /2.

    Schema /1 is kept for callers that cannot yet supply a tokenizer/contract
    identity (historical tooling); new training code must pass provenance, and
    the loader treats /1 as the explicit legacy contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "schema": "quantsmind.checkpoint/1",
    }
    if provenance is not None:
        from training.checkpoint_schema import SCHEMA_V2, validate_provenance

        problems = validate_provenance(provenance)
        if problems:
            raise ValueError("incomplete checkpoint provenance: " + "; ".join(problems))
        payload["schema"] = SCHEMA_V2
        payload["provenance"] = provenance
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    tmp.replace(path)


def capture_rng_state() -> dict:
    """RNG state for reproducible resume.

    Without this, resuming restarts numpy's window sampling from wherever the
    process happens to be, so a resumed run sees different data than an
    uninterrupted one and the two losses are not comparable.
    """
    import random as _random

    import numpy as np

    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    return {
        "torch": torch_state,
        "numpy": numpy_state,
        "python": _random.getstate(),
    }


def restore_rng_state(state: dict | None) -> bool:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    if not state:
        return False
    import random as _random

    import numpy as np

    try:
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("python") is not None:
            _random.setstate(state["python"])
        return True
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to a resume
        print(f"WARNING: could not fully restore RNG state ({type(exc).__name__}: {exc}); "
              "resume will be best-effort rather than bitwise identical")
        return False


def warmstart_from(model, path: str):
    """Initialise model weights from a checkpoint with a narrow safe loader.

    The operator supplies the path explicitly (never a request). Safe
    deserialisation (``weights_only=True``) is the only mode; the single
    documented retry admits exactly the NumPy reconstruction helpers Taksh's
    own training loop stores in its RNG state. No arbitrary globals, no
    unpickler bypass, no weights_only=False.
    """
    import hashlib
    import numpy as np
    import numpy.dtypes as _dtypes
    from numpy._core.multiarray import _reconstruct
    from torch.serialization import safe_globals

    allow = [_reconstruct, np.ndarray, np.dtype]
    allow += [getattr(_dtypes, name) for name in dir(_dtypes) if name.endswith("DType")]

    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - classify, never fall back
        if "numpy" not in str(exc).lower():
            raise SystemExit(
                f"Refusing to warmstart from {path}: safe deserialisation failed "
                f"({type(exc).__name__}: {exc}). Check the file is a trusted "
                "Taksh checkpoint."
            ) from exc
        try:
            with safe_globals(allow):
                blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as retry_exc:  # noqa: BLE001 - refuse
            raise SystemExit(
                f"Refusing to warmstart from {path}: checkpoint needs NumPy globals "
                f"beyond the reviewed allowlist ({type(retry_exc).__name__}: {retry_exc})."
            ) from retry_exc

    state = blob.get("model") if isinstance(blob, dict) else None
    if state is None:
        raise SystemExit(
            f"Refusing to warmstart from {path}: no 'model' entry. Partial "
            "initialisation is not supported here; use training.sft.init_from_checkpoint "
            "with allow_partial explicitly if that is what you want."
        )

    from training.checkpoint import load_model_weights

    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    report = load_model_weights(model, state, source=path)
    print(f"warmstart from {path} (sha256 {sha.hexdigest()}): "
          f"loaded {len(report.loaded)} tensors")
    return sha.hexdigest()


def checkpoint_extra(step, dataset_tokens, seq_len, tokens_per_step,
                     resumed_tokens=0, val_loss=None, interrupted=False,
                     start_step=0):
    """Everything a resume needs beyond weights and optimizer state.

    ``step`` is the absolute optimizer step, so tokens already covered by the
    run that produced this checkpoint must be added separately. Without
    ``start_step`` a resumed run double-counts the resumed span and reports more
    tokens than the dataset contains.
    """
    data_cfg = run_context.get("data", {})
    tokens_processed = resumed_tokens + max(0, step - start_step) * tokens_per_step
    extra = {
        "dataset_tokens": dataset_tokens,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "optimizer_steps_completed": step,
        "epoch_tokens": step * tokens_per_step,
        "tokens_processed": tokens_processed,
        "rng_state": capture_rng_state(),
        "val_loss": val_loss,
        "interrupted": interrupted,
        "dataset_identity": {
            "train_tokens": str(run_context.get("tokens_path") or ""),
            "val_tokens": str(run_context.get("val_tokens_path") or ""),
            "dataset_manifest": str(data_cfg.get("dataset_manifest") or ""),
        },
        "tokenizer_identity": {
            "path": str(data_cfg.get("tokenizer") or ""),
            "vocab_size": run_context.get("tokenizer_vocab"),
        },
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    return extra


def write_m3_manifest(cfg, out_dir, model_cfg, model, *, status, stop_reason,
                      steps_run, tokens_per_step, resumed_tokens, best_val_loss,
                      final_loss, history, warnings, deviations, interruptions,
                      tokens_path, val_tokens_path, elapsed,
                      checkpoint_events=None):
    """Write the run manifest. A non-complete run is never marked successful."""
    from training.manifest import build_run_manifest, write_run_manifest

    data_cfg = cfg.get("data") or {}
    logging_cfg = cfg.get("logging") or {}
    params = sum(p.numel() for p in model.parameters())
    total_tokens = resumed_tokens + steps_run * tokens_per_step
    train_cfg = cfg.get("training") or {}

    metrics = {
        "final_train_loss": final_loss,
        "best_validation_loss": best_val_loss,
        "uniform_random_baseline_loss": (
            math.log(model_cfg["vocab_size"]) if model_cfg.get("vocab_size") else None
        ),
        "tokens_per_step": tokens_per_step,
    }

    manifest = build_run_manifest(
        run_id=(cfg.get("run") or {}).get("name") or out_dir.name,
        status=status,
        config=cfg,
        config_path=cfg.get("__config_path__"),
        model_cfg=model_cfg,
        model_config_path=(cfg.get("run") or {}).get("model_config"),
        params=params,
        tokenizer_path=data_cfg.get("tokenizer"),
        tokenizer_vocab=run_context.get("tokenizer_vocab"),
        dataset_manifest=data_cfg.get("dataset_manifest"),
        dataset_paths=[p for p in (tokens_path, val_tokens_path) if p],
        seed=(cfg.get("run") or {}).get("seed"),
        checkpoint_path=str(out_dir / "checkpoint_last.pt"),
        steps_completed=steps_run,
        planned_steps=train_cfg.get("max_steps"),
        tokens_processed=total_tokens,
        metrics=metrics,
        start_time=run_context.get("start_time"),
        end_time=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        elapsed_seconds=elapsed,
        warnings=warnings,
        deviations=deviations,
        interruptions=interruptions,
        stopping_criteria={
            "planned_max_steps": train_cfg.get("max_steps"),
            "stop_reason": stop_reason,
            "criteria": run_context.get("stopping_criteria") or [],
        },
        initialization=run_context.get("initialization"),
        history=history,
        notes=[
            f"checkpoint events: {len(checkpoint_events or [])}",
        ],
    )
    manifest["checkpoint_events"] = checkpoint_events or []
    manifest["validation"] = {
        "configured": run_context.get("val_tokens_path") is not None,
        "stream": str(run_context.get("val_tokens_path") or ""),
        "batches_per_pass": train_cfg.get("val_batches"),
        "seed": data_cfg.get("val_seed"),
        "eval_every_steps": train_cfg.get("eval_every_steps"),
        "note": (
            "Validation windows are drawn from a fixed seed, so every pass "
            "scores the same data and the trend is comparable."
        ),
    }
    manifest["holdout_policy"] = (
        "The M2 holdout (test split) was not used for training, validation, or "
        "any threshold choice. It remains unspent."
    )

    target = logging_cfg.get("run_manifest_path") or (out_dir / "run_manifest.json")
    return write_run_manifest(target, manifest)


def load_checkpoint(path: Path, model, optimizer):
    """Restore model/optimizer/step. Validates the weights instead of assuming.

    Resuming onto a different architecture would otherwise silently produce a
    model that is neither the old one nor the new one, so compatibility is
    checked here too (A2/B3).

    Returns just the step, for backwards compatibility with ``training/sft.py``.
    Use :func:`load_checkpoint_full` when the whole payload is needed (M3 needs
    the RNG state and recorded identities to resume reproducibly).
    """
    step, _ckpt = load_checkpoint_full(path, model, optimizer)
    return step


def load_checkpoint_full(path: Path, model, optimizer):
    """Like :func:`load_checkpoint` but returns ``(step, full_payload)``."""
    path = Path(path)
    if not path.exists():
        raise CheckpointIncompatible(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - corrupt or truncated file
        raise CheckpointIncompatible(
            f"{path} could not be read ({type(exc).__name__}: {exc}). The file "
            "is corrupt or incomplete; refusing to resume from it."
        ) from exc
    if not isinstance(ckpt, dict):
        raise CheckpointIncompatible(
            f"{path} is not a checkpoint dictionary (got "
            f"{type(ckpt).__name__}); refusing to resume"
        )
    schema = ckpt.get("schema")
    if schema not in ("quantsmind.checkpoint/1", "quantsmind.checkpoint/2"):
        raise CheckpointIncompatible(
            f"{path} has schema {schema!r}, expected 'quantsmind.checkpoint/1' "
            "or 'quantsmind.checkpoint/2'"
        )
    if schema == "quantsmind.checkpoint/2":
        from training.checkpoint_schema import validate_provenance

        problems = validate_provenance(ckpt.get("provenance"))
        if problems:
            raise CheckpointIncompatible(
                f"{path} is schema /2 but its provenance is incomplete: "
                + "; ".join(problems)
            )
    state = ckpt.get("model")
    if state is None:
        raise CheckpointIncompatible(
            f"{path} has no 'model' entry; refusing to resume from it"
        )
    if "optimizer" not in ckpt:
        raise CheckpointIncompatible(
            f"{path} has no 'optimizer' entry; a resume cannot continue training "
            "from it without optimizer state"
        )
    report = load_model_weights(model, state, source=str(path))
    optimizer.load_state_dict(ckpt["optimizer"])
    if "step" not in ckpt:
        raise CheckpointIncompatible(f"{path} records no 'step'")
    return int(ckpt["step"]), ckpt


def main():
    p = argparse.ArgumentParser(description="Quantsmind-LLM pre-training loop")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="runs/pretrain")
    p.add_argument("--tokens", default=None, help="override token file path")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--init-from",
        default=None,
        help="warmstart: initialize model weights from this checkpoint "
        "(new optimizer state; distinct from --resume, which also restores "
        "the optimizer and step counter)",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)

    # A4: fail before building a model, with a message that names the fix.
    problems = validate_train_config(cfg)
    if problems:
        joined = "\n".join(f"  - {p}" for p in problems)
        raise SystemExit(f"invalid train config {args.config}:\n{joined}")
    unread = unused_config_keys(cfg)
    if unread:
        print(
            "WARNING: these config keys are not read by this training loop and "
            "have no effect: " + ", ".join(unread)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    print(f"device={device}")

    data_cfg = cfg["data"]
    tokens_path = resolve_tokens_path(cfg, args.tokens)
    if not tokens_path.exists():
        raise SystemExit(
            f"token file not found: {tokens_path}\n"
            f"Encode a corpus first, e.g.\n"
            f"  python -m data.encode_dataset --corpus data/code_corpus_v0.jsonl \\\n"
            f"    --tokenizer tokenizer/qm_code_bpe_v0.json --out {tokens_path}"
        )
    dataset = TokenDataset(tokens_path, seed=cfg.get("run", {}).get("seed", 42))

    # M3: validation data must come from a separate artifact. Loading training
    # windows for validation produces a number that looks meaningful and is not.
    val_dataset = None
    val_tokens_path = (cfg.get("data") or {}).get("val_tokens_path")
    if val_tokens_path:
        val_tokens_path = Path(val_tokens_path)
        if not val_tokens_path.exists():
            raise SystemExit(
                f"validation token file not found: {val_tokens_path}\n"
                "data.val_tokens_path must point at data separate from "
                "data.tokens_path."
            )
        try:
            from training.validation import assert_separate

            assert_separate(tokens_path, val_tokens_path)
        except ValueError as exc:
            raise SystemExit(f"refusing to train: {exc}")
        val_dataset = TokenDataset(
            val_tokens_path, seed=cfg.get("data", {}).get("val_seed", 1234)
        )
        print(
            f"validation stream: {val_tokens_path} "
            f"({len(val_dataset)} tokens, evaluation only)"
        )
    else:
        print(
            "WARNING: no data.val_tokens_path configured. This run will record "
            "training loss only and cannot report validation loss or perplexity."
        )

    model_cfg = flatten_model_cfg(cfg)
    tokenizer_vocab = None
    tokenizer_path = cfg.get("data", {}).get("tokenizer")
    if tokenizer_path and Path(tokenizer_path).exists():
        try:
            from tokenizers import Tokenizer

            tokenizer_vocab = Tokenizer.from_file(tokenizer_path).get_vocab_size()
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            print(f"WARNING: could not read tokenizer {tokenizer_path}: {exc}")

    artifact_problems = dataset.validate_for_model(
        model_cfg["vocab_size"], tokenizer_vocab
    )
    if any("would index outside" in p for p in artifact_problems):
        raise SystemExit(
            f"token artifact is incompatible with the model:\n"
            + "\n".join(f"  - {p}" for p in artifact_problems)
        )
    for problem in artifact_problems:
        print(f"WARNING: token artifact: {problem}")

    for problem in check_vocab_alignment(model_cfg["vocab_size"], dataset.vocab_size):
        print(f"WARNING: {problem}")

    model = QMModel(model_cfg).to(device)
    if args.init_from:
        sha = warmstart_from(model, args.init_from)
        run_context["initialization"] = (
            f"warmstarted from {args.init_from} (sha256 {sha}); "
            "optimizer state not carried over"
        )
    opt_cfg = cfg["optimizer"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr_peak"],
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )

    train_cfg = cfg["training"]
    total_steps = args.max_steps or train_cfg["max_steps"]
    start_step = 0
    last_ckpt = out_dir / "checkpoint_last.pt"
    resume_warnings = []
    resumed_tokens = 0
    if args.resume:
        if not last_ckpt.exists():
            raise SystemExit(
                f"--resume was requested but no checkpoint exists at {last_ckpt}. "
                "Start a run without --resume, or point --out-dir at the run "
                "directory that holds the checkpoint."
            )
        start_step, payload = load_checkpoint_full(last_ckpt, model, optim)
        resumed_tokens = int(payload.get("tokens_processed") or 0)
        if restore_rng_state(payload.get("rng_state")):
            print(f"resumed from {last_ckpt} at step {start_step} (RNG state restored)")
        else:
            resume_warnings.append(
                "resumed without RNG state; this run is best-effort continuous, "
                "not bitwise identical to an uninterrupted run"
            )
            print(f"resumed from {last_ckpt} at step {start_step} ({resume_warnings[-1]})")
        if start_step >= total_steps:
            print(
                f"checkpoint is at step {start_step} and this run was asked for "
                f"{total_steps} steps; nothing to do. Raise --max-steps to continue."
            )
    if device == "cuda":
        torch.compile(model)

    batch_cfg = cfg["batch"]
    micro_bs = batch_cfg["micro_batch_size"]
    accum = batch_cfg.get("grad_accum_steps", 1)
    seq_len = data_cfg["seq_len"]

    eval_every = train_cfg.get("eval_every_steps") or 0
    val_batches = train_cfg.get("val_batches", 8)
    val_seed = (cfg.get("data") or {}).get("val_seed", 1234)
    best_val_path = out_dir / "checkpoint_best.pt"
    best_val_loss = None
    last_val_loss = None

    run_context.update({
        "data": cfg.get("data") or {},
        "tokens_path": tokens_path,
        "val_tokens_path": val_tokens_path,
        "tokenizer_vocab": tokenizer_vocab,
        "start_time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stopping_criteria": [
            f"all {total_steps} steps complete with finite loss",
            f"final train loss < {run_context.get('loss_target', 3.0)}",
            "validation loss finite and <= final train loss + 2.0",
            "checkpoint saves, verifies, reloads in a fresh process, generates",
        ],
    })
    cfg["__config_path__"] = str(args.config)

    use_wandb = os.environ.get("QM_WANDB") == "1"
    if use_wandb:
        import wandb

        wandb.init(project="quantsmind-llm", name=cfg.get("run", {}).get("name"), config=cfg)

    nan_every = cfg.get("watchdog", {}).get("nan_check_every_steps", 100)
    log_every = train_cfg.get("log_every_steps", 10)
    ckpt_every = train_cfg.get("checkpoint_every_steps", 1000)

    logger = LossLogger(
        out_dir / "loss_log.jsonl",
        run_kind="pretrain",
        model_config=cfg["run"].get("model_config"),
        dataset=str(tokens_path),
        config=cfg,
    )
    tokens_per_step = micro_bs * accum * seq_len
    logger.log_header(
        total_steps=total_steps,
        dataset_tokens=len(dataset),
        seq_len=seq_len,
        tokens_per_step=tokens_per_step,
        micro_batch_size=micro_bs,
        grad_accum_steps=accum,
        seed=cfg.get("run", {}).get("seed", 42),
    )
    print(f"loss log: {out_dir / 'loss_log.jsonl'}")

    run_warnings = list(resume_warnings)
    deviations: list[str] = []
    interruptions: list[str] = []
    history: list[dict] = []
    checkpoint_events: list[dict] = []
    stop_reason = None

    interrupt_requested = {"value": False}

    def _on_sigint(signum, frame):  # pragma: no cover - signal delivery path
        interrupt_requested["value"] = True
        print("\ninterrupt received; finishing this step and saving a checkpoint...")

    previous_sigint = None
    try:
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:  # noqa: BLE001 - not available on every platform
        previous_sigint = None

    ema_loss = None
    model.train()
    t0 = time.time()
    final_loss = None
    steps_run = 0

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, opt_cfg["lr_peak"], opt_cfg.get("warmup_steps", 0), opt_cfg.get("min_lr_ratio", 0.1))
        for g in optim.param_groups:
            g["lr"] = lr

        loss_val = 0.0
        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = dataset.batch(micro_bs, seq_len)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += loss.item() / accum

        # Gradient health is checked BEFORE clipping: clipping rescales
        # non-finite values into finite ones and would hide the fault.
        if nan_every and step % nan_every == 0:
            bad = [
                name for name, param in model.named_parameters()
                if param.grad is not None and not torch.isfinite(param.grad).all()
            ]
            if bad:
                raise RuntimeError(
                    f"non-finite gradients in {len(bad)} parameter(s) at step "
                    f"{step} (first: {bad[0]}); aborting per watchdog policy"
                )

        grad_clip = opt_cfg.get("grad_clip")
        grad_norm = None
        if grad_clip:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            )
        optim.step()
        steps_run += 1
        final_loss = loss_val

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}; aborting per watchdog policy")

        ema = 0.05 if ema_loss is None else 0.95
        ema_loss = ema * loss_val + (1 - ema) * (ema_loss or loss_val)

        if use_wandb:
            wandb.log({"loss": loss_val, "ema_loss": ema_loss, "lr": lr, "step": step})

        # ---- validation: no weight updates, fixed windows ----
        val_loss = None
        if (
            val_dataset is not None
            and eval_every
            and ((step + 1) % eval_every == 0 or (step + 1) == total_steps)
        ):
            from training.validation import evaluate

            report = evaluate(
                model, val_dataset, seq_len, val_batches, val_seed,
                device=device, batch_size=micro_bs,
                vocab_size=model_cfg["vocab_size"],
            )
            val_loss = report.loss
            last_val_loss = val_loss if val_loss is not None else last_val_loss
            for message in report.errors:
                run_warnings.append(f"validation: {message}")
            improved = val_loss is not None and (
                best_val_loss is None or val_loss < best_val_loss
            )
            ppl_text = (
                f" ppl={report.perplexity:.2f}" if report.perplexity else ""
            )
            if improved:
                best_val_loss = val_loss
                save_checkpoint(
                    best_val_path, model, optim, step + 1, cfg,
                    extra=checkpoint_extra(
                        step + 1, len(dataset), seq_len, tokens_per_step,
                        resumed_tokens, val_loss, start_step=start_step,
                    ),
                )
                checkpoint_events.append({
                    "event": "best_validation", "step": step + 1,
                    "val_loss": val_loss, "path": str(best_val_path),
                })
                print(f"  val_loss={val_loss:.4f}{ppl_text} -> new best, "
                      f"saved {best_val_path.name}")
            else:
                shown = f"{val_loss:.4f}" if val_loss is not None else "n/a"
                best_text = f" (best {best_val_loss:.4f})" if best_val_loss else ""
                print(f"  val_loss={shown}{ppl_text}{best_text}")
            if use_wandb and val_loss is not None:
                wandb.log({"val_loss": val_loss, "step": step})

        elapsed = time.time() - t0
        if step % log_every == 0 or (step + 1) == total_steps:
            rate = micro_bs * accum * seq_len * max(1, step - start_step + 1) / (elapsed or 1)
            # A5: persist before printing, so an interrupted run still leaves
            # the records it reached.
            logger.log_step(
                step, loss_val, lr=lr, ema_loss=ema_loss, tokens_per_second=rate,
                grad_norm=grad_norm,
            )
            print(f"step {step}/{total_steps} loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr:.2e} tok/s={rate:.0f}")
            history.append({
                "step": step,
                "train_loss": loss_val,
                "ema_loss": ema_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "val_loss": val_loss,
                "tokens_per_second": rate,
                "elapsed_seconds": elapsed,
            })

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            save_checkpoint(
                last_ckpt, model, optim, step + 1, cfg,
                extra=checkpoint_extra(
                    step + 1, len(dataset), seq_len, tokens_per_step,
                    resumed_tokens, val_loss, start_step=start_step,
                ),
            )
            checkpoint_events.append({
                "event": "periodic", "step": step + 1, "path": str(last_ckpt),
            })
            print(f"checkpoint saved: {last_ckpt} (step {step + 1})")

        if interrupt_requested["value"]:
            stop_reason = "interrupted by SIGINT"
            interruptions.append({
                "at_step": step + 1,
                "tokens_processed":
                    resumed_tokens + (step + 1 - start_step) * tokens_per_step,
                "elapsed_seconds": time.time() - t0,
            })
            print("saving interrupted run state...")
            save_checkpoint(
                last_ckpt, model, optim, step + 1, cfg,
                extra=checkpoint_extra(
                    step + 1, len(dataset), seq_len, tokens_per_step,
                    resumed_tokens, val_loss, interrupted=True,
                    start_step=start_step,
                ),
            )
            break

    if previous_sigint is not None:
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except Exception:  # noqa: BLE001
            pass

    completed_steps = start_step + steps_run
    tokens_processed = resumed_tokens + steps_run * tokens_per_step
    finished = stop_reason is None and completed_steps >= total_steps
    status = "complete" if finished else (
        "interrupted" if stop_reason and "SIGINT" in stop_reason else "stopped_early"
    )

    logger.close(
        status=status,
        optimizer_steps_completed=steps_run,
        total_tokens_processed=tokens_processed,
    )

    manifest_path = write_m3_manifest(
        cfg, out_dir, model_cfg, model, status=status, stop_reason=stop_reason,
        steps_run=steps_run, tokens_per_step=tokens_per_step,
        resumed_tokens=resumed_tokens, best_val_loss=best_val_loss,
        final_loss=final_loss, history=history, warnings=run_warnings,
        deviations=deviations, interruptions=interruptions,
        tokens_path=tokens_path, val_tokens_path=val_tokens_path,
        elapsed=(time.time() - t0), checkpoint_events=checkpoint_events,
    )

    print(f"run manifest: {manifest_path}")
    print(f"status={status} steps={completed_steps}/{total_steps} "
          f"tokens={tokens_processed}")
    if best_val_loss is not None:
        print(f"best validation loss: {best_val_loss:.4f}")
    print("training complete" if finished else f"training ended: {stop_reason or status}")


if __name__ == "__main__":
    main()
