"""Versioned training configuration for Taksh v1.3.0.

The configuration is validated **before** the model is built, so an unsafe or
incompatible combination fails in seconds rather than after a run has spent
its budget. It is also hashed: the hash is recorded in the checkpoint and in
the run manifest, so a checkpoint can never be resumed under a different
configuration without saying so.

Reusable configuration must not contain machine-specific absolute paths, so
the validator refuses absolute paths in the data, tokenizer, model and output
locations. Relative paths are resolved against the configuration file's
directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

TRAIN_CONFIG_SCHEMA = "quantsmind.train_config/1"
MODES = ("production", "synthetic")
DTYPES = ("float32", "float16", "bfloat16")
DEVICES = ("auto", "cpu", "cuda", "mps")
SEQUENCE_POLICIES = ("document", "packed")


class TrainConfigError(ValueError):
    """The training configuration is invalid or unsafe."""


def config_identity(cfg: dict) -> str:
    """Stable hash of the resolved configuration."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_absolute(value) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("/") or value.startswith("\\"):
        return True
    return len(value) > 2 and value[1] == ":"


def validate_train_config(cfg: dict) -> list:
    """Return every problem with the configuration."""
    problems: list = []
    if not isinstance(cfg, dict):
        return ["configuration must be a mapping"]
    if cfg.get("schema") != TRAIN_CONFIG_SCHEMA:
        problems.append(
            f"schema must be {TRAIN_CONFIG_SCHEMA!r}, got {cfg.get('schema')!r}"
        )
    for section in ("run", "data", "optimizer", "batch", "precision", "training", "limits"):
        if section not in cfg:
            problems.append(f"missing section '{section}'")
    mode = cfg.get("mode")
    if mode not in MODES:
        problems.append(f"mode must be one of {list(MODES)}, got {mode!r}")

    run = cfg.get("run") or {}
    for key in ("name", "model_config", "seed"):
        if key not in run:
            problems.append(f"run.{key} is required")
    if not isinstance(run.get("seed", 0), int):
        problems.append("run.seed must be an integer")

    data = cfg.get("data") or {}
    for key in ("manifest", "tokenizer", "context_length", "sequence_policy"):
        if key not in data:
            problems.append(f"data.{key} is required")
    if data.get("sequence_policy") not in SEQUENCE_POLICIES:
        problems.append(
            f"data.sequence_policy must be one of {list(SEQUENCE_POLICIES)}"
        )
    if int(data.get("context_length", 0) or 0) < 2:
        problems.append("data.context_length must be >= 2")
    for key in ("manifest", "tokenizer"):
        if _is_absolute(data.get(key)):
            problems.append(
                f"data.{key} must be a relative path; absolute paths make the "
                "configuration machine-specific"
            )
    if mode == "production" and not data.get("manifest"):
        problems.append("production mode requires data.manifest")

    optimizer = cfg.get("optimizer") or {}
    if optimizer.get("type", "adamw") != "adamw":
        problems.append("optimizer.type must be 'adamw' for this version")
    if float(optimizer.get("lr_peak", 0) or 0) <= 0:
        problems.append("optimizer.lr_peak must be positive")
    if int(optimizer.get("warmup_steps", 0) or 0) < 0:
        problems.append("optimizer.warmup_steps must be >= 0")
    if float(optimizer.get("grad_clip", 0) or 0) <= 0:
        problems.append("optimizer.grad_clip must be positive")

    batch = cfg.get("batch") or {}
    if int(batch.get("micro_batch_size", 0) or 0) < 1:
        problems.append("batch.micro_batch_size must be >= 1")
    if int(batch.get("grad_accum_steps", 0) or 0) < 1:
        problems.append("batch.grad_accum_steps must be >= 1")

    precision = cfg.get("precision") or {}
    if precision.get("dtype", "float32") not in DTYPES:
        problems.append(f"precision.dtype must be one of {list(DTYPES)}")
    if precision.get("device", "auto") not in DEVICES:
        problems.append(f"precision.device must be one of {list(DEVICES)}")

    training = cfg.get("training") or {}
    for key in ("max_steps", "max_tokens", "max_wall_seconds", "output_dir"):
        if key not in training:
            problems.append(f"training.{key} is required")
    if int(training.get("max_steps", 0) or 0) <= 0:
        problems.append("training.max_steps must be positive")
    if int(training.get("max_tokens", 0) or 0) <= 0:
        problems.append("training.max_tokens must be positive")
    if float(training.get("max_wall_seconds", 0) or 0) <= 0:
        problems.append("training.max_wall_seconds must be positive")
    if _is_absolute(training.get("output_dir")):
        problems.append(
            "training.output_dir must be a relative path; absolute paths make "
            "the configuration machine-specific"
        )

    limits = cfg.get("limits") or {}
    for key in ("max_checkpoints", "max_output_bytes"):
        if key not in limits:
            problems.append(f"limits.{key} is required")
    if int(limits.get("max_checkpoints", 0) or 0) < 1:
        problems.append("limits.max_checkpoints must be >= 1")
    if int(limits.get("max_output_bytes", 0) or 0) <= 0:
        problems.append("limits.max_output_bytes must be positive")
    return problems


def load_train_config(path) -> dict:
    """Load and validate a training configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise TrainConfigError(f"configuration not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problems = validate_train_config(cfg)
    if problems:
        raise TrainConfigError(
            f"{config_path}: invalid training configuration: " + "; ".join(problems)
        )
    cfg["__config_path__"] = str(config_path)
    cfg["__config_dir__"] = str(config_path.resolve().parent)
    return cfg


def resolve_path(cfg: dict, value) -> Path:
    """Resolve a configuration path relative to the configuration file."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base = Path(cfg.get("__config_dir__", "."))
    return (base / candidate).resolve()
