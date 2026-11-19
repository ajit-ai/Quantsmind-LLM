"""Machine-readable run manifest (M3 §12).

One file per run that contains everything needed to identify and reload the
model: architecture, parameter count, tokenizer and dataset identities with
hashes, the full config, seed, library versions, hardware, what actually
happened, and the checkpoint's own hash.

Two rules it enforces:

* A run that did not finish is never recorded as a successful training
  artifact. ``status`` is ``complete``, ``interrupted``, ``failed`` or
  ``stopped_early``, and ``success`` is derived from it.
* Warnings, interruptions and deviations are recorded, not discarded. A run
  that recovered from a NaN or resumed from a checkpoint says so here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.run_manifest/1"

STATUS_COMPLETE = "complete"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_STOPPED_EARLY = "stopped_early"
STATUS_SMOKE = "smoke"

# Only these count as a successful training artifact.
SUCCESS_STATUSES = {STATUS_COMPLETE}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "numpy", "tokenizers", "yaml"):
        module = name if name != "yaml" else "yaml"
        try:
            imported = __import__(module)
            versions[name] = getattr(imported, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absent optional dependency
            versions[name] = "not installed"
    return versions


def hardware_info() -> dict:
    info = {"logical_cpus": None, "cuda_available": False, "device": "cpu"}
    try:
        import os

        info["logical_cpus"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = (
            torch.cuda.get_device_name(0) if info["cuda_available"] else None
        )
        info["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return info


def peak_memory_info() -> dict:
    from training.preflight import peak_rss_bytes

    peak = peak_rss_bytes()
    return {
        "peak_rss_bytes": peak,
        "kind": "measured" if peak else "not measurable on this platform",
    }


def build_run_manifest(*, run_id, status, config, config_path, model_cfg,
                       model_config_path, params, tokenizer_path=None,
                       tokenizer_vocab=None, dataset_manifest=None,
                       dataset_paths=None, seed=None, checkpoint_path=None,
                       steps_completed=0, planned_steps=0, tokens_processed=0,
                       metrics=None, start_time=None, end_time=None,
                       elapsed_seconds=None, warnings=None, deviations=None,
                       interruptions=None, preflight=None, stopping_criteria=None,
                       history=None, notes=None):
    """Assemble the manifest. Nothing is inferred; unrecorded values stay None."""
    checkpoint = Path(checkpoint_path) if checkpoint_path else None

    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": status,
        "success": status in SUCCESS_STATUSES,
        "success_rule": (
            f"success is true only for status in {sorted(SUCCESS_STATUSES)}; an "
            "interrupted, failed or stopped-early run is never a successful "
            "training artifact"
        ),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed_seconds,
        "libraries": library_versions(),
        "hardware": hardware_info(),
        "memory": peak_memory_info(),
        "model": {
            "architecture": "decoder-only causal transformer "
                            "(RMSNorm, RoPE, GQA, SwiGLU)",
            "objective": "next-token prediction",
            "parameters": params,
            "config": model_cfg,
            "config_path": str(model_config_path) if model_config_path else None,
            "initialization": "random (torch.nn.init.normal_, std=0.02); "
                              "no pretrained weights were loaded",
        },
        "tokenizer": {
            "path": str(tokenizer_path) if tokenizer_path else None,
            "vocab_size": tokenizer_vocab,
            "sha256": sha256_file(tokenizer_path) if tokenizer_path else None,
        },
        "dataset": {
            "manifest_path": str(dataset_manifest) if dataset_manifest else None,
            "manifest_sha256": sha256_file(dataset_manifest)
            if dataset_manifest else None,
            "artifacts": {
                str(path): {
                    "sha256": sha256_file(path),
                    "bytes": Path(path).stat().st_size if Path(path).exists() else None,
                }
                for path in (dataset_paths or [])
            },
        },
        "config": {
            "path": str(config_path) if config_path else None,
            "sha256": sha256_file(config_path) if config_path else None,
            "resolved": config,
        },
        "seed": seed,
        "training": {
            "steps_completed": steps_completed,
            "planned_steps": planned_steps,
            "tokens_processed": tokens_processed,
            "metrics": metrics or {},
            "history": history or [],
            "stopping_criteria": stopping_criteria or {},
        },
        "checkpoint": {
            "path": str(checkpoint) if checkpoint else None,
            "sha256": sha256_file(checkpoint) if checkpoint else None,
            "bytes": checkpoint.stat().st_size
            if checkpoint and checkpoint.exists() else None,
            "format": "quantsmind.checkpoint/1",
        },
        "preflight": preflight,
        "warnings": list(warnings or []),
        "deviations": list(deviations or []),
        "interruptions": list(interruptions or []),
        "notes": list(notes or []),
    }
    return manifest


def write_run_manifest(path, manifest) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)