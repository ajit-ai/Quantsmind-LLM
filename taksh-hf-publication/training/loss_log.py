"""Append-only machine-readable training logs (A5).

Every code run in this repository discarded its loss history, which is why the
loss figures quoted in the docs cannot be checked against any artifact
(``OPEN_ISSUES`` F2). This module makes retention the default: one JSON object
per line, flushed as it is written, so an interrupted run still leaves a usable
record of everything that happened up to the interruption.

Design notes
------------
* **Append, never truncate.** A second run against the same path adds to the file
  and is distinguishable by its ``run_id``, so history is not silently replaced.
* **A run header** is written once when a run starts, identifying the model
  config, dataset and config fingerprint. Records that follow carry that identity
  through ``run_id`` only.
* **Failures surface.** If the log cannot be written, the training loop is told
  and stops, rather than continuing with no record.
* **No secrets, no examples.** Only scalars, identifiers and timing are logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.training_loss_log/1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_fingerprint(cfg: dict) -> str:
    """Stable short digest of a training config, for run identification."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class LossLogger:
    """Append-only JSONL sink for training/validation loss records."""

    def __init__(
        self,
        path: str | Path,
        run_id: str | None = None,
        run_kind: str = "pretrain",
        model_config: str | None = None,
        dataset: str | None = None,
        config: dict | None = None,
        enabled: bool = True,
    ):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_kind = run_kind
        self.enabled = enabled
        self.records_written = 0
        self._fh = None
        self.header = {
            "schema": SCHEMA,
            "event": "run_start",
            "run_id": self.run_id,
            "run_kind": run_kind,
            "started_at": utc_now_iso(),
            "model_config": model_config,
            "dataset": dataset,
            "config_fingerprint": config_fingerprint(config) if config else None,
            "pid": os.getpid(),
        }
        if self.enabled:
            self._open()

    def _open(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode is the whole point: an existing log from an earlier run
            # is evidence and is never overwritten.
            self._fh = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"loss log cannot be opened at {self.path}: {exc}. "
                "Refusing to train without a place to record results."
            ) from exc

    def _write(self, payload: dict):
        if not self.enabled:
            return
        if self._fh is None:
            self._open()
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except OSError as exc:
            # Surface it. A run that cannot record its loss is not a run whose
            # results can be trusted later.
            raise RuntimeError(
                f"loss log write failed for {self.path}: {exc}. "
                "Training is stopping rather than continuing unrecorded."
            ) from exc
        self.records_written += 1

    def log_header(self, **extra):
        payload = dict(self.header)
        payload.update(extra)
        self._write(payload)

    def log_step(
        self,
        step: int,
        loss: float,
        *,
        epoch: float | None = None,
        lr: float | None = None,
        ema_loss: float | None = None,
        grad_norm: float | None = None,
        tokens_per_second: float | None = None,
        consumed_examples: int | None = None,
        phase: str = "train",
        extra: dict | None = None,
    ):
        payload = {
            "schema": SCHEMA,
            "event": "step",
            "run_id": self.run_id,
            "phase": phase,
            "step": step,
            "loss": loss,
            "timestamp": utc_now_iso(),
        }
        for key, value in (
            ("epoch", epoch),
            ("lr", lr),
            ("ema_loss", ema_loss),
            ("grad_norm", grad_norm),
            ("tokens_per_second", tokens_per_second),
            ("consumed_examples", consumed_examples),
        ):
            if value is not None:
                payload[key] = value
        if extra:
            payload["extra"] = extra
        self._write(payload)

    def log_validation(self, step: int, val_loss: float, **extra):
        self.log_step(step, val_loss, phase="validation", extra=extra or None)

    def log_end(self, status: str = "complete", **extra):
        payload = {
            "schema": SCHEMA,
            "event": "run_end",
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "status": status,
            "finished_at": utc_now_iso(),
            "records_written": self.records_written,
        }
        payload.update(extra)
        self._write(payload)

    def close(self, status: str = "complete", **extra):
        try:
            self.log_end(status=status, **extra)
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "interrupted" if exc_type is not None else "complete"
        try:
            self.close(status=status)
        except Exception:  # noqa: BLE001 - never mask the original error
            pass
        return False


def read_log(path: str | Path) -> list[dict]:
    """Parse a loss log. Returns an empty list when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records