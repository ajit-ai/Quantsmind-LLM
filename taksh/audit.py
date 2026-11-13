"""Durable audit logging with redaction and rotation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_TOKEN_PATTERNS = (
    "token",
    "api_key",
    "authorization",
    "password",
    "secret",
)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        for marker in _TOKEN_PATTERNS:
            if marker in value.lower():
                return "<redacted>"
        return value
    if isinstance(value, dict):
        return {key: ("<redacted>" if any(marker in key.lower() for marker in _TOKEN_PATTERNS) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class AuditLog:
    def __init__(self, path: str | Path | None = None, max_bytes: int = 1_000_000) -> None:
        self.path = Path(path).resolve() if path else None
        self.max_bytes = max_bytes
        self.records: list[dict[str, Any]] = []

    def write(self, identity: str, operation: str, decision: str, outcome: str) -> dict[str, Any]:
        record = _redact({"identity": identity, "operation": operation, "decision": decision, "outcome": outcome, "timestamp": time.time()})
        self.records.append(record)
        if self.path is not None:
            self._append(record)
        return record

    def read(self, limit: int = 1000) -> list[dict[str, Any]]:
        if self.path is not None and self.path.is_file():
            records = []
            for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return records
        return list(self.records[-limit:])

    def _append(self, record: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                backup = self.path.with_suffix(self.path.suffix + f".{int(time.time())}")
                self.path.rename(backup)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            raise RuntimeError(f"audit persistence failed; authorization must fail closed: {exc}") from exc


def audit_log_from_env() -> AuditLog:
    path = os.environ.get("TAKSH_AUDIT_PATH")
    max_bytes = int(os.environ.get("TAKSH_AUDIT_MAX_BYTES", "1000000"))
    return AuditLog(path=path, max_bytes=max_bytes)
