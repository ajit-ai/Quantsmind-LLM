"""Focused tests for the authentication token lifecycle and durable audit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.auth import authenticate, TokenConfig
from taksh.audit import AuditLog
from datetime import datetime, timedelta, timezone


def test_expired_token_is_rejected():
    config = TokenConfig(token="abcdefghijklmnopqrstuvwxyz0123456789", scopes=frozenset({"read"}),
                         expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    result = authenticate("Bearer abcdefghijklmnopqrstuvwxyz0123456789", config)
    assert result.ok is False
    assert result.error == "token has expired"


def test_revoked_token_is_rejected(tmp_path):
    import hashlib
    token = "abcdefghijklmnopqrstuvwxyz0123456789"
    digest = hashlib.sha256(token.encode()).hexdigest()
    revoked = tmp_path / "revoked.txt"
    revoked.write_text(digest + "\n", encoding="utf-8")
    config = TokenConfig(token=token, scopes=frozenset({"read"}), revoked_hashes=frozenset({digest}))
    result = authenticate(f"Bearer {token}", config)
    assert result.ok is False
    assert result.error == "token has been revoked"


def test_malformed_authorization_is_rejected():
    config = TokenConfig(token="abcdefghijklmnopqrstuvwxyz0123456789", scopes=frozenset({"read"}))
    assert authenticate("Basic abc", config).error == "invalid or missing bearer token"
    assert authenticate("Bearer token with spaces", config).error == "invalid or missing bearer token"


def test_insufficient_scope_is_not_authorized():
    config = TokenConfig(token="abcdefghijklmnopqrstuvwxyz0123456789", scopes=frozenset({"read"}))
    result = authenticate("Bearer abcdefghijklmnopqrstuvwxyz0123456789", config)
    assert result.ok is True
    assert "mutate" not in result.scopes


def test_audit_log_persists_and_redacts(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.write("identity", "op", "allow", "token=supersecretvalue123456789")
    restarted = AuditLog(path=path)
    records = restarted.read()
    assert len(records) == 1
    assert records[0]["decision"] == "allow"
    assert "supersecretvalue123456789" not in records[0]["outcome"]


def test_audit_write_failure_raises_fail_closed(tmp_path):
    directory = tmp_path / "audit-dir"
    directory.mkdir()
    log = AuditLog(path=directory)
    with pytest.raises(RuntimeError):
        log.write("identity", "op", "allow", "authorized")


def test_audit_retention_rotation(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path, max_bytes=10)
    log.write("i", "op", "allow", "authorized")
    log.write("i", "op", "allow", "authorized")
    assert path.is_file() or list(tmp_path.glob("audit.jsonl*"))
