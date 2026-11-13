"""Authentication token lifecycle for the TAKSH engineering service."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_SCOPES = {"read", "execute", "mutate", "memory", "audit"}


@dataclass(frozen=True)
class TokenConfig:
    token: str | None
    scopes: frozenset[str]
    expires_at: datetime | None = None
    revoked_hashes: frozenset[str] = frozenset()
    source: str = "env"

    @classmethod
    def from_env(cls) -> "TokenConfig":
        token = os.environ.get("TAKSH_ENGINEERING_TOKEN") or os.environ.get("QM_API_KEY") or None
        scopes = frozenset(scope.strip() for scope in os.environ.get("TAKSH_TOKEN_SCOPES", "read").split(",") if scope.strip())
        expires_at = _parse_expires_at(os.environ.get("TAKSH_TOKEN_EXPIRES_AT"))
        revoked_hashes = _load_revoked_hashes(os.environ.get("TAKSH_REVOKED_TOKENS_FILE"))
        return cls(token=token, scopes=scopes, expires_at=expires_at, revoked_hashes=revoked_hashes)


@dataclass(frozen=True)
class AuthResult:
    identity: str
    scopes: frozenset[str]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _parse_expires_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        value = raw.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _load_revoked_hashes(path: str | None) -> frozenset[str]:
    if not path:
        return frozenset()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    hashes = set()
    for line in lines:
        candidate = line.strip().lower()
        if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
            hashes.add(candidate)
    return frozenset(hashes)


def authenticate(authorization_header: str | None, config: TokenConfig | None = None) -> AuthResult:
    config = config or TokenConfig.from_env()
    if config.token is None:
        return AuthResult("unauthenticated", frozenset(), "engineering authentication is not configured")
    if not authorization_header or not authorization_header.lower().startswith("bearer "):
        return AuthResult("malformed-token", frozenset(), "invalid or missing bearer token")
    supplied = authorization_header[7:].strip()
    if " " in supplied or not supplied:
        return AuthResult("malformed-token", frozenset(), "invalid or missing bearer token")

    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
    if digest in config.revoked_hashes:
        return AuthResult("revoked-token", frozenset(), "token has been revoked")
    if config.expires_at is not None and datetime.now(timezone.utc) >= config.expires_at:
        return AuthResult("expired-token", frozenset(), "token has expired")
    if not hmac.compare_digest(supplied, config.token):
        return AuthResult("invalid-token", frozenset(), "invalid or missing bearer token")
    return AuthResult("token", config.scopes)


def authorized(result: AuthResult, required_scope: str) -> bool:
    return result.ok and required_scope in result.scopes
