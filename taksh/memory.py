"""Project-scoped persistent memory for TAKSH."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryFact:
    fact_id: str
    project_id: str
    kind: str
    value: Any
    source: str
    confidence: float
    verified: bool
    created_at: float
    updated_at: float
    expires_at: float | None = None
    invalidated: bool = False
    invalidation_reason: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_stale(self, source_hash: str | None = None) -> bool:
        if self.invalidated:
            return True
        if self.expires_at is not None and time.time() > self.expires_at:
            return True
        return False


class ProjectMemory:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.facts: dict[str, MemoryFact] = {}
        self._load()

    def _project_key(self, project_id: str) -> str:
        return hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> None:
        if not self.storage_path.is_file():
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for item in data.get("facts", []):
            fact = MemoryFact(**item)
            self.facts[fact.fact_id] = fact

    def _save(self) -> None:
        payload = {"facts": [fact.to_dict() for fact in self.facts.values()]}
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def store(self, project_id: str, kind: str, value: Any, *, source: str, confidence: float = 0.5, verified: bool = False, ttl_seconds: float | None = None, tags: list[str] | None = None) -> MemoryFact:
        if any(secret in str(value).lower() for secret in ("password", "api_key", "secret", "token")):
            raise ValueError("refusing to store secret-like value")
        fact_id = f"{self._project_key(project_id)}:{kind}:{len(self.facts)}"
        now = time.time()
        fact = MemoryFact(
            fact_id=fact_id,
            project_id=project_id,
            kind=kind,
            value=value,
            source=source,
            confidence=confidence,
            verified=verified,
            created_at=now,
            updated_at=now,
            expires_at=(now + ttl_seconds) if ttl_seconds else None,
            tags=tags or [],
        )
        self.facts[fact_id] = fact
        self._save()
        return fact

    def get(self, project_id: str, kind: str | None = None) -> list[MemoryFact]:
        result = [fact for fact in self.facts.values() if fact.project_id == project_id]
        if kind is not None:
            result = [fact for fact in result if fact.kind == kind]
        return result

    def invalidate(self, fact_id: str, reason: str) -> None:
        if fact_id in self.facts:
            self.facts[fact_id].invalidated = True
            self.facts[fact_id].invalidation_reason = reason
            self.facts[fact_id].updated_at = time.time()
            self._save()

    def invalidate_project(self, project_id: str, reason: str) -> int:
        count = 0
        for fact in self.facts.values():
            if fact.project_id == project_id and not fact.invalidated:
                fact.invalidated = True
                fact.invalidation_reason = reason
                fact.updated_at = time.time()
                count += 1
        self._save()
        return count

    def delete(self, fact_id: str) -> bool:
        if fact_id in self.facts:
            del self.facts[fact_id]
            self._save()
            return True
        return False

    def clear_project(self, project_id: str) -> int:
        to_delete = [fid for fid, fact in self.facts.items() if fact.project_id == project_id]
        for fid in to_delete:
            del self.facts[fid]
        self._save()
        return len(to_delete)

    def inspect(self, project_id: str | None = None) -> dict[str, Any]:
        facts = list(self.facts.values())
        if project_id is not None:
            facts = [fact for fact in facts if fact.project_id == project_id]
        return {
            "storage_path": str(self.storage_path),
            "fact_count": len(facts),
            "facts": [fact.to_dict() for fact in facts],
        }
