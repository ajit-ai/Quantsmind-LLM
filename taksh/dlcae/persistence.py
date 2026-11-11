"""Persistence and recovery for DLCAE sequence and knowledge state.

Atomic updates (``.tmp_partial`` + fsync + ``os.replace``), a content-hashed
manifest, integrity verification and tamper detection, interrupted-operation
recovery, idempotent re-saves, and an explicit schema-version gate. No file is
ever silently repaired: any mismatch is reported.

Nothing here touches model checkpoints, benchmarks, corpora, or committed
registries.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class PersistenceError(Exception):
    pass


class CorruptionError(PersistenceError):
    """Raised when a persisted artifact fails integrity verification."""


class PersistenceStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "MANIFEST.json"

    # ------------------------------------------------------------------ #
    @staticmethod
    def atomic_write_bytes(target: Path, payload: bytes) -> str:
        tmp = target.with_name(target.name + ".tmp_partial")
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return sha256(target.read_bytes()).hexdigest()

    def atomic_write_json(self, relative: str, obj: Any) -> str:
        target = self._target(relative)
        payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
        return self.atomic_write_bytes(target, payload)

    def _target(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            raise PersistenceError(f"path escapes the persistence root: {relative!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------------ #
    def save_json(self, relative: str, obj: Any) -> str:
        rel = Path(relative).as_posix()
        digest = self.atomic_write_json(rel, obj)
        manifest = self._load_manifest()
        manifest["files"][rel] = digest
        manifest["generation"] += 1
        self._save_manifest(manifest)
        return digest

    def save_json_idempotent(self, relative: str, obj: Any) -> dict[str, Any]:
        """Skip the write when the content is already persisted identically."""
        rel = Path(relative).as_posix()
        payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
        digest = sha256(payload).hexdigest()
        manifest = self._load_manifest()
        target = self._target(rel)
        existing = manifest["files"].get(rel)
        if target.is_file() and existing == digest and not self.verify_integrity()["problems"]:
            return {"relative": rel, "sha256": digest, "written": False,
                    "reason": "identical content already persisted"}
        return {"relative": rel,"sha256": self.save_json(rel, obj),
                "written": True, "reason": "written"}

    def load_json(self, relative: str, *, verify: bool = True) -> Any:
        rel = Path(relative).as_posix()
        target = self._target(rel)
        if not target.is_file():
            raise PersistenceError(f"missing artifact: {rel}")
        if verify:
            manifest = self._load_manifest()
            expected = manifest["files"].get(rel)
            actual = sha256(target.read_bytes()).hexdigest()
            if expected != actual:
                raise CorruptionError(
                    f"integrity mismatch for {rel}: manifest={expected} actual={actual}")
        return json.loads(target.read_text(encoding="utf-8"))

    def list_artifacts(self) -> list[str]:
        manifest = self._load_manifest()
        return sorted(manifest["files"])

    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schema": "taksh.dlcae.persistence/1",
                    "schema_version": SCHEMA_VERSION,
                    "generation": 0,
                    "files": {}}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise PersistenceError(
                f"schema version mismatch: {manifest.get('schema_version')} != {SCHEMA_VERSION}; "
                f"refusing silent migration — a schema migration must be an explicit, "
                f"reviewed operation")
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        self.atomic_write_bytes(self.manifest_path, payload)

    def verify_integrity(self) -> dict[str, Any]:
        problems: list[str] = []
        manifest = self._load_manifest()
        for rel, expected in sorted(manifest["files"].items()):
            target = self._target(rel)
            if not target.is_file():
                problems.append(f"missing file recorded in manifest: {rel}")
                continue
            actual = sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                problems.append(f"hash mismatch (tampered or corrupt): {rel}")
        # detect unrecorded files (excluding the manifest itself and tmp debris)
        recorded = set(manifest["files"]) | {"MANIFEST.json"}
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                if rel.endswith(".tmp_partial"):
                    problems.append(f"stale interrupted write: {rel}")
                elif rel not in recorded:
                    problems.append(f"unrecorded file (possible tamper): {rel}")
        return {"ok": not problems, "problems": problems,
                "generation": manifest["generation"],
                "file_count": len(manifest["files"])}

    def interrupt(self, relative: str, payload_bytes: bytes) -> Path:
        """Simulate an interrupted write, for recovery testing only."""
        target = self._target(relative)
        tmp = target.with_name(target.name + ".tmp_partial")
        tmp.write_bytes(payload_bytes)
        return tmp

    def recover(self, relative: str) -> dict[str, Any]:
        """Remove interrupted-write debris only when the target is absent or invalid.

        Recovery never fabricates content: a missing file is reported as
        missing, and the caller must rebuild from the source of truth.
        """
        rel = Path(relative).as_posix()
        target = self._target(rel)
        tmp = target.with_name(target.name + ".tmp_partial")
        interrupted = tmp.is_file()
        if interrupted:
            # verify_integrity() would report it; we clean only the debris here
            # because the target is either absent or intact.
            if target.is_file():
                try:
                    self.load_json(rel)
                    tmp.unlink()
                except PersistenceError:
                    pass  # target not trustworthy; leave evidence in place
            # if the target is missing, keep the partial for inspection
        return {"relative": rel, "interrupted_debris_present": interrupted,
                "target_exists": target.is_file(),
                "recovered": interrupted and target.is_file()}
