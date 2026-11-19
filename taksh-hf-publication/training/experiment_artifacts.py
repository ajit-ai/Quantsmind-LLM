"""Immutable experiment artifacts (Taksh v2.0.0).

v1.9.0 lost its trained checkpoint to a later diagnostic probe that wrote
into the same run directory. This module removes that failure mode: run
directories are **exclusive** and finalized artifacts are **immutable**.

Rules enforced here:

* A run identifier may be used once; creating it again fails.
* A checkpoint destination may be written once; writing it again fails.
* Writes are atomic: the payload goes to a temporary file and is renamed only
  after it is complete, so an interrupted write never leaves a plausible
  looking artifact behind.
* Finalization writes a manifest and a hash inventory, then makes the
  finalized files read-only.
* Verification recomputes the hashes and fails on any mismatch.
* Nothing identifies an artifact by filename alone: every evaluation record
  carries the checkpoint hash it was produced from.

The output root is outside the Git repository by construction: the caller
supplies it, and this module refuses a root inside the working tree unless
``allow_repository_root`` is set explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"
SUBDIRECTORIES = ("checkpoints", "evaluations", "metrics", "integrity")


class ArtifactError(RuntimeError):
    """An artifact operation was refused."""


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RunArtifacts:
    """One immutable experiment run directory."""

    def __init__(self, root, run_id: str, *, create: bool = True,
                 allow_repository_root: bool = False):
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise ArtifactError(f"invalid run id: {run_id!r}")
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.files = {}
        self.created = False
        self.finalized = False
        if not allow_repository_root:
            repo = Path(__file__).resolve().parents[1]
            if self.root == repo or repo in self.root.parents:
                raise ArtifactError(
                    f"the experiment root {self.root} is inside the repository; "
                    "experiments must not write into the source tree"
                )
        self.directory = self.root / run_id
        if create:
            if self.directory.exists():
                raise ArtifactError(
                    f"run directory already exists: {self.directory}; a run "
                    "identifier may be used only once"
                )
            self.directory.mkdir(parents=True)
            for name in SUBDIRECTORIES:
                (self.directory / name).mkdir()
            self.created = True
        elif not self.directory.is_dir():
            raise ArtifactError(f"run directory does not exist: {self.directory}")

    def path(self, *parts) -> Path:
        return self.directory.joinpath(*parts)

    # ------------------------------------------------------------------ writes
    def _write_atomic(self, path: Path, data: bytes) -> Path:
        if self.finalized:
            raise ArtifactError(
                f"run {self.run_id} is finalized; no artifact may be added to it"
            )
        if path.exists():
            raise ArtifactError(
                f"refusing to overwrite an existing artifact: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        return path

    def write_checkpoint(self, name: str, payload: dict) -> Path:
        """Write a checkpoint exactly once, atomically."""
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        path = self.path("checkpoints", name)
        import torch

        self._write_atomic(path, _torch_bytes(payload))
        digest = file_sha256(path)
        self.files[f"checkpoints/{name}"] = digest
        return path

    def write_json(self, kind: str, name: str, payload: dict) -> Path:
        """Write a JSON artifact exactly once, atomically."""
        if kind not in ("evaluations", "metrics", "integrity"):
            raise ArtifactError(f"unknown artifact kind: {kind!r}")
        path = self.path(kind, name if name.endswith(".json") else f"{name}.json")
        data = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
        self._write_atomic(path, data)
        self.files[f"{kind}/{path.name}"] = file_sha256(path)
        return path

    def write_evaluation(self, name: str, payload: dict) -> Path:
        """Write an evaluation record, which must name its checkpoint hash."""
        checkpoint_hash = payload.get("checkpoint_sha256")
        if not checkpoint_hash:
            raise ArtifactError(
                "an evaluation record must carry the checkpoint hash it was "
                "produced from"
            )
        return self.write_json("evaluations", name, payload)

    def write_metric(self, name: str, payload: dict) -> Path:
        return self.write_json("metrics", name, payload)

    # ------------------------------------------------------------- finalisation
    def finalize(self, manifest: dict) -> Path:
        """Finalize the run: write the manifest and the hash inventory.

        A finalized run cannot be finalized again, and its files are made
        read-only at the application level.
        """
        if self.finalized:
            raise ArtifactError(
                f"run {self.run_id} is already finalized and cannot be changed"
            )
        manifest = dict(manifest)
        manifest.setdefault("run_id", self.run_id)
        manifest.setdefault(
            "finalized_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        manifest["files"] = dict(sorted(self.files.items()))
        manifest["manifest_sha256"] = canonical_digest(manifest)
        path = self.path(MANIFEST_NAME)
        data = (json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
        self._write_atomic(path, data)
        self.files[MANIFEST_NAME] = file_sha256(path)
        self.write_json("integrity", "hashes.json", {"files": dict(sorted(self.files.items()))})
        for relative in list(self.files):
            target = self.path(relative)
            if target.exists():
                target.chmod(stat.S_IREAD)
        self.finalized = True
        return path

    def verify(self) -> dict:
        """Recompute every recorded hash and report any mismatch."""
        inventory_path = self.path("integrity", "hashes.json")
        if not inventory_path.is_file():
            raise ArtifactError(
                f"run {self.run_id} has no hash inventory; it was never finalized"
            )
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        mismatches = []
        missing = []
        for relative, expected in sorted(inventory.get("files", {}).items()):
            target = self.path(relative)
            if not target.is_file():
                missing.append(relative)
                continue
            actual = file_sha256(target)
            if actual != expected:
                mismatches.append(
                    {"file": relative, "expected": expected, "actual": actual}
                )
        return {
            "run_id": self.run_id,
            "files_checked": len(inventory.get("files", {})),
            "mismatches": mismatches,
            "missing": missing,
            "intact": not mismatches and not missing,
        }


def _torch_bytes(payload: dict) -> bytes:
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()
