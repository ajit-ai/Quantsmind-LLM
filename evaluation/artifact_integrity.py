"""Read-only integrity checks for optional model artifacts.

The canonical v180_sft checkpoint is intentionally not committed. These helpers
make absence explicit instead of letting a missing file masquerade as a hash
verification failure or success.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

V180_SFT_REL = Path("runs/v180_sft/checkpoint_last.pt")
V180_SFT_SHA256 = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"


def resolve_checkpoint_path(repo: Path, relative: Path = V180_SFT_REL) -> Path:
    override = os.environ.get("TAKSH_V180_SFT_CHECKPOINT")
    return Path(override) if override else Path(repo) / relative


def checkpoint_hash_status(path: Path, expected_sha256: str = V180_SFT_SHA256) -> str:
    """Return ABSENT/MATCH/MISMATCH without treating absence as success."""
    if not path.is_file():
        return "ABSENT"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return "MATCH" if digest.hexdigest() == expected_sha256 else "MISMATCH"


def checkpoint_evidence(repo: Path, expected_sha256: str = V180_SFT_SHA256) -> dict[str, str]:
    path = resolve_checkpoint_path(repo)
    return {
        "path": path.as_posix(),
        "status": checkpoint_hash_status(path, expected_sha256),
        "expected_sha256": expected_sha256,
    }
