"""``quantsmind.checkpoint/2`` provenance contract (v1.1.0).

A checkpoint is only trustworthy if it answers, from the file alone: which model,
which tokenizer, which prompt contract, which data and which code produced it.
Schema ``/1`` recorded weights, optimizer, step and the resolved config, which is
not enough: the M7 incident was a checkpoint served under a prompt contract it
never trained on, and nothing in the file made that visible.

Schema ``/2`` adds a ``provenance`` block with eleven fields:

1. model geometry          2. parameter count        3. tokenizer identity
4. config hash             5. prompt contract id     6. code revision
7. corpus identity         8. command                 9. seed
10. library versions       11. creation time

Loading is strict for ``/2``. Legacy ``/1`` files remain loadable through an
explicit legacy branch (they carry no contract, so they are treated as
``qm.plain_instruction/1``), never silently as if they were ``/2``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_V1 = "quantsmind.checkpoint/1"
SCHEMA_V2 = "quantsmind.checkpoint/2"

PROVENANCE_FIELDS = (
    "model_geometry",
    "parameter_count",
    "tokenizer",
    "config_sha256",
    "contract_id",
    "code_revision",
    "corpus",
    "command",
    "seed",
    "libraries",
    "created_at",
)


class ContractMismatch(RuntimeError):
    """A checkpoint's recorded prompt/tokenizer contract does not match the loader."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(model_cfg: dict) -> str:
    """Stable hash of the model geometry (key order and whitespace invariant)."""
    return _sha256_text(json.dumps(model_cfg, sort_keys=True, separators=(",", ":")))


def code_revision(cwd: str | Path | None = None) -> str:
    """Best-effort code revision; never raises, never blocks a save."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rev = out.stdout.strip()
        return rev if out.returncode == 0 and rev else "unknown"
    except Exception:  # noqa: BLE001 - provenance must never break training
        return "unknown"


def library_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "tokenizers", "numpy", "pyyaml"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def tokenizer_identity(tokenizer=None, tokenizer_path: str | None = None,
                       tokenizer_sha256: str | None = None) -> dict:
    """Record path, digest and vocabulary so a loader can prove the pairing."""
    from training.tokenizer_contract import (
        contract_ids,
        tokenizer_fingerprint,
        tokenizer_file_fingerprint,
    )

    digest = tokenizer_sha256
    if digest is None and tokenizer_path and Path(tokenizer_path).is_file():
        digest = tokenizer_file_fingerprint(tokenizer_path)
    if digest is None and tokenizer is not None:
        digest = tokenizer_fingerprint(tokenizer)
    return {
        "path": str(tokenizer_path) if tokenizer_path else None,
        "sha256": digest,
        "vocab_size": tokenizer.get_vocab_size() if tokenizer is not None else None,
        "control_ids": contract_ids(tokenizer) if tokenizer is not None else None,
    }


def build_provenance(*, model_cfg: dict, parameter_count: int, contract_id: str,
                     config: dict, seed: int | None, tokenizer=None,
                     tokenizer_path: str | None = None,
                     tokenizer_sha256: str | None = None,
                     corpus_manifest: str | None = None,
                     corpus_sha256: str | None = None,
                     command: str | None = None) -> dict:
    """Assemble the eleven required fields. Missing identity is ``None``, never invented."""
    return {
        "model_geometry": dict(model_cfg),
        "parameter_count": int(parameter_count),
        "tokenizer": tokenizer_identity(tokenizer, tokenizer_path, tokenizer_sha256),
        "config_sha256": config_sha256(model_cfg),
        "contract_id": contract_id,
        "code_revision": code_revision(),
        "corpus": {"manifest": corpus_manifest, "sha256": corpus_sha256},
        "command": command if command is not None else " ".join(sys.argv),
        "seed": seed,
        "libraries": library_versions(),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def validate_provenance(provenance: dict | None) -> list[str]:
    """Return every missing/invalid field. An empty list means complete."""
    if not isinstance(provenance, dict):
        return ["provenance block is missing"]
    problems = [f"missing field: {f}" for f in PROVENANCE_FIELDS if f not in provenance]
    tok = provenance.get("tokenizer")
    if not isinstance(tok, dict) or not tok.get("sha256"):
        problems.append("tokenizer identity has no sha256")
    if not provenance.get("contract_id"):
        problems.append("contract_id is empty")
    if not provenance.get("config_sha256"):
        problems.append("config_sha256 is empty")
    if not isinstance(provenance.get("parameter_count"), int):
        problems.append("parameter_count is not an integer")
    if not isinstance(provenance.get("seed"), int):
        problems.append("seed is not an integer")
    return problems


def verify_against(provenance: dict, *, tokenizer=None, tokenizer_sha256: str | None = None,
                   contract_id: str | None = None, model_cfg: dict | None = None) -> None:
    """Refuse a checkpoint whose recorded identity disagrees with the loader.

    Only identities the caller can actually check are compared; an unrecorded
    field is a refusal for ``/2`` because the point of the schema is that the
    answer is present.
    """
    problems = validate_provenance(provenance)
    if problems:
        raise ContractMismatch("checkpoint provenance incomplete: " + "; ".join(problems))

    if contract_id is not None and provenance["contract_id"] != contract_id:
        raise ContractMismatch(
            f"checkpoint was trained under contract {provenance['contract_id']!r}, "
            f"but the loader requires {contract_id!r}"
        )
    if model_cfg is not None and provenance["config_sha256"] != config_sha256(model_cfg):
        raise ContractMismatch(
            "checkpoint model geometry hash does not match the loader's config"
        )

    recorded = provenance["tokenizer"].get("sha256")
    if tokenizer is not None and recorded:
        from training.tokenizer_contract import tokenizer_fingerprint

        if tokenizer_fingerprint(tokenizer) != recorded:
            raise ContractMismatch(
                "checkpoint tokenizer hash does not match the tokenizer being loaded"
            )
    if tokenizer_sha256 is not None and recorded and tokenizer_sha256 != recorded:
        raise ContractMismatch(
            "checkpoint tokenizer file hash does not match the tokenizer file supplied"
        )


def schema_of(ckpt: dict) -> str | None:
    return ckpt.get("schema") if isinstance(ckpt, dict) else None
