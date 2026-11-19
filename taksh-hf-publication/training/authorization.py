"""Dataset authorization gate for Taksh v1.3.0.

Training data reaches the trainer only through this gate. The gate reads a
v1.2.0 dataset manifest, verifies it, and decides whether the configured mode
may use it.

Two modes, and they never mix:

``production``
    Requires a manifest whose declared class is ``production``, whose
    ``corpus_ready`` flag is true, that has at least one approved source, and
    whose splits are ready. Every artifact checksum is verified. A manifest
    labelled ``synthetic`` is refused: a qualification fixture can never be
    promoted to production.

``synthetic``
    Requires a manifest whose declared class is ``synthetic``. Production
    manifests are refused in this mode too, so a synthetic run can never be
    confused with a production one. Outputs of a synthetic run are labelled
    as synthetic in the checkpoint, the run manifest and the evaluation
    report.

There is no fallback. A missing manifest, a tampered artifact, a tokenizer
that does not match the manifest, or a not-ready split is an error, never a
silent switch to a historical corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from data.python_v1 import MANIFEST_SCHEMA
from data.python_v1.interface import DatasetInterfaceError, PythonDataset
from data.python_v1.registry import file_sha256

MODES = ("production", "synthetic")


class AuthorizationError(RuntimeError):
    """The dataset is not authorized for this run."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Authorization:
    """The verified facts about the dataset a run may use."""

    mode: str
    manifest_path: Path
    manifest: dict
    manifest_sha256: str
    corpus_class: str
    corpus_ready: bool
    approved_sources: list = field(default_factory=list)
    tokenizer_path: Path | None = None
    tokenizer_sha256: str | None = None
    contract_id: str | None = None
    splits: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    @property
    def dataset_id(self) -> str:
        return self.manifest_sha256

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})


def _load_manifest(manifest_path) -> dict:
    path = Path(manifest_path)
    if not path.is_file():
        raise AuthorizationError(
            "manifest_missing", f"no dataset manifest at {path}. Refusing to fall back."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthorizationError("manifest_unreadable", str(exc)) from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AuthorizationError(
            "manifest_schema_mismatch",
            f"schema is {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}",
        )
    return manifest


def authorize_dataset(
    manifest_path,
    *,
    mode: str,
    tokenizer_path=None,
    verify_artifacts: bool = True,
) -> Authorization:
    """Authorize a dataset for training, or refuse.

    Returns an :class:`Authorization` describing what was verified. Raises
    :class:`AuthorizationError` with a stable reason code otherwise.
    """
    if mode not in MODES:
        raise AuthorizationError("unknown_mode", f"mode={mode!r}")
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    notes = manifest.get("notes") or {}
    corpus_class = notes.get("corpus_class", "production")
    corpus_ready = bool(notes.get("corpus_ready", False))
    approved = list(manifest.get("sources", {}).get("approved", []))
    splits = manifest.get("splits", {})

    authorization = Authorization(
        mode=mode,
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=manifest.get("manifest_sha256", ""),
        corpus_class=corpus_class,
        corpus_ready=corpus_ready,
        approved_sources=approved,
        splits=splits,
    )

    if mode == "production":
        if corpus_class != "production":
            raise AuthorizationError(
                "synthetic_manifest_refused",
                f"the manifest declares class {corpus_class!r}; a synthetic "
                "qualification fixture cannot authorize production training",
            )
        if not corpus_ready:
            raise AuthorizationError(
                "corpus_not_ready",
                "the manifest does not declare corpus_ready: true; an owner "
                "decision on the source set and licensing is required",
            )
        if not approved:
            raise AuthorizationError(
                "no_approved_sources", "the manifest lists no approved source"
            )
        if not splits.get("ready", False):
            raise AuthorizationError(
                "splits_not_ready",
                "; ".join(splits.get("insufficient", ["no reason recorded"])),
            )
        authorization.record("production_class", True)
        authorization.record("corpus_ready", True)
        authorization.record("approved_sources", bool(approved), ",".join(approved))
    else:
        if corpus_class != "synthetic":
            raise AuthorizationError(
                "production_manifest_refused_in_synthetic_mode",
                f"the manifest declares class {corpus_class!r}; synthetic "
                "qualification runs must use a manifest that declares itself "
                "synthetic, so the two can never be confused",
            )
        if not notes.get("synthetic", False):
            raise AuthorizationError(
                "synthetic_flag_missing",
                "the manifest is not labelled synthetic in its notes",
            )
        authorization.record("synthetic_class", True)
        authorization.record("synthetic_labelled", True)

    # Tokenizer identity ---------------------------------------------------
    recorded = manifest.get("tokenizer", {}) or {}
    if tokenizer_path is not None:
        tokenizer = Path(tokenizer_path)
        if not tokenizer.is_file():
            raise AuthorizationError("tokenizer_missing", str(tokenizer))
        digest = file_sha256(tokenizer)
        declared = recorded.get("sha256")
        if declared and digest != declared:
            raise AuthorizationError(
                "tokenizer_mismatch",
                f"the manifest was prepared with tokenizer {declared}, the "
                f"supplied tokenizer hashes to {digest}",
            )
        authorization.tokenizer_path = tokenizer
        authorization.tokenizer_sha256 = digest
        authorization.contract_id = recorded.get("contract_id")
        authorization.record("tokenizer_identity", True, digest)
    else:
        raise AuthorizationError(
            "tokenizer_required",
            "a tokenizer path is required so the data can be encoded with the "
            "artifact the manifest was prepared with",
        )

    # Artifact integrity ----------------------------------------------------
    if verify_artifacts:
        for split in ("train", "val", "test"):
            try:
                PythonDataset(
                    path,
                    split,
                    tokenizer_path=tokenizer,
                    verify=True,
                    require_ready=False,
                )
            except DatasetInterfaceError as exc:
                raise AuthorizationError(
                    "artifact_verification_failed", f"{split}: {exc}"
                ) from exc
        authorization.record("artifact_checksums", True)
    return authorization
