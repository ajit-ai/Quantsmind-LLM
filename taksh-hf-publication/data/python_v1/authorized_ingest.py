"""Authorized corpus ingestion contract (Taksh v1.6.0).

The ingestion contract is the boundary between "a human approved a source"
and "bytes enter a training corpus". It refuses unapproved material *before*
reading it, and it records enough evidence that the resulting corpus can be
rebuilt or challenged later.

The pipeline reuses the v1.2.0 machinery (traversal, quality filters,
deduplication, source-group splitting, contamination screening, manifest) and
adds the v1.6.0 authorization gate in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from data.python_v1.authorization import evaluate_authorization
from data.python_v1.registry import file_sha256, snapshot_digest

BUILD_MANIFEST_SCHEMA = "quantsmind.authorized_corpus_build/1"


class IngestRefused(RuntimeError):
    """The material may not be ingested."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class SourceEntry:
    """One approved source, its manifest and the local path it lives at."""

    manifest: dict
    local_path: Path
    manifest_path: Path | None = None
    expected_hashes: dict = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return self.manifest.get("source_id", "unknown")


@dataclass
class IngestRequest:
    """Everything an authorized build needs, stated up front."""

    sources: list
    out_dir: Path
    config: dict = field(default_factory=dict)
    pipeline_version: str = "python-v1.6.0"
    intended_use: str | None = None


def verify_revision(entry: SourceEntry) -> str:
    """Recompute the source snapshot and compare it with the approved revision."""
    root = Path(entry.local_path)
    entries = [
        (path.relative_to(root).as_posix(), file_sha256(path))
        for path in sorted(root.rglob("*.py"))
    ]
    observed = snapshot_digest(entries)
    approved = entry.manifest.get("revision")
    if approved and observed != approved:
        raise IngestRefused(
            "revision_mismatch",
            f"{entry.source_id}: approved {approved}, observed {observed}",
        )
    return observed


def authorize_entries(entries: list, *, intended_use: str | None = None) -> list:
    """Authorize every source. Any refusal stops the whole build."""
    decisions = []
    for entry in entries:
        decision = evaluate_authorization(entry.manifest, intended_use=intended_use)
        decisions.append(decision)
        if not decision.allowed:
            raise IngestRefused(
                "source_not_authorized",
                f"{decision.source_id}: {', '.join(decision.reasons)}",
            )
    return decisions


def _contract_for(request: IngestRequest) -> dict:
    """Translate approved source manifests into the v1.2.0 contract shape."""
    return {
        "schema": "quantsmind.python_data_contract/1",
        "version": "authorized_v16",
        "dataset_class": "production",
        "policy": {
            "default_decision": "deny",
            "allowed_spdx": sorted(
                {e.manifest.get("declared_license") for e in request.sources}
            ),
            "require_pinned_revision": True,
            "require_license_evidence": True,
        },
        "limits": request.config.get(
            "limits",
            {
                "max_files": 1000,
                "max_total_bytes": 50_000_000,
                "max_file_bytes": 200_000,
                "max_path_depth": 40,
            },
        ),
        "encoding": request.config.get(
            "encoding", {"primary": "utf-8", "fallbacks": []}
        ),
        "normalization": request.config.get(
            "normalization", {"newline": "\n", "strip_bom": True}
        ),
        "quality": request.config.get("quality", {}),
        "dedup": request.config.get("dedup", {}),
        "splits": request.config.get(
            "splits",
            {
                "proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 120,
                "min_groups_per_split": 1,
                "min_records_per_split": 1,
            },
        ),
        "benchmarks": request.config.get("benchmarks", {}),
        "tokenizer": request.config.get("tokenizer", {}),
        "sources": [
            {
                "id": entry.source_id,
                "kind": "local_directory",
                "location": str(entry.local_path),
                "revision": entry.manifest.get("revision"),
                "project": entry.manifest.get("project", entry.source_id),
                "attribution": entry.manifest.get("attribution", ""),
                "allowed_extensions": [".py"],
                "excluded_paths": [],
                "license": {
                    "spdx": entry.manifest.get("declared_license"),
                    "decision": "approved",
                    "evidence": (entry.manifest.get("license_evidence") or {}).get(
                        "path", "LICENSE"
                    ),
                    "evidence_sha256": (entry.manifest.get("license_evidence") or {}).get(
                        "evidence_hash", ""
                    ),
                },
                "approval": {
                    "status": "approved",
                    "reviewer": entry.manifest.get("reviewer"),
                    "rationale": entry.manifest.get("decision_rationale", ""),
                },
            }
            for entry in request.sources
        ],
    }


def build_authorized_corpus(request: IngestRequest) -> dict:
    """Build a corpus from approved sources, or refuse.

    The order matters: authorization and revision verification happen before
    a single file is read, and the build manifest records the authorization
    references that made the build possible.
    """
    import json
    import platform
    import sys

    import yaml

    from data.python_v1.pipeline import build_dataset

    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = authorize_entries(request.sources, intended_use=request.intended_use)
    observed = {}
    for entry in request.sources:
        observed[entry.source_id] = verify_revision(entry)

    contract_path = out_dir / "authorized_contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(_contract_for(request), sort_keys=True), encoding="utf-8"
    )
    manifest = build_dataset(contract_path, out_dir / "corpus")

    build_manifest = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pipeline_version": request.pipeline_version,
        "source_revisions": observed,
        "authorization_references": [
            {
                "source_id": decision.source_id,
                "state": decision.state,
                "allowed": decision.allowed,
            }
            for decision in decisions
        ],
        "configuration_digest": manifest["pipeline"]["contract_fingerprint"],
        "input_hashes": {
            artifact["name"]: artifact["sha256"]
            for artifact in manifest.get("artifacts", [])
        },
        "accepted_records": manifest["filtering"]["kept"],
        "rejected_records": manifest["filtering"]["rejected_total"],
        "rejection_reasons": manifest["filtering"]["rejected"],
        "deduplication": manifest["deduplication"],
        "contamination": {
            "verdict": manifest["contamination"]["verdict"],
            "hit_count": manifest["contamination"]["hit_count"],
            "not_checked": manifest["contamination"]["coverage"].get("not_checked", []),
        },
        "split_counts": manifest["splits"]["counts"],
        "split_ready": manifest["splits"]["ready"],
        "output_hashes": {
            artifact["name"]: artifact["sha256"]
            for artifact in manifest.get("artifacts", [])
        },
        "environment": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "platform": platform.platform(),
        },
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return build_manifest
