"""Deterministic dataset manifest, artifact checksums and data card (v1.2.0).

The manifest is the reproducibility record of a build. It carries the source
identities and revisions, the processing configuration hash, the filtering and
deduplication statistics, the split configuration, the contamination coverage,
the tokenizer identity, the final counts and the checksums of the artifacts.

Two properties are deliberate:

* **No timestamps and no absolute paths.** Two builds of the same inputs
  produce byte-identical manifests and byte-identical shards. Anything needed
  for an audit but changing per run belongs in a separate audit record, not in
  the content-addressed identity.
* **The manifest describes reality.** The checksums are computed from the
  files that were written, and the counts are read back from those files, so
  a manifest cannot claim a corpus the shards do not contain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.python_v1 import MANIFEST_SCHEMA, PIPELINE_VERSION
from data.python_v1.registry import canonical_json, sha256_text


def artifact_entry(path, name: str, records: int) -> dict:
    """Checksum and size of one written artifact."""
    data = Path(path).read_bytes()
    return {
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "records": records,
    }


def manifest_digest(manifest: dict) -> str:
    """Content hash of the manifest, excluding the hash field itself."""
    material = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return "sha256:" + sha256_text(canonical_json(material))


def build_manifest(
    *,
    contract: dict,
    contract_fingerprint: str,
    registry_fingerprint: str,
    code_revision: str,
    decisions: list,
    filtering: dict,
    dedup_stats: dict,
    splits: dict,
    contamination: dict,
    tokenizer: dict,
    artifacts: list,
    source_revisions: dict,
    notes: dict | None = None,
) -> dict:
    """Assemble the manifest. Deterministic: no timestamp, no absolute path."""
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "pipeline": {
            "version": PIPELINE_VERSION,
            "code_revision": code_revision,
            "contract_fingerprint": contract_fingerprint,
            "registry_fingerprint": registry_fingerprint,
        },
        "sources": {
            "decisions": decisions,
            "revisions": source_revisions,
            "approved": sorted(d["id"] for d in decisions if d.get("allowed")),
            "denied": sorted(d["id"] for d in decisions if not d.get("allowed")),
        },
        "configuration": {
            "normalization": contract.get("normalization", {}),
            "quality": contract.get("quality", {}),
            "dedup": contract.get("dedup", {}),
            "limits": contract.get("limits", {}),
            "encoding": contract.get("encoding", {}),
            "policy": contract.get("policy", {}),
        },
        "filtering": filtering,
        "deduplication": dedup_stats,
        "splits": {
            "proportions": splits.get("proportions"),
            "seed": splits.get("seed"),
            "counts": splits.get("counts"),
            "component_counts": splits.get("component_counts"),
            "ready": splits.get("split_ready"),
            "insufficient": splits.get("insufficient", []),
            "integrity": splits.get("integrity", {}),
            "assignment": {
                component_id: split
                for component_id, split in sorted(
                    splits.get("assignment", {}).items()
                )
            },
        },
        "contamination": {
            "verdict": contamination.get("verdict"),
            "hit_count": contamination.get("hit_count", 0),
            "coverage": contamination.get("coverage", {}),
            "method": contamination.get("method", {}),
            "limitations": contamination.get("limitations", []),
            "hits": contamination.get("hits", [])[:50],
        },
        "tokenizer": tokenizer,
        "artifacts": sorted(artifacts, key=lambda a: a["name"]),
        "notes": notes or {},
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_manifest(path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_data_card(manifest: dict) -> str:
    """Plain-language data card generated from the manifest."""
    splits = manifest.get("splits", {})
    contamination = manifest.get("contamination", {})
    filtering = manifest.get("filtering", {})
    dedup = manifest.get("deduplication", {})
    tokenizer = manifest.get("tokenizer", {})
    lines = [
        "# Data card: Python dataset foundation (v1.2.0)",
        "",
        f"- Pipeline: `{manifest['pipeline']['version']}`",
        f"- Manifest hash: `{manifest.get('manifest_sha256', '')}`",
        f"- Contract hash: `{manifest['pipeline']['contract_fingerprint']}`",
        "",
        "## Sources",
        "",
    ]
    for decision in manifest.get("sources", {}).get("decisions", []):
        state = "approved" if decision.get("allowed") else "denied"
        reasons = ", ".join(decision.get("reasons", [])) or "-"
        lines.append(
            f"- `{decision.get('id')}` ({state}): licence "
            f"{decision.get('license_spdx') or 'unknown'}, revision "
            f"`{decision.get('revision') or 'none'}`, reasons: {reasons}"
        )
    lines += [
        "",
        "## Composition",
        "",
        f"- Samples kept: {filtering.get('kept', 0)}",
        f"- Samples rejected: {filtering.get('rejected_total', 0)}",
        f"- Exact duplicates removed: {dedup.get('exact_duplicate_removed', 0)}",
        f"- Near-duplicate groups: {dedup.get('near_groups', 0)}",
        "",
        "## Splits",
        "",
        f"- Counts: {splits.get('counts')}",
        f"- Ready for evaluation: {splits.get('ready')}",
        "",
        "## Contamination",
        "",
        f"- Verdict: {contamination.get('verdict')}",
        f"- Checked suites: {contamination.get('coverage', {}).get('checked')}",
        f"- Not checked: {contamination.get('coverage', {}).get('not_checked')}",
        "",
        "## Tokenizer",
        "",
        f"- Path: `{tokenizer.get('path')}`",
        f"- SHA-256: `{tokenizer.get('sha256')}`",
        f"- Contract: `{tokenizer.get('contract_id')}`",
        f"- Vocabulary: {tokenizer.get('vocab_size')}",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in manifest.get("artifacts", []):
        lines.append(
            f"- `{artifact['name']}`: {artifact['records']} records, "
            f"{artifact['bytes']} bytes, sha256 `{artifact['sha256']}`"
        )
    lines += [
        "",
        "## Limits of this card",
        "",
        "- The contamination verdict covers only the benchmark suites listed "
        "above. It is lexical, so paraphrases and re-implementations are not "
        "detected.",
        "- Source approval and licence decisions are owner decisions recorded "
        "in the contract; this card does not make a legal determination.",
        "- The card describes one build. Reproducibility is claimed only for "
        "the same inputs, contract and tokenizer.",
        "",
    ]
    return "\n".join(lines)
