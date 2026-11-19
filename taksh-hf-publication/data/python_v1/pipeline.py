"""One-command build of the Python data foundation (v1.2.0).

The pipeline is the composition of the stages, each of which is a module that
can be tested on its own:

    source registry -> licence gate -> local ingestion -> normalisation and
    quality filtering -> exact/near deduplication -> contamination screening ->
    source-group aware splitting -> manifest, checksums and data card

Every stage is deterministic and bounded. The pipeline never clones a
repository, never downloads a dataset and never trains a model. A source that
is not approved is denied and recorded; it is never quietly skipped.

Usage
-----
    python -m data.python_v1 --contract data/contracts/python_v1.yaml \\
        --out data/python_v1_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from data.python_v1 import RECORD_SCHEMA
from data.python_v1.benchmarks import load_benchmark_records, load_registry, screen_records
from data.python_v1.dedup import deduplicate
from data.python_v1.ingest import DEFAULT_PRUNE_DIRS, IngestError, ingest_source, read_source
from data.python_v1.manifest import (
    artifact_entry,
    build_manifest,
    render_data_card,
    write_manifest,
)
from data.python_v1.quality import assess_python, normalize_source
from data.python_v1.registry import (
    contract_fingerprint,
    load_contract,
    registry_fingerprint,
    resolve_source_root,
    snapshot_digest,
    source_decision,
)
from data.python_v1.splits import build_splits

# ``code_revision`` is a provenance helper from v1.1.0; it has no model
# dependency, so reusing it keeps one implementation of "which code produced
# this artifact".
from training.checkpoint_schema import code_revision

DEFAULT_CONTRACT = "data/contracts/python_v1.yaml"


def _read_bytes_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ingest_source_records(contract, contract_path, source, stats) -> list:
    """Ingest one approved source and return its accepted records."""
    root = resolve_source_root(contract_path, source)
    decision = source_decision(contract, source, root=root if root.is_dir() else None)
    if not root.is_dir():
        decision["allowed"] = False
        decision["reasons"] = list(decision["reasons"]) + ["source_root_missing"]
        stats["decisions"].append(decision)
        return []

    limits = contract.get("limits", {}) or {}
    report = ingest_source(
        source,
        root,
        limits=limits,
        excluded_paths=source.get("excluded_paths", []) or [],
        allowed_extensions=tuple(source.get("allowed_extensions") or (".py",)),
        prune_dirs=tuple(contract.get("prune_dirs", DEFAULT_PRUNE_DIRS)),
    )
    stats["ingest_diagnostics"].update(
        {k: stats["ingest_diagnostics"].get(k, 0) + v for k, v in report.diagnostics.items()}
    )
    if report.truncated:
        stats["truncated"] = True
        stats["truncated_reasons"][report.truncated_reason] = (
            stats["truncated_reasons"].get(report.truncated_reason, 0) + 1
        )

    # The pinned revision must describe the exact file set that was found.
    entries = []
    for ref in report.files:
        try:
            entries.append((ref.rel_path, _read_bytes_hash(ref.abs_path)))
        except OSError:
            continue
    digest = snapshot_digest(entries)
    if source.get("revision") and digest != source["revision"]:
        decision["allowed"] = False
        decision["reasons"] = list(decision["reasons"]) + ["revision_mismatch"]
        decision["observed_revision"] = digest
        stats["decisions"].append(decision)
        return []
    decision["observed_revision"] = digest
    stats["decisions"].append(decision)
    if not decision["allowed"]:
        return []

    encoding_cfg = contract.get("encoding", {}) or {}
    fallbacks = tuple(encoding_cfg.get("fallbacks", []) or ())
    quality_cfg = contract.get("quality", {}) or {}
    records = []
    for ref in report.files:
        try:
            text, encoding, used_fallback = read_source(
                ref, fallbacks=fallbacks, max_bytes=int(limits.get("max_file_bytes", 200_000))
            )
        except IngestError as exc:
            stats["rejected"][exc.reason] = stats["rejected"].get(exc.reason, 0) + 1
            stats["rejected_total"] += 1
            continue
        text, applied = normalize_source(text, contract.get("normalization", {}))
        verdict = assess_python(text, ref.rel_path, quality_cfg)
        if not verdict.ok:
            stats["rejected"][verdict.reason] = stats["rejected"].get(verdict.reason, 0) + 1
            stats["rejected_total"] += 1
            continue
        if used_fallback:
            stats["encoding_fallbacks"][encoding] = (
                stats["encoding_fallbacks"].get(encoding, 0) + 1
            )
        records.append(
            {
                "record_schema": RECORD_SCHEMA,
                "text": text,
                "source_id": source["id"],
                "source_revision": source.get("revision"),
                "project": source.get("project") or source["id"],
                "file_path": ref.rel_path,
                "license_spdx": (source.get("license") or {}).get("spdx"),
                "attribution": source.get("attribution", ""),
                "language": "python",
                "encoding": encoding,
                "normalization": list(applied),
            }
        )
    stats["seen"] += len(report.files)
    return records


def build_dataset(contract_path, out_dir, *, dry_run: bool = False) -> dict:
    """Run every stage and write the artifacts. Returns the manifest."""
    contract_path = Path(contract_path)
    contract = load_contract(contract_path)
    stats = {
        "decisions": [],
        "rejected": {},
        "rejected_total": 0,
        "ingest_diagnostics": {},
        "encoding_fallbacks": {},
        "seen": 0,
        "truncated": False,
        "truncated_reasons": {},
    }

    records: list = []
    for source in contract.get("sources", []):
        records.extend(_ingest_source_records(contract, contract_path, source, stats))

    # Contamination screening happens before splitting so a contaminated file
    # can never be assigned to a holdout.
    benchmark_cfg = contract.get("benchmarks", {}) or {}
    registry_path = benchmark_cfg.get("registry")
    contamination = {
        "verdict": "unavailable",
        "hits": [],
        "hit_count": 0,
        "coverage": {"suites": [], "checked": [], "not_checked": []},
        "method": {},
        "limitations": ["no benchmark registry configured"],
    }
    if registry_path:
        resolved = (contract_path.parent / registry_path).resolve()
        if not resolved.is_file():
            resolved = Path(registry_path)
        registry = load_registry(resolved)
        benchmarks = load_benchmark_records(registry, base_dir=resolved.parent)
        contamination = screen_records(records, benchmarks, benchmark_cfg)
    contaminated = {
        (hit.get("source_id"), hit.get("file_path")) for hit in contamination["hits"]
    }
    if contaminated:
        kept = []
        for record in records:
            if (record.get("source_id"), record.get("file_path")) in contaminated:
                stats["rejected"]["benchmark_contaminated"] = (
                    stats["rejected"].get("benchmark_contaminated", 0) + 1
                )
                stats["rejected_total"] += 1
                continue
            kept.append(record)
        records = kept

    # Duplicate detection and removal.
    dedup_config = contract.get("dedup", {}) or {}
    dedup_result = deduplicate(records, dedup_config)
    removed_indices = set()
    for group_id, representative in dedup_result.representatives.items():
        for index in dedup_result.groups[group_id]:
            if index != representative:
                removed_indices.add(index)
    if removed_indices:
        records = [
            record
            for index, record in enumerate(records)
            if index not in removed_indices
        ]
        # Recompute the component map over the survivors so the split map
        # matches the records that are actually written.
        dedup_result = deduplicate(records, dedup_config)

    # Splits.
    splits = build_splits(
        records, dedup_result.component_of, contract.get("splits", {}) or {}
    )
    split_of = splits["split_of_record"]
    for index, record in enumerate(records):
        record["split"] = split_of[index]
        record["component_id"] = dedup_result.component_of[index]
        record["content_sha256"] = hashlib.sha256(
            record["text"].encode("utf-8")
        ).hexdigest()

    out_dir = Path(out_dir)
    artifacts = []
    for split_name, file_name in (
        ("train", "train.jsonl"),
        ("val", "val.jsonl"),
        ("test", "test.jsonl"),
    ):
        indexes = splits["per_split"].get(split_name, [])
        path = out_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in indexes:
                record = dict(records[index])
                handle.write(
                    json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                )
        artifacts.append(artifact_entry(path, file_name, len(indexes)))

    filtering = {
        "seen": stats["seen"],
        "kept": len(records),
        "rejected_total": stats["rejected_total"],
        "rejected": dict(sorted(stats["rejected"].items())),
        "ingest_diagnostics": dict(sorted(stats["ingest_diagnostics"].items())),
        "encoding_fallbacks": dict(sorted(stats["encoding_fallbacks"].items())),
        "truncated": stats["truncated"],
        "truncated_reasons": dict(sorted(stats["truncated_reasons"].items())),
    }
    # ``dataset_class`` is declared by the contract, never inferred. A
    # ``synthetic`` build is labelled as such everywhere it is consumed, and
    # the training authorization gate refuses to promote one to production.
    dataset_class = contract.get("dataset_class", "production")
    corpus_ready = bool(
        dataset_class == "production"
        and stats["decisions"]
        and all(d.get("allowed") for d in stats["decisions"])
        and not stats["truncated"]
        and splits.get("split_ready")
    )
    manifest = build_manifest(
        contract=contract,
        contract_fingerprint=contract_fingerprint(contract),
        registry_fingerprint=registry_fingerprint(contract),
        code_revision=code_revision(),
        decisions=stats["decisions"],
        filtering=filtering,
        dedup_stats=dedup_result.stats,
        splits=splits,
        contamination=contamination,
        tokenizer=_tokenizer_identity(contract, contract_path.parent),
        artifacts=artifacts,
        source_revisions={
            d["id"]: d.get("observed_revision") or d.get("revision")
            for d in stats["decisions"]
        },
        notes={
            "corpus_class": dataset_class,
            "synthetic": dataset_class == "synthetic",
            "corpus_ready": corpus_ready,
            "corpus_readiness_note": (
                "A build is corpus_ready only when every declared source is "
                "approved, nothing was truncated and the split is ready. A "
                "synthetic build is never corpus_ready: it is a qualification "
                "fixture, not a corpus."
            ),
        },
    )
    if not dry_run:
        write_manifest(out_dir / "manifest.json", manifest)
        (out_dir / "DATA_CARD.md").write_text(
            render_data_card(manifest), encoding="utf-8"
        )
    return manifest


def _tokenizer_identity(contract: dict, base_dir=None) -> dict:
    """Record the tokenizer the corpus is prepared for."""
    spec = contract.get("tokenizer") or {}
    path = spec.get("path")
    identity = {
        "path": path,
        "sha256": None,
        "vocab_size": None,
        "contract_id": spec.get("contract_id"),
    }
    if not path:
        return identity
    candidate = Path(path)
    if not candidate.is_file() and base_dir is not None:
        candidate = Path(base_dir) / path
    if not candidate.is_file():
        identity["error"] = "tokenizer file not found"
        return identity
    identity["sha256"] = _read_bytes_hash(candidate)
    try:
        from tokenizers import Tokenizer

        from training.tokenizer_contract import detect_contract

        tokenizer = Tokenizer.from_file(str(candidate))
        identity["vocab_size"] = tokenizer.get_vocab_size()
        identity["contract_id"] = detect_contract(tokenizer)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        identity["error"] = f"{type(exc).__name__}: {exc}"
    return identity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the governed Python dataset from approved local sources"
    )
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--out", default="data/python_v1_out")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every stage and report, but write no artifacts",
    )
    args = parser.parse_args(argv)

    manifest = build_dataset(args.contract, args.out, dry_run=args.dry_run)
    splits = manifest.get("splits", {})
    filtering = manifest.get("filtering", {})
    print(f"sources approved : {manifest['sources']['approved']}")
    print(f"sources denied   : {manifest['sources']['denied']}")
    print(f"samples kept     : {filtering.get('kept')}")
    print(f"samples rejected : {filtering.get('rejected_total')}")
    print(f"splits           : {splits.get('counts')} ready={splits.get('ready')}")
    print(
        "contamination    : "
        f"{manifest.get('contamination', {}).get('verdict')}"
    )
    print(f"manifest hash    : {manifest.get('manifest_sha256')}")
    if not splits.get("ready"):
        print(
            "NOTE: the split is reported as not ready: "
            + "; ".join(splits.get("insufficient", []))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
