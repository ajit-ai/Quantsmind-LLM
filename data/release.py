"""Dataset release artifacts (M2.8).

A corpus is not "released" because a JSONL file exists. A release is a named,
immutable, verifiable set: the data, its provenance manifest, its licence
evidence, checksums that detect drift, and a record saying what the release
contains and what it does not.

This module builds and verifies that set. It deliberately does **not** publish,
upload, or tag anything -- those are separate, explicitly authorised operations.

Release layout (written next to the data by default)::

    <release_dir>/
        RELEASE.json            # the record: identity, contents, checksums
        CHECKSUMS.sha256        # verify with sha256sum -c
        <shard>.jsonl            # the data itself
        <shard>.manifest.json    # provenance, if the corpus builder produced one

Verify later with :func:`verify_release`, which re-hashes every file and
re-checks the recorded invariants. Nothing is trusted because it was written
down; it is trusted because it still hashes correctly.

Usage
-----
    python -m data.release build --corpus data/code_corpus_v2.jsonl \
        --manifest data/code_corpus_v2.manifest.json \
        --out data/releases/code_corpus-v2 --dataset-id code_corpus-v2

    python -m data.release verify --release data/releases/code_corpus-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.dataset_release/1"
CHECKSUM_FILE = "CHECKSUMS.sha256"
RECORD_FILE = "RELEASE.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_records(path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _summarise_manifest(manifest_path):
    """Extract the parts of a corpus manifest a release must carry forward."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        "corpus": data.get("corpus", {}),
        "status": data.get("status"),
        "complete": data.get("complete"),
        "truncated": data.get("truncated"),
        "counts": data.get("counts", {}),
        "splits": data.get("splits", {}),
        "licensing": {
            "policy_version": data.get("licensing", {}).get("policy_version"),
            "by_spdx": data.get("licensing", {}).get("by_spdx", {}),
            "per_source": data.get("licensing", {}).get("per_source", []),
        },
        "leakage_check": {
            k: v for k, v in data.get("leakage_check", {}).items()
            if k != "limitations"
        },
        "integrity_failures": data.get("integrity_failures", []),
        "content_hashes_count": len(data.get("content_hashes", [])),
    }


def build_release(corpus_path, out_dir, dataset_id, manifest_path=None,
                  notes=None, copy=True, now=None, extra_paths=None):
    """Assemble a release set and return the release record.

    ``copy=False`` records checksums against existing files without copying,
    which is useful for verifying an already-materialised corpus in place.
    ``extra_paths`` releases additional artifacts that belong to the same
    training input -- typically the ``.tokens.bin`` and its ``.meta.json``.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / corpus_path.name
    if copy and corpus_path.resolve() != target.resolve():
        target.write_bytes(corpus_path.read_bytes())

    manifest_summary = None
    if manifest_path and Path(manifest_path).exists():
        manifest_summary = _summarise_manifest(manifest_path)
        m_target = out_dir / Path(manifest_path).name
        if copy and Path(manifest_path).resolve() != m_target.resolve():
            m_target.write_bytes(Path(manifest_path).read_bytes())

    released_extras = []
    for extra in extra_paths or []:
        extra = Path(extra)
        if not extra.exists():
            raise FileNotFoundError(f"release input not found: {extra}")
        e_target = out_dir / extra.name
        if copy and extra.resolve() != e_target.resolve():
            e_target.write_bytes(extra.read_bytes())
        released_extras.append(extra.name)

    # Link the token artifact back to the shard it was encoded from. A token
    # artifact whose recorded corpus hash disagrees with the released shard is
    # not part of this release, however well-formed it looks.
    chain = {"checked": False}
    token_meta_name = next(
        (n for n in released_extras if n.endswith(".meta.json")), None
    )
    if token_meta_name:
        token_meta = json.loads(
            (out_dir / token_meta_name).read_text(encoding="utf-8")
        )
        shard_path = out_dir / corpus_path.name
        shard_sha = sha256_file(shard_path) if shard_path.exists() else None
        chain = {
            "checked": True,
            "token_meta": token_meta_name,
            "recorded_corpus": token_meta.get("corpus"),
            "recorded_corpus_sha256": token_meta.get("corpus_sha256"),
            "released_shard_sha256": shard_sha,
            "matches": shard_sha is not None
            and token_meta.get("corpus_sha256") == shard_sha,
            "num_tokens": token_meta.get("num_tokens"),
            "num_documents": token_meta.get("num_documents"),
            "dataset_schema_validated": token_meta.get("dataset_schema_validated"),
            "vocab_size": token_meta.get("vocab_size"),
        }

    # Checksum every released file, excluding the record and checksum file which
    # are written afterwards.
    files = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name not in (RECORD_FILE, CHECKSUM_FILE):
            files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

    record = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "version": manifest_summary["corpus"].get("version") if manifest_summary else None,
        "created_at": now or utc_now_iso(),
        "released": False,
        "release_note": (
            "This record marks a local, verified dataset set. It is NOT a "
            "publication: nothing has been uploaded, tagged or released. "
            "Publication is a separate, explicitly authorised operation."
        ),
        "contents": {
            "shard": corpus_path.name,
            "records": _count_records(target),
            "manifest": Path(manifest_path).name if manifest_summary else None,
        },
        "files": files,
        "provenance": manifest_summary,
        "token_chain": chain,
        "notes": list(notes or []),
        "verification": {
            "command": f"python -m data.release verify --release {out_dir}",
            "method": "sha256 over every released file, re-read on verify",
        },
    }

    (out_dir / RECORD_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    lines = [f"{meta['sha256']}  {name}" for name, meta in sorted(files.items())]
    (out_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def verify_release(release_dir, deep=True):
    """Re-hash a release and re-check its recorded invariants.

    Returns a report dict. ``ok`` is False for *any* problem, so a caller never
    has to decide which findings are fatal.
    """
    release_dir = Path(release_dir)
    record_path = release_dir / RECORD_FILE
    report = {
        "schema": SCHEMA,
        "release_dir": str(release_dir),
        "exists": record_path.exists(),
        "ok": False,
        "checks": {},
        "problems": [],
    }
    if not record_path.exists():
        report["problems"].append(f"missing release record: {record_path}")
        return report

    record = json.loads(record_path.read_text(encoding="utf-8"))
    report["dataset_id"] = record.get("dataset_id")
    report["checks"]["schema_ok"] = record.get("schema") == SCHEMA

    missing, mismatched = [], []
    for name, meta in sorted(record.get("files", {}).items()):
        path = release_dir / name
        if not path.exists():
            missing.append(name)
            continue
        if deep and sha256_file(path) != meta.get("sha256"):
            mismatched.append(
                {"file": name, "expected": meta.get("sha256"),
                 "actual": sha256_file(path)}
            )
    report["checks"]["files_present"] = not missing
    report["checks"]["checksums_match"] = not mismatched
    if missing:
        report["problems"].append(f"missing released files: {missing}")
    for item in mismatched:
        report["problems"].append(
            f"checksum mismatch for {item['file']}: expected "
            f"{item['expected'][:12]}..., got {item['actual'][:12]}..."
        )

    # CHECKSUMS.sha256 must agree with the release record.
    checksum_file = release_dir / CHECKSUM_FILE
    if checksum_file.exists():
        listed = {}
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, name = line.partition("  ")
                listed[name.strip()] = digest
        agree = listed == {
            name: meta["sha256"] for name, meta in record.get("files", {}).items()
        }
        report["checks"]["checksum_file_agrees"] = agree
        if not agree:
            report["problems"].append(f"{CHECKSUM_FILE} disagrees with {RECORD_FILE}")
    else:
        report["checks"]["checksum_file_agrees"] = False
        report["problems"].append(f"missing {CHECKSUM_FILE}")

    # A released shard must actually hold the records the record claims.
    shard = release_dir / record.get("contents", {}).get("shard", "")
    if shard.name and shard.exists():
        actual_records = _count_records(shard)
        claimed = record["contents"].get("records")
        report["checks"]["record_count_matches"] = actual_records == claimed
        if actual_records != claimed:
            report["problems"].append(
                f"{shard.name}: record claims {claimed} records, found {actual_records}"
            )
    else:
        report["checks"]["record_count_matches"] = False
        report["problems"].append(f"missing released shard: {shard}")

    chain = record.get("token_chain") or {}
    if chain.get("checked"):
        report["checks"]["token_chain_matches_shard"] = bool(chain.get("matches"))
        report["checks"]["token_artifact_schema_validated"] = bool(
            chain.get("dataset_schema_validated")
        )
        if not chain.get("matches"):
            report["problems"].append(
                "token artifact was encoded from a different corpus than the "
                "released shard (recorded corpus_sha256 does not match)"
            )
        if not chain.get("dataset_schema_validated"):
            report["problems"].append(
                "token artifact was encoded with dataset schema validation "
                "disabled"
            )

    provenance = record.get("provenance") or {}
    if provenance:
        status = provenance.get("status")
        report["checks"]["corpus_status_complete"] = status == "complete"
        if status != "complete":
            report["problems"].append(
                f"corpus manifest status is {status!r}, not 'complete'"
            )
        failures = provenance.get("integrity_failures") or []
        report["checks"]["no_integrity_failures"] = not failures
        if failures:
            report["problems"].append(
                f"manifest records integrity failures: {failures}"
            )

    report["ok"] = not report["problems"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify a dataset release")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="assemble a release set")
    build.add_argument("--corpus", required=True)
    build.add_argument("--manifest", default=None)
    build.add_argument("--out", required=True)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--note", action="append", default=None)
    build.add_argument(
        "--extra", action="append", default=None,
        help="additional file belonging to the release (e.g. the .tokens.bin)",
    )

    verify = sub.add_parser("verify", help="verify an existing release")
    verify.add_argument("--release", required=True)

    args = parser.parse_args(argv)

    if args.command == "build":
        record = build_release(
            args.corpus, args.out, args.dataset_id,
            manifest_path=args.manifest, notes=args.note,
            extra_paths=args.extra,
        )
        print(f"release: {record['dataset_id']} -> {args.out}")
        print(f"  records : {record['contents']['records']}")
        print(f"  files   : {len(record['files'])}")
        for name, meta in sorted(record["files"].items()):
            print(f"    {name}  {meta['sha256'][:16]}...  {meta['bytes']} bytes")
        print(f"  released: {record['released']} (local verified set only)")
        print(f"  verify  : {record['verification']['command']}")
        return 0

    report = verify_release(args.release)
    print(f"release verify: {report['release_dir']}")
    print(f"  dataset_id: {report.get('dataset_id')}")
    for name, value in sorted(report["checks"].items()):
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    for problem in report["problems"]:
        print(f"  problem: {problem}")
    print(f"  result: {'OK' if report['ok'] else 'NOT OK'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())