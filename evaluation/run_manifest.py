"""Evaluation run records (M2.10: reproducibility requirements).

An evaluation number without the inputs that produced it is not reproducible
and is not reviewable. This module defines what an evaluation run must record,
and refuses to certify a run that is missing any of it.

Required for every evaluation run:

* the **checkpoint** evaluated, with its sha256;
* the **seed**, so sampling and shuffling can be reproduced;
* the **dataset release** and **holdout** that were measured on, with checksums;
* the **tokenizer** and the **config**, with checksums;
* the **task set** identifier and its version.

A run missing any of these is reported ``certified: False`` with the specific
gaps named. There is no "mostly reproducible".

Usage
-----
    python -m evaluation.run_manifest build \
        --checkpoint runs/qm-code-v2.pt --seed 42 \
        --release data/releases/code_corpus-v2 \
        --holdout evaluation/code_holdout_v2.jsonl \
        --tokenizer tokenizer/qm_code_bpe_v0.json \
        --config configs/train/pretrain_code_v2.yaml \
        --out evaluation/run_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.eval_run/1"
REQUIRED = (
    "checkpoint", "checkpoint_sha256", "seed", "release", "holdout",
    "holdout_sha256", "tokenizer", "tokenizer_sha256", "config",
    "config_sha256", "task_set", "task_set_version",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path, label):
    path = Path(path)
    return {"label": label, "path": str(path), "exists": path.exists(),
            "sha256": sha256_file(path)}


def build_run_record(run_id, checkpoint=None, seed=None, release=None,
                     holdout=None, tokenizer=None, config=None,
                     task_set=None, task_set_version=None, environment=None,
                     notes=None, now=None):
    """Assemble a run record and check it for reproducibility gaps."""
    files = {
        "checkpoint": describe(checkpoint, "model checkpoint") if checkpoint else None,
        "tokenizer": describe(tokenizer, "tokenizer") if tokenizer else None,
        "config": describe(config, "training/eval config") if config else None,
    }
    holdout_info = None
    if holdout:
        holdout_info = describe(holdout, "held-out evaluation set")
    release_info = None
    if release:
        record_path = Path(release) / "RELEASE.json"
        release_info = {
            "label": "dataset release",
            "path": str(release),
            "record": str(record_path),
            "exists": record_path.exists(),
            "sha256": sha256_file(record_path),
        }
        if release_info["exists"]:
            try:
                data = json.loads(record_path.read_text(encoding="utf-8"))
                release_info["dataset_id"] = data.get("dataset_id")
                files_recorded = data.get("files", {})
                release_info["files"] = {
                    name: meta.get("sha256") for name, meta in files_recorded.items()
                }
            except json.JSONDecodeError:
                release_info["error"] = "RELEASE.json is not valid JSON"
                release_info["files"] = {}

    record = {
        "schema": SCHEMA,
        "run_id": run_id,
        "created_at": now or utc_now_iso(),
        "files": files,
        "release": release_info,
        "holdout": holdout_info,
        "seed": seed,
        "task_set": task_set,
        "task_set_version": task_set_version,
        "environment": environment or {},
        "notes": list(notes or []),
    }

    gaps = []
    if seed is None:
        gaps.append("seed is not recorded")
    if not task_set:
        gaps.append("task_set is not recorded")
    if task_set_version is None:
        gaps.append("task_set_version is not recorded")
    for label, info in files.items():
        if info is None:
            gaps.append(f"{label} is not recorded")
        elif not info["exists"]:
            gaps.append(f"{label} file does not exist: {info['path']}")
        elif info["sha256"] is None:
            gaps.append(f"{label} sha256 could not be computed")
    if holdout_info is None:
        gaps.append("holdout is not recorded")
    elif not holdout_info["exists"]:
        gaps.append(f"holdout file does not exist: {holdout_info['path']}")
    if release_info is None:
        gaps.append("release is not recorded")
    elif not release_info["exists"]:
        gaps.append(f"release record does not exist: {release_info['record']}")
    elif release_info.get("error"):
        gaps.append(f"release record is unusable: {release_info['error']}")
    elif not release_info.get("files"):
        gaps.append("release record lists no files")

    # The holdout must be the one the release was built from. If the holdout's
    # recorded corpus is not among the released files, the two have drifted.
    if holdout_info and holdout_info["exists"]:
        manifest = Path(holdout).with_suffix(".manifest.json")
        if manifest.exists():
            try:
                corpus_sha = json.loads(
                    manifest.read_text(encoding="utf-8")
                ).get("corpus_sha256")
            except json.JSONDecodeError:
                corpus_sha = None
                gaps.append("holdout manifest is not valid JSON")
            if corpus_sha and release_info and release_info.get("files"):
                if corpus_sha not in set(release_info["files"].values()):
                    gaps.append(
                        "holdout was built from a corpus that is not in the "
                        "recorded release"
                    )

    record["gaps"] = gaps
    record["certified"] = not gaps
    record["certification"] = (
        "Run is reproducible: every required input is recorded with a checksum."
        if not gaps else
        "Run is NOT reproducible: " + "; ".join(gaps)
    )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build an evaluation run record")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--release")
    parser.add_argument("--holdout")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config")
    parser.add_argument("--task-set")
    parser.add_argument("--task-set-version")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    record = build_run_record(
        args.run_id, checkpoint=args.checkpoint, seed=args.seed,
        release=args.release, holdout=args.holdout, tokenizer=args.tokenizer,
        config=args.config, task_set=args.task_set,
        task_set_version=args.task_set_version,
    )
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"eval run record: {args.run_id} -> {args.out}")
    print(f"  certified : {record['certified']}")
    for gap in record["gaps"]:
        print(f"  gap: {gap}")
    if record["certified"]:
        print("  every required input is recorded with a checksum")
    return 0 if record["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())