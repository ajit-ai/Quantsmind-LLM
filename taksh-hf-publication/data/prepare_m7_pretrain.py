"""Prepare M7 pretraining-token artifacts from licence-verified source files.

M7 reuses the M3 source set for pretraining (the same five verified
repositories), but not M3's validation stream: M7's development partition must
never select a checkpoint. This module therefore builds its own:

* ``pretrain.train`` -- every verified record from the five M7 training repos
* ``pretrain.val``   -- aiohttp records in the even file-hash bucket only
* ``pretrain.dev``   -- aiohttp records in the odd bucket are deliberately NOT
  encoded here; they are reserved for development evaluation

The split unit is the source file, not the example. Two functions from one file
share imports and helpers, so an example-level split would leak. The module
verifies that no repository and no content hash crosses train/val.

No external model, no synthetic text: the token stream is concatenated licensed
source with ``<eos>`` document boundaries.

Usage
-----
    python -m data.prepare_m7_pretrain --encode
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.prepare_m3_dataset import (
    LicenceGateError,
    sha256_file,
    utc_now_iso,
    verify_licences,
    write_split,
)

SCHEMA = "quantsmind.m7_pretrain/1"

TRAIN_SOURCES = (
    "pallets-click",
    "pallets-flask",
    "pytest-dev-pytest",
    "tiangolo-fastapi",
    "urllib3-urllib3",
)
VAL_SOURCE = "aio-libs-aiohttp"


def _file_bucket(file_path: str) -> str:
    digest = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    return "val" if int(digest, 16) % 2 == 0 else "dev"


def select(records):
    train = [r for r in records if r.get("source_id") in TRAIN_SOURCES]
    val = [r for r in records
           if r.get("source_id") == VAL_SOURCE
           and _file_bucket(r.get("file_path", "")) == "val"]
    dev = [r for r in records
           if r.get("source_id") == VAL_SOURCE
           and _file_bucket(r.get("file_path", "")) == "dev"]
    return train, val, dev


def check_isolation(train, val):
    train_repos = {r.get("repo") for r in train}
    val_repos = {r.get("repo") for r in val}
    train_hashes = {r.get("content_sha1") for r in train}
    val_hashes = {r.get("content_sha1") for r in val}
    train_files = {r.get("file_path") for r in train}
    val_files = {r.get("file_path") for r in val}
    return {
        "train_records": len(train),
        "val_records": len(val),
        "train_repositories": sorted(r for r in train_repos if r),
        "val_repositories": sorted(r for r in val_repos if r),
        "shared_repositories": sorted(train_repos & val_repos),
        "shared_content_sha1": sorted(train_hashes & val_hashes),
        "shared_file_paths": sorted(train_files & val_files),
        "ok": not (train_repos & val_repos) and not (train_hashes & val_hashes)
        and not (train_files & val_files),
    }


def prepare(corpus_path="data/code_corpus_v2.jsonl",
            licence_report="data/licence_verification.json",
            tokenizer="tokenizer/qm_code_bpe_v0.json",
            out_prefix="data/m7/pretrain", encode=True,
            dataset_id="m7-pretrain") -> dict:
    corpus_path = Path(corpus_path)
    records = [json.loads(line) for line in
               corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gate = verify_licences(records, licence_report)
    authorised = gate["authorised"]

    train, val, dev = select(authorised)
    if not train or not val:
        raise ValueError(
            f"empty M7 pretraining partition: train={len(train)} val={len(val)}")
    isolation = check_isolation(train, val)
    if not isolation["ok"]:
        raise ValueError(f"pretraining partitions are not isolated: {isolation}")

    for record in train:
        record["m7_partition"] = "train"
    for record in val:
        record["m7_partition"] = "val"

    splits = {}
    for name, subset in (("train", train), ("val", val)):
        splits[name] = write_split(
            subset, out_prefix, name, tokenizer=tokenizer, encode=encode)

    # Prove the M7 train content is the same authorised source set M3 trained on.
    m3_train = Path("data/m3_code_verified.train.jsonl")
    overlap = None
    if m3_train.exists():
        m3_hashes = set()
        for line in m3_train.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m3_hashes.add(json.loads(line).get("content_sha1"))
        m7_hashes = {r.get("content_sha1") for r in train}
        overlap = {
            "m3_train_records": len(m3_hashes),
            "m7_train_records": len(m7_hashes),
            "shared_content_sha1": len(m3_hashes & m7_hashes),
            "m7_only": len(m7_hashes - m3_hashes),
            "m3_only": len(m3_hashes - m7_hashes),
        }

    manifest = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "generated_at": utc_now_iso(),
        "source_corpus": str(corpus_path),
        "source_corpus_sha256": sha256_file(corpus_path),
        "licence_report": str(licence_report),
        "licence_report_sha256": sha256_file(licence_report),
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": sha256_file(tokenizer),
        "licence_gate": {
            "verified_sources": gate["verified_ids"],
            "withheld_sources": gate["withheld_sources"],
            "withheld_records": gate["withheld_records"],
            "policy": gate["policy"],
            "unreviewed_allowed": gate["unreviewed_allowed"],
        },
        "partitions": {
            "train": {
                "sources": sorted({r["source_id"] for r in train}),
                "records": len(train),
                "by_source": dict(Counter(r["source_id"] for r in train)),
            },
            "val": {
                "sources": sorted({r["source_id"] for r in val}),
                "records": len(val),
                "file_bucket": "even sha256(file_path)",
            },
            "dev": {
                "sources": sorted({r["source_id"] for r in dev}),
                "records": len(dev),
                "file_bucket": "odd sha256(file_path)",
                "encoded": False,
                "reason": ("Reserved for development evaluation; encoding it "
                           "here would make it available to validation."),
            },
        },
        "splits": splits,
        "isolation": isolation,
        "m3_overlap": overlap,
        "reproduce": (
            f"python -m data.prepare_m7_pretrain --corpus {corpus_path} "
            f"--licence {licence_report} --tokenizer {tokenizer} "
            f"--out-prefix {out_prefix} --encode"),
    }
    manifest_path = Path(f"{out_prefix}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build M7 pretraining token artifacts")
    parser.add_argument("--corpus", default="data/code_corpus_v2.jsonl")
    parser.add_argument("--licence", default="data/licence_verification.json")
    parser.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    parser.add_argument("--out-prefix", default="data/m7/pretrain")
    parser.add_argument("--dataset-id", default="m7-pretrain")
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.corpus, args.licence, args.tokenizer,
                           args.out_prefix, encode=not args.no_encode,
                           dataset_id=args.dataset_id)
    except (LicenceGateError, ValueError, FileNotFoundError) as exc:
        print(f"M7 pretraining data refused: {exc}")
        return 1
    print(f"M7 pretraining data: {manifest['dataset_id']}")
    for name, info in manifest["splits"].items():
        print(f"  {name:5}: {info['records']:3} records, "
              f"{info['token_count']} tokens, repos={info['repositories']}")
    print(f"  isolation: {'OK' if manifest['isolation']['ok'] else 'VIOLATED'}")
    if manifest["m3_overlap"]:
        print(f"  content overlap with M3 train: "
              f"{manifest['m3_overlap']['shared_content_sha1']} shared, "
              f"{manifest['m3_overlap']['m7_only']} M7-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
