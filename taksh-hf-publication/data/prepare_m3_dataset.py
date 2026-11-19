"""M3 training-data preparation: licence-gated, split-separated.

M2 produced one corpus with a source-group train/val/test split. M2's own
readiness gate reports the corpus as **BLOCKED**, and the reason matters here:
two sources in the corpus -- ``psf-black`` and ``pypa-setuptools`` -- have
licences that a human must review before their code is used as training data.

Measured composition of ``data/code_corpus_v2.jsonl``:

===============  =======  ==========  =========
split            records  verified    unresolved
===============  =======  ==========  =========
train            193      193         0
val              198      63          135 (pypa-setuptools)
test             46       19          27  (psf-black)
===============  =======  ==========  =========

So the *training* split is already licence-clean, while the validation and
holdout splits are majority unresolved. Training the whole corpus would put
unreviewed licence material into optimizer updates, which this module refuses to
do.

What it produces
----------------

* ``<name>.train.jsonl`` / ``<name>.train.tokens.bin`` -- licence-verified records
  from the train split only.
* ``<name>.val.jsonl`` / ``<name>.val.tokens.bin`` -- licence-verified records
  from the val split, used only for evaluation, never for gradients.
* ``<name>.manifest.json`` -- provenance, hashes, licence evidence, and the
  separation proof.

The **holdout (test) split is deliberately not emitted**. Tuning against it would
destroy its only purpose, and its licence composition is unresolved anyway.

This module never mutates the M2 corpus or token artifacts.

Usage
-----
    python -m data.prepare_m3_dataset \
        --corpus data/code_corpus_v2.jsonl \
        --licence data/licence_verification.json \
        --tokenizer tokenizer/qm_code_bpe_v0.json \
        --out-prefix data/m3_code_verified \
        --encode
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.m3_dataset/1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if stripped:
            yield number, stripped


class LicenceGateError(RuntimeError):
    """Training data would contain material whose licence is unresolved."""


def verify_licences(records, licence_report_path, allow_unreviewed=False):
    """Split records into (authorised, withheld) by licence evidence.

    ``authorised`` is decided by ``data/licence_verification.json``: a source
    counts only when it appears in ``verified_ids``. Nothing is inferred from a
    record's own ``spdx`` field, because a record asserting MIT is not evidence
    that MIT was reviewed and accepted -- the verification report is.
    """
    path = Path(licence_report_path)
    if not path.exists():
        raise LicenceGateError(
            f"licence verification report not found: {path}\n"
            "Training data cannot be authorised without it. Run "
            "`python -m data.verify_licenses` first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    verified = set(report.get("verified_ids") or [])
    needs_review = set(report.get("needs_human_review_ids") or [])

    if not verified:
        raise LicenceGateError(
            f"{path} lists no verified sources; no record can be authorised"
        )

    authorised, withheld = [], []
    for record in records:
        source_id = record.get("source_id")
        if source_id in verified:
            authorised.append(record)
        elif source_id in needs_review or True:
            withheld.append(record)

    withheld_by_source = Counter(
        r.get("source_id") for r in withheld
    )
    return {
        "authorised": authorised,
        "withheld": withheld,
        "report": report,
        "verified_ids": sorted(verified),
        "withheld_sources": dict(withheld_by_source),
        "withheld_records": len(withheld),
        "policy": (
            "A source must appear in licence_verification.json:verified_ids. "
            "Records from any other source are withheld from training and "
            "validation data."
        ),
        "unreviewed_allowed": bool(allow_unreviewed),
    }


def check_split_isolation(records, train_split="train", val_split="val"):
    """Prove the train and val sets share no repository and no content hash."""
    train = [r for r in records if r.get("split") == train_split]
    val = [r for r in records if r.get("split") == val_split]

    train_repos = {r.get("repo") for r in train}
    val_repos = {r.get("repo") for r in val}
    train_hashes = {r.get("content_sha1") for r in train}
    val_hashes = {r.get("content_sha1") for r in val}

    shared_repos = sorted(train_repos & val_repos)
    shared_hashes = sorted(train_hashes & val_hashes)

    return {
        "train_records": len(train),
        "val_records": len(val),
        "train_repositories": sorted(r for r in train_repos if r),
        "val_repositories": sorted(r for r in val_repos if r),
        "shared_repositories": shared_repos,
        "shared_content_sha1": shared_hashes,
        "grouping": "source repository (source_group)",
        "ok": not shared_repos and not shared_hashes,
        "problem": (
            (f"repositories shared between train and val: {shared_repos}")
            if shared_repos else
            (f"content hashes shared between train and val: {shared_hashes}")
            if shared_hashes else ""
        ),
    }


def write_split(records, out_prefix, split, tokenizer=None, encode=False,
                max_tokens=None):
    """Write one split's JSONL and, optionally, its token artifact.

    Split names are appended with plain string concatenation, not
    ``Path.with_suffix``. ``Path("x.train").with_suffix(".jsonl")`` yields
    ``x.jsonl`` -- it replaces ``.train`` instead of adding to it -- which
    silently gave both splits the same filename and let the val split overwrite
    the train split.
    """
    stem = f"{out_prefix}.{split}"
    jsonl_path = Path(f"{stem}.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    result = {
        "split": split,
        "records": len(records),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "repositories": sorted({
            r.get("repo") for r in records if r.get("repo")
        }),
        "licenses": dict(Counter(r.get("spdx") for r in records)),
        "token_bytes": None,
        "token_count": None,
        "token_meta": None,
        "token_chain_matches_jsonl": None,
    }
    if not encode:
        return result

    from data.encode_dataset import iter_records, encode_records
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    token_path = Path(f"{stem}.tokens.bin")
    ids, emitted = encode_records(tok, iter_records(jsonl_path), max_tokens=max_tokens)
    if not ids:
        raise ValueError(f"{jsonl_path} produced no tokens; refusing to continue")
    import numpy as np

    max_id = max(ids)
    dtype = np.uint16 if max_id < 65536 else np.uint32
    np.asarray(ids, dtype=dtype).tofile(token_path)

    meta = {
        "schema": "quantsmind.token_stream/1",
        "generated_at": utc_now_iso(),
        "vocab_size": tok.get_vocab_size(),
        "num_tokens": len(ids),
        "num_documents": emitted,
        "dtype": np.dtype(dtype).name,
        "corpus": str(jsonl_path),
        "corpus_sha256": result["jsonl_sha256"],
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": sha256_file(tokenizer),
        "document_separator": "<eos>",
        "dataset_schema_validated": True,
        "purpose": f"M3 {split} split; derived from the licence-gated M3 dataset",
    }
    meta_path = Path(f"{stem}.tokens.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result.update({
        "token_bytes": token_path.stat().st_size,
        "token_count": len(ids),
        "token_meta": str(meta_path),
        "token_chain_matches_jsonl": meta["corpus_sha256"] == result["jsonl_sha256"],
    })
    return result


def prepare(corpus_path, licence_report, tokenizer, out_prefix,
            train_split="train", val_split="val", encode=True,
            allow_unreviewed=False, dataset_id="m3-code-verified"):
    """Build the M3 training and validation sets from the M2 corpus.

    Raises :class:`LicenceGateError` if any record that would enter training has
    no verified licence, and ``ValueError`` if train and val are not isolated.
    """
    corpus_path = Path(corpus_path)
    records = [json.loads(raw) for _, raw in read_jsonl(corpus_path)]

    gate = verify_licences(records, licence_report, allow_unreviewed)
    authorised = gate["authorised"]

    train_records = [r for r in authorised if r.get("split") == train_split]
    val_records = [r for r in authorised if r.get("split") == val_split]
    if not train_records:
        raise ValueError(
            f"no licence-authorised records in the {train_split!r} split; "
            "refusing to build an empty training set"
        )
    if not val_records:
        raise ValueError(
            f"no licence-authorised records in the {val_split!r} split; a "
            "validation set is required to produce honest metrics"
        )

    isolation = check_split_isolation(authorised, train_split, val_split)
    if not isolation["ok"]:
        raise ValueError(
            f"train and val are not isolated: {isolation['problem']}\n"
            "Refusing to build a dataset whose validation data can leak into "
            "training."
        )

    splits = {}
    for split, subset in ((train_split, train_records), (val_split, val_records)):
        splits[split] = write_split(
            subset, out_prefix, split, tokenizer=tokenizer, encode=encode
        )

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
        "splits": splits,
        "isolation": isolation,
        "holdout": {
            "split": "test",
            "emitted": False,
            "reason": (
                "The test split is withheld from M3 on purpose. Its licence "
                "composition is unresolved (psf-black), and tuning against it "
                "would spend the only data reserved for final measurement."
            ),
        },
        "reproduce": (
            f"python -m data.prepare_m3_dataset --corpus {corpus_path} "
            f"--licence {licence_report} --tokenizer {tokenizer} "
            f"--out-prefix {out_prefix} --encode"
        ),
    }
    manifest_path = Path(f"{out_prefix}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build licence-gated, split-separated M3 training data"
    )
    parser.add_argument("--corpus", default="data/code_corpus_v2.jsonl")
    parser.add_argument("--licence", default="data/licence_verification.json")
    parser.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    parser.add_argument("--out-prefix", default="data/m3_code_verified")
    parser.add_argument("--dataset-id", default="m3-code-verified")
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument(
        "--allow-unreviewed-licences", action="store_true",
        help="DANGEROUS: include sources whose licence needs human review. "
             "Requires an explicit human decision recorded outside this tool.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = prepare(
            args.corpus, args.licence, args.tokenizer, args.out_prefix,
            encode=not args.no_encode, dataset_id=args.dataset_id,
            allow_unreviewed=args.allow_unreviewed_licences,
        )
    except (LicenceGateError, ValueError, FileNotFoundError) as exc:
        print(f"M3 dataset preparation refused: {exc}")
        return 1

    gate = manifest["licence_gate"]
    print(f"M3 dataset: {manifest['dataset_id']}")
    print(f"  verified sources : {len(gate['verified_sources'])}")
    print(f"  withheld records : {gate['withheld_records']} "
          f"from sources {sorted(gate['withheld_sources'])}")
    for split, info in manifest["splits"].items():
        print(f"  {split:16}: {info['records']:4} records, "
              f"{info['token_count']} tokens, "
              f"repos={len(info['repositories'])}")
        print(f"  {'':16}  token/jsonl chain verified: "
              f"{info['token_chain_matches_jsonl']}")
    isolation = manifest["isolation"]
    print(f"  isolation        : {'OK' if isolation['ok'] else 'VIOLATED'} "
          f"(shared repos={isolation['shared_repositories']})")
    print(f"  holdout          : not emitted ({manifest['holdout']['reason'][:60]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())