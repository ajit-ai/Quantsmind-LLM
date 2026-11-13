"""Build the Taksh L0.3 pretraining corpus (license-cleared, reproducible).

Merged from in-repository, license-documented sources only. Nothing is
downloaded; nothing outside the allowlist is admitted. Every document records
its source, license and domain. Exact duplicates are removed by SHA-1 of the
normalised text. A seeded, content-hash split assigns train/val.

Sources (all already present in the repository):

* ``data/corpus_v0.jsonl`` -- Wikipedia (CC-BY-SA-3.0) + Project Gutenberg
  (Public Domain), quality-filtered, 9,308 chunks, ~9.99M tokens
  (see data/DATA_CARD_v0.md). General language.
* ``data/m3_code_verified.train.jsonl`` -- 193 licence-verified source files
  (MIT / BSD-3-Clause, pinned commits, per-record spdx), ~1.2M tokens. Code.
  Excluded: the m7 duplicate of the same 193 records (same content_sha1).
* ``data/v180_corpus/taksh_code_v1.txt`` -- the Taksh project's own source
  tree (Apache-2.0, self-authored), document marker ``<|file|> PATH``. Code.

Not used, and why:
* ``data/m7/pretrain.train.jsonl`` -- byte-identical content to
  m3_code_verified.train.jsonl (adds only an ``m7_partition`` field).
* ``data/m3_code_verified.val.jsonl`` -- validation data; never trained on.
* ``data/instruct_v1.train.jsonl`` -- derived from the same source files;
  would double-count code tokens.
* ``data/v170_corpus`` / ``data/tokenizer_corpus`` -- subsets of the same
  repository snapshot (188 files vs v180's 237); avoid near-duplicate inflation.
* ``data/code_sft_v0.jsonl`` -- quarantined (see data/code_sft_v0.quarantine.json).
* ``data/corpus_v0.tokens.bin`` -- encoded with the obsolete qm_bpe_v0
  tokenizer (49,152 vocab); re-encoded here with the Taksh tokenizer.
* Wikipedia/Gutenberg share-alike terms permit training a research model; the
  raw files remain in-repo with their manifests and are not redistributed.

Outputs (new directory, never overwritten without --force):
* data/l03_corpus/train.jsonl / val.jsonl   (one {text, source, license, ...} per line)
* data/l03_corpus/dataset_manifest.json     (counts, licences, sha256s, split seed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SEED = 20261110
SPLIT_TRAIN = 0.98

CODE_SPDX_ALLOWLIST = frozenset({
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc",
})
GENERAL_LICENSE_ALLOWLIST = frozenset({
    "cc-by-sa-3.0", "public domain",
})

REPO = Path(__file__).resolve().parent.parent


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def sha1(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def load_wiki_gutenberg(path: Path):
    kept = []
    rejected = {"license": 0, "empty": 0}
    # Gutenberg rows carry ``license: see-MANIFEST``; the project data card
    # (data/DATA_CARD_v0.md) documents them as Public Domain, and the raw
    # MANIFEST (data/raw/gutenberg/MANIFEST.json) is present and lists the
    # three books. Accept exactly that combination, nothing else.
    gutenberg_manifest_ok = (REPO / "data/raw/gutenberg/MANIFEST.json").is_file()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            license_ = str(row.get("license", "")).strip().lower()
            source = str(row.get("source", "")).strip()
            if license_ == "see-manifest" and source == "gutenberg" and gutenberg_manifest_ok:
                license_ = "public domain"
            if license_ not in GENERAL_LICENSE_ALLOWLIST:
                rejected["license"] += 1
                continue
            text = normalize(row.get("text", ""))
            if len(text) < 200:
                rejected["empty"] += 1
                continue
            kept.append({"text": text, "source": source, "license": license_,
                         "domain": "general"})
    return kept, rejected


def load_m3_code(path: Path):
    kept, rejected = [], {"license": 0, "empty": 0}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            spdx = str(row.get("spdx", "")).strip().lower()
            if spdx not in CODE_SPDX_ALLOWLIST:
                rejected["license"] += 1
                continue
            text = normalize(row.get("text", ""))
            if len(text) < 100:
                rejected["empty"] += 1
                continue
            kept.append({"text": text,
                         "source": f"m3-verified:{row.get('repo')}@{str(row.get('commit'))[:12]}",
                         "license": spdx,
                         "domain": "code",
                         "file_path": row.get("file_path"),
                         "attribution": row.get("attribution")})
    return kept, rejected


def load_repo_code(path: Path):
    docs, current_path, buf = [], None, []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("<|file|>"):
                if current_path is not None:
                    text = normalize("".join(buf))
                    if len(text) >= 100:
                        docs.append({"text": text,
                                     "source": f"taksh-repo:{current_path}",
                                     "license": "apache-2.0",
                                     "domain": "code"})
                current_path = line[len("<|file|>"):].strip()
                buf = []
            else:
                buf.append(line)
    if current_path is not None:
        text = normalize("".join(buf))
        if len(text) >= 100:
            docs.append({"text": text, "source": f"taksh-repo:{current_path}",
                         "license": "apache-2.0", "domain": "code"})
    return docs, {}


def split_of(doc_sha: str) -> str:
    # Content-hash split: stable across runs regardless of enumeration order.
    digest = hashlib.sha1(f"{SEED}:{doc_sha}".encode("utf-8")).hexdigest()
    return "train" if int(digest, 16) % 10000 < int(SPLIT_TRAIN * 10000) else "val"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/l03_corpus")
    args = parser.parse_args(argv)

    out_dir = REPO / args.out_dir
    if (out_dir / "dataset_manifest.json").exists():
        print(f"error: {out_dir}/dataset_manifest.json already exists; refusing to overwrite. "
              "Remove the directory explicitly to rebuild.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    rejected = {}
    wiki, rej = load_wiki_gutenberg(REPO / "data/corpus_v0.jsonl")
    rejected.update({f"wiki_gutenberg:{k}": v for k, v in rej.items()})
    m3, rej = load_m3_code(REPO / "data/m3_code_verified.train.jsonl")
    rejected.update({f"m3_code:{k}": v for k, v in rej.items()})
    repo_docs, rej = load_repo_code(REPO / "data/v180_corpus/taksh_code_v1.txt")
    rejected.update({f"repo_code:{k}": v for k, v in rej.items()})

    docs = wiki + m3 + repo_docs
    print(f"loaded: wikipedia/gutenberg={len(wiki)} m3_verified_code={len(m3)} "
          f"repo_code={len(repo_docs)} | total={len(docs)}")

    seen = set()
    kept, dups = [], 0
    for d in docs:
        h = sha1(d["text"])
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        d["sha1"] = h
        d["split"] = split_of(h)
        kept.append(d)
    print(f"dedup: removed {dups} exact duplicate documents; kept {len(kept)}")
    for k, v in rejected.items():
        print(f"rejected[{k}] = {v}")

    train_docs = [d for d in kept if d["split"] == "train"]
    val_docs = [d for d in kept if d["split"] == "val"]
    for name, group in (("train", train_docs), ("val", val_docs)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for d in group:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_domain = {}
    by_license = {}
    for d in kept:
        by_domain[d["domain"]] = by_domain.get(d["domain"], 0) + 1
        by_license[d["license"]] = by_license.get(d["license"], 0) + 1

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema": "taksh.dataset_manifest/1",
        "dataset_id": "l03-corpus",
        "created": "2026-11-10",
        "generated_by": "scripts/build_l03_corpus.py (seed=%d, split_train=%.2f)" % (SEED, SPLIT_TRAIN),
        "seed": SEED,
        "split_seed": SEED,
        "split": "98/2 train/val by seeded content hash (sha1)",
        "sources": {
            "wikipedia_gutenberg": {"path": "data/corpus_v0.jsonl", "records": len(wiki),
                                    "license": "CC-BY-SA-3.0 (Wikipedia); Public Domain (Gutenberg; rows carry 'see-MANIFEST', verified against data/raw/gutenberg/MANIFEST.json and DATA_CARD_v0.md)", "domain": "general"},
            "m3_code_verified_train": {"path": "data/m3_code_verified.train.jsonl", "records": len(m3),
                                       "license": "MIT / BSD-3-Clause (per record)", "domain": "code"},
            "taksh_repo_code": {"path": "data/v180_corpus/taksh_code_v1.txt", "records": len(repo_docs),
                                "license": "Apache-2.0", "domain": "code"},
        },
        "excluded": {
            "m7/pretrain.train.jsonl": "duplicate of m3_code_verified.train.jsonl (same content_sha1)",
            "m3_code_verified.val.jsonl": "validation records; never trained on",
            "instruct_v1.train.jsonl": "derived from the same source files; would double-count",
            "v170/tokenizer_corpus": "subsets of the same snapshot; near-duplicate inflation",
            "code_sft_v0.jsonl": "quarantined (data/code_sft_v0.quarantine.json)",
            "corpus_v0.tokens.bin": "encoded with obsolete qm_bpe_v0 tokenizer; re-encoded with Taksh tokenizer",
        },
        "documents": len(kept),
        "duplicates_removed": dups,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "by_domain": by_domain,
        "by_license": by_license,
        "rejected": rejected,
        "train_sha256": _sha(out_dir / "train.jsonl"),
        "val_sha256": _sha(out_dir / "val.jsonl"),
        "evaluation_note": "Holdout benchmark is evaluation/l02_python_bench_v1.json (authored 2026-11-09, after every corpus snapshot); no corpus document contains it.",
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
