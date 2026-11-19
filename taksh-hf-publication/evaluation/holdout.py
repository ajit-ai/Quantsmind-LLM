"""Held-out evaluation data preparation (M2.10).

Turns the corpus's held-out split into an evaluation artifact and proves it is
actually held out. Three properties matter, and all three are checked here
rather than assumed:

1. **Separation.** No content hash in the held-out set appears in the training
   split, and no source repository appears in both. The corpus builder already
   groups by source repository; this re-verifies it independently against the
   shipped file.
2. **Provenance.** Every held-out item keeps its repository, commit, path and
   licence, so a result can be traced back to what it was measured on.
3. **Honest labelling.** These items are *held-out language-modelling* data --
   unseen source files from repositories the model never trained on. They are
   NOT an executable pass@1 benchmark. The published HumanEval and MBPP suites
   are not available in this repository, and this module never claims they are.

Usage
-----
    python -m evaluation.holdout --corpus data/code_corpus_v2.jsonl \
        --out evaluation/code_holdout_v2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.code_holdout/1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path):
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped:
            yield number, stripped


def load_records(path):
    return [json.loads(raw) for _, raw in read_jsonl(path)]


def check_separation(records, holdout_split="test"):
    """Verify the held-out split shares no content and no repository with train.

    Returns a report. ``ok`` is False whenever any overlap is found; a violation
    is never downgraded to a warning.
    """
    holdout = [r for r in records if r.get("split") == holdout_split]
    train = [r for r in records if r.get("split") == "train"]

    holdout_hashes = {r.get("content_sha1") for r in holdout if r.get("content_sha1")}
    train_hashes = {r.get("content_sha1") for r in train if r.get("content_sha1")}
    shared_hashes = sorted(holdout_hashes & train_hashes)

    holdout_repos = {r.get("repo") for r in holdout if r.get("repo")}
    train_repos = {r.get("repo") for r in train if r.get("repo")}
    shared_repos = sorted(holdout_repos & train_repos)

    report = {
        "holdout_split": holdout_split,
        "holdout_records": len(holdout),
        "train_records": len(train),
        "shared_content_sha1": shared_hashes,
        "shared_repositories": shared_repos,
        "grouping": "source repository (source_group)",
        "ok": not shared_hashes and not shared_repos,
    }
    if shared_hashes:
        report["problem"] = (
            f"{len(shared_hashes)} content hash(es) appear in both the held-out "
            "and training splits"
        )
    elif shared_repos:
        report["problem"] = (
            f"{len(shared_repos)} repository/repositories span both the held-out "
            "and training splits"
        )
    return report


def build_holdout(corpus_path, out_path, holdout_split="test", min_chars=200,
                  dataset_id="code_corpus-v2", now=None, vocab_size=None):
    """Write a held-out evaluation set and its manifest.

    Items shorter than ``min_chars`` are skipped: a 3-line docstring is not a
    meaningful held-out sample and would let a memorising model look good.
    ``vocab_size`` records the uniform-random baseline loss in the manifest.
    """
    corpus_path = Path(corpus_path)
    records = load_records(corpus_path)
    separation = check_separation(records, holdout_split)
    if not separation["ok"]:
        raise ValueError(
            f"held-out split is not actually held out: {separation['problem']}"
        )

    candidates = [
        r for r in records
        if r.get("split") == holdout_split and len(r.get("text") or "") >= min_chars
    ]
    if not candidates:
        raise ValueError(
            f"no held-out records of at least {min_chars} characters in {corpus_path}"
        )

    items = []
    for index, record in enumerate(candidates):
        text = record["text"]
        items.append({
            "task_id": f"{holdout_split}/{index:04d}",
            "task_type": "held_out_language_modelling",
            "language": record.get("language"),
            "prompt": text,
            "token_count_hint": len(text) // 4,
            "source": {
                "repo": record.get("repo"),
                "commit": record.get("commit"),
                "file_path": record.get("file_path"),
                "license": record.get("spdx"),
                "attribution": record.get("attribution"),
                "retrieved_at": record.get("retrieved_at"),
            },
            "content_sha1": record.get("content_sha1"),
            "split": holdout_split,
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")

    manifest = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "holdout_split": holdout_split,
        "generated_at": now or utc_now_iso(),
        "corpus": str(corpus_path),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "items": len(items),
        "skipped_short": len(
            [r for r in records
             if r.get("split") == holdout_split
             and len(r.get("text") or "") < min_chars]
        ),
        "min_chars": min_chars,
        "languages": dict(Counter(i["language"] for i in items)),
        "repositories": sorted({
            i["source"]["repo"] for i in items if i["source"]["repo"]
        }),
        "separation": separation,
        "separation_method": (
            "content SHA-1 set intersection and repository set intersection "
            "against the training split, computed from the shipped corpus"
        ),
        "evaluation_protocol": {
            "metric": "per-token cross-entropy loss on held-out source files",
            "why_loss": (
                "These are real source files from unseen repositories, not "
                "executable function specifications. Loss is the only metric "
                "they support honestly."
            ),
            "executable_benchmark": False,
            "published_suites_status": "unavailable",
            "baseline": (
                {
                    "uniform_random_loss": load_baseline_loss(vocab_size),
                    "vocab_size": vocab_size,
                    "note": (
                        "A model at or above this loss has learned nothing; "
                        "report it as the floor for any held-out loss figure."
                    ),
                }
                if vocab_size else
                {
                    "uniform_random_loss": None,
                    "vocab_size": None,
                    "note": "pass vocab_size to record the random-model floor",
                }
            ),
            "limitations": [
                "Not a pass@1 code-generation benchmark; cannot report pass@1.",
                "HumanEval (164) and MBPP (500) are not present in this "
                "repository; the 5-problem subsets in evaluation/eval_code.py "
                "exist only to prove the harness runs.",
                "Loss on held-out files measures generalisation over unseen "
                "repositories, not task capability.",
                "Split is grouped by repository, so the held-out set spans "
                f"{len({i['source']['repo'] for i in items})} repositories only.",
            ],
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_baseline_loss(vocab_size):
    """Loss a uniform-random model achieves: the floor a real model must beat."""
    import math
    return math.log(vocab_size)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare held-out evaluation data")
    parser.add_argument("--corpus", default="data/code_corpus_v2.jsonl")
    parser.add_argument("--out", default="evaluation/code_holdout_v2.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument(
        "--tokenizer", default="tokenizer/qm_code_bpe_v0.json",
        help="only used to record the random-baseline loss",
    )
    args = parser.parse_args(argv)

    vocab_size = None
    if args.tokenizer and Path(args.tokenizer).exists():
        try:
            from tokenizers import Tokenizer
            vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()
        except Exception as exc:  # noqa: BLE001 - baseline is optional context
            print(f"warning: could not read vocab size from {args.tokenizer}: {exc}")

    try:
        manifest = build_holdout(
            args.corpus, args.out, holdout_split=args.split,
            min_chars=args.min_chars, vocab_size=vocab_size,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"holdout preparation failed: {exc}")
        return 1

    print(f"holdout: {args.out}")
    print(f"  items        : {manifest['items']} (skipped short: {manifest['skipped_short']})")
    print(f"  split        : {manifest['holdout_split']}")
    print(f"  repositories : {len(manifest['repositories'])}")
    print(f"  languages    : {manifest['languages']}")
    print(f"  separation   : {'OK' if manifest['separation']['ok'] else 'VIOLATED'}")
    if not manifest["separation"]["ok"]:
        print(f"    problem: {manifest['separation']['problem']}")
    protocol = manifest["evaluation_protocol"]
    print(f"  metric       : {protocol['metric']}")
    print(f"  executable   : {protocol['executable_benchmark']}")
    print(f"  published    : {protocol['published_suites_status']}")
    print("  limitations  :")
    for item in protocol["limitations"]:
        print(f"    - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())