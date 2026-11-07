"""G2: benchmark contamination audit.

Answers one question with evidence: does ``data/code_corpus_v1.jsonl`` contain
text that also appears in a benchmark we evaluate on?

Design constraints that shape this module:

* Only benchmark data that is **already in this repository**, or explicitly
  configured by a local path, is used. Nothing is downloaded from an unapproved
  location.
* The repo ships 5-problem hand-transcribed subsets of HumanEval and MBPP in
  ``evaluation/eval_code.py``, not the published suites (164 / 500). So the audit
  reports on what it actually has and marks the published suites
  ``unavailable``. Absence of benchmark data is reported as ``unavailable``,
  never as ``clean``.
* Detection is lexical: exact match and normalised/near-duplicate match using the
  same MinHash shingling as the corpus filters. This is **not** semantic
  decontamination and does not claim to be.

Usage
-----
    python -m data.contamination_audit
    python -m data.contamination_audit --corpus data/code_corpus_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.filters import MinHashDedup, word_shingles

SUBSET_NGRAM = 13  # GPT-3 style n-gram overlap floor


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalise_for_match(text: str) -> str:
    """Normalisation used for the exact-match arm.

    NFKC, lowercase, collapse all whitespace, drop punctuation-only spacing.
    Deliberately conservative: it does not remove comments or rename identifiers,
    because that would inflate the match rate and hide real overlap.
    """
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip()


def ngrams(text: str, n: int = SUBSET_NGRAM):
    """Word n-grams of a normalised string."""
    words = normalise_for_match(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_local_benchmarks(extra_paths=None):
    """Collect benchmark records from local sources only.

    Returns (records, provenance) where each record is
    ``{suite, task_id, field, text}``.
    """
    records = []
    provenance = []

    try:
        from evaluation import eval_code

        for suite, constant in (
            ("humaneval", eval_code.HUMANEVAL_PROBLEMS),
            ("mbpp", eval_code.MBPP_PROBLEMS),
        ):
            for i, prob in enumerate(constant):
                for field in ("prompt", "test"):
                    text = prob.get(field)
                    if text:
                        records.append(
                            {
                                "suite": suite,
                                "task_id": prob.get("task_id", f"{suite}/{i}"),
                                "field": field,
                                "text": text,
                            }
                        )
            provenance.append(
                {
                    "suite": suite,
                    "origin": "evaluation/eval_code.py (in-repo constant)",
                    "problems": len(constant),
                    "is_published_suite": False,
                    "note": (
                        "hand-transcribed subset used to exercise the harness; "
                        "NOT the published benchmark"
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        provenance.append(
            {
                "suite": "humaneval+mbpp",
                "origin": "evaluation/eval_code.py",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    for path in extra_paths or []:
        p = Path(path)
        if not p.exists():
            provenance.append({"suite": None, "origin": str(p), "error": "missing"})
            continue
        suite = p.stem
        count = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("prompt", "canonical_solution", "test", "code"):
                    text = row.get(field)
                    if text:
                        records.append(
                            {
                                "suite": suite,
                                "task_id": row.get("task_id", f"{suite}/{count}"),
                                "field": field,
                                "text": text,
                            }
                        )
                count += 1
        provenance.append(
            {
                "suite": suite,
                "origin": str(p),
                "problems": count,
                "is_published_suite": True,
                "note": "explicitly configured local benchmark file",
            }
        )

    return records, provenance


def _build_ngram_index(records, n):
    index = defaultdict(set)
    for i, rec in enumerate(records):
        for gram in ngrams(rec["text"], n):
            index[gram].add(i)
    return index


def _minhash_index(records, detector):
    """LSH index over benchmark records for near-duplicate lookup."""
    buckets = defaultdict(list)
    for i, rec in enumerate(records):
        sig = detector.signature(rec["text"])
        if sig.size == 0:
            continue
        per_band = detector.rows
        for band in range(detector.num_perms // per_band):
            key = (band, sig[band * per_band : (band + 1) * per_band].tobytes())
            buckets[key].append(i)
    return buckets


def audit_corpus(
    corpus_path,
    extra_benchmark_paths=None,
    ngram_n=SUBSET_NGRAM,
    near_threshold=0.8,
    minhash_perms=64,
    minhash_bands=16,
    log=print,
):
    """Run the audit. Returns a report dict; never claims clean on no data."""
    records, provenance = load_local_benchmarks(extra_benchmark_paths)

    published = {
        "humaneval": {"published_problems": 164, "available": False},
        "mbpp": {"published_problems": 500, "available": False},
    }
    for entry in provenance:
        suite = entry.get("suite")
        if suite in published and entry.get("is_published_suite"):
            published[suite]["available"] = True

    report = {
        "schema": "quantsmind.contamination_audit/1",
        "generated_at": utc_now_iso(),
        "corpus": str(corpus_path),
        "benchmark_provenance": provenance,
        "published_suite_availability": published,
        "method": {
            "exact": "normalised full-text containment (NFKC, lowercase, collapse whitespace)",
            "ngram": f"{ngram_n}-word shingle overlap, any shared shingle is a match",
            "near_duplicate": (
                f"MinHash LSH over 8-word shingles, {minhash_perms} perms / "
                f"{minhash_bands} bands, jaccard >= {near_threshold}"
            ),
            "not_semantic": (
                "Lexical only. Paraphrases, renamed identifiers and reimplemented "
                "logic are NOT detected. This is not semantic decontamination."
            ),
        },
        "status": None,
        "records_checked": 0,
        "matches": [],
        "match_counts": {"exact": 0, "ngram": 0, "near_duplicate": 0},
        "limitations": [],
    }

    if not records:
        report["status"] = "unavailable"
        report["limitations"].append(
            "no benchmark records were available locally, so contamination is "
            "UNKNOWN. This is not a clean result."
        )
        return report

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        report["status"] = "unavailable"
        report["limitations"].append(f"corpus not found: {corpus_path}")
        return report

    index = _build_ngram_index(records, ngram_n)
    detector = MinHashDedup(
        num_perms=minhash_perms, bands=minhash_bands, threshold=near_threshold
    )
    buckets = _minhash_index(records, detector)

    norm_records = [normalise_for_match(r["text"]) for r in records]
    rows = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    report["records_checked"] = len(rows)

    for row in rows:
        text = row["text"]
        norm = normalise_for_match(text)
        digest = row.get("content_sha1") or hashlib.sha1(
            text.encode("utf-8")
        ).hexdigest()

        hits = set()
        for gram in ngrams(text, ngram_n):
            hits |= index.get(gram, set())

        for i, rec_norm in enumerate(norm_records):
            if rec_norm and rec_norm in norm and len(rec_norm) > 40:
                hits.add(i)
                break

        near_hits = set()
        sig = detector.signature(text)
        if sig.size:
            per_band = detector.rows
            for band in range(detector.num_perms // per_band):
                key = (band, sig[band * per_band : (band + 1) * per_band].tobytes())
                near_hits |= set(buckets.get(key, ()))

        for i in sorted(hits | near_hits):
            rec = records[i]
            # Precedence: a full-text containment is the strongest signal, then
            # LSH near-duplicate, then shingle overlap alone.
            if rec_norm and len(rec_norm) > 40 and rec_norm in norm:
                kind = "exact"
            elif i in near_hits:
                kind = "near_duplicate"
            else:
                kind = "ngram"
            report["matches"].append(
                {
                    "corpus_source_id": row.get("source_id"),
                    "corpus_repo": row.get("repo"),
                    "corpus_file_path": row.get("file_path"),
                    "corpus_content_sha1": digest,
                    "corpus_split": row.get("split"),
                    "suite": rec["suite"],
                    "task_id": rec["task_id"],
                    "field": rec["field"],
                    "match_kind": kind,
                }
            )
            report["match_counts"][kind] = (
                report["match_counts"].get(kind, 0) + 1
            )

    missing_suites = [
        s for s, info in published.items() if not info["available"]
    ]
    if report["matches"]:
        report["status"] = "contaminated"
        report["limitations"].append(
            f"{len(report['matches'])} overlapping record(s) found against the "
            "locally available benchmark subsets"
        )
    else:
        report["status"] = "no_overlap_detected"
        report["limitations"].append(
            "no overlap found against the benchmark subsets present in this "
            "repository"
        )
    if missing_suites:
        report["limitations"].append(
            "the published suites "
            + ", ".join(
                f"{s} ({published[s]['published_problems']} problems)"
                for s in missing_suites
            )
            + " are NOT available here and were not checked; the corpus is "
            "therefore not demonstrated to be free of published-benchmark overlap"
        )
    report["limitations"].append(
        "lexical matching only: semantic or paraphrased overlap is not detectable "
        "by this audit"
    )

    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark contamination audit")
    p.add_argument("--corpus", default="data/code_corpus_v1.jsonl")
    p.add_argument("--out", default="data/contamination_audit.json")
    p.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="local benchmark JSONL to also check (repeatable)",
    )
    p.add_argument("--ngram", type=int, default=SUBSET_NGRAM)
    args = p.parse_args(argv)

    report = audit_corpus(args.corpus, args.benchmark, ngram_n=args.ngram)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"corpus records checked : {report['records_checked']}")
    print(f"status                 : {report['status']}")
    print(f"matches                : {report['match_counts']}")
    for entry in report["benchmark_provenance"]:
        print(
            f"  benchmark {entry.get('suite')}: {entry.get('problems', '?')} "
            f"problems from {entry.get('origin')}"
        )
    for lim in report["limitations"]:
        print(f"  limitation: {lim}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())