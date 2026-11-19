"""Leakage detection for the M5 benchmark.

A score is meaningless until the evaluation examples have been checked against
everything the model could have trained on. This module compares the benchmark
against:

* M3 pretraining sources (the licence-gated corpus files);
* M4 instruction-tuning examples;
* prior evaluation prompts (the M4 frozen set).

Detection layers, weakest to strongest:

``exact``
    identical prompt strings
``normalised``
    case/whitespace/unicode-folded equality -- catches reformatting
``hash``
    SHA-256 over the normalised text -- same thing, cryptographically
``source``
    the same upstream repository, which is a *risk* signal, not contamination

Repository-disjoint is **not** content-disjoint. A repository can contribute two
nearly identical files, and this module reports source overlap as an unresolved
risk rather than silently treating it as clean.

Near-duplicate detection is explicitly out of scope here. A hash is not semantic
similarity, and nothing in the output claims otherwise.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.m5_schema import load, normalise, text_hash


def _index_with_sources(path, field):
    """Like :func:`_index` but keeps records so sources can be extracted."""
    exact, normalised, hashed, sources = set(), set(), set(), set()
    count = 0
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "available": False, "records": 0,
                "exact": exact, "normalised": normalised, "hashes": hashed,
                "sources": sources}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        count += 1
        record = json.loads(line)
        text = (record.get(field) or "").strip()
        exact.add(text)
        normalised.add(normalise(record.get(field) or ""))
        hashed.add(text_hash(record.get(field) or ""))
        sources.add(record.get("source_id") or record.get("repo"))
    return {"path": str(path), "available": True, "records": count,
            "exact": exact, "normalised": normalised, "hashes": hashed,
            "sources": sources}


def leakage_report(benchmark_path, reference_paths):
    """Compare the benchmark against each reference corpus.

    ``reference_paths`` is a list of ``(label, path, text_field)`` triples.
    Returns an auditable report; ``leaked`` is True when any exact, normalised
    or hash overlap is found.
    """
    benchmark = load(benchmark_path)
    findings = []
    total_overlap = 0

    for label, path, field in reference_paths:
        index = _index_with_sources(path, field)
        exact, normalised, hashes = [], [], []
        for record in benchmark:
            prompt = (record.get("prompt") or "").strip()
            if prompt and prompt in index["exact"]:
                exact.append(record["id"])
            if normalise(record.get("prompt") or "") in index["normalised"]:
                normalised.append(record["id"])
            if text_hash(record.get("prompt") or "") in index["hashes"]:
                hashes.append(record["id"])

        benchmark_sources = {r.get("source_id") for r in benchmark}
        shared_sources = sorted(
            s for s in (index["sources"] & benchmark_sources) if s
        )
        overlap = sorted(set(exact) | set(normalised) | set(hashes))
        total_overlap += len(overlap)
        findings.append({
            "reference": label,
            "path": str(path),
            "available": index["available"],
            "reference_records": index["records"],
            "text_field": field,
            "exact_overlaps": sorted(exact),
            "normalised_overlaps": sorted(normalised),
            "hash_overlaps": sorted(hashes),
            "overlapping_examples": overlap,
            "shared_sources": shared_sources,
            "source_overlap_is_risk_not_contamination": True,
        })

    return {
        "benchmark": str(benchmark_path),
        "benchmark_examples": len(benchmark),
        "references": findings,
        "total_overlapping_examples": total_overlap,
        "leaked": total_overlap > 0,
        "near_duplicate_detection": (
            "not implemented; hashing detects exact normalised matches only "
            "and is not semantic similarity"
        ),
        "unresolved_risks": sorted({
            source
            for finding in findings
            for source in finding["shared_sources"]
        }),
    }


def default_reference_paths():
    """Every corpus a QuantsMind checkpoint could have trained on."""
    return [
        ("m3_pretrain_train", "data/m3_code_verified.train.jsonl", "text"),
        ("m3_pretrain_val", "data/m3_code_verified.val.jsonl", "text"),
        ("m3_raw_corpus", "data/code_corpus_v2.jsonl", "text"),
        ("m4_sft_train", "data/instruct_v1.train.jsonl", "instruction"),
        ("m4_sft_val", "data/instruct_v1.val.jsonl", "instruction"),
        ("prior_eval_prompts", "evaluation/m4_eval_set.jsonl", "prompt"),
    ]