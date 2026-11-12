"""Benchmark contamination registry and screening (v1.2.0).

The registry records which benchmark families are *known* to the pipeline and
which of them are actually *available* locally. Nothing is downloaded: a
family that has no local copy is recorded as ``unavailable`` and its absence
is reported as a coverage limitation.

A clean result is only ever a statement about the fingerprints that were
available and checked. The screening is lexical -- exact normalised
containment, shared n-gram shingles, and an optional MinHash near-match arm --
so it cannot detect a paraphrase or a re-implementation. The verdict says so.

Local benchmark files are read from paths named in the registry. Registering
HumanEval or MBPP is a registry edit plus a local copy; the pipeline never
fetches a suite on its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from data.contamination_audit import normalise_for_match
from data.filters import word_shingles

BENCHMARK_SCHEMA = "quantsmind.benchmark_registry/1"
REGISTRY_KEYS = {
    "schema",
    "version",
    "suites",
}
SUITE_KEYS = {
    "id",
    "version",
    "availability",
    "path",
    "fields",
    "published_problems",
    "notes",
    "exclude_sources",
    "near_threshold",
}


class BenchmarkRegistryError(ValueError):
    """The benchmark registry is malformed."""


@dataclass
class BenchmarkRecords:
    """Loaded benchmark content plus how it was obtained."""

    records: list = field(default_factory=list)
    provenance: list = field(default_factory=list)

    def fingerprints(self) -> set:
        return {r["sha256"] for r in self.records}

    def shingle_index(self, size: int) -> dict:
        index: dict = {}
        for record in self.records:
            for gram in word_shingles(record["text"], size):
                index.setdefault(gram, set()).add(record["task_id"])
        return index


def load_registry(path) -> dict:
    """Load and validate a benchmark registry."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise BenchmarkRegistryError(f"{path}: registry must be a mapping")
    if data.get("schema") != BENCHMARK_SCHEMA:
        raise BenchmarkRegistryError(
            f"{path}: schema must be {BENCHMARK_SCHEMA!r}, got {data.get('schema')!r}"
        )
    suites = data.get("suites")
    if not isinstance(suites, list) or not suites:
        raise BenchmarkRegistryError(f"{path}: 'suites' must be a non-empty list")
    seen = set()
    for index, suite in enumerate(suites):
        if not isinstance(suite, dict):
            raise BenchmarkRegistryError(f"{path}: suites[{index}] must be a mapping")
        unknown = set(suite) - SUITE_KEYS
        if unknown:
            raise BenchmarkRegistryError(
                f"{path}: suites[{index}] has unknown keys: {sorted(unknown)}"
            )
        suite_id = suite.get("id")
        if not isinstance(suite_id, str) or not suite_id:
            raise BenchmarkRegistryError(f"{path}: suites[{index}].id is required")
        if suite_id in seen:
            raise BenchmarkRegistryError(f"{path}: duplicate suite id {suite_id!r}")
        seen.add(suite_id)
        availability = suite.get("availability")
        if availability not in ("available", "unavailable"):
            raise BenchmarkRegistryError(
                f"{path}: suites[{index}].availability must be 'available' or "
                "'unavailable'"
            )
        if availability == "available" and not suite.get("path"):
            raise BenchmarkRegistryError(
                f"{path}: suites[{index}] is 'available' but has no local path"
            )
    return data


def _logical_origin(suite: dict) -> str | None:
    """Portable identity for a suite: the declared relative path as written,
    or a machine-invariant ``local://<basename>`` for absolute declarations.
    Absolute machine-specific paths never enter manifests."""
    declared = suite.get("path")
    if not declared:
        return None
    path = Path(declared)
    if path.is_absolute():
        return f"local://{path.name}"
    return declared


def _file_sha256(path: Path) -> str | None:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
    except (OSError, ValueError):
        return None
    return digest.hexdigest()


def load_benchmark_records(registry: dict, base_dir=None) -> BenchmarkRecords:
    """Load the locally available benchmark content.

    ``base_dir`` resolves relative suite paths against the registry location
    rather than the process working directory. The *declared* path is what the
    manifest records — never the resolved absolute path.
    """
    result = BenchmarkRecords()
    for suite in registry.get("suites", []):
        suite_id = suite["id"]
        version = suite.get("version", "unversioned")
        if suite.get("availability") != "available":
            result.provenance.append(
                {
                    "id": suite_id,
                    "version": version,
                    "availability": "unavailable",
                    "problems": 0,
                    "origin": None,
                    "note": suite.get("notes")
                    or "no local copy; the suite was not checked",
                }
            )
            continue
        path = Path(suite["path"])
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        origin = _logical_origin(suite)
        if not path.is_file():
            result.provenance.append(
                {
                    "id": suite_id,
                    "version": version,
                    "availability": "unavailable",
                    "problems": 0,
                    "origin": origin,
                    "note": "configured local file is missing; the suite was not checked",
                }
            )
            continue
        content_sha256 = _file_sha256(path)
        fields = tuple(suite.get("fields") or ("prompt", "test"))
        count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for field_name in fields:
                        text = row.get(field_name)
                        if not isinstance(text, str) or not text.strip():
                            continue
                        result.records.append(
                            {
                                "suite": suite_id,
                                "version": version,
                                "task_id": str(
                                    row.get("task_id")
                                    or row.get("problem_id")
                                    or f"{suite_id}/{number}"
                                ),
                                "field": field_name,
                                "text": text,
                                "sha256": hashlib.sha256(
                                    normalise_for_match(text).encode("utf-8")
                                ).hexdigest(),
                            }
                        )
                    count += 1
        except (OSError, json.JSONDecodeError) as exc:
            result.provenance.append(
                {
                    "id": suite_id,
                    "version": version,
                    "availability": "unavailable",
                    "problems": 0,
                    "origin": origin,
                    "content_sha256": content_sha256,
                    "note": f"unreadable: {type(exc).__name__}",
                }
            )
            continue
        result.provenance.append(
            {
                "id": suite_id,
                "version": version,
                "availability": "available",
                "problems": count,
                "origin": origin,
                "content_sha256": content_sha256,
                "note": suite.get("notes") or "local copy screened",
            }
        )
    return result


def screen_records(records: list, benchmarks: BenchmarkRecords, config: dict | None = None) -> dict:
    """Screen corpus records against the available benchmark content.

    Returns the hits, the per-suite coverage and an explicit verdict. The
    verdict is ``contaminated`` when a hit exists, ``no_overlap_detected`` when
    the available suites were checked and found nothing, and ``unavailable``
    when no benchmark content was available at all -- which is never reported
    as clean.
    """
    config = config or {}
    ngram_size = int(config.get("ngram_size", 13))
    near_threshold = config.get("near_threshold")

    fingerprints = benchmarks.fingerprints()
    index = benchmarks.shingle_index(ngram_size)
    available = [
        p for p in benchmarks.provenance if p.get("availability") == "available"
    ]
    unavailable = [
        p for p in benchmarks.provenance if p.get("availability") != "available"
    ]

    hits = []
    fingerprint_map: dict = {}
    for record in benchmarks.records:
        fingerprint_map.setdefault(record["sha256"], set()).add(record["suite"])
    for record in records:
        text = record.get("text", "")
        normalised = normalise_for_match(text)
        digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
        if digest in fingerprint_map:
            hits.append(
                {
                    "source_id": record.get("source_id"),
                    "file_path": record.get("file_path"),
                    "match_kind": "exact",
                    "benchmark": sorted(fingerprint_map[digest]),
                }
            )
            continue
        grams = word_shingles(text, ngram_size)
        shared = sorted(grams & set(index))
        if shared:
            hits.append(
                {
                    "source_id": record.get("source_id"),
                    "file_path": record.get("file_path"),
                    "match_kind": "ngram",
                    "benchmark": sorted(index[shared[0]]),
                    "shared_ngrams": len(shared),
                }
            )
            continue
        if near_threshold:
            best = _near_match(text, benchmarks, float(near_threshold))
            if best:
                hits.append(
                    {
                        "source_id": record.get("source_id"),
                        "file_path": record.get("file_path"),
                        "match_kind": "near_duplicate",
                        "benchmark": [best],
                    }
                )

    checked_ids = {p["id"] for p in available}
    by_suite = {
        p["id"]: {"checked": p["id"] in checked_ids, "hits": 0}
        for p in benchmarks.provenance
    }
    for hit in hits:
        for suite_id in hit.get("benchmark") or []:
            entry = by_suite.get(suite_id)
            if entry:
                entry["hits"] += 1

    if hits:
        verdict = "contaminated"
    elif available:
        verdict = "no_overlap_detected"
    else:
        verdict = "unavailable"

    limitations = [
        "lexical screening only: paraphrases, renamed identifiers and "
        "re-implemented logic are not detected"
    ]
    if unavailable:
        limitations.append(
            "not checked (no local copy): "
            + ", ".join(sorted(p["id"] for p in unavailable))
        )
    if not available:
        limitations.append(
            "no benchmark content was available locally, so contamination is "
            "UNKNOWN; this is not a clean result"
        )
    return {
        "verdict": verdict,
        "hits": hits,
        "hit_count": len(hits),
        "coverage": {
            "suites": benchmarks.provenance,
            "checked": sorted(p["id"] for p in available),
            "not_checked": sorted(p["id"] for p in unavailable),
            "records_screened": len(records),
        },
        "method": {
            "exact": "sha256 of NFKC/lowercase/whitespace-collapsed text",
            "ngram": f"{ngram_size}-word shingles of the benchmark text",
            "near_duplicate": (
                f"minhash near match at {near_threshold}"
                if near_threshold
                else "disabled"
            ),
        },
        "limitations": limitations,
    }


def _near_match(text: str, benchmarks: BenchmarkRecords, threshold: float):
    """Return the task id of the closest benchmark record, if within threshold."""
    from data.python_v1.dedup import jaccard, minhash_signature, _permutations

    grams = set(word_shingles(text, 8))
    if not grams:
        return None
    permutations = _permutations(64, 120)
    signature = minhash_signature(grams, permutations)
    best = None
    best_score = 0.0
    for record in benchmarks.records:
        other = set(word_shingles(record["text"], 8))
        if not other:
            continue
        score = jaccard(grams, other)
        if score > best_score:
            best_score = score
            best = record["task_id"]
    return best if best_score >= threshold else None
