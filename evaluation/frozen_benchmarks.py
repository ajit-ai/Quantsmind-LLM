"""Inspect the frozen-benchmark manifest without touching the benchmarks.

The manifest pins expected SHA-256 values and historical results. This module
only reads files and hashes them. It never downloads, modifies or re-scores a
benchmark, and it never trains, creates a checkpoint or publishes anything.

Run:

    python -m evaluation.frozen_benchmarks
    python -m evaluation.frozen_benchmarks --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFEST = "configs/roadmap/frozen_benchmarks.yaml"
DEFAULT_OUT = "reports/taksh_frozen_benchmarks.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FrozenBenchmarkError(RuntimeError):
    """The manifest is malformed or claims a prohibited action."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path = MANIFEST) -> dict[str, Any]:
    manifest_path = _path(path)
    if not manifest_path.is_file():
        raise FrozenBenchmarkError(f"manifest not found: {manifest_path}")
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise FrozenBenchmarkError(f"manifest is not a mapping: {manifest_path}")
    return document


def _validate(manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    if manifest.get("schema") != "taksh.frozen_benchmark_manifest/1":
        errors.append(f"unexpected schema: {manifest.get('schema')!r}")

    policy = manifest.get("policy") or {}
    for key in ("download", "modify", "rescore", "capability_claim"):
        if policy.get(key) is not False:
            errors.append(f"policy.{key} must be false")
    if policy.get("evaluation_requires_owner_authorization") is not True:
        errors.append("evaluation must require owner authorization")

    benchmarks = manifest.get("benchmarks") or []
    if not benchmarks:
        errors.append("benchmarks must not be empty")
    seen: set[str] = set()
    for index, benchmark in enumerate(benchmarks):
        if not isinstance(benchmark, dict):
            errors.append(f"benchmarks[{index}] is not a mapping")
            continue
        benchmark_id = str(benchmark.get("id", ""))
        if not benchmark_id:
            errors.append(f"benchmarks[{index}] is missing id")
        if benchmark_id in seen:
            errors.append(f"duplicate benchmark id: {benchmark_id}")
        seen.add(benchmark_id)
        if not benchmark.get("path"):
            errors.append(f"benchmark {benchmark_id or index} is missing path")
        expected = str(benchmark.get("expected_sha256", ""))
        if not SHA256.match(expected):
            errors.append(
                f"benchmark {benchmark_id or index} has invalid expected_sha256"
            )
        if not isinstance(benchmark.get("expected_tasks"), int) or benchmark["expected_tasks"] <= 0:
            errors.append(
                f"benchmark {benchmark_id or index} must declare positive expected_tasks"
            )
        if benchmark.get("status") not in ALLOWED_STATUSES:
            errors.append(
                f"benchmark {benchmark_id or index} has invalid status "
                f"{benchmark.get('status')!r}"
            )
        sources = benchmark.get("hash_sources") or []
        if not sources:
            errors.append(
                f"benchmark {benchmark_id or index} must declare hash_sources"
            )
        historical = benchmark.get("historical") or {}
        if historical.get("preserved") is not True:
            errors.append(
                f"benchmark {benchmark_id or index} must preserve its historical result"
            )
        if historical.get("re_scored") is not False:
            errors.append(
                f"benchmark {benchmark_id or index} must not re-score historical results"
            )

    if errors:
        raise FrozenBenchmarkError("; ".join(errors))


def _provenance(benchmark: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    expected = benchmark["expected_sha256"]
    records = []
    for source in benchmark.get("hash_sources") or []:
        source_path = _path(source)
        present = source_path.is_file()
        confirms = False
        if present:
            confirms = expected in source_path.read_text(encoding="utf-8", errors="replace")
        records.append(
            {
                "path": _relative(source_path),
                "present": present,
                "confirms_expected_sha256": confirms,
            }
        )
    return records, bool(records) and all(record["confirms_expected_sha256"] for record in records)


def _inspect_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    path = _path(benchmark["path"])
    provenance, provenance_ok = _provenance(benchmark)
    expected = benchmark["expected_sha256"]

    if not path.is_file():
        availability = "ABSENT"
        actual = None
        status = "BLOCKED"
    else:
        actual = _sha256(path)
        if actual == expected:
            availability = "PRESENT_VERIFIED"
            status = "PASS" if provenance_ok else "BLOCKED"
        else:
            availability = "PRESENT_HASH_MISMATCH"
            status = "BLOCKED"

    historical = dict(benchmark.get("historical") or {})
    historical["preserved"] = True
    historical["re_scored"] = False

    return {
        "id": benchmark["id"],
        "name": benchmark["name"],
        "path": _relative(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "expected_tasks": benchmark["expected_tasks"],
        "availability": availability,
        "status": status,
        "provenance_ok": provenance_ok,
        "provenance": provenance,
        "historical": historical,
        "note": benchmark.get("note"),
    }


def inspect(path: str | Path = MANIFEST) -> dict[str, Any]:
    """Return deterministic frozen-benchmark evidence; never writes a benchmark."""
    manifest = load_manifest(path)
    _validate(manifest)

    benchmarks = [_inspect_benchmark(item) for item in manifest["benchmarks"]]
    absent = [item["id"] for item in benchmarks if item["availability"] == "ABSENT"]
    mismatched = [
        item["id"]
        for item in benchmarks
        if item["availability"] == "PRESENT_HASH_MISMATCH"
    ]
    provenance_failed = [
        item["id"] for item in benchmarks if item["provenance_ok"] is not True
    ]

    if mismatched:
        verdict = "FROZEN_BENCHMARK_INTEGRITY_FAILURE"
    elif absent:
        verdict = "FROZEN_BENCHMARKS_ABSENT"
    elif provenance_failed:
        verdict = "FROZEN_BENCHMARK_PROVENANCE_UNVERIFIED"
    else:
        verdict = "FROZEN_BENCHMARKS_INTACT"

    status = "PASS" if verdict == "FROZEN_BENCHMARKS_INTACT" else "BLOCKED"
    if status not in ALLOWED_STATUSES:
        raise FrozenBenchmarkError(f"invalid overall status: {status}")

    return {
        "schema": "taksh.frozen_benchmarks/1",
        "generated_by": "evaluation/frozen_benchmarks.py",
        "manifest": _relative(_path(path)),
        "status": status,
        "verdict": verdict,
        "benchmarks": benchmarks,
        "absent": absent,
        "hash_mismatches": mismatched,
        "provenance_failures": provenance_failed,
        "historical_results": [
            {
                "id": item["id"],
                "result": item["historical"].get("result"),
                "measured": item["historical"].get("measured"),
                "preserved": True,
                "re_scored": False,
            }
            for item in benchmarks
        ],
        "policy": {
            "download": False,
            "modify": False,
            "rescore": False,
            "capability_claim": False,
            "evaluation_requires_owner_authorization": True,
            "historical_results_immutable": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=MANIFEST,
        help=f"frozen benchmark manifest (default: {MANIFEST})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = inspect(args.manifest)
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"verdict: {report['verdict']}")
        print("benchmarks:")
        for benchmark in report["benchmarks"]:
            print(
                f"  {benchmark['status']:<10} {benchmark['availability']:<22} "
                f"{benchmark['id']}"
            )
        print(f"report: {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
