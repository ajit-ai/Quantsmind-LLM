"""Build a local-only, benchmark-clean open-code capability corpus.

Sources (no network, no money, no external data):
  - this repository's own committed Python source (Apache-2.0)
  - installed open-source packages in taksh-env (read-only)

Benchmark safety is structural, not assumed:
  - evaluation/, tests/, benchmarks/ and eval/bench harness files are excluded
  - the finished corpus is scanned for frozen-benchmark markers and the build
    fails if any marker survives

Produces data/cap_corpus.jsonl with {text, language, source} records and a
manifest with SHA-256s, byte counts, and per-source counts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.build_code_corpus import is_minified, has_valid_structure, chunk_code  # noqa: E402
from data.filters import ExactDedup, MinHashDedup, quality_check, scrub_pii  # noqa: E402

CORPUS = Path("data/cap_corpus.jsonl")
MANIFEST = Path("data/cap_corpus.manifest.json")
BUILDER_PATH = Path(__file__).resolve()

# Directories that may contain benchmark tasks, expected answers, fixtures, or
# local data. None of these may enter a training corpus.
EXCLUDED_PARTS = {
    ".git", "node_modules", "taksh-env", "taksh.egg-info", "hf_export",
    "build", "dist", "runs", "data", "evaluation", "tests", "test",
    "benchmarks", "benchmark", "native_v01", "native_v02",
    "taksh-hf-publication",
}

# Filenames that are evaluation/benchmark harnesses rather than application code.
def excluded_name(name: str) -> bool:
    low = name.lower()
    return (
        low.startswith("eval_")
        or low.startswith("bench_")
        or "benchmark" in low
        or low in {"conftest.py"}
    )


FROZEN_MARKERS = (
    "NV01-BASIC-001",
    "Benchmark task inventory for TAKSH Native v0.1",
    "baseline_results.json",
)

stats = {
    "files_seen": 0,
    "files_excluded": 0,
    "files_yielded": 0,
    "chunks_seen": 0,
    "kept": 0,
    "rejected_quality": 0,
    "rejected_dup": 0,
    "rejected_fuzzy": 0,
    "rejected_minified": 0,
    "rejected_syntax": 0,
    "rejected_short": 0,
    "rejected_decode": 0,
    "bytes_in": 0,
    "bytes_out": 0,
    "by_source_kind": {},
    "total_sources": 0,
    "excluded_roots": sorted(EXCLUDED_PARTS),
    "corpus_sha256": None,
    "size_bytes": 0,
}

dedup = ExactDedup()
fuzzy = MinHashDedup()


def is_excluded(path: Path) -> bool:
    # The builder defines the frozen marker strings used by the safety scan, so
    # harvesting its own source would make the scan match itself. It is never
    # training data.
    try:
        if path.resolve() == BUILDER_PATH:
            return True
    except OSError:
        pass
    low_parts = {part.lower() for part in path.parts}
    if low_parts & EXCLUDED_PARTS:
        return True
    return excluded_name(path.name)


def iter_repo_py_files():
    out = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout
    paths = []
    for line in out.splitlines():
        p = Path(line)
        if p.name.endswith("_pb2.py"):
            continue
        if is_excluded(p):
            stats["files_excluded"] += 1
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: p.as_posix())


def iter_pip_py_files():
    root = Path("taksh-env/Lib/site-packages")
    if not root.exists():
        return
    paths = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if is_excluded(rel):
            stats["files_excluded"] += 1
            continue
        if p.name.endswith(".d.py"):
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: p.as_posix())


def attribution(path: Path, kind: str) -> str:
    if kind == "repo":
        return f"repo:{path.as_posix()}"
    root = Path("taksh-env/Lib/site-packages")
    try:
        rel = path.relative_to(root)
    except ValueError:
        return f"pip:{path.name}"
    return f"pip:{rel.parts[0]}/{Path(*rel.parts[1:]).as_posix()}" if len(rel.parts) > 1 else f"pip:{rel.name}"


def harvest_file(p: Path, kind: str, out) -> int:
    stats["files_seen"] += 1
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        stats["rejected_decode"] += 1
        return 0
    stats["bytes_in"] += len(text.encode("utf-8"))
    if len(text) < 120:
        stats["rejected_short"] += 1
        return 0
    if is_minified(text):
        stats["rejected_minified"] += 1
        return 0
    written = 0
    for piece in chunk_code(text, 400):
        stats["chunks_seen"] += 1
        if not has_valid_structure(piece, "python"):
            stats["rejected_syntax"] += 1
            continue
        ok, _ = quality_check(piece, 150, 120_000, 0.30)
        if not ok:
            stats["rejected_quality"] += 1
            continue
        if not dedup.add(piece):
            stats["rejected_dup"] += 1
            continue
        if not fuzzy.add(piece):
            stats["rejected_fuzzy"] += 1
            continue
        clean = scrub_pii(piece)
        out.write(json.dumps({
            "text": clean,
            "language": "python",
            "source": attribution(p, kind),
        }, ensure_ascii=False) + "\n")
        written += 1
        stats["bytes_out"] += len(clean.encode("utf-8"))
    stats["kept"] += written
    if written:
        stats["files_yielded"] += 1
        stats["total_sources"] += 1
        stats["by_source_kind"][kind] = stats["by_source_kind"].get(kind, 0) + 1
    return written


def verify_benchmark_clean(path: Path) -> None:
    """Fail closed if any frozen-benchmark marker reached the corpus."""
    hits: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            for marker in FROZEN_MARKERS:
                if marker in line:
                    hits.append(f"line {lineno}: {marker}")
                    break
            if len(hits) >= 5:
                break
    if hits:
        raise SystemExit(
            "refusing to publish contaminated corpus; frozen-benchmark markers found: "
            + "; ".join(hits)
        )


def main():
    with CORPUS.open("w", encoding="utf-8") as out:
        seen_files: set[Path] = set()
        for p in iter_repo_py_files():
            if p.resolve() in seen_files:
                continue
            seen_files.add(p.resolve())
            harvest_file(p, "repo", out)
        for p in iter_pip_py_files():
            if p.resolve() in seen_files:
                continue
            seen_files.add(p.resolve())
            harvest_file(p, "pip", out)

    verify_benchmark_clean(CORPUS)
    h = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    stats["corpus_sha256"] = h
    stats["size_bytes"] = CORPUS.stat().st_size
    MANIFEST.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"files_seen={stats['files_seen']} excluded={stats['files_excluded']} "
        f"yielded={stats['files_yielded']} kept={stats['kept']} "
        f"sources={stats['total_sources']} bytes_in={stats['bytes_in'] // 2**20}MB "
        f"out={stats['size_bytes'] // 2**20}MB"
    )
    print(f"benchmark-clean: PASS; manifest: {MANIFEST} sha256={h}")


if __name__ == "__main__":
    main()
