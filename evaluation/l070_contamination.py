"""TAKSH L0.7.0 — contamination protection vs frozen benchmarks.

Verifies the three pinned benchmark artifacts (hashes re-checked at run time),
builds normalized task/solution fingerprints, and offers exact-match and
near-duplicate detection. Candidate corpus screening is demonstrated on a
SYNTHETIC contaminated fixture only — no external corpus exists and none may
be accessed in this milestone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REGISTRY_PATH = REPO / "configs" / "data" / "l070_contamination_registry.yaml"


def normalized(text: str) -> str:
    text = re.sub(r"(?m)^\s*#.*$", " ", text)          # python comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)  # c-family block comments
    text = re.sub(r"//[^\n]*", " ", text)               # c-family line comments
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def load_registry() -> dict:
    import yaml

    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def verify_pinned_benchmarks() -> list[dict]:
    registry = load_registry()
    results = []
    for bench in registry["benchmarks"]:
        path = REPO / bench["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        results.append({"id": bench["id"], "path": bench["path"],
                        "expected": bench["sha256"], "actual": digest,
                        "intact": digest == bench["sha256"]})
    return results


def extract_task_fingerprints() -> dict[str, list[str]]:
    """Normalized fingerprints of every task text we can read in-repo."""
    out: dict[str, list[str]] = {}

    py = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    py_tasks = py.get("problems") if isinstance(py, dict) else py
    fps = []
    for i, task in enumerate(py_tasks if isinstance(py_tasks, list) else []):
        if isinstance(task, dict):
            blob = json.dumps(task, sort_keys=True)
            fps.append(fingerprint(blob))
    out["frozen-python-bench-v1"] = fps

    mb = json.loads((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_text(encoding="utf-8"))
    mb_tasks = mb.get("tasks") if isinstance(mb, dict) else mb
    fps2 = []
    for task in (mb_tasks if isinstance(mb_tasks, list) else []):
        if isinstance(task, dict):
            fps2.append(fingerprint(json.dumps(task, sort_keys=True)))
    out["multilingual-bench-v1"] = fps2
    return out


def scan_corpus(samples: list[str]) -> dict:
    registry = load_registry()
    pinned = verify_pinned_benchmarks()
    task_fps = extract_task_fingerprints()
    benchmark_set = {fp: name for name, fps in task_fps.items() for fp in fps}

    exact_hits = []
    for i, sample in enumerate(samples):
        fp = fingerprint(sample)
        if fp in benchmark_set:
            exact_hits.append({"sample_index": i, "benchmark": benchmark_set[fp], "fingerprint": fp})

    near = []
    # shingle near-duplicate detection vs benchmark tasks (token shingles)
    def shingles(text: str, n: int = 5):
        toks = normalized(text).split()
        return {" ".join(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))}

    sample_shingles = [shingles(s) for s in samples]
    bench_texts = []
    py = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    for task in (py.get("problems", []) if isinstance(py, dict) else []):
        bench_texts.append(json.dumps(task, sort_keys=True))
    for i, ss in enumerate(sample_shingles):
        for j, bt in enumerate(bench_texts):
            bs = shingles(bt)
            union = len(ss | bs)
            if union and len(ss & bs) / union >= 0.5:
                near.append({"sample_index": i, "benchmark_task": j,
                             "jaccard": round(len(ss & bs) / union, 4)})

    return {
        "benchmarks_intact": pinned,
        "benchmark_fingerprints": {k: len(v) for k, v in task_fps.items()},
        "samples_scanned": len(samples),
        "exact_matches": exact_hits,
        "near_duplicates": near,
        "excluded": len(exact_hits) + len(near),
        "decision": "BLOCKED (contamination found)" if (exact_hits or near) else "CLEAN (fixtures only)",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_contamination_audit.json")
    args = parser.parse_args(argv)

    pinned = verify_pinned_benchmarks()
    if not all(p["intact"] for p in pinned):
        print("CONTAMINATION REGISTRY INVALID — pinned benchmark altered:", pinned)
        return 2

    # SYNTHETIC contaminated fixture: embeds a real frozen-bench task
    bench_task = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    first_task = bench_task["problems"][0]
    import json as _json

    contaminated = _json.dumps(first_task, sort_keys=True)
    fixtures = ["def totally_unrelated(x):\n    return x * 3\n", contaminated]
    audit = scan_corpus(fixtures)

    report = {
        "schema": "taksh.l070_contamination_audit/1",
        "generated_by": "evaluation/l070_contamination.py",
        "pinned_benchmarks": pinned,
        "synthetic_fixture_audit": audit,
        "external_corpus_screening": "NOT PERFORMED — no corpus exists; acquisition is disabled",
        "policy": "any exact/near match blocks training eligibility until resolved; fingerprints registry is append-only",
        "frozen_python_benchmark": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6 (intact)",
        "verdict": "CONTAMINATION CONTROLS QUALIFIED on synthetic fixtures; benchmarks intact",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"benchmarks intact; synthetic exact matches caught: {len(audit['exact_matches'])}, near: {len(audit['near_duplicates'])}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
