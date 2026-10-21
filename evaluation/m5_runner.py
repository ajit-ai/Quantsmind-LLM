"""Deterministic M5 evaluation runner.

One command produces one run directory containing:

* ``results.jsonl``    -- one record per example, appended as it completes
* ``run_manifest.json`` -- identity, environment, benchmark hash, timings
* ``summary.json``      -- aggregated metrics with explicit denominators

Properties that matter for auditability on a constrained machine:

* **Resumable.** Completed examples are skipped on re-run, so one failure does
  not cost a full re-evaluation.
* **Selective.** ``--only`` re-runs named example ids without destroying others.
* **Honest about blocking.** A model that cannot be loaded produces a run
  directory marked ``blocked`` with the reason -- never an empty success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m5_adapters import AdapterUnavailable, system_snapshot
from evaluation.m5_schema import SCHEMA, load, manifest_hash
from evaluation.m5_scorers import SCORING_VERSION, aggregate, score_example

RUN_SCHEMA = "quantsmind.m5_run/1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_state() -> dict:
    """Commit and dirty state, or an honest 'unavailable'."""
    def run(args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True,
                                 timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    commit = run(["rev-parse", "HEAD"])
    dirty = run(["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty_tree": bool(dirty) if dirty is not None else None,
        "note": None if commit else "git unavailable; commit not recorded",
    }


def existing_results(path) -> dict:
    """Completed example ids, so a resumed run does not redo work."""
    done = {}
    path = Path(path)
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated final line from an interrupted run
        if record.get("complete"):
            done[record["example_id"]] = record
    return done


def _append(path, entry):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def evaluate(adapter, benchmark_path, out_dir, max_new_tokens=96, greedy=True,
             temperature=0.0, top_k=0, top_p=1.0, seed=11, resume=True,
             only=None, label=None, notes=None):
    """Run one model over the benchmark and return the run manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    records = load(benchmark_path)
    bench_hash = manifest_hash(records)
    done = existing_results(results_path) if resume else {}
    if not resume and results_path.exists():
        # A fresh run must not inherit records from a previous attempt. Leaving
        # them in place would double every count in results.jsonl while the
        # manifest reported only this run's examples.
        results_path.unlink()
    if only:
        wanted = set(only)
        done = {k: v for k, v in done.items() if k not in wanted}

    info = adapter.info()
    generation_config = {
        "strategy": "greedy" if greedy else "sampling",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature, "top_k": top_k, "top_p": top_p,
        "seed": seed,
        "note": ("greedy by default so sampling noise cannot decide a "
                 "before/after comparison"),
    }

    run_id = label or f"m5-{Path(str(adapter.model_id)).stem}"
    started = utc_now_iso()
    t0 = time.perf_counter()
    completed = list(done.values())
    errors = []

    for record in records:
        if record["id"] in done:
            continue
        if only and record["id"] not in set(only):
            continue
        entry = {
            "example_id": record["id"],
            "task_type": record.get("task_type"),
            "language": record.get("language"),
            "prompt": record["prompt"],
            "source_id": record.get("source_id"),
            "model_id": adapter.model_id,
            "complete": False,
        }
        try:
            output = adapter.generate(
                record["prompt"], max_new_tokens=max_new_tokens, greedy=greedy,
                temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - one bad example must not kill the run
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["complete"] = True
            errors.append({"example_id": record["id"], "error": entry["error"]})
            completed.append(entry)
            _append(results_path, entry)
            continue

        entry.update({
            "output": output["text"],
            "output_tokens": output.get("completion_tokens"),
            "finish_reason": output.get("finish_reason"),
            "generation_seconds": output.get("seconds"),
            "tokens_per_second": output.get("tokens_per_second"),
            "metrics": score_example(record, output["text"]),
            "complete": True,
        })
        completed.append(entry)
        _append(results_path, entry)

    elapsed = time.perf_counter() - t0
    with_metrics = [e for e in completed if e.get("metrics")]
    by_task = {}
    for entry in with_metrics:
        by_task.setdefault(entry["task_type"], []).append(entry["metrics"])

    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "benchmark_schema": SCHEMA,
        "benchmark_path": str(benchmark_path),
        "benchmark_manifest_hash": bench_hash,
        "scoring_version": SCORING_VERSION,
        "started_at": started,
        "completed_at": utc_now_iso(),
        "elapsed_seconds": round(elapsed, 2),
        "status": "complete" if not errors else "complete_with_errors",
        "complete": True,
        "examples_total": len(records),
        "examples_evaluated": len(with_metrics),
        "examples_reused_from_previous_run": len(done),
        "errors": errors,
        "generation_config": generation_config,
        "model": info.as_dict(),
        "environment": system_snapshot(),
        "git": git_state(),
        "results_file": str(results_path),
        "summary": aggregate([e["metrics"] for e in with_metrics]),
        "by_task": {task: aggregate(metrics)
                    for task, metrics in sorted(by_task.items())},
        "notes": list(notes or []),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(manifest["summary"], indent=2), encoding="utf-8")
    return manifest


def write_blocked_run(out_dir, model_id, reason, benchmark_path=None):
    """A run that never executed, recorded as blocked rather than empty success."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": f"m5-blocked-{Path(str(model_id)).stem}",
        "status": "blocked",
        "complete": False,
        "evaluated": False,
        "model_id": model_id,
        "blocker": reason,
        "started_at": utc_now_iso(),
        "completed_at": utc_now_iso(),
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "environment": system_snapshot(),
        "git": git_state(),
        "note": ("No inference was performed. This run carries no scores and "
                 "must not be compared against a completed run."),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_adapter(args):
    from evaluation.m5_adapters import ExternalHFAdapter, QMCkptAdapter

    if args.adapter == "qm":
        if not args.ckpt:
            raise SystemExit("--adapter qm requires --ckpt")
        return QMCkptAdapter(args.ckpt, tokenizer_path=args.tokenizer)
    if not args.model_id:
        raise SystemExit("--adapter external requires --model-id")
    licences = {}
    record_path = Path("configs/evaluation/candidate_licences.json")
    if record_path.exists():
        licences = json.loads(record_path.read_text(encoding="utf-8"))
    return ExternalHFAdapter(
        args.model_id, license_record=licences.get(args.model_id, {}))


def print_summary(manifest) -> None:
    print(f"\nrun: {manifest['run_id']}  status={manifest['status']}")
    print(f"  examples : {manifest['examples_evaluated']}/{manifest['examples_total']}")
    print(f"  hash     : {manifest['benchmark_manifest_hash'][:16]}...")
    print(f"  elapsed  : {manifest['elapsed_seconds']}s")
    for name, metric in manifest["summary"].items():
        rate = metric["rate"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        print(f"  {name:12} {metric['numerator']}/{metric['denominator']} "
              f"= {rate_text}"
              + ("  [PRELIMINARY]" if metric["preliminary"] else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the M5 evaluation")
    parser.add_argument("--adapter", required=True, choices=["qm", "external"])
    parser.add_argument("--ckpt", help="checkpoint path for the qm adapter")
    parser.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    parser.add_argument("--model-id", help="HF model id for the external adapter")
    parser.add_argument("--benchmark", default="evaluation/m5_benchmark.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sample", action="store_true",
                        help="use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        default=True)
    parser.add_argument("--only", action="append", default=None,
                        help="re-run only these example ids")
    args = parser.parse_args(argv)

    try:
        adapter = build_adapter(args)
    except SystemExit as exc:
        print(exc)
        return 2

    try:
        ok, reason = adapter.preflight()
        print(f"preflight: {'OK' if ok else 'BLOCKED'} - {reason}")
        if not ok:
            write_blocked_run(args.out_dir, getattr(adapter, "model_id", "?"),
                              reason, args.benchmark)
            print(f"blocked run recorded: {args.out_dir}/run_manifest.json")
            return 3
        adapter.load()
    except AdapterUnavailable as exc:
        print(f"adapter unavailable: {exc}")
        write_blocked_run(args.out_dir, getattr(adapter, "model_id", "?"),
                          str(exc), args.benchmark)
        return 3

    try:
        manifest = evaluate(
            adapter, args.benchmark, args.out_dir,
            max_new_tokens=args.max_new_tokens, greedy=not args.sample,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            seed=args.seed, resume=args.resume, only=args.only, label=args.label,
        )
    finally:
        adapter.release()

    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())