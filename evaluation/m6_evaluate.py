"""M6 evaluation integration.

Bridges the M6 runtime contract onto the **existing** M5 harness. Scoring,
resume, result persistence and manifest logic all come from ``m5_runner`` and
``m5_scorers``; nothing here reimplements them. This module's only job is to
present an M6 runtime in the shape ``m5_runner.evaluate`` already expects, and
to attach M6-specific provenance to the resulting run manifest.

A candidate that never produced output is recorded as blocked. It is never
reported as a zero score.

    python -m evaluation.m6_evaluate --candidate quantsmind/m3-base \
        --checkpoint runs/m3_base/checkpoint_best.pt --out-dir runs/m6/m3_base
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m5_adapters import AdapterInfo
from evaluation.m6_capability import compliance_evidence
from evaluation.m6_runtime import (GenerationRequest, LocalModel,
                                   RuntimeUnavailable)

M6_RUN_SCHEMA = "quantsmind.m6_run/1"
DEFAULT_BENCHMARK = "evaluation/m5_benchmark.jsonl"


class M5BridgeAdapter:
    """Presents an M6 :class:`LocalModel` through the M5 adapter contract.

    ``m5_runner.evaluate`` calls ``info()`` once and ``generate(prompt, ...)``
    per example. Those two methods are all that is needed to reuse the entire
    harness, including its scoring and resume behaviour.
    """

    def __init__(self, runtime: LocalModel, license_record=None,
                 artifact_sha256=None, revision=None):
        self.runtime = runtime
        self.license_record = license_record or {}
        self.artifact_sha256 = artifact_sha256
        self.revision = revision
        self.model_id = getattr(runtime, "model_id", None) or "unknown"

    def info(self) -> AdapterInfo:
        d = self.runtime.describe()
        return AdapterInfo(
            adapter=self.runtime.runtime_name,
            model_id=d.get("model_id") or self.model_id,
            family=d.get("task_head", "unknown"),
            tokenizer_id=d.get("tokenizer") or d.get("tokenizer_path"),
            checkpoint_path=d.get("artifact_path") or d.get("local_path"),
            artifact_sha256=self.artifact_sha256 or d.get("artifact_sha256"),
            revision=d.get("revision") or self.revision,
            parameters=d.get("parameters"),
            context_length=d.get("context_window"),
            vocab_size=d.get("vocab_size"),
            parameters_source="runtime",
            license_id=(self.license_record.get("license_id")
                        or d.get("license_id")),
            license_verified=bool(self.license_record.get("license_verified")),
            extra={"task_head": d.get("task_head"),
                   "hosted_api_used": False},
        )

    def generate(self, prompt, max_new_tokens=96, greedy=True, temperature=0.0,
                 top_k=0, top_p=1.0, seed=None, stop=None) -> dict:
        request = GenerationRequest(
            prompt=prompt, max_new_tokens=max_new_tokens, do_sample=not greedy,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
            stop=list(stop or []))
        response = self.runtime.generate(request)
        # Translate back into the dict shape m5_runner persists.
        return {
            "text": response.text,
            "token_ids": None,
            "finish_reason": response.finish_reason,
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "seconds": response.duration_seconds,
            "tokens_per_second": (
                round(response.output_tokens / response.duration_seconds, 2)
                if response.output_tokens and response.duration_seconds
                else None),
            "unsupported_settings": response.unsupported_settings,
            "template_applied": response.template_applied,
        }

    def release(self) -> None:
        self.runtime.close()


def validate_benchmark(benchmark_path, verified_sources=None) -> dict:
    """Validate the frozen M5 benchmark before spending compute on it."""
    from evaluation.m5_schema import validate_dataset

    return validate_dataset(str(benchmark_path),
                            verified_sources=verified_sources,
                            require_provenance=True)


def leakage_report(benchmark_path) -> dict:
    """Reuse the M5 leakage detector against the prior evaluation sets."""
    from evaluation import m5_leakage

    fn = getattr(m5_leakage, "leakage_report", None)
    if fn is None:
        return {"available": False,
                "reason": "no leakage_report entry point in evaluation.m5_leakage"}
    references = m5_leakage.default_reference_paths()
    return fn(str(benchmark_path), references)


def per_task_summary(manifest) -> dict:
    """Per-task breakdown taken from the M5 manifest, not recomputed."""
    by_task = manifest.get("by_task") or {}
    return {
        task: {name: {"numerator": m.get("numerator"),
                      "denominator": m.get("denominator"),
                      "rate": m.get("rate")}
               for name, m in metrics.items()}
        for task, metrics in by_task.items()}


def read_results(results_file) -> list[dict]:
    """Example records persisted by ``m5_runner``. Unreadable lines are skipped."""
    path = Path(results_file) if results_file else None
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def evaluate_candidate(runtime: LocalModel, out_dir, *, candidate_id=None,
                       license_record=None, benchmark=DEFAULT_BENCHMARK,
                       max_new_tokens=96, greedy=True, resume=True,
                       label=None, benchmark_validation=None,
                       leakage=None) -> dict:
    """Run one model over the frozen M5 benchmark via the M5 harness."""
    from evaluation import m5_runner

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = candidate_id or getattr(runtime, "model_id", "unknown")

    # 1. The benchmark must be valid before anything is generated.
    validation = benchmark_validation or validate_benchmark(benchmark)
    rejected = len(validation.get("rejections", []))
    if validation.get("accepted", 0) == 0:
        return record_blocked(
            out_dir, candidate_id,
            f"benchmark {benchmark} has no acceptable records "
            f"({rejected} rejected)")

    # 2. Leakage is recorded, never silently ignored.
    leak = leakage if leakage is not None else leakage_report(benchmark)

    # 3. Preflight before loading any weights.
    ok, reason = runtime.preflight()
    if not ok:
        return record_blocked(out_dir, candidate_id, reason, benchmark)

    # The M5 harness calls ``info()`` before generating but never loads; the M5
    # CLI loads explicitly. M6 must do the same, or every run blocks on
    # "model not loaded".
    runtime.load()

    described = runtime.describe()
    bridge = M5BridgeAdapter(
        runtime, license_record=license_record,
        artifact_sha256=described.get("artifact_sha256"),
        revision=described.get("revision"))

    started = datetime.now(timezone.utc).replace(microsecond=0)
    manifest = m5_runner.evaluate(
        bridge, benchmark, out_dir, max_new_tokens=max_new_tokens,
        greedy=greedy, resume=resume, label=label)
    finished = datetime.now(timezone.utc).replace(microsecond=0)
    evidence = compliance_evidence(read_results(manifest.get("results_file")))

    manifest["m6"] = {
        "schema": M6_RUN_SCHEMA,
        "candidate_id": candidate_id,
        "task_head": described.get("task_head"),
        "runtime": described.get("runtime"),
        "revision": described.get("revision"),
        "license_id": (license_record or {}).get("license_id"),
        "license_verified": bool((license_record or {}).get("license_verified")),
        "artifact_sha256": described.get("artifact_sha256"),
        "hosted_api_used": False,
        "credentials_used": False,
        "benchmark_validation": {
            "accepted": validation.get("accepted"),
            "rejected": rejected,
            "rejections": validation.get("rejections", []),
        },
        "leakage": leak,
        "compliance_evidence": evidence,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
    }
    manifest["per_task"] = per_task_summary(manifest)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def record_blocked(out_dir, candidate_id, reason, benchmark=None) -> dict:
    """Persist a blocked run. Never reports evaluation, never a zero."""
    from evaluation.m5_runner import write_blocked_run

    manifest = write_blocked_run(out_dir, candidate_id, reason, benchmark)
    manifest["m6"] = {
        "schema": M6_RUN_SCHEMA,
        "candidate_id": candidate_id,
        "evaluated": False,
        "blocked_reason": reason,
        "hosted_api_used": False,
    }
    Path(out_dir, "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_runtime(candidate_id, *, checkpoint=None, license_record=None,
                  local_path=None, tokenizer=None, revision=None):
    """Construct the right runtime for a candidate id."""
    from evaluation.m6_runtime import HFLocalRuntime, SovereignCheckpointRuntime

    if candidate_id.startswith("quantsmind/") or checkpoint:
        if not checkpoint:
            raise RuntimeUnavailable(
                f"{candidate_id} is a sovereign candidate and needs --checkpoint")
        return SovereignCheckpointRuntime(checkpoint, tokenizer_path=tokenizer)
    return HFLocalRuntime(candidate_id, revision=revision,
                          license_record=license_record, local_path=local_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an M6 candidate over the frozen M5 benchmark")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path for a quantsmind/* candidate")
    parser.add_argument("--local-path", default=None,
                        help="explicit local snapshot directory")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--license-id", default=None)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        default=True)
    args = parser.parse_args(argv)

    license_record = {}
    if args.candidate.startswith("quantsmind/"):
        license_record = {"license_id": "project-owned",
                          "license_verified": True}
    else:
        from evaluation.m6_licensing import lookup
        reg = lookup(args.candidate) or {}
        license_record = {
            "license_id": args.license_id or reg.get("license_id"),
            "license_verified": bool(args.license_id or
                                     reg.get("license_verified")),
        }

    try:
        runtime = build_runtime(args.candidate, checkpoint=args.checkpoint,
                                license_record=license_record,
                                local_path=args.local_path,
                                tokenizer=args.tokenizer)
    except RuntimeUnavailable as exc:
        print(f"cannot build runtime: {exc}")
        return 2

    manifest = evaluate_candidate(
        runtime, args.out_dir, candidate_id=args.candidate,
        license_record=license_record, benchmark=args.benchmark,
        max_new_tokens=args.max_new_tokens, greedy=not args.sample,
        resume=args.resume, label=args.label)

    status = manifest.get("status")
    print(f"candidate : {args.candidate}")
    print(f"status    : {status}")
    if status == "blocked":
        print(f"blocked   : {manifest.get('m6', {}).get('blocked_reason')}")
        return 3
    print(f"examples  : {manifest.get('examples_evaluated')}"
          f"/{manifest.get('examples_total')}")
    print(f"hash      : {(manifest.get('benchmark_manifest_hash') or '')[:16]}")
    for name, metric in sorted((manifest.get("summary") or {}).items()):
        rate = metric["rate"]
        text = "n/a" if rate is None else f"{rate:.1%}"
        flag = "  [PRELIMINARY]" if metric.get("preliminary") else ""
        print(f"  {name:12} {metric['numerator']}/{metric['denominator']}"
              f" = {text}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())