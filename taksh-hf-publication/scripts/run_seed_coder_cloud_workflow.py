"""Reproducible Seed-Coder cloud execution workflow (D3.1).

Default mode prints a plan and performs no execution. Real execution requires
--execute, a local model directory, a frozen suite, and passing preflight.
It never downloads weights, calls hosted services, or falls back to S2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.preflight_seed_coder_cloud import preflight
from scripts.run_provider_qualification import (
    build_provider,
    canonical_digest,
    evaluate_case,
    file_sha256,
    summarize,
)
from serving.providers import qualification_metadata

SEED_CODER_PROVIDER = "seed-coder-8b-instruct"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan(args) -> dict:
    return {
        "mode": "plan_only",
        "provider": SEED_CODER_PROVIDER,
        "model_path": str(args.model_path) if args.model_path else None,
        "suite": str(args.suite),
        "expected_suite_sha256": args.expected_sha,
        "output_directory": str(args.out_dir),
        "resource_controls": {
            "max_new_tokens": args.max_new_tokens,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "timeout_seconds": args.timeout,
            "device": args.device,
            "dtype": args.dtype,
        },
        "execution": {
            "steps": [
                "environment verification",
                "local model artifact availability check",
                "Seed-Coder initialization",
                "small generation smoke test",
                "AD-1 evaluation",
                "raw versus binding-repair comparison",
                "result and log export",
                "runtime shutdown and cleanup",
            ],
            "automatic_downloads": False,
            "provider_fallback": False,
            "hosted_inference": False,
        },
    }


def _run_one(
    provider,
    suite,
    *,
    binding_repair: bool,
    args,
    partial_records=None,
):
    records = []
    started = time.perf_counter()
    try:
        for case in suite["cases"]:
            records.append(
                evaluate_case(
                    provider,
                    case,
                    binding_repair=binding_repair,
                    timeout_seconds=args.timeout,
                    max_new_tokens=args.max_new_tokens,
                )
            )
    except KeyboardInterrupt:
        if partial_records is not None:
            partial_records.extend(records)
        raise
    return records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-Coder cloud execution workflow")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--expected-artifact-sha256", default=None)
    parser.add_argument("--expected-config-sha256", default=None)
    parser.add_argument("--expected-tokenizer-sha256", default=None)
    args = parser.parse_args()

    if not (1 <= args.max_new_tokens <= 512):
        raise SystemExit("REFUSED: max_new_tokens must be between 1 and 512")
    if not (1 <= args.context_length <= 32768):
        raise SystemExit("REFUSED: context_length must be between 1 and 32768")
    if args.batch_size != 1:
        raise SystemExit("REFUSED: Seed-Coder cloud workflow supports batch_size=1 only")
    if not args.out_dir:
        raise SystemExit("REFUSED: --out-dir is required")

    if not args.execute:
        print(json.dumps(_plan(args), indent=2))
        return

    suite_path = Path(args.suite)
    out_dir = Path(args.out_dir)
    suite_sha = file_sha256(suite_path)
    if suite_sha != args.expected_sha.lower():
        raise SystemExit("REFUSED: suite SHA-256 does not match expected hash")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if canonical_digest(suite["cases"]) != suite["digest"]:
        raise SystemExit("REFUSED: suite canonical digest changed")

    preflight_report = preflight(
        model_path=Path(args.model_path) if args.model_path else None,
        expected_config_sha256=args.expected_config_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    _write_json(out_dir / "logs" / "preflight.json", preflight_report)
    if not preflight_report["ready"]:
        raise SystemExit("REFUSED: cloud preflight failed; see logs/preflight.json")

    provider = build_provider(
        SEED_CODER_PROVIDER,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )
    metadata = qualification_metadata(
        provider,
        suite_version=suite["version"],
        suite_sha256=suite_sha,
    )
    if hasattr(provider, "artifact_hashes"):
        metadata["artifact_hashes"] = provider.artifact_hashes()
    _write_json(out_dir / "provider_metadata.json", metadata)

    started = _now()
    records = {"raw": [], "repaired": []}
    status = "completed"
    try:
        records["raw"] = _run_one(provider, suite, binding_repair=False, args=args)
        records["repaired"] = _run_one(provider, suite, binding_repair=True, args=args)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        result = {
            "status": status,
            "provider_qualification_version": "1.0.0",
            "suite": suite["suite"],
            "suite_version": suite["version"],
            "suite_sha256": suite_sha,
            "suite_digest": suite["digest"],
            "provider_metadata": metadata,
            "resource_controls": {
                "max_new_tokens": args.max_new_tokens,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "timeout_seconds": args.timeout,
                "device": args.device,
                "dtype": args.dtype,
            },
            "started_utc": started,
            "finished_utc": _now(),
            "records": records,
            "summary": {
                "raw": summarize(records["raw"]),
                "repaired": summarize(records["repaired"]),
            },
        }
        _write_json(out_dir / "results.json", result)
        _write_json(out_dir / "summary.json", result["summary"])
        _write_json(
            out_dir / "run_manifest.json",
            {
                "status": status,
                "started_utc": started,
                "finished_utc": result["finished_utc"],
                "suite_sha256": suite_sha,
                "provider": metadata["provider_identity"],
                "resource_controls": result["resource_controls"],
                "hosted_inference": False,
                "automatic_downloads": False,
                "provider_fallback": False,
            },
        )
    print(json.dumps({"out": str(out_dir), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
