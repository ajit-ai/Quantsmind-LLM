"""Offline provider qualification harness.

Records provider identity and metrics for the frozen suites. Only the built-in
Taksh S2 provider can execute checkpoints. Foundation placeholders report an
unavailable status and never download weights or call hosted endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.safe_execute import execute_tests
from scripts.run_binding_repair_qualification import (
    canonical_digest,
    file_sha256,
    score_static,
)
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    ProviderTimeoutError,
    ProviderUnavailableError,
    get_provider,
    qualification_metadata,
)

DEFAULT_CHECKPOINT = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
DEFAULT_TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"
EXPECTED_CHECKPOINT = "02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac"


def source_for(case, completion: str) -> tuple[str, str]:
    if completion.lstrip().startswith("def "):
        return completion, "full_function"
    from training.generalization_eval import _indent

    return case["signature"] + "\n" + _indent(completion), "body"


def evaluate_case(
    provider,
    case,
    *,
    binding_repair: bool,
    timeout_seconds: float,
    max_new_tokens: int = 64,
    top_k: int = 1,
    top_p: float = 1.0,
    seed: int = 20261023,
):
    prompt = case["instruction"] + "\n" + case["signature"] + "\n"
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        timeout_seconds=timeout_seconds,
        binding_repair=binding_repair,
        language="python",
    )
    request = GenerationRequest(prompt=prompt, config=config, metadata={"case_id": case["case_id"]})
    started = time.perf_counter()
    try:
        from serving.providers import generate_with_timeout

        result = generate_with_timeout(provider, request)
    except (ProviderUnavailableError, ProviderTimeoutError) as exc:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "negative": bool(case.get("negative")),
            "prompt": prompt,
            "completion": "",
            "raw_text": "",
            "provider_status": "unavailable" if isinstance(exc, ProviderUnavailableError) else "timeout",
            "provider_error": {"type": type(exc).__name__, "message": str(exc)},
            "binding_repair": binding_repair,
            "latency_seconds": time.perf_counter() - started,
            "parses": False,
            "exact": False,
            "structural": None,
            "binding": False,
            "functional": {"passed": None, "reason": "provider_not_executed"},
            "repair": {},
        }

    completion = result.text
    record = {
        "case_id": case["case_id"],
        "category": case["category"],
        "negative": bool(case.get("negative")),
        "prompt": prompt,
        "completion": completion,
        "raw_text": result.raw_text,
        "provider_status": "completed",
        "binding_repair": binding_repair,
        "finish_reason": result.finish_reason,
        "tokens": result.completion_tokens,
        "latency_seconds": result.latency_seconds,
        **score_static(case, completion),
        "functional": {"passed": None, "reason": "not_executed"},
        "repair": result.repair,
    }
    if case.get("execution_safe") and not case.get("negative"):
        source, form = source_for(case, completion)
        execution = execute_tests(source, case["function_name"], case["tests"])
        record["functional"] = {
            "passed": execution.passed,
            "reason": execution.reason,
            "source_form": form,
        }
    return record


def summarize(records):
    attempted = [record for record in records if record["provider_status"] == "completed"]
    functional = [record for record in attempted if record["functional"]["passed"] is not None]
    structural = [record for record in attempted if record["structural"] is not None]
    return {
        "cases": len(records),
        "completed": len(attempted),
        "unavailable": sum(1 for record in records if record["provider_status"] == "unavailable"),
        "timed_out": sum(1 for record in records if record["provider_status"] == "timeout"),
        "parses": sum(bool(record["parses"]) for record in attempted),
        "exact": sum(bool(record["exact"]) for record in attempted),
        "structural": sum(record["structural"] is True for record in structural),
        "structural_cases": len(structural),
        "binding": sum(bool(record["binding"]) for record in attempted),
        "functional_passed": sum(record["functional"]["passed"] is True for record in functional),
        "functional_cases": len(functional),
        "abstentions": sum(1 for record in attempted if record.get("repair", {}).get("status", "").startswith("abstained")),
        "repairs": sum(1 for record in attempted if record.get("repair", {}).get("status") == "repaired"),
        "negative_cases": sum(1 for record in records if record["negative"]),
        "negative_safe": sum(
            1 for record in attempted
            if record["negative"] and not record["exact"] and record["structural"] is not True
        ),
    }


def build_provider(
    provider_name: str,
    *,
    model_path: str | None = None,
    revision: str | None = None,
    device: str = "cpu",
    dtype: str = "auto",
    expected_artifact_sha256: str | None = None,
):
    if provider_name != "taksh-s2":
        if provider_name == "seed-coder-8b-instruct":
            kwargs = {}
            if model_path is not None:
                kwargs["model_path"] = model_path
            if revision is not None:
                kwargs["revision"] = revision
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if expected_artifact_sha256 is not None:
                kwargs["expected_artifact_sha256"] = expected_artifact_sha256
            return get_provider(provider_name, **kwargs)
        return get_provider(provider_name)
    checkpoint_sha = file_sha256(DEFAULT_CHECKPOINT)
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise SystemExit("REFUSED: checkpoint hash does not match frozen S2")
    from tokenizers import Tokenizer

    from serving.local_inference import load_model

    model, _config = load_model(str(DEFAULT_CHECKPOINT), "cpu")
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    return get_provider(
        "taksh-s2",
        model=model,
        tokenizer=tokenizer,
        model_id="taksh-s2",
        revision=checkpoint_sha,
        source=str(DEFAULT_CHECKPOINT),
        artifact_sha256=checkpoint_sha,
        tokenizer_path=str(DEFAULT_TOKENIZER),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline provider qualification")
    parser.add_argument("--provider", default="taksh-s2")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--binding-repair", choices=["true", "false"], default="true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--expected-artifact-sha256", default=None)
    parser.add_argument(
        "--out",
        default=r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\provider_qualification_results.json",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite)
    suite_sha = file_sha256(suite_path)
    if suite_sha != args.expected_sha.lower():
        raise SystemExit("REFUSED: suite SHA-256 does not match expected hash")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if canonical_digest(suite["cases"]) != suite["digest"]:
        raise SystemExit("REFUSED: suite canonical digest changed")

    provider = build_provider(
        args.provider,
        model_path=args.model_path,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )
    metadata = qualification_metadata(
        provider,
        suite_version=suite["version"],
        suite_sha256=suite_sha,
        config=GenerationConfig(binding_repair=(args.binding_repair == "true")),
    )
    if hasattr(provider, "artifact_hashes"):
        metadata["artifact_hashes"] = provider.artifact_hashes()
    records = [
        evaluate_case(
            provider,
            case,
            binding_repair=(args.binding_repair == "true"),
            timeout_seconds=args.timeout,
        )
        for case in suite["cases"]
    ]
    result = {
        "provider_qualification_version": "1.0.0",
        "suite": suite["suite"],
        "suite_version": suite["version"],
        "suite_sha256": suite_sha,
        "suite_digest": suite["digest"],
        "provider_metadata": metadata,
        "binding_repair": args.binding_repair == "true",
        "executed_fixture_cases_only": True,
        "records": records,
        "summary": summarize(records),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(output), "summary": result["summary"],
                      "provider_metadata": result["provider_metadata"]["provider_identity"]}, indent=2))


if __name__ == "__main__":
    main()
