"""Compare baseline and candidate checkpoints on fixed held-out suites.

This script evaluates raw model output only (`binding_repair=False`), so it
measures the checkpoint rather than the post-processing repair. Arbitrary
generated code is never executed by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.functional_eval import (
    FunctionalResult,
    classify_failure,
    functional_report,
)
from training.curriculum_quality import detect_mode_collapse, diversity_report
from training.program_analysis import analyse

TOKENIZER_PATH = REPO / "tokenizer" / "qm_code_bpe_v0.json"
DEFAULT_AD1 = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
DEFAULT_C3 = REPO / "evaluation" / "python_benchmark_heldout_v1.json"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def score_ad1_case(case, completion: str) -> dict:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], _indent(completion))
    exact = bool(generated["parses"] and expected["parses"] and generated["body_dump"] == expected["body_dump"])
    structural = bool(generated["parses"] and expected["operations"] and set(expected["operations"]).issubset(set(generated["operations"])))
    binding = bool(
        generated["parses"]
        and generated["params_used"] == expected["params_used"]
        and not generated["undefined_names"]
    )
    if case.get("negative"):
        exact = False
        structural = False
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "parses": bool(generated["parses"]),
        "exact": exact,
        "structural": structural,
        "binding": binding,
        "operations": sorted(generated["operations"]),
        "undefined_names": generated["undefined_names"],
        "failure_category": classify_failure(case["signature"], case["reference_body"], completion),
        "completion": completion,
    }


def score_c3_case(case, completion: str) -> dict:
    from scripts.run_binding_repair_qualification import score_static

    result = score_static(case, completion)
    functional = {"passed": None, "reason": "not_executed"}
    if case.get("execution_safe") and not case.get("negative"):
        from evaluation.safe_execute import execute_tests

        if completion.lstrip().startswith("def "):
            source = completion
        else:
            source = case["signature"] + "\n" + _indent(completion)
        execution = execute_tests(source, case["function_name"], case["tests"])
        functional = {"passed": execution.passed, "reason": execution.reason}
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "completion": completion,
        **result,
        "functional": functional,
        "failure_category": classify_failure(
            case["signature"],
            case["reference_body"],
            completion,
            functional=(
                FunctionalResult(functional["passed"], "pass" if functional["passed"] else "fail", functional["reason"])
                if functional["passed"] is not None
                else None
            ),
        ),
    }


def aggregate(ad1_records, c3_records) -> dict:
    """Summarise scored records, including template diversity and collapse.

    The original parse/exact/structural/binding/functional keys are
    unchanged (Increment 2 extension only adds ``ad1_diversity``,
    ``c3_diversity`` and ``mode_collapse``), so historical summaries stay
    comparable.
    """
    structural_cases = [r for r in c3_records if r["structural"] is not None]
    functional_cases = [r for r in c3_records if r["functional"]["passed"] is not None]
    return {
        "ad1_diversity": diversity_report(ad1_records),
        "c3_diversity": diversity_report(c3_records),
        # Increment 4: separate functional layer, never merged into one score.
        "functional_analysis": {
            "ad1": functional_report(ad1_records),
            "c3": functional_report(c3_records),
        },
        "mode_collapse": {
            "ad1": detect_mode_collapse(ad1_records),
            "c3": detect_mode_collapse(c3_records),
        },
        "ad1": {
            "cases": len(ad1_records),
            "parses": sum(bool(r["parses"]) for r in ad1_records),
            "exact": sum(bool(r["exact"]) for r in ad1_records),
            "structural": sum(bool(r["structural"]) for r in ad1_records),
            "binding": sum(bool(r["binding"]) for r in ad1_records),
        },
        "c3_heldout": {
            "cases": len(c3_records),
            "parses": sum(bool(r["parses"]) for r in c3_records),
            "exact": sum(bool(r["exact"]) for r in c3_records),
            "structural": sum(r["structural"] is True for r in structural_cases),
            "structural_cases": len(structural_cases),
            "binding": sum(bool(r["binding"]) for r in c3_records),
            "functional_passed": sum(r["functional"]["passed"] is True for r in functional_cases),
            "functional_cases": len(functional_cases),
        },
        "latency": {
            "ad1_mean_seconds": round(statistics.mean(r["latency_seconds"] for r in ad1_records), 4) if ad1_records else None,
            "c3_mean_seconds": round(statistics.mean(r["latency_seconds"] for r in c3_records), 4) if c3_records else None,
        },
    }


def evaluate_checkpoint(checkpoint: Path, *, ad1_cases, c3_cases, tokenizer):
    import torch
    from serving.local_inference import generate_ids, load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    ad1_records = []
    c3_records = []
    try:
        for case in ad1_cases:
            prompt = case["instruction"] + "\n" + case["signature"] + "\n"
            started = time.perf_counter()
            result = generate_ids(
                model,
                tokenizer,
                prompt,
                max_new_tokens=64,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                device="cpu",
                seed=20261023,
                binding_repair=False,
            )
            elapsed = time.perf_counter() - started
            record = score_ad1_case(case, result["text"])
            record["latency_seconds"] = round(elapsed, 6)
            ad1_records.append(record)
        for case in c3_cases:
            prompt = case["instruction"] + "\n" + case["signature"] + "\n"
            started = time.perf_counter()
            result = generate_ids(
                model,
                tokenizer,
                prompt,
                max_new_tokens=64,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                device="cpu",
                seed=20261023,
                binding_repair=False,
            )
            elapsed = time.perf_counter() - started
            record = score_c3_case(case, result["text"])
            record["latency_seconds"] = round(elapsed, 6)
            c3_records.append(record)
    finally:
        del model
        gc.collect()
    return ad1_records, c3_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1.2.0 Increment 1 candidate")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--ad1", default=str(DEFAULT_AD1))
    parser.add_argument("--c3", default=str(DEFAULT_C3))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ad1_suite = json.loads(Path(args.ad1).read_text(encoding="utf-8"))
    c3_suite = json.loads(Path(args.c3).read_text(encoding="utf-8"))
    baseline_ad1, baseline_c3 = evaluate_checkpoint(
        Path(args.baseline), ad1_cases=ad1_suite["cases"], c3_cases=c3_suite["cases"], tokenizer=tokenizer
    )
    candidate_ad1, candidate_c3 = evaluate_checkpoint(
        Path(args.candidate), ad1_cases=ad1_suite["cases"], c3_cases=c3_suite["cases"], tokenizer=tokenizer
    )
    result = {
        "baseline_checkpoint": str(args.baseline),
        "candidate_checkpoint": str(args.candidate),
        "ad1": {"baseline": baseline_ad1, "candidate": candidate_ad1},
        "c3": {"baseline": baseline_c3, "candidate": candidate_c3},
        "summary": {
            "baseline": aggregate(baseline_ad1, baseline_c3),
            "candidate": aggregate(candidate_ad1, candidate_c3),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
