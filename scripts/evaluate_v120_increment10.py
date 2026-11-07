"""Increment 10 evaluation: candidate vs baseline (+ stored Inc5/Inc7 results).

Evaluates frozen identifier-copy probes (13), the supplemental behavior
suite (48), and frozen AD-1/C3 with identical greedy decoding. Writes a
machine-readable JSON report; never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.behavior_suite import build_cases
from evaluation.copy_eval import copy_report, score_copy_case
from evaluation.diagnostics import COPY_CASES, copy_case_bindings
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.stabilization import per_behavior_summary
from training.program_analysis import analyse
from scripts.evaluate_v120_increment1 import score_ad1_case, score_c3_case

TOKENIZER_PATH = REPO / "tokenizer" / "qm_code_bpe_v0.json"
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


def _generate(model, tokenizer, prompt):
    import torch  # noqa: F401
    from serving.local_inference import generate_ids

    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=False,
    )
    return result["text"]


def _score_behavior(case, completion: str) -> dict:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], _indent(completion))
    exact = bool(generated["parses"] and expected["parses"] and generated["body_dump"] == expected["body_dump"])
    binding = bool(generated["parses"] and generated["params_used"] == expected["params_used"] and not generated["undefined_names"])
    structural = bool(generated["parses"] and expected["operations"] and set(expected["operations"]).issubset(set(generated["operations"])))
    functional = evaluate_functional(case["signature"] + "\n" + _indent(completion), case["function_name"], case["tests"])
    return {
        "case_id": case["case_id"],
        "behavior": case["behavior"],
        "parses": bool(generated["parses"]),
        "exact": exact,
        "structural": structural,
        "binding": binding,
        "functional": {"passed": functional.passed, "reason": functional.reason},
        "completion": completion,
    }


def evaluate_all(checkpoint: Path, *, tokenizer, ad1_cases, c3_cases, supp_cases):
    import torch
    from serving.local_inference import load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    copy_records = []
    supp_records = []
    ad1_records = []
    c3_records = []
    try:
        for case_id, kind, instruction, signature, fragment in COPY_CASES:
            completion = _generate(model, tokenizer, instruction + "\n" + signature + "\n")
            scored = score_copy_case(kind, signature, completion)
            scored.update({
                "case_id": case_id,
                "completion": completion,
                "fragment_present": fragment in completion,
            })
            copy_records.append(scored)
        for case in supp_cases:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            supp_records.append(_score_behavior(case, completion))
        for case in ad1_cases:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            ad1_records.append(score_ad1_case(case, completion))
        for case in c3_cases:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            c3_records.append(score_c3_case(case, completion))
    finally:
        del model
        gc.collect()
    return {
        "copy": copy_records,
        "copy_report": copy_report(copy_records),
        "supplemental": supp_records,
        "supplemental_report": functional_report(supp_records),
        "supplemental_by_behavior": per_behavior_summary(supp_records),
        "ad1": ad1_records,
        "c3": c3_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Increment 10 evaluation")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--ad1", default=r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
    parser.add_argument("--c3", default=str(REPO / "evaluation" / "python_benchmark_heldout_v1.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ad1_suite = json.loads(Path(args.ad1).read_text(encoding="utf-8"))
    c3_suite = json.loads(Path(args.c3).read_text(encoding="utf-8"))
    supp_cases = build_cases()
    baseline = evaluate_all(
        Path(args.baseline), tokenizer=tokenizer,
        ad1_cases=ad1_suite["cases"], c3_cases=c3_suite["cases"], supp_cases=supp_cases,
    )
    candidate = evaluate_all(
        Path(args.candidate), tokenizer=tokenizer,
        ad1_cases=ad1_suite["cases"], c3_cases=c3_suite["cases"], supp_cases=supp_cases,
    )
    result = {
        "baseline_checkpoint": str(args.baseline),
        "candidate_checkpoint": str(args.candidate),
        "baseline": baseline,
        "candidate": candidate,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "copy_baseline": baseline["copy_report"],
        "copy_candidate": candidate["copy_report"],
        "supplemental_baseline": baseline["supplemental_report"],
        "supplemental_candidate": candidate["supplemental_report"],
        "ad1_baseline": {
            "parses": sum(r["parses"] for r in baseline["ad1"]),
            "binding": sum(r["binding"] for r in baseline["ad1"]),
        },
        "ad1_candidate": {
            "parses": sum(r["parses"] for r in candidate["ad1"]),
            "binding": sum(r["binding"] for r in candidate["ad1"]),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
