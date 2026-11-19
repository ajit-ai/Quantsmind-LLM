"""Increment 13 adapter feasibility evaluation (inference only).

Runs the frozen copy probes, supplemental suite, and AD-1/C3 with the
copy adapter enabled vs disabled, and reports each dimension separately.
Writes a machine-readable JSON report; never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.behavior_suite import build_cases
from evaluation.copy_eval import copy_report, score_copy_case
from evaluation.copy_mechanism import CopyGate, copy_generate
from evaluation.diagnostics import COPY_CASES
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.stabilization import per_behavior_summary
from training.program_analysis import analyse
from scripts.evaluate_v120_increment1 import score_ad1_case, score_c3_case

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
AD1_PATH = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
C3_PATH = REPO / "evaluation" / "python_benchmark_heldout_v1.json"


def _generate(model, tokenizer, prompt):
    from serving.local_inference import generate_ids

    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=False,
    )
    return result["text"]


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def _behavior_record(case, completion, new_completion):
    expected = analyse(case["signature"], case["reference_body"])
    for label, text in (("raw", completion), ("adapted", new_completion)):
        pass
    generated = analyse(case["signature"], _indent(new_completion))
    functional = evaluate_functional(
        case["signature"] + "\n" + _indent(new_completion), case["function_name"], case["tests"]
    )
    return {
        "case_id": case["case_id"],
        "behavior": case["behavior"],
        "parses": bool(generated["parses"]),
        "exact": bool(generated["parses"] and generated["body_dump"] == expected["body_dump"]),
        "structural": bool(generated["parses"] and expected["operations"] and set(expected["operations"]).issubset(set(generated["operations"]))),
        "binding": bool(generated["parses"] and generated["params_used"] == expected["params_used"] and not generated["undefined_names"]),
        "functional": {"passed": functional.passed, "reason": functional.reason},
        "completion": new_completion,
        "raw_completion": completion,
    }


def evaluate(checkpoint: Path, *, tokenizer, enabled: bool):
    import torch
    from serving.local_inference import load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    gate = CopyGate(enabled=enabled)
    ad1_suite = json.loads(AD1_PATH.read_text(encoding="utf-8"))
    c3_suite = json.loads(C3_PATH.read_text(encoding="utf-8"))
    copy_records, supp_records, ad1_records, c3_records = [], [], [], []
    decisions = []
    try:
        for case_id, kind, instruction, signature, _fragment in COPY_CASES:
            completion = _generate(model, tokenizer, instruction + "\n" + signature + "\n")
            result = copy_generate(gate, instruction, signature, completion, tokenizer)
            scored = score_copy_case(kind, signature, result["rewritten"])
            scored.update({"case_id": case_id, "completion": result["rewritten"], "raw": completion,
                           "decision": result["decision"]["action"]})
            copy_records.append(scored)
            decisions.append(result["decision"]["action"])
        for case in build_cases():
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            result = copy_generate(gate, case["instruction"], case["signature"], completion, tokenizer)
            supp_records.append(_behavior_record(case, completion, result["rewritten"]))
        for case in ad1_suite["cases"]:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            result = copy_generate(gate, case["instruction"], case["signature"], completion, tokenizer)
            ad1_records.append(score_ad1_case(case, result["rewritten"]))
        for case in c3_suite["cases"]:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            result = copy_generate(gate, case["instruction"], case["signature"], completion, tokenizer)
            # score_c3_case re-scores from completion text; pass rewritten body.
            ad1_equiv = score_c3_case(case, result["rewritten"])
            ad1_equiv["raw_completion"] = completion
            c3_records.append(ad1_equiv)
    finally:
        del model
        gc.collect()
    return {
        "copy": copy_records,
        "copy_report": copy_report(copy_records),
        "copy_decisions": decisions,
        "supplemental": supp_records,
        "supplemental_report": functional_report(supp_records),
        "supplemental_by_behavior": per_behavior_summary(supp_records),
        "ad1": ad1_records,
        "c3": c3_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Increment 13 adapter evaluation")
    parser.add_argument("--checkpoint", default=str(BASELINE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    checkpoint = Path(args.checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "disabled": evaluate(checkpoint, tokenizer=tokenizer, enabled=False),
        "enabled": evaluate(checkpoint, tokenizer=tokenizer, enabled=True),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label in ("disabled", "enabled"):
        side = result[label]
        print(json.dumps({
            label: {
                "copy_exact_copy": side["copy_report"]["all"]["exact_copy"],
                "copy_cases": side["copy_report"]["all"]["cases"],
                "copy_decisions": side["copy_decisions"],
                "supp_functional": side["supplemental_report"]["functional_rate"],
                "ad1": {
                    "parses": sum(r["parses"] for r in side["ad1"]),
                    "binding": sum(r["binding"] for r in side["ad1"]),
                    "exact": sum(r["exact"] for r in side["ad1"]),
                },
                "c3": {
                    "parses": sum(r["parses"] for r in side["c3"]),
                    "binding": sum(r["binding"] for r in side["c3"]),
                    "functional": sum(r["functional"]["passed"] is True for r in side["c3"]),
                },
            }
        }))


if __name__ == "__main__":
    main()
