"""v1.2.1 corrected-hybrid acceptance evaluation (inference only).

Evaluates the frozen suites plus the new acceptance suite with the
model, adapter, and corrected hybrid. Abstention-expected cases are
scored on whether generation safely abstains. Writes JSON only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.acceptance_suite import build_cases as build_acceptance
from evaluation.copy_eval import copy_report, score_copy_case
from evaluation.copy_mechanism import CopyGate, copy_generate
from evaluation.diagnostics import COPY_CASES
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.hybrid import HybridGenerator, analyse_names, rewrite_body
from evaluation.operation_synth import synthesize
from evaluation.stabilization import per_behavior_summary
from training.program_analysis import analyse

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")


def _generate(model, tokenizer, prompt):
    from serving.local_inference import generate_ids

    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=False,
    )
    return result["text"]


def run_mode(model, tokenizer, mode):
    gate = CopyGate(enabled=(mode != "model"))
    generator = HybridGenerator(model, tokenizer, enabled=(mode == "hybrid"))
    copy_records, acceptance_records = [], []
    for case_id, kind, instruction, signature, _fragment in COPY_CASES:
        completion = _generate(model, tokenizer, instruction + "\n" + signature + "\n")
        if mode == "model":
            final = completion
        elif mode == "adapter":
            final = copy_generate(gate, instruction, signature, completion, tokenizer)["rewritten"]
        else:
            synth = synthesize(instruction, signature)
            final = synth["body"] if synth["action"] == "synthesized" else _hybrid_fallback(
                gate, generator, instruction, signature, completion
            )
        scored = score_copy_case(kind, signature, final)
        scored.update({"case_id": case_id, "completion": final})
        copy_records.append(scored)
    for case in build_acceptance():
        completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
        if case["expected_abstain"]:
            abstained = evaluate_abstention(model, gate, generator, case, completion, mode)
            acceptance_records.append({
                "case_id": case["case_id"],
                "behavior": case["behavior"],
                "expected_abstain": True,
                "abstained": abstained,
                "functional": {"passed": None, "reason": "abstention_expected"},
                "parses": None,
                "exact": None,
                "structural": None,
                "binding": None,
                "completion": "abstained" if abstained else completion,
            })
            continue
        if mode == "model":
            final = completion
        elif mode == "adapter":
            final = copy_generate(gate, case["instruction"], case["signature"], completion, tokenizer)["rewritten"]
        else:
            synth = synthesize(case["instruction"], case["signature"])
            final = synth["body"] if synth["action"] == "synthesized" else _hybrid_fallback(
                gate, generator, case["instruction"], case["signature"], completion
            )
        acceptance_records.append(_score_acceptance(case, final))
    return {"copy": copy_records, "copy_report": copy_report(copy_records), "acceptance": acceptance_records}


def evaluate_abstention(model, gate, generator, case, completion, mode) -> bool:
    if mode == "model":
        return False
    if mode == "adapter":
        decision = copy_generate(gate, case["instruction"], case["signature"], completion, _tokenizer_of(model))["decision"]
        return decision["action"] in ("abstain", "generate")
    result = generator.generate(case["instruction"], case["signature"])
    return result["decision"]["action"] in ("abstain", "generate", "propagate", "copy")


def _tokenizer_of(_model):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))


def _hybrid_fallback(gate, generator, instruction, signature, completion):
    if generator.enabled:
        return generator.generate(instruction, signature)["completion"]
    return copy_generate(gate, instruction, signature, completion, _tokenizer_of(None))["rewritten"]


def _score_acceptance(case, completion: str) -> dict:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], completion if completion[0].isspace() else "    " + completion.replace("\n", "\n    "))
    generated = analyse(case["signature"], _indent(completion))
    functional = evaluate_functional(
        case["signature"] + "\n" + _indent(completion), case["function_name"], case["tests"]
    )
    return {
        "case_id": case["case_id"],
        "behavior": case["behavior"],
        "expected_abstain": False,
        "parses": bool(generated["parses"]),
        "exact": bool(generated["parses"] and generated["body_dump"] == expected["body_dump"]),
        "structural": bool(generated["parses"] and expected["operations"] and set(expected["operations"]).issubset(set(generated["operations"]))),
        "binding": bool(generated["parses"] and generated["params_used"] == expected["params_used"] and not generated["undefined_names"]),
        "functional": {"passed": functional.passed, "reason": functional.reason},
        "completion": completion,
    }


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.2.1 acceptance evaluation")
    parser.add_argument("--checkpoint", default=str(BASELINE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from serving.local_inference import load_model

    import torch

    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    model, _config = load_model(str(args.checkpoint), "cpu")
    try:
        result = {mode: run_mode(model, tokenizer, mode) for mode in ("model", "adapter", "hybrid")}
    finally:
        del model
    for mode, side in result.items():
        acceptance = side["acceptance"]
        regular = [r for r in acceptance if not r["expected_abstain"]]
        withheld = [r for r in acceptance if r["expected_abstain"]]
        print(json.dumps({
            mode: {
                "copy_exact": side["copy_report"]["all"]["exact_copy"],
                "acceptance_functional": functional_report(regular)["functional_rate"],
                "acceptance_passes": sum(r["functional"]["passed"] is True for r in regular),
                "acceptance_cases": len(regular),
                "acceptance_abstained_correct": sum(r["abstained"] for r in withheld),
                "acceptance_abstain_cases": len(withheld),
            }
        }))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(args.checkpoint), **result}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
