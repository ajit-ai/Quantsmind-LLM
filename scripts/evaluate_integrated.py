"""Integrated campaign evaluation (inference only).

Compares three modes with identical frozen cases and decoding:
``model`` (raw neural output), ``adapter`` (Increment 13 gate only),
``hybrid`` (model + operation-aware identifier resolution). All modes
execute through the existing safe gate. Writes JSON only.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.behavior_suite import build_cases as build_supp
from evaluation.copy_eval import copy_report, score_copy_case
from evaluation.copy_mechanism import CopyGate, copy_generate
from evaluation.diagnostics import COPY_CASES
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.generalization_suite import build_cases as build_generalization
from evaluation.hybrid import HybridGenerator
from evaluation.stabilization import per_behavior_summary
from training.program_analysis import analyse

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
AD1_PATH = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
C3_PATH = REPO / "evaluation" / "python_benchmark_heldout_v1.json"


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def _score_behavior(case, completion: str) -> dict:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], _indent(completion))
    functional = evaluate_functional(
        case["signature"] + "\n" + _indent(completion), case["function_name"], case["tests"]
    )
    return {
        "case_id": case["case_id"],
        "behavior": case.get("behavior", case.get("kind")),
        "parses": bool(generated["parses"]),
        "exact": bool(generated["parses"] and generated["body_dump"] == expected["body_dump"]),
        "structural": bool(generated["parses"] and expected["operations"] and set(expected["operations"]).issubset(set(generated["operations"]))),
        "binding": bool(generated["parses"] and generated["params_used"] == expected["params_used"] and not generated["undefined_names"]),
        "functional": {"passed": functional.passed, "reason": functional.reason},
        "completion": completion,
    }


def _generate_text(model, tokenizer, prompt):
    from serving.local_inference import generate_ids

    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=False,
    )
    return result["text"]


def evaluate_mode(model, tokenizer, mode: str) -> dict:
    gate = CopyGate(enabled=(mode != "model"))
    generator = HybridGenerator(model, tokenizer, enabled=(mode == "hybrid"))
    ad1_suite = json.loads(AD1_PATH.read_text(encoding="utf-8"))
    c3_suite = json.loads(C3_PATH.read_text(encoding="utf-8"))
    copy_records, supp_records, gen_records, ad1_records, c3_records = [], [], [], [], []
    decisions = []
    for case_id, kind, instruction, signature, _fragment in COPY_CASES:
        completion = _generate_text(model, tokenizer, instruction + "\n" + signature + "\n")
        if mode == "model":
            hybrid_record = None
            final = completion
        elif mode == "adapter":
            result = copy_generate(gate, instruction, signature, completion, tokenizer)
            final = result["rewritten"]
            decisions.append(result["decision"]["action"])
        else:
            result = copy_generate(gate, instruction, signature, completion, tokenizer)
            # Hybrid path: operation-aware resolution on the model completion.
            hybrid_record = generator.generate(instruction, signature)
            final = hybrid_record["completion"]
        scored = score_copy_case(kind, signature, final)
        scored.update({"case_id": case_id, "completion": final})
        copy_records.append(scored)
    for case in build_supp():
        record = _behavior_case(model, tokenizer, gate, generator, case, mode)
        supp_records.append(record)
    for case in build_generalization():
        gen_records.append(_behavior_case(model, tokenizer, gate, generator, case, mode))
    for case in ad1_suite["cases"]:
        from scripts.evaluate_v120_increment1 import score_ad1_case

        completion = _generate_text(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
        final = _transform(gate, generator, case["instruction"], case["signature"], completion, mode)
        ad1_records.append(score_ad1_case(case, final))
    for case in c3_suite["cases"]:
        from scripts.evaluate_v120_increment1 import score_c3_case

        completion = _generate_text(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
        final = _transform(gate, generator, case["instruction"], case["signature"], completion, mode)
        c3_records.append(score_c3_case(case, final))
    return {
        "copy": copy_records,
        "copy_report": copy_report(copy_records),
        "copy_decisions": decisions,
        "supplemental": supp_records,
        "supplemental_report": functional_report(supp_records),
        "supplemental_by_behavior": per_behavior_summary(supp_records),
        "generalization": gen_records,
        "generalization_report": functional_report(gen_records),
        "ad1": ad1_records,
        "c3": c3_records,
    }


def _transform(gate, generator, instruction, signature, completion, mode):
    if mode == "model":
        return completion
    if mode == "adapter":
        return copy_generate(gate, instruction, signature, completion, _tokenizer())["rewritten"]
    return generator.generate(instruction, signature)["completion"]


def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))


def _behavior_case(model, tokenizer, gate, generator, case, mode):
    completion = _generate_text(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
    if mode == "model":
        final = completion
    elif mode == "adapter":
        final = copy_generate(gate, case["instruction"], case["signature"], completion, _tokenizer())["rewritten"]
    else:
        final = generator.generate(case["instruction"], case["signature"])["completion"]
    return _score_behavior(case, final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated generation evaluation")
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
        result = {mode: evaluate_mode(model, tokenizer, mode) for mode in ("model", "adapter", "hybrid")}
    finally:
        del model
        gc.collect()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(args.checkpoint), **result}, indent=2), encoding="utf-8")
    for mode in ("model", "adapter", "hybrid"):
        side = result[mode]
        print(json.dumps({
            mode: {
                "copy_exact": side["copy_report"]["all"]["exact_copy"],
                "supp_functional": side["supplemental_report"]["functional_rate"],
                "gen_functional": side["generalization_report"]["functional_rate"],
                "ad1_parses_binding": [sum(r["parses"] for r in side["ad1"]), sum(r["binding"] for r in side["ad1"])],
                "c3_parses_binding_functional": [
                    sum(r["parses"] for r in side["c3"]),
                    sum(r["binding"] for r in side["c3"]),
                    sum(r["functional"]["passed"] is True for r in side["c3"]),
                ],
            }
        }))


if __name__ == "__main__":
    main()
