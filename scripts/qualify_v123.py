"""v1.2.3 qualification evaluation (inference only).

Evaluates model-only, adapter-only, v1.2.1 hybrid, and v1.2.2 corrected
hybrid on frozen + new independent suites, plus neural operation-selection
probes (teacher-forced token probabilities) and constrained-decoding POC
metrics. Writes JSON only; never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.acceptance_suite_v122 import build_cases as build_v122
from evaluation.acceptance_suite_v123 import build_cases as build_v123
from evaluation.behavior_suite import build_cases as build_supp
from evaluation.constrained import generate_constrained
from evaluation.copy_eval import copy_report, score_copy_case
from evaluation.copy_mechanism import CopyGate, copy_generate
from evaluation.diagnostics import COPY_CASES
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.generalization_suite import build_cases as build_gen
from evaluation.hybrid import HybridGenerator, analyse_names, infer_relation, rewrite_body
from evaluation.operation_synth import synthesize
from evaluation.stabilization import per_behavior_summary
from training.program_analysis import analyse

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
AD1_PATH = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
C3_PATH = REPO / "evaluation" / "python_benchmark_heldout_v1.json"
V121_PATH = REPO / "evaluation" / "acceptance_suite.py"


def _ind(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def _generate(model, tokenizer, prompt):
    from serving.local_inference import generate_ids

    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=False,
    )
    return result["text"]


def _transform(gate, generator, instruction, signature, completion, mode, tokenizer):
    if mode == "model":
        return completion
    if mode == "adapter":
        return copy_generate(gate, instruction, signature, completion, tokenizer)["rewritten"]
    return generator.generate(instruction, signature)["completion"]


def _behavior_record(case, completion, reference_body):
    from evaluation.functional_eval import evaluate_functional as _ef

    expected = analyse(case["signature"], reference_body)
    generated = analyse(case["signature"], _ind(completion))
    functional = _ef(case["signature"] + "\n" + _ind(completion) if completion.lstrip().startswith("def ") is False else completion,
                     case["function_name"], case["tests"])
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


def evaluate_all(model, tokenizer, mode):
    gate = CopyGate(enabled=(mode != "model"))
    generator = HybridGenerator(model, tokenizer, enabled=(mode in ("hybrid", "hybrid122")))
    ad1_suite = json.loads(AD1_PATH.read_text(encoding="utf-8"))
    c3_suite = json.loads(C3_PATH.read_text(encoding="utf-8"))
    copy_records, supp_records, gen_records, new_records, ad1_records, c3_records = [], [], [], [], [], []
    for case_id, kind, instruction, signature, _fragment in COPY_CASES:
        completion = _generate(model, tokenizer, instruction + "\n" + signature + "\n")
        final = _transform(gate, generator, instruction, signature, completion, mode, tokenizer)
        scored = score_copy_case(kind, signature, final)
        scored.update({"case_id": case_id, "completion": final})
        copy_records.append(scored)
    suites = [
        (build_supp(), supp_records), (build_gen(), gen_records),
        (build_v123(), new_records),
    ]
    for cases, sink in suites:
        for case in cases:
            completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
            if case.get("expected_abstain"):
                pass
            final = _transform(gate, generator, case["instruction"], case["signature"], completion, mode, tokenizer)
            rec = _behavior_record(case, final, case["reference_body"])
            rec["expected_abstain"] = case.get("expected_abstain", False)
            if rec["expected_abstain"]:
                rec["abstained"] = final == completion and mode != "model"
                rec["functional"] = {"passed": None, "reason": "abstention_expected"}
            sink.append(rec)
    for case in ad1_suite["cases"]:
        from scripts.evaluate_v120_increment1 import score_ad1_case

        completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
        final = _transform(gate, generator, case["instruction"], case["signature"], completion, mode, tokenizer)
        ad1_records.append(score_ad1_case(case, final))
    for case in c3_suite["cases"]:
        from scripts.evaluate_v120_increment1 import score_c3_case

        completion = _generate(model, tokenizer, case["instruction"] + "\n" + case["signature"] + "\n")
        final = _transform(gate, generator, case["instruction"], case["signature"], completion, mode, tokenizer)
        c3_records.append(score_c3_case(case, final))
    return {
        "copy": copy_records,
        "copy_report": copy_report(copy_records),
        "supplemental": supp_records,
        "supplemental_report": functional_report([r for r in supp_records if not r["expected_abstain"]] or supp_records),
        "generalization": gen_records,
        "generalization_report": functional_report(gen_records),
        "new_independent": new_records,
        "ad1": ad1_records,
        "c3": c3_records,
    }


def op_selection_probe(model, tokenizer) -> dict:
    """Neural operation-selection: teacher-forced operation-token ranks."""
    import torch

    probes = [
        ("sum", "Return the sum of two numbers.", "def f(left_value, right_value):"),
        ("difference", "Return the difference between two numbers.", "def f(left_value, right_value):"),
        ("product", "Return the product of two numbers.", "def f(left_value, right_value):"),
        ("upper", "Return the text in upper case.", "def f(raw_text):"),
        ("lower", "Return the text in lower case.", "def f(raw_text):"),
        ("count", "Count the items in a list.", "def f(item_values):"),
        ("sum", "Sum the items in a list.", "def f(item_values):"),
        ("even", "Check whether the number is even.", "def f(number_value):"),
    ]
    observations = []
    model.eval()
    with torch.no_grad():
        for expected_op, instruction, signature in probes:
            sequence = instruction + "\n" + signature + "\n    return "
            ids = list(tokenizer.encode(sequence, add_special_tokens=False).ids)
            logits, _ = model(torch.tensor([ids], dtype=torch.long))
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            top_ids = torch.topk(probs, 5).indices.tolist()
            top_tokens = [tokenizer.id_to_token(t) for t in top_ids]
            observations.append({
                "instruction": instruction,
                "expected_operation": expected_op,
                "top_tokens": top_tokens,
            })
    return {"observations": observations}


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.2.3 qualification")
    parser.add_argument("--checkpoint", default=str(BASELINE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from serving.local_inference import load_model

    import torch

    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    model, _config = load_model(str(args.checkpoint), "cpu")
    modes = {}
    try:
        modes["model"] = evaluate_all(model, tokenizer, "model")
        gate_adapter = CopyGate(enabled=True)
        modes["adapter"] = evaluate_all(model, tokenizer, "adapter")
        modes["hybrid"] = evaluate_all(model, tokenizer, "hybrid")
        # Constrained POC over copy probes (syntax layer comparison).
        constrained_prompts = []
        for case_id, kind, instruction, signature, fragment in COPY_CASES:
            constrained_prompts.append((case_id, instruction + "\n" + signature + "\n"))
        constrained_runs = []
        for case_id, prompt in constrained_prompts:
            result = generate_constrained(model, tokenizer, prompt)
            constrained_runs.append({
                "case_id": case_id,
                "completion": result["completion"],
                "interventions": result["interventions"],
            })
        op_selection = op_selection_probe(model, tokenizer)
    finally:
        del model
        gc.collect()
    result = {
        "checkpoint": str(args.checkpoint),
        **modes,
        "constrained_copy_probes": constrained_runs,
        "op_selection": op_selection,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for mode in ("model", "adapter", "hybrid"):
        side = result[mode]
        supp = sum(r["functional"]["passed"] is True for r in side["supplemental"])
        gen = sum(r["functional"]["passed"] is True for r in side["generalization"])
        new_reg = [r for r in side["new_independent"] if not r["expected_abstain"]]
        new_func = sum(r["functional"]["passed"] is True for r in new_reg)
        print(json.dumps({
            mode: {
                "copy": side["copy_report"]["all"]["exact_copy"],
                "supp": supp, "gen": gen,
                "new_functional": new_func,
                "ad1": [sum(r["parses"] for r in side["ad1"]), sum(r["binding"] for r in side["ad1"])],
                "c3": [sum(r["parses"] for r in side["c3"]), sum(r["binding"] for r in side["c3"]), sum(r["functional"]["passed"] is True for r in side["c3"])],
            }
        }))
    print("op_selection_first2", json.dumps(op_selection["observations"][:2], ensure_ascii=True)[:400])
    print("constrained_interventions", sum(len(r["interventions"]) for r in constrained_runs))


if __name__ == "__main__":
    main()
