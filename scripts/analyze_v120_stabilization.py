"""Increment 6 stabilization analysis: per-behaviour functional evidence.

Evaluates the baseline and Increment 5 candidate on the supplemental
behavior suite (evaluation/behavior_suite.py, separate from frozen AD-1/C3)
with identical greedy decoding, plus static analysis of the frozen-suite
evaluation JSONs. Writes a JSON report; never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.behavior_suite import BEHAVIORS, build_cases
from evaluation.functional_eval import evaluate_functional, functional_report
from evaluation.stabilization import compare_behaviors, decide_stabilization, per_behavior_summary
from training.program_analysis import analyse

TOKENIZER_PATH = REPO / "tokenizer" / "qm_code_bpe_v0.json"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
CANDIDATE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\checkpoints\trained.pt")
DEFAULT_EVAL = Path(
    r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\evaluations\heldout_comparison.json"
)


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def score_behavior_case(case, completion: str) -> dict:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], _indent(completion))
    exact = bool(generated["parses"] and expected["parses"] and generated["body_dump"] == expected["body_dump"])
    binding = bool(
        generated["parses"]
        and generated["params_used"] == expected["params_used"]
        and not generated["undefined_names"]
    )
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


def evaluate_checkpoint(checkpoint: Path, *, cases, tokenizer):
    import torch
    from serving.local_inference import generate_ids, load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    records = []
    try:
        for case in cases:
            prompt = case["instruction"] + "\n" + case["signature"] + "\n"
            started = time.perf_counter()
            result = generate_ids(
                model, tokenizer, prompt,
                max_new_tokens=64, temperature=0.0, top_k=1, top_p=1.0,
                device="cpu", seed=20261023, binding_repair=False,
            )
            started = time.perf_counter() - started
            record = score_behavior_case(case, result["text"])
            record["latency_seconds"] = round(started, 6)
            records.append(record)
    finally:
        del model
        gc.collect()
    return records


def frozen_analysis(eval_path: Path) -> dict:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    summaries = payload["summary"]
    analysis = {}
    for side in ("baseline", "candidate"):
        analysis[side] = {}
        for suite in ("ad1", "c3"):
            records = payload[suite][side]
            analysis[side][suite] = {
                "metrics": summaries[side][suite if suite == "ad1" else "c3_heldout"],
                "functional_analysis": summaries[side]["functional_analysis"][suite],
                "diversity": summaries[side][suite + "_diversity"],
                "mode_collapse": summaries[side]["mode_collapse"][suite],
                "records": records,
            }
    base_bind = {r["case_id"] for r in analysis["baseline"]["ad1"]["records"] if r["binding"]}
    cand_bind = {r["case_id"] for r in analysis["candidate"]["ad1"]["records"] if r["binding"]}
    analysis["ad1_binding_regressed"] = sorted(base_bind - cand_bind)
    analysis["ad1_binding_gained"] = sorted(cand_bind - base_bind)
    analysis["ad1_binding_loss"] = len(base_bind - cand_bind)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Increment 6 stabilization analysis")
    parser.add_argument("--evaluation", default=str(DEFAULT_EVAL))
    parser.add_argument("--candidate", default=str(CANDIDATE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    cases = build_cases()
    baseline_records = evaluate_checkpoint(BASELINE, cases=cases, tokenizer=tokenizer)
    candidate_records = evaluate_checkpoint(Path(args.candidate), cases=cases, tokenizer=tokenizer)

    baseline_by_behavior = per_behavior_summary(baseline_records)
    candidate_by_behavior = per_behavior_summary(candidate_records)
    comparison = compare_behaviors(baseline_by_behavior, candidate_by_behavior)

    baseline_report = functional_report(baseline_records)
    candidate_report = functional_report(candidate_records)

    frozen = frozen_analysis(Path(args.evaluation))
    base_func = frozen["baseline"]["c3"]["functional_analysis"]
    cand_func = frozen["candidate"]["c3"]["functional_analysis"]
    decision = decide_stabilization(
        baseline_functional_rate=base_func["functional_rate"] or 0.0,
        candidate_functional_rate=cand_func["functional_rate"] or 0.0,
        baseline_conditional=base_func["functional_given_binding"],
        candidate_conditional=cand_func["functional_given_binding"],
        behavior_regressions=comparison["regression_count"],
        ad1_binding_loss=frozen["ad1_binding_loss"],
        ad1_parse_fall=(
            frozen["candidate"]["ad1"]["metrics"]["parses"] < frozen["baseline"]["ad1"]["metrics"]["parses"]
        ),
        candidate_collapsed=(
            frozen["candidate"]["ad1"]["mode_collapse"]["collapsed"]
            or frozen["candidate"]["c3"]["mode_collapse"]["collapsed"]
        ),
    )

    # Case-level improve / regress / match on the supplemental suite.
    base_by_id = {r["case_id"]: r for r in baseline_records}
    cand_by_id = {r["case_id"]: r for r in candidate_records}
    transitions = Counter()
    for case_id in base_by_id:
        base_pass = bool(base_by_id[case_id]["functional"]["passed"])
        cand_pass = bool(cand_by_id[case_id]["functional"]["passed"])
        if cand_pass and not base_pass:
            transitions["improved"] += 1
        elif base_pass and not cand_pass:
            transitions["regressed"] += 1
        elif base_pass and cand_pass:
            transitions["both_pass"] += 1
        else:
            transitions["both_fail"] += 1

    result = {
        "behaviors": list(BEHAVIORS),
        "supplemental": {
            "cases": len(cases),
            "baseline": {
                "report": baseline_report,
                "by_behavior": baseline_by_behavior,
                "records": baseline_records,
            },
            "candidate": {
                "report": candidate_report,
                "by_behavior": candidate_by_behavior,
                "records": candidate_records,
            },
            "comparison": comparison,
            "transitions": dict(transitions),
        },
        "frozen": {
            "baseline_c3_functional_analysis": base_func,
            "candidate_c3_functional_analysis": cand_func,
            "ad1_binding_regressed": frozen["ad1_binding_regressed"],
            "ad1_binding_gained": frozen["ad1_binding_gained"],
            "ad1_binding_loss": frozen["ad1_binding_loss"],
            "candidate_mode_collapse": {
                "ad1": frozen["candidate"]["ad1"]["mode_collapse"],
                "c3": frozen["candidate"]["c3"]["mode_collapse"],
            },
            "candidate_diversity": {
                "ad1": frozen["candidate"]["ad1"]["diversity"],
                "c3": frozen["candidate"]["c3"]["diversity"],
            },
        },
        "decision": decision,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "supplemental_baseline": baseline_report,
        "supplemental_candidate": candidate_report,
        "by_behavior": comparison,
        "transitions": dict(transitions),
        "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
