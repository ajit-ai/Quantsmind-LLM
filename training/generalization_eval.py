"""Evaluation of frozen v2.2.0 cases without executing generated code."""

from __future__ import annotations

from collections import defaultdict

from training.program_analysis import analyse
from training.supervised import build_example
from training.supervised_train import reconstruct_program

KNOWN_OPERATIONS = frozenset({
    "add", "subtract", "multiply", "max_builtin", "min_builtin",
    "comparison_gt", "comparison_lt", "if_selection", "filter_positive",
    "filter_negative", "comprehension", "length", "upper", "lower",
    "mapping_get", "mapping_keys", "mapping_contains", "iteration",
    "count_items", "sum_items", "attr_get",
})


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def score_case(case: dict, continuation: str, finish_reason=None,
               tokens=0) -> dict:
    expected = analyse(case["signature"], case["body"])
    generated = analyse(case["signature"], _indent(continuation))
    reference_ops = set(expected["operations"])
    generated_ops = set(generated["operations"])
    structural = bool(generated["parses"] and reference_ops
                      and reference_ops.issubset(generated_ops))
    exact = bool(generated["parses"] and expected["parses"]
                 and generated["body_dump"] == expected["body_dump"])
    binding = bool(generated["parses"]
                   and generated["params_used"] == expected["params_used"]
                   and not generated["undefined_names"])
    control_flow = bool(generated["parses"]
                        and generated["shape"] == expected["shape"])
    composition = bool(generated["parses"] and generated["key"] == expected["key"])
    template_hit = bool(generated["parses"]
                        and generated_ops.intersection(KNOWN_OPERATIONS))
    if case.get("negative"):
        structural = False
        exact = False
        composition = False
    return {
        "case_id": case["case_id"],
        "partition": case["partition"],
        "operation": case["operation"],
        "family": case["family"],
        "composition": case.get("composition"),
        "seen_in_training": (case.get("metadata") or {}).get("seen_in_training"),
        "negative": case.get("negative"),
        "instruction": case["instruction"],
        "signature": case["signature"],
        "reference_body": case["body"],
        "continuation": continuation,
        "generated_body": _indent(continuation),
        "parses": bool(generated["parses"]),
        "error": generated["error"],
        "exact": exact,
        "structural": structural,
        "binding": binding,
        "control_flow": control_flow,
        "composition_match": composition,
        "template_hit": template_hit,
        "reference_operations": sorted(reference_ops),
        "generated_operations": sorted(generated_ops),
        "params_used": generated["params_used"],
        "undefined_names": generated["undefined_names"],
        "eos_terminated": finish_reason == "stop",
        "truncated": finish_reason == "length",
        "tokens": tokens,
    }

def _summarize(records):
    total = len(records)

    def count(name):
        return sum(1 for record in records if record.get(name))

    summary = {
        "cases": total,
        "parses": count("parses"),
        "exact": count("exact"),
        "structural": count("structural"),
        "binding": count("binding"),
        "control_flow": count("control_flow"),
        "composition_match": count("composition_match"),
        "eos_terminated": count("eos_terminated"),
        "truncated": count("truncated"),
        "template_hit": count("template_hit"),
    }
    for key in ("parses", "exact", "structural", "binding", "control_flow",
                "composition_match", "eos_terminated", "truncated"):
        summary[key + "_rate"] = round(summary[key] / total, 4) if total else 0.0
    return summary


def summarize(records) -> dict:
    by_partition = defaultdict(list)
    by_operation = defaultdict(list)
    for record in records:
        by_partition[record["partition"]].append(record)
        by_operation[record["operation"]].append(record)
    result = {"overall": _summarize(records),
              "by_partition": {key: _summarize(value)
                               for key, value in sorted(by_partition.items())},
              "by_operation": {key: _summarize(value)
                               for key, value in sorted(by_operation.items())}}
    negatives = [r for r in records if r["partition"] == "E6"]
    if negatives:
        unsafe = [r for r in negatives if r["exact"] or r["structural"]]
        result["negative_controls"] = {
            "cases": len(negatives),
            "template_hits": sum(1 for r in negatives if r["template_hit"]),
            "unsafe_credits": len(unsafe),
            "safe": len(negatives) - len(unsafe),
            "safe_rate": round((len(negatives) - len(unsafe)) / len(negatives), 4),
        }
    compositions = [r for r in records if r["partition"] == "E4"]
    if compositions:
        seen = [r for r in compositions if r["seen_in_training"]]
        unseen = [r for r in compositions if not r["seen_in_training"]]
        result["composition_transfer"] = {
            "seen": _summarize(seen), "unseen": _summarize(unseen),
        }
    return result


def paired_delta(baseline_records, candidate_records) -> dict:
    baseline = {r["case_id"]: r for r in baseline_records}
    candidate = {r["case_id"]: r for r in candidate_records}
    shared = sorted(set(baseline) & set(candidate))
    result = {"cases": len(shared)}
    for metric in ("parses", "exact", "structural", "binding", "control_flow",
                   "composition_match"):
        before = sum(1 for key in shared if baseline[key].get(metric))
        after = sum(1 for key in shared if candidate[key].get(metric))
        gained = sum(1 for key in shared
                     if candidate[key].get(metric) and not baseline[key].get(metric))
        lost = sum(1 for key in shared
                   if baseline[key].get(metric) and not candidate[key].get(metric))
        result[metric] = {
            "baseline": before, "candidate": after, "delta": after - before,
            "gained": gained, "lost": lost,
        }
    return result


def compare_partitions(baseline: dict, candidate: dict) -> dict:
    return {"overall": paired_delta(baseline["records"], candidate["records"]),
            "by_partition": {
                key: paired_delta(baseline["by_partition_records"][key],
                                  candidate["by_partition_records"][key])
                for key in sorted(set(baseline["by_partition_records"])
                                  & set(candidate["by_partition_records"]))
            }}

def evaluate_cases(model, tokenizer, cases, *, budget: int = 64,
                   device: str = "cpu", progress=None) -> dict:
    from serving.local_inference import generate_ids

    records = []
    for index, case in enumerate(cases, 1):
        example = build_example(
            example_id=case["case_id"], instruction=case["instruction"],
            signature=case["signature"], body=case["body"],
            operation=case["operation"], family=case["family"],
            split=case["partition"], seed=20261022,
            metadata=case.get("metadata") or {},
        )
        result = generate_ids(
            model, tokenizer, example.prompt_text, max_new_tokens=budget,
            temperature=0.0, device=device, seed=20261022,
        )
        continuation = result.get("text", "")
        record = score_case(
            case, continuation, finish_reason=result.get("finish_reason"),
            tokens=len(result.get("token_ids") or []),
        )
        record["prompt"] = example.prompt_text
        records.append(record)
        if progress is not None:
            progress(index, len(cases), record)
    by_partition = defaultdict(list)
    for record in records:
        by_partition[record["partition"]].append(record)
    return {"version": "2.2.0", "records": records,
            "by_partition_records": dict(by_partition),
            "metrics": summarize(records)}

