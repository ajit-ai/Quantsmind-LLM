"""Run the frozen binding-repair qualification suite with the S2 checkpoint.

No generated code is executed except for explicitly approved, pure fixture
cases in the frozen dataset. Unsupported/ambiguous cases are never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.local_inference import generate_ids, load_model
from training.binding_repair import repair_from_prompt
from training.generalization_eval import _indent
from training.program_analysis import analyse
from evaluation.safe_execute import execute_tests

DEFAULT_CHECKPOINT = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
DEFAULT_TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"
SUITE = REPO / "evaluation" / "binding_repair_qualification_v1.json"
EXPECTED_CHECKPOINT = "02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac"
EXPECTED_SUITE_SHA = "961936f8b5f098e0d9d863a8753d7430929ba87b0a6880fe27acb81c461f3d2f"


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def score_static(case: dict[str, Any], completion: str) -> dict[str, Any]:
    expected = analyse(case["signature"], case["reference_body"])
    generated = analyse(case["signature"], _indent(completion))
    exact = bool(generated["parses"] and expected["parses"]
                 and generated["body_dump"] == expected["body_dump"])
    reference_ops = set(expected["operations"])
    generated_ops = set(generated["operations"])
    if reference_ops:
        structural: bool | None = bool(generated["parses"] and reference_ops.issubset(generated_ops))
    else:
        structural = None
    binding = bool(generated["parses"]
                   and generated["params_used"] == expected["params_used"]
                   and not generated["undefined_names"])
    if case.get("negative"):
        exact = False
        structural = False
    return {
        "parses": bool(generated["parses"]),
        "error": generated["error"],
        "exact": exact,
        "structural": structural,
        "binding": binding,
        "operations": sorted(generated_ops),
        "reference_operations": expected["operations"],
        "undefined_names": generated["undefined_names"],
    }


def source_for(case: dict[str, Any], completion: str) -> tuple[str, str]:
    stripped = completion.lstrip()
    if stripped.startswith("def "):
        return completion, "full_function"
    return case["signature"] + "\n" + _indent(completion), "body"


def run_case(model, tokenizer, case: dict[str, Any], binding_repair: bool):
    prompt = case["instruction"] + "\n" + case["signature"] + "\n"
    result = generate_ids(
        model, tokenizer, prompt,
        max_new_tokens=64, temperature=0.0, top_k=1, top_p=1.0,
        device="cpu", seed=20261023, binding_repair=binding_repair,
    )
    completion = result["text"]
    static = score_static(case, completion)
    functional = {"passed": None, "reason": "not_executed"}
    if case.get("execution_safe") and not case.get("negative"):
        source, form = source_for(case, completion)
        execution = execute_tests(source, case["function_name"], case["tests"])
        functional = {"passed": execution.passed, "reason": execution.reason,
                      "source_form": form}
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "negative": bool(case.get("negative")),
        "prompt": prompt,
        "completion": completion,
        "binding_repair": binding_repair,
        "finish_reason": result["finish_reason"],
        "tokens": result["completion_tokens"],
        **static,
        "functional": functional,
        "repair": result.get("repair", {}),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    structural = [r for r in records if r["structural"] is not None]
    positives = [r for r in records if not r["negative"]]
    return {
        "cases": total,
        "parses": sum(bool(r["parses"]) for r in records),
        "exact": sum(bool(r["exact"]) for r in records),
        "structural": sum(r["structural"] is True for r in structural),
        "structural_cases": len(structural),
        "binding": sum(bool(r["binding"]) for r in records),
        "negative_safe": sum(1 for r in records
                             if r["negative"] and not r["exact"] and not r["structural"]),
        "negative_cases": sum(1 for r in records if r["negative"]),
        "functional_passed": sum(1 for r in records if r["functional"]["passed"] is True),
        "functional_failed": sum(1 for r in records if r["functional"]["passed"] is False),
        "functional_cases": sum(1 for r in records if r["functional"]["passed"] is not None),
        "positive_cases": len(positives),
        "exact_rate": round(sum(bool(r["exact"]) for r in records) / total, 4),
        "structural_rate": round(
            sum(r["structural"] is True for r in structural) / len(structural), 4
        ) if structural else None,
        "binding_rate": round(sum(bool(r["binding"]) for r in records) / total, 4),
        "functional_rate": round(
            sum(r["functional"]["passed"] is True for r in records
                if r["functional"]["passed"] is not None)
            / max(1, sum(r["functional"]["passed"] is not None for r in records)), 4
        ),
    }


def paired(raw: list[dict[str, Any]], repaired: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {r["case_id"]: r for r in raw}
    lookup_r = {r["case_id"]: r for r in repaired}
    changed = sum(1 for k in lookup if lookup[k]["completion"] != lookup_r[k]["completion"])
    improved = {
        metric: sum(1 for k in lookup
                    if bool(lookup_r[k][metric]) and not bool(lookup[k][metric]))
        for metric in ("parses", "exact", "binding")
    }
    regressions = {
        metric: sum(1 for k in lookup
                    if bool(lookup[k][metric]) and not bool(lookup_r[k][metric]))
        for metric in ("parses", "exact", "binding")
    }
    changed_negative = sum(
        1 for k in lookup if lookup[k]["negative"] and lookup[k]["completion"] != lookup_r[k]["completion"]
    )
    return {
        "changed": changed,
        "changed_negative": changed_negative,
        "improved": improved,
        "regressions": regressions,
        "changed_rate": round(changed / len(lookup), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--suite", default=str(SUITE))
    parser.add_argument("--expected-sha", default=EXPECTED_SUITE_SHA)
    parser.add_argument("--out", default=r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\binding_repair_qualification_results.json")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    expected_sha = args.expected_sha.lower()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_sha = file_sha256(suite_path)
    if suite_sha != expected_sha:
        raise SystemExit("REFUSED: qualification dataset hash does not match expected SHA-256")
    if canonical_digest(suite["cases"]) != suite["digest"]:
        raise SystemExit("REFUSED: qualification dataset digest changed")
    checkpoint_sha = file_sha256(Path(args.checkpoint))
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise SystemExit("REFUSED: checkpoint hash does not match frozen S2")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    model, _config = load_model(args.checkpoint, "cpu")

    raw_records, repaired_records = [], []
    for case in suite["cases"]:
        started = time.perf_counter()
        raw_records.append(run_case(model, tokenizer, case, binding_repair=False))
        raw_records[-1]["latency_seconds"] = time.perf_counter() - started
    for case in suite["cases"]:
        started = time.perf_counter()
        repaired_records.append(run_case(model, tokenizer, case, binding_repair=True))
        repaired_records[-1]["latency_seconds"] = time.perf_counter() - started

    def abstentions(records):
        counts = {}
        for record in records:
            status = record.get("repair", {}).get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def latency(records):
        values = [record["latency_seconds"] for record in records]
        if not values:
            return None
        mean = sum(values) / len(values)
        return {"mean_seconds": mean, "median_seconds": sorted(values)[len(values) // 2]}

    raw_summary = summarize(raw_records)
    repaired_summary = summarize(repaired_records)
    result = {
        "suite": suite["suite"],
        "suite_version": suite["version"],
        "suite_sha256": suite_sha,
        "suite_digest": suite["digest"],
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer": str(DEFAULT_TOKENIZER),
        "decoding": suite["decoding"],
        "executed_fixture_cases_only": True,
        "records": {"raw": raw_records, "repaired": repaired_records},
        "summary": {"raw": raw_summary, "repaired": repaired_summary},
        "repair": {
            "statuses": {
                "raw": abstentions(raw_records),
                "repaired": abstentions(repaired_records),
            },
            "latency_seconds": {
                "raw": latency(raw_records),
                "repaired": latency(repaired_records),
            },
        },
        "paired": paired(raw_records, repaired_records),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(output),
        "raw": result["summary"]["raw"],
        "repaired": result["summary"]["repaired"],
        "paired": result["paired"],
    }, indent=2))


if __name__ == "__main__":
    main()
