"""TAKSH L0.6.2 — Karkain inference integration (TAKSH-side contract).

Implements TAKSH-side *inference support structures* for Karkain — the 54th
first-class language. This module never claims Karkain capability, never runs
KCC, never fabricates compile results, and never writes the Karkain repo.

Supported here:
  * .kark / "karkain" prompt identification
  * Karkain specification + compiler-version metadata (pinned to the contract)
  * Karkain task classification (declared intent, never verification)
  * KIR / compiler-feedback context interfaces (explicitly empty while KCC is
    unavailable on this host)
  * Separate model_only vs compiler_assisted evaluation modes — never merged
  * Eligibility gate: NOT MET (mirrors the L0.6.1 contract)

KCC remains the authoritative compiler oracle. Host state (unchanged, recorded):
kcc.exe exits 0xC0000135; all compile-pipeline commands exit 3, zero diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

KARK_CONTRACT = REPO / "configs" / "foundation" / "l0601_karkain_learning_contract.yaml"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"

TASK_CLASSES = frozenset({
    "syntax_read", "kata", "feature_impl", "bugfix", "code_review",
    "kir_inspection", "compiler_feedback_loop", "benchmark_probe",
})


def load_karkain_contract() -> dict:
    import yaml

    return yaml.safe_load(KARK_CONTRACT.read_text(encoding="utf-8"))


def identify_karkain_prompt(text: str | None = None, *, filename: str | None = None,
                            language_hint: str | None = None) -> dict:
    declared = (language_hint or "").strip().lower() == "karkain"
    suffix = bool(filename) and Path(filename).suffix.lower() == ".kark"
    detected = declared or suffix
    if text and not detected:
        lowered = text.lower()
        detected = ".kark" in lowered or "karkain" in lowered
    return {
        "is_karkain": detected,
        "detection": "declared" if declared else ("filename_extension" if suffix else
                                                  ("content_marker" if detected else "none")),
        "language_id": "karkain" if detected else None,
        "note": "identification never implies capability",
    }


def karkain_inference_metadata() -> dict:
    contract = load_karkain_contract()
    spec = contract["identity"]["spec_reference"]
    return {
        "language_id": "karkain",
        "extensions": [".kark"],
        "spec_version": spec["spec_version"],
        "compiler_version": spec["compiler_version"],
        "kcc_engine": spec["kcc_engine"],
        "kir_version": spec["kir_version"],
        "bench": contract["evaluation_references"]["bench_schema"],
        "qualification_status": contract["identity"]["registry_linkage"]["qualification_status"],
        "kcc_state_on_host": contract["kcc_state_on_host"],
        "training_eligibility": contract["training_eligibility"]["currently_met"],
        "evaluation_modes": ["model_only", "compiler_assisted"],
        "mode_separation": "compiler_assisted results are never reported as model capability",
    }


def classify_karkain_task(instruction: str, declared: str | None = None) -> dict:
    if declared and declared in TASK_CLASSES:
        return {"task_class": declared, "source": "declared", "confidence": 1.0}
    lowered = (instruction or "").lower()
    for cls, needles in {
        "kir_inspection": ("kir", "ssa", "ir dump"),
        "compiler_feedback_loop": ("compiler", "kcc", "diagnostics"),
        "bugfix": ("bug", "fix", "error"),
        "code_review": ("review", "audit"),
        "kata": ("kata", "exercise"),
        "benchmark_probe": ("benchmark", "probe"),
        "syntax_read": ("read", "explain", "syntax"),
    }.items():
        if any(n in lowered for n in needles):
            return {"task_class": cls, "source": "heuristic", "confidence": 0.6}
    return {"task_class": "feature_impl", "source": "heuristic_default", "confidence": 0.2}


def build_karkain_context(instruction: str, *, mode: str = "model_only",
                          kir_snippet: str | None = None,
                          compiler_feedback: str | None = None) -> dict:
    if mode not in ("model_only", "compiler_assisted"):
        raise ValueError(f"mode must be 'model_only' or 'compiler_assisted', got {mode!r}")
    if mode == "compiler_assisted" and not (kir_snippet or compiler_feedback):
        compiler_feedback = None
        note = ("compiler_assisted requested but KCC is unavailable on this host "
                "(0xC0000135 / exit 3): context carries NO compiler evidence")
    else:
        note = "model_only: no compiler evidence attached by design"
    return {
        "language_id": "karkain",
        "mode": mode,
        "instruction": instruction,
        "kir_snippet": kir_snippet,
        "compiler_feedback": compiler_feedback,
        "note": note,
        "evidence": "none" if compiler_feedback is None else "caller_supplied",
    }


def validate() -> list[str]:
    problems: list[str] = []
    if not KARK_CONTRACT.is_file():
        problems.append(f"karkain learning contract missing: {KARK_CONTRACT}")
        return problems
    contract = load_karkain_contract()
    spec = contract.get("identity", {}).get("spec_reference", {})
    if spec.get("spec_version") != "1.1.0":
        problems.append("spec_version must remain pinned to 1.1.0")
    if contract.get("training_eligibility", {}).get("currently_met") != []:
        problems.append("karkain training eligibility must remain NOT MET without owner authorization")
    bench = REPO / "evaluation" / "l02_python_bench_v1.json"
    if bench.is_file() and hashlib.sha256(bench.read_bytes()).hexdigest() != FROZEN_PY_SHA:
        problems.append("frozen Python benchmark changed")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l062_karkain_inference.json")
    args = parser.parse_args(argv)

    problems = validate()
    if problems:
        for p in problems:
            print("KARK INFERENCE INVALID:", p)
        return 2

    metadata = karkain_inference_metadata()
    probe = identify_karkain_prompt(filename="examples/app.kark")
    classification = classify_karkain_task("Fix the borrow error in this function", None)
    context_model_only = build_karkain_context("Write a Karkain struct", mode="model_only")
    context_assisted = build_karkain_context("Write a Karkain struct", mode="compiler_assisted")

    report = {
        "schema": "taksh.l062_karkain_inference/1",
        "generated_by": "evaluation/l062_karkain_inference.py",
        "valid": True,
        "metadata": metadata,
        "probe": {
            "identify": probe,
            "classification": classification,
            "context_model_only": context_model_only,
            "context_compiler_assisted": context_assisted,
        },
        "capability": "NOT CLAIMED — no compiler evidence; no frozen KARK-BENCH results",
        "verdict": "KARK INFERENCE CONTRACT QUALIFIED (interfaces only; no Karkain execution)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"karkain inference contract valid; kcc_state recorded; no compile attempted")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
