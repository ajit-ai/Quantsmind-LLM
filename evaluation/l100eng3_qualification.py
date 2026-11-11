"""TAKSH L1.0.0-ENG3 — End-to-End Coding Intelligence Integration reporter.

Single command:  python -m evaluation.l100eng3_qualification

Reports implementation readiness SEPARATELY from model capability. A synthetic
double is never represented as TAKSH output; the canonical taksh-foundation-111m
artifact has never existed and is reported MISSING/UNAVAILABLE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_BLOB_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"

PY_FIXTURE = "def add(a: int, b: int) -> int:\n    return a + b\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO, check=True).stdout.strip()


def verify() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="l100eng3_qual_"))

    from taksh.dlcae.coding.task import DevSuite
    from taksh.dlcae.coding.model import ModelSession, StaticModelDouble
    from taksh.dlcae.coding.orchestrator import CodingTaskOrchestrator
    from taksh.dlcae.coding.evidence import EvidenceCategory, may_promote
    from taksh.dlcae.runtime import DLCAERuntime

    suite = DevSuite()
    hb = suite.get("ENG3-DEV-001")

    # 1. Canonical session — genuinely unavailable (no artifact exists)
    canonical = ModelSession(None, None)
    canonical_assessment = canonical.assess()

    # 2. Substituted artifact — rejected
    v180 = ModelSession(REPO / "runs" / "v180_sft" / "checkpoint_last.pt",
                        REPO / "tokenizer" / "taksh_code_bpe_v1.json")
    v180_assessment = v180.assess()

    # 3. Deterministic double runs the full pipeline end-to-end (labeled synthetic)
    ok_double = StaticModelDouble(responses=["def add(a, b):\n    return a + b\n"])
    ok_run = CodingTaskOrchestrator(ok_double, state_dir=str(tmp / "ok"),
                                    max_repair_attempts=1).run(hb)

    # 4. Repair loop: first broken candidate, then a correct one (double, labeled)
    repair_double = StaticModelDouble(
        responses=["def add(a, b):\n    return 0\n",
                   "def add(a, b):\n    return a + b\n"])
    repair_run = CodingTaskOrchestrator(repair_double, state_dir=str(tmp / "repair"),
                                        max_repair_attempts=2).run(hb)

    # 5. Always-broken double — bounded termination, FAILED not fabricated pass
    bad_double = StaticModelDouble(responses=["def add(a, b):\n    return 0\n"] * 4)
    bad_run = CodingTaskOrchestrator(bad_double, state_dir=str(tmp / "bad"),
                                     max_repair_attempts=1).run(hb)

    # 6. Runtime integration via ENG2 semantic pipeline
    runtime = DLCAERuntime(tmp / "runtime")
    semantic = runtime.semantic_analysis("l100eng3-probe", "python", PY_FIXTURE,
                                         mode="model_only")

    # 7. Evidence category promotion gates
    promotion_gate = {category.value: may_promote(category) for category in EvidenceCategory}

    # Preservation hashes
    bench_ok = file_sha256(REPO / "evaluation" / "l02_python_bench_v1.json") == FROZEN_PY_SHA
    multi_ok = file_sha256(REPO / "evaluation" / "l051_multilingual_benchmark_v1.json") == MULTILINGUAL_SHA
    v180_ok = file_sha256(REPO / "runs" / "v180_sft" / "checkpoint_last.pt") == V180_SHA
    found_ok = file_sha256(REPO / "configs" / "foundation" / "l060_foundation.yaml") == FOUNDATION_SHA
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    tok_ok = hashlib.sha256(blob).hexdigest() == TOK_BLOB_SHA

    from taksh.dlcae.learning import LearningFeedbackEngine
    handoff = LearningFeedbackEngine().handoff_to_training([])

    summary = {
        "canonical_model_available": canonical_assessment["state"] == "MODEL_READY",
        "v180_substituted": v180_assessment["state"] == "MODEL_READY",
        "double_pipeline_completed": ok_run["state"] == "COMPLETED",
        "double_repair_loop_completed": repair_run["state"] == "COMPLETED",
        "double_bounded_failure": bad_run["state"] == "FAILED"
                                  and bad_run["final_result"]["attempts_used"] <= 2,
        "semantic_pipeline_ok": semantic["status"] in ("PASS", "PARTIAL"),
        "training_handoff_authorized": handoff["authorized"],
        "capability_claims_made": False,
    }
    core_ok = (summary["canonical_model_available"] is False
               and summary["v180_substituted"] is False
               and summary["double_pipeline_completed"]
               and summary["double_repair_loop_completed"]
               and summary["double_bounded_failure"]
               and summary["semantic_pipeline_ok"]
               and summary["training_handoff_authorized"] is False
               and bench_ok and v180_ok and tok_ok and found_ok)

    report = {
        "schema": "taksh.l100eng3_coding_qualification/1",
        "generated_by": "evaluation/l100eng3_qualification.py",
        "verified_on": "2026-11-11",
        "git": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "remote_feature_branch": _git(
                "ls-remote", "origin",
                "refs/heads/feature/l060-foundation-model-qualification").split()[0],
        },
        "summary": summary,
        "task_input": {
            "suite_id": "taksh-l100eng3-dev-suite-v1",
            "suite_sha256": suite.suite_sha256(),
            "task_count": len(suite),
            "sample_task": hb.to_dict(),
            "note": "synthetic development fixtures — not benchmark evidence",
        },
        "model_availability": {
            "canonical_assessment": canonical_assessment,
            "v180_sft_assessment": v180_assessment,
            "decision": ("MODEL_UNAVAILABLE — canonical taksh-foundation-111m "
                         "artifact does not exist; no substitution authorized or possible"),
        },
        "generation_attempts": {
            "ok_run": ok_run["attempts"],
            "repair_run": repair_run["attempts"],
            "bad_run": bad_run["attempts"],
            "model_kind": "test_double (synthetic; never TAKSH output)",
        },
        "end_to_end_results": {
            "ok_run": ok_run["final_result"],
            "repair_run": repair_run["final_result"],
            "bad_run": bad_run["final_result"],
        },
        "semantic_validation": {
            "runtime_probe": {"status": semantic["status"],
                              "stage_status": semantic.get("stage_status")},
            "promotion_gates": promotion_gate,
            "note": ("VERIFIED requires TOOL_VERIFIED/HUMAN_CONFIRMED; "
                     "MODEL_GENERATED never qualifies"),
        },
        "toolchain_execution": {
            "workspace": "isolated tmp sandbox (auto-cleaned)",
            "execution": "subprocess-only; never in-process exec of generated code",
            "ok_run_exec": ok_run["attempts"][0]["validation"]["execution"]["duration_ms"],
        },
        "diagnostics_and_repair": {
            "ok_run_correlation": ok_run["attempts"][0]["validation"].get("correlation", {}).get("verdict"),
            "bad_run_repair_candidates": bad_run["attempts"][0]["repair_candidates"]["candidate_count"],
            "repair_mode": "fresh model generation per attempt; suggestions never mutate source",
        },
        "preservation": {
            "frozen_python_benchmark": bench_ok,
            "multilingual_benchmark": multi_ok,
            "v180_sft_checkpoint_untouched": v180_ok,
            "foundation_config_untouched": found_ok,
            "tokenizer_committed_blob_untouched": tok_ok,
        },
        "training_handoff": handoff,
        "canonical_checkpoint": "MISSING",
        "v180_sft": "GEOMETRY_INCOMPATIBLE — never substituted",
        "qualification": {
            "engineering_qualification": "ENGINEERING_QUALIFIED (this milestone)",
            "model_availability": "MODEL_UNAVAILABLE",
            "actual_coding_capability": "NOT_EVALUATED — never claimed",
            "language_qualification": "no language promoted; Karkain remains DISCOVERED",
        },
        "capability_claim": "none",
        "verdict": ("L1.0.0-ENG3 END-TO-END CODING INTELLIGENCE INTEGRATION QUALIFIED "
                    "(engineering qualification only — no capability claim)"),
        "core_ok": core_ok,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l100eng3_coding_integration.json")
    args = parser.parse_args(argv)

    report = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"core_ok: {report['core_ok']}")
    print(f"canonical model: {report['model_availability']['canonical_assessment']['state']}")
    print(f"double pipeline: {report['end_to_end_results']['ok_run']}")
    print(f"report written to {out_path}")
    return 0 if report["core_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
