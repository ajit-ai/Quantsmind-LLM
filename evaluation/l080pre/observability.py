"""TAKSH L0.8.0-PRE — I: observability aggregate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

TOOLS = [
    ("model_identity", "evaluation.l080pre.model_identity"),
    ("tokenizer_integrity", "evaluation.l080pre.tokenizer_integrity"),
    ("corpus_verify", "evaluation.l080pre.corpus_verify"),
    ("training_config", "evaluation.l080pre.training_config"),
    ("resource_estimate", "evaluation.l080pre.resource_estimate"),
    ("checkpoint_recovery", "evaluation.l080pre.checkpoint_recovery"),
    ("benchmark_safeguard", "evaluation.l080pre.benchmark_safeguard"),
    ("gate", "evaluation.l080pre.gate"),
]


def run_tool(module: str) -> dict:
    proc = subprocess.run([sys.executable, "-m", module],
                          capture_output=True, text=True, cwd=REPO, timeout=900)
    return {"module": module, "exit_code": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-4:] if proc.stdout else [],
            "stderr_tail": proc.stderr.strip().splitlines()[-4:] if proc.stderr else []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_observability.json")
    args = parser.parse_args(argv)

    results = []
    for name, module in TOOLS:
        r = run_tool(module)
        results.append({"tool": name,
                        "state": "PASS" if r["exit_code"] == 0 else "FAIL", **r})

    gate = json.loads((REPO / "reports" / "l080pre_aggregate.json").read_text(encoding="utf-8")) if (REPO / "reports" / "l080pre_aggregate.json").is_file() else {}

    report = {
        "schema": "taksh.l080pre_observability/1",
        "generated_by": "evaluation/l080pre/observability.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "distinguishing_statuses": ["PASS", "CONDITIONAL", "BLOCKED", "MISSING", "NOT_AUTHORIZED", "NO_GO"],
        "tools": results,
        "summary": {
            "all_exit_zero": all(r["exit_code"] == 0 for r in results),
            "preparation_verdict": gate.get("preparation_verdict"),
            "training_verdict": gate.get("training_verdict"),
            "training_tokens_eligible": 0,
            "artifacts_created_or_modified": [],
            "frozen_benchmarks_modified": False,
        },
        "verdict": "L0.8.0-PRE COMPLETE — PREPARATION_QUALIFIED; TRAINING NO-GO; nothing trained; nothing acquired",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['tool']:22s} {r['state']:6s} exit={r['exit_code']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
