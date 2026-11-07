"""Taksh v2.3.0 — Generation quality canary.

Runs against selected checkpoints to detect degeneration independently of
training loss. Produces an explicit PASS / WARNING / FAIL outcome.
Work Package 5 of Taksh v2.3.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from taksh.v2_3_detection import detect_degeneration, DegenerationResult


class CanaryResult:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_hash: str,
        config: Dict[str, Any],
        generations: List[DegenerationResult],
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_hash = checkpoint_hash
        self.config = config
        self.generations = generations
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.outcome: str = "pending"
        self.summary: Dict[str, Any] = {}
    def compute_outcome(self) -> str:
        if not self.generations:
            self.outcome = "FAIL"
            self.summary = {"error": "no generations evaluated"}
            return self.outcome

        fails = sum(1 for g in self.generations if g.classification == "fail")
        warnings = sum(1 for g in self.generations if g.classification == "warning")
        passes = sum(1 for g in self.generations if g.classification == "pass")
        total = len(self.generations)

        fail_rate = fails / total
        warning_rate = warnings / total

        has_eos_failure = any(g.premature_eos for g in self.generations)
        has_identity_failure = any(
            w.type == "degenerate_semantics" and w.severity == "failure"
            for g in self.generations
            for w in g.warnings
        )

        self.summary = {
            "total_generations": total,
            "passes": passes,
            "warnings": warnings,
            "fails": fails,
            "pass_rate": round(passes / total, 4),
            "warning_rate": round(warning_rate, 4),
            "fail_rate": round(fail_rate, 4),
            "has_premature_eos": has_eos_failure,
            "has_identity_failure": has_identity_failure,
        }

        if fail_rate > 0.20 or has_identity_failure:
            self.outcome = "FAIL"
        elif warning_rate > 0.10 or has_eos_failure:
            self.outcome = "WARNING"
        else:
            self.outcome = "PASS"

        return self.outcome
        return self.outcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
            "summary": self.summary,
            "generations": [
                {
                    "generation_id": g.generation_id,
                    "classification": g.classification,
                    "total_tokens": g.total_tokens,
                    "unique_tokens": g.unique_tokens,
                    "repeat_rate": g.repeat_rate,
                    "premature_eos": g.premature_eos,
                    "excessive_repetition": g.excessive_repetition,
                    "degenerate_semantics": g.degenerate_semantics,
                    "warnings": [
                        {
                            "type": w.type,
                            "severity": w.severity,
                            "message": w.message,
                            "evidence": w.evidence,
                        }
                        for w in g.warnings
                    ],
                }
                for g in self.generations
            ],
        }

def run_canary(
    checkpoint_path: str,
    checkpoint_hash: str,
    generations: List[DegenerationResult],
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> CanaryResult:
    if config is None:
        config = {}

    result = CanaryResult(
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        config=config,
        generations=generations,
    )
    result.compute_outcome()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


def compare_to_baseline(
    canary: CanaryResult,
    baseline_results: List[DegenerationResult],
    baseline_hash: str,
) -> Dict[str, Any]:
    canary_fail = sum(1 for g in canary.generations if g.classification == "fail") / max(1, len(canary.generations))
    baseline_fail = sum(1 for g in baseline_results if g.classification == "fail") / max(1, len(baseline_results))

    canary_warn = sum(1 for g in canary.generations if g.classification == "warning") / max(1, len(canary.generations))
    baseline_warn = sum(1 for g in baseline_results if g.classification == "warning") / max(1, len(baseline_results))

    canary_repeat = sum(g.repeat_rate for g in canary.generations) / max(1, len(canary.generations))
    baseline_repeat = sum(g.repeat_rate for g in baseline_results) / max(1, len(baseline_results))

    return {
        "baseline_checkpoint_hash": baseline_hash,
        "candidate_checkpoint_hash": canary.checkpoint_hash,
        "baseline_fail_rate": round(baseline_fail, 4),
        "candidate_fail_rate": round(canary_fail, 4),
        "fail_rate_delta": round(canary_fail - baseline_fail, 4),
        "baseline_warning_rate": round(baseline_warn, 4),
        "candidate_warning_rate": round(canary_warn, 4),
        "warning_rate_delta": round(canary_warn - baseline_warn, 4),
        "baseline_avg_repeat_rate": round(baseline_repeat, 4),
        "candidate_avg_repeat_rate": round(canary_repeat, 4),
        "repeat_rate_delta": round(canary_repeat - baseline_repeat, 4),
        "regression_detected": canary_fail > baseline_fail,
        "improvement_detected": canary_fail < baseline_fail,
    }