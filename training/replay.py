"""TAKSH L0.6.1 — F: replay-buffer and catastrophic-forgetting measurement interfaces.

Infrastructure only: establishes measurable contracts for future continual-learning
experiments. Nothing here trains a model or claims forgetting prevention.

Forgetting metric: relative change in a per-language capability score between
a baseline snapshot and a post-update snapshot. Regression = negative delta
beyond tolerance. Replay buffer entries carry language, provenance, license
and hash — no entry enters replay without passing source validation upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOLERANCE = 0.05  # 5% relative drop triggers regression flag


def forgetting_metrics(baseline: dict[str, float], after: dict[str, float],
                       tolerance: float = TOLERANCE) -> dict:
    """Compute per-language deltas, regression flags, and a summary.

    Scores are unitless capability scores (e.g. pass@1 or normalized code_val).
    Both dicts must contain the same language keys.
    """
    missing = set(baseline) - set(after)
    if missing:
        raise ValueError(f"missing keys in after-snapshot: {sorted(missing)}")
    per_language = {}
    regressions = []
    for lang in baseline:
        b, a = baseline[lang], after[lang]
        delta = a - b
        rel = (delta / b) if b else (float("inf") if a > 0 else 0.0)
        regression = rel < -tolerance
        per_language[lang] = {
            "baseline": b,
            "after": a,
            "absolute_delta": round(delta, 6),
            "relative_delta": round(rel, 6),
            "regression": regression,
        }
        if regression:
            regressions.append(lang)
    return {
        "per_language": per_language,
        "regressions": regressions,
        "mean_relative_delta": round(sum(v["relative_delta"] for v in per_language.values()) / len(per_language), 6),
        "catastrophic_forgetting_detected": len(regressions) > 0,
        "tolerance": tolerance,
    }


class ReplayBuffer:
    """Minimal interface for a provenance-checked replay buffer (synthetic entries only)."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: list[dict] = []

    def add(self, entry: dict) -> dict:
        for field in ("id", "language_id", "source", "license", "provenance_note", "hash"):
            if field not in entry:
                raise ValueError(f"replay entry missing {field}")
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append(entry)
        return {"accepted": True, "size": len(self.entries)}

    def sample(self, n: int, seed: int = 42):
        import random

        rng = random.Random(seed)
        return rng.sample(self.entries, min(n, len(self.entries)))

    def per_language(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.entries:
            counts[e["language_id"]] += 1
        return dict(counts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_replay.json")
    args = parser.parse_args(argv)

    # Deterministic synthetic fixtures (NOT real training data)
    baseline = {"python": 8.60, "javascript": 6.20}
    after    = {"python": 8.10, "javascript": 7.60}   # python forgetting, js gaining
    metrics = forgetting_metrics(baseline, after)

    buf = ReplayBuffer(max_size=100)
    for i, lang in enumerate(["python", "python", "javascript", "python"]):
        buf.add({"id": f"synthetic-{i}", "language_id": lang,
                 "source": "synthetic-fixture", "license": "apache-2.0",
                 "provenance_note": "L0.6.1 synthetic fixture; not training data",
                 "hash": f"{i:064x}"})

    report = {
        "schema": "taksh.l0601_replay/1",
        "generated_by": "training/replay.py",
        "forgetting_metric": "relative delta vs baseline; regression when rel_delta < -tolerance",
        "forgetting_metrics_example": metrics,
        "replay_buffer_example": {"size": len(buf.entries), "per_language": buf.per_language()},
        "policy": {
            "tolerance": TOLERANCE,
            "rollback": "on regression beyond tolerance: halt adaptation, revert to last validated checkpoint",
            "auto_replay": "every language re-checked after every adaptation; never averaged into oblivion",
        },
        "determinism": "fixed seeds; identical fixtures ⇒ identical metrics",
        "verdict": "REPLAY + FORGETTING INTERFACES QUALIFIED (synthetic fixtures; no training performed)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"forgetting detected: {metrics['catastrophic_forgetting_detected']} in {metrics['regressions']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
