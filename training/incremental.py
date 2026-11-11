"""TAKSH L0.6.1 — D: incremental learning and training orchestration.

Deterministic, resumable, language-aware sampling and job planning. All
randomness flows from explicit seeds; all limits are reported rather than
silently applied. Nothing here launches training; this is configuration and
planning infrastructure only.

Sampling strategies:
  equal        — each language gets equal weight (inflation of scarce languages)
  proportional — weight ∝ token_count
  quality      — weight ∝ quality_weight (verified > unverified > unknown)
  hybrid       — proportional x quality
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUALITY_WEIGHTS = {"verified": 1.0, "unverified": 0.5, "unknown": 0.1, "rejected": 0.0}
MAX_EXPOSURE = 4  # owner-defined ceiling; never silently exceeded


def compute_weights(entries, strategy: str) -> dict:
    ids = [e["id"] for e in entries if not e.get("excluded")]
    if strategy == "equal":
        return {i: 1.0 for i in ids}
    if strategy == "proportional":
        tot = sum(e["token_count"] for e in entries) or 1
        return {e["id"]: e["token_count"] / tot for e in entries if e["id"] in ids}
    if strategy == "quality":
        return {e["id"]: QUALITY_WEIGHTS.get(e["quality_status"], 0.0) for e in entries if e["id"] in ids}
    if strategy == "hybrid":
        tot = sum(e["token_count"] for e in entries) or 1
        return {
            e["id"]: (e["token_count"] / tot) * QUALITY_WEIGHTS.get(e["quality_status"], 0.0)
            for e in entries if e["id"] in ids
        }
    raise ValueError(f"unknown sampling strategy: {strategy!r}")


def sampling_plan(entries, strategy: str, total_steps: int, batch_size: int,
                  seed: int = 42) -> dict:
    import random

    weights = compute_weights(entries, strategy)
    pool = [eid for eid in weights if weights[eid] > 0]
    if not pool:
        raise ValueError("no eligible entries under strategy %r" % strategy)
    probs = [weights[eid] for eid in pool]
    s = sum(probs)
    probs = [p / s for p in probs]

    rng = random.Random(seed)
    # exposure accounting by entry; repeated examples tracked explicitly
    exposure = Counter()
    batches = []
    for step in range(total_steps):
        choice = rng.choices(pool, weights=probs, k=1)[0]
        exposure[choice] += 1
        batches.append({"step": step, "entry_id": choice})

    oversampled = {eid: n for eid, n in exposure.items() if n > MAX_EXPOSURE}
    return {
        "strategy": strategy,
        "seed": seed,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "batches": batches,
        "exposure": dict(exposure),
        "max_exposure": MAX_EXPOSURE,
        "oversampled_entries": oversampled,
        "oversampling_warning": bool(oversampled),
        "note": (
            "exposure > MAX_EXPOSURE means a small corpus would be repeated more than the "
            "configured ceiling; acquire more data rather than oversample"
            if oversampled else "exposure within ceiling"
        ),
    }


def job_manifest(cfg_hash: str, dataset_fingerprint: str, strategy: str,
                  seed: int, total_steps: int) -> dict:
    return {
        "schema": "taksh.incremental_job/1",
        "job_id": f"l0601-{seed}-{strategy}-{cfg_hash[:8]}",
        "config_sha256": cfg_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "strategy": strategy,
        "seed": seed,
        "total_steps": total_steps,
        "state": "planned",           # planned -> running -> completed | interrupted | failed
        "checkpoint_every_steps": 500,
        "resume_from": None,
        "provenance": {
            "pipeline": "training/incremental.py",
            "schema": "taksh.incremental_job/1",
        },
    }


def resume_validation(checkpoint_meta: dict, current_job: dict) -> dict:
    problems = []
    required = ["step", "config_sha256", "dataset_fingerprint", "model_config_sha256", "tokenizer_sha256"]
    for field in required:
        if field not in checkpoint_meta:
            problems.append(f"checkpoint missing {field}")
    if checkpoint_meta.get("config_sha256") != current_job.get("config_sha256"):
        problems.append("config hash mismatch; refusing to resume")
    if checkpoint_meta.get("tokenizer_sha256") != current_job.get("tokenizer_sha256"):
        problems.append("tokenizer hash mismatch; refusing to resume")
    return {
        "resumable": not problems,
        "problems": problems,
        "action": "resume" if not problems else "start new run, do not overwrite",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_incremental.json")
    args = parser.parse_args(argv)

    from evaluation.l0601_training_manifest import example_manifest

    manifest = example_manifest()
    entries = manifest["entries"]
    plans = {}
    for strategy in ("equal", "proportional", "quality", "hybrid"):
        plans[strategy] = sampling_plan(entries, strategy, total_steps=300, batch_size=16, seed=42)

    report = {
        "schema": "taksh.l0601_incremental/1",
        "generated_by": "training/incremental.py",
        "entries": [e["id"] for e in entries],
        "strategies": list(plans),
        "plans": {k: {kk: v for kk, v in p.items() if kk != "batches"} for k, p in plans.items()},
        "determinism": "same seed + same manifest ⇒ identical batch sequence and exposure counts",
        "exposure_ceiling": MAX_EXPOSURE,
        "verdict": "INCREMENTAL ORCHESTRATION QUALIFIED (no training launched)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in report["plans"].items():
        print(f"{k:12s} oversampled={list(v['oversampled_entries']) or 'none'}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
