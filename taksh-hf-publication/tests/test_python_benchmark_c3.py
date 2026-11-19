"""Frozen tests for the C3 independent Python benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_binding_repair_qualification import canonical_digest

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "evaluation" / "python_benchmark_development_v1.json"
HELD = REPO / "evaluation" / "python_benchmark_heldout_v1.json"
DEV_SHA = "4b2536aac52696bb45fa4e2ecf4513c26cc35d64af37352504e79d98da8722fd"
DEV_DIGEST = "e53427edcb5a0af002c00a1abc12f63f8c443ff181dda62be359c44888e8696a"
HELD_SHA = "1b72f5388d86372582f03ae8cdf78ca031bfe34d27c0e3d9bbdc7cac4532a26e"
HELD_DIGEST = "c47fee4725749dae56a35e99fffeb9dc18ef29b9da4f9cc4beb7ef06f13e9c12"


def load_suite(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_development_dataset_is_frozen():
    assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA
    suite = load_suite(DEV)
    assert suite["status"] == "frozen_before_evaluation"
    assert suite["case_count"] == len(suite["cases"]) == 38
    assert canonical_digest(suite["cases"]) == DEV_DIGEST == suite["digest"]
    pairs = [(case["instruction"], case["signature"]) for case in suite["cases"]]
    assert len(pairs) == len(set(pairs))


def test_heldout_dataset_is_frozen():
    assert hashlib.sha256(HELD.read_bytes()).hexdigest() == HELD_SHA
    suite = load_suite(HELD)
    assert suite["status"] == "frozen_before_evaluation"
    assert suite["case_count"] == len(suite["cases"]) == 54
    assert canonical_digest(suite["cases"]) == HELD_DIGEST == suite["digest"]
    pairs = [(case["instruction"], case["signature"]) for case in suite["cases"]]
    assert len(pairs) == len(set(pairs))


def test_all_negative_cases_are_execution_locked():
    for path in (DEV, HELD):
        suite = load_suite(path)
        negatives = [case for case in suite["cases"] if case["negative"]]
        assert negatives
        assert all(case["execution_safe"] is False for case in negatives)
