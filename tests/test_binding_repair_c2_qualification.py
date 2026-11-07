"""Frozen tests for the C2 binding-repair qualification dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_binding_repair_qualification import canonical_digest

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "evaluation" / "binding_repair_qualification_v2.json"
EXPECTED_SHA = "5ff3561878c3674c627323105018071dd8f23521723a7b897e5478fc5a918230"
EXPECTED_DIGEST = "d96ee2de638236bc743dd96280dbefeb1c93a481eed2bdf6bf0188c720087d21"


def load_suite():
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_c2_dataset_is_frozen():
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == EXPECTED_SHA
    suite = load_suite()
    assert suite["status"] == "frozen_before_evaluation"
    assert suite["case_count"] == len(suite["cases"]) == 35
    assert canonical_digest(suite["cases"]) == EXPECTED_DIGEST == suite["digest"]
    pairs = [(case["instruction"], case["signature"]) for case in suite["cases"]]
    assert len(pairs) == len(set(pairs))


def test_c2_negative_cases_are_never_execution_approved():
    suite = load_suite()
    negatives = [case for case in suite["cases"] if case["negative"]]
    assert negatives
    assert all(case["execution_safe"] is False for case in negatives)
