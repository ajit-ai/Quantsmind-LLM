"""Focused tests for deployment helpers and performance benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.deployment import validate_environment, health_check, readiness_check
from evaluation.performance import run_benchmarks


def test_environment_validation_reports_python_and_files():
    env = validate_environment()
    assert env["python_ok"] is True
    assert env["all_required_files_present"] is True


def test_health_and_readiness_are_available():
    health = health_check(".")
    readiness = readiness_check(".")
    assert health["status"] == "ok"
    assert readiness["status"] == "ready"
    assert readiness["project"]["python_files"] > 0


def test_performance_benchmark_runs_and_records_measurements(tmp_path):
    report = run_benchmarks(REPO, iterations=2)
    assert report["schema"] == "taksh.performance_benchmark/1"
    assert "project_indexing" in report["results"]
    assert report["results"]["project_indexing"]["iterations"] == 2
    assert report["results"]["project_indexing"]["mean_seconds"] >= 0
