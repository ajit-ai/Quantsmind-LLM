"""Tests for the C# evaluation surface and execution filter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from evaluation.csharp.diagnostics import (
    analyse_csharp_source,
    category_for,
    parse_roslyn_diagnostics,
    summarize_diagnostics,
)
from evaluation.csharp.runner import (
    compile_csharp,
    detect_csharp_environment,
    validate_target_framework,
)
from evaluation.csharp.suite import CSHARP_PROBES, evaluate_probe, probes_by_category
from serving.languages import is_supported_language, normalize_language, prompt_language
from training.exec_filter import evaluate_candidate

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "data" / "contracts" / "csharp_v1.yaml"
DOTNET_AVAILABLE = shutil.which("dotnet") is not None


def test_roslyn_diagnostics_are_typed():
    output = (
        "Program.cs(3,1): warning CS8602: Dereference of a possibly null reference.\n"
        "Program.cs(5,10): error CS1002: ; expected\n"
    )
    diagnostics = parse_roslyn_diagnostics(output)
    assert [item.code for item in diagnostics] == ["CS8602", "CS1002"]
    assert diagnostics[0].category == "nullable"
    assert diagnostics[0].line == 3
    summary = summarize_diagnostics(diagnostics)
    assert summary["errors"] == 1
    assert summary["nullable"] == 1


def test_async_heuristic_reports_blocking_and_async_void():
    source = (
        "async void Run() { var work = Task.Delay(1); var x = work.Result; await Task.CompletedTask; }\n"
        "async Task EmptyAsync() {}\n"
    )
    result = analyse_csharp_source(source)
    codes = {item["code"] for item in result["diagnostics"]}
    assert "TAK-ASYNC001" in codes
    assert "TAK-ASYNC002" in codes
    assert "TAK-ASYNC003" in codes
    assert result["summary"]["async"] == 3


def test_compiler_error_categories_use_numeric_code_ranges():
    # Parse errors live in the CS1xxx numeric band and are syntax errors.
    assert category_for("CS1002") == "syntax"
    assert category_for("CS1001") == "syntax"
    # Type/conversion compiler errors are semantic errors despite low numbers.
    assert category_for("CS0029") == "semantic"
    assert category_for("CS0101") == "semantic"
    # Nullable warnings keep their own category.
    assert category_for("CS8602") == "nullable"
    # Async completion warnings keep their own category.
    assert category_for("CS1998") == "async"
    assert category_for("CS4014") == "async"


def test_csharp_contract_is_explicit_and_not_a_release_claim():
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "proposed-not-released"
    assert contract["safety"]["execute_untrusted_code"] is False
    assert contract["safety"]["execution_requires_flag"] is True
    assert contract["target"]["framework"]
    assert contract["data"]["license_cleared_required"] is True
    assert contract["data"]["claims"] == "no C# capability measured"


def test_serving_language_normalization():
    assert normalize_language("C#") == "csharp"
    assert normalize_language("dotnet") == "csharp"
    assert normalize_language("CS") == "csharp"
    assert is_supported_language("csharp")
    assert not is_supported_language("cobol")
    assert prompt_language("csharp") == "C#"


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_detect_csharp_environment():
    env = detect_csharp_environment("net10.0")
    assert env.sdk_version.startswith("10.")
    assert env.target_framework == "net10.0"
    assert "NETCore" in (env.runtime_version or "") or env.runtime_version is not None


def test_target_framework_is_validated():
    assert validate_target_framework("net10.0") == "net10.0"
    with pytest.raises(ValueError):
        validate_target_framework("../net10.0")


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_linq_probe_compiles_without_execution():
    result = compile_csharp(
        "using System.Linq;\nvar xs=new[]{1,2,3}; System.Console.WriteLine(xs.Sum());\n",
        target_framework="net10.0",
        allow_execution=False,
        timeout=60,
    )
    assert result.outcome == "compiled"
    assert result.exit_code == 0
    assert "PASS" not in result.stdout


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_compile_failure_preserves_diagnostics():
    result = compile_csharp(
        "public static class Broken { public static void Main() { System.Console.WriteLine(1) } }",
        target_framework="net10.0",
        timeout=60,
    )
    assert result.outcome == "compile_failed"
    assert result.diagnostics


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_allow_execution_runs_hand_written_fixture():
    probe = next(item for item in CSHARP_PROBES if item.case_id == "csharp-unit-001")
    result = compile_csharp(
        probe.source, target_framework="net10.0",
        allow_execution=True, timeout=60,
    )
    assert result.outcome == "executed"
    assert result.exit_code == 0
    assert "PASS" in result.stdout


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_evaluate_probe_reports_async_source_diagnostic():
    probe = next(item for item in CSHARP_PROBES if item.case_id == "csharp-async-001")
    result = evaluate_probe(probe, target_framework="net10.0", allow_execution=False)
    assert result["matched_codes"] is True
    codes = {item["code"] for item in result["result"]["source_diagnostics"]}
    assert "TAK-ASYNC001" in codes


def test_probe_categories_are_grouped():
    grouped = probes_by_category()
    assert set(grouped) >= {"linq", "nullable", "async", "unit"}
    assert all(grouped[category] for category in grouped)


@pytest.mark.skipif(not DOTNET_AVAILABLE, reason="dotnet SDK is required")
def test_execution_filter_passes_compilable_candidate_without_running_it():
    result = evaluate_candidate(
        "csharp",
        "System.Console.WriteLine(\"compiled\");\n",
        allow_execution=False,
        target_framework="net10.0",
        timeout=60,
    )
    assert result.passed is True
    assert result.execution_authorized is False
    assert result.outcome == "compiled"


def test_execution_filter_rejects_unknown_language():
    with pytest.raises(ValueError, match="unsupported execution language"):
        evaluate_candidate("cobol", "DISPLAY \"HELLO\".")
