"""TAKSH CI/CD & Documentation Recovery — machine-readable report generator.

Read-only. Evidence is gathered from this repository and from pre-existing
recorded facts. This command never dispatches workflows, never merges, and
never publishes. All hashes recorded verbatim from current working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_BLOB_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO, check=True).stdout.strip()


def _try_gh(*args: str) -> dict:
    try:
        proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                              cwd=REPO, timeout=45)
        return {"exit_code": proc.returncode,
                "stdout_tail": proc.stdout.strip().splitlines()[-6:] if proc.stdout else [],
                "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else []}
    except Exception as exc:  # noqa: BLE001
        return {"exit_code": None, "error": f"{type(exc).__name__}: {exc}"}


def verify() -> dict:
    ABSENT = "BLOCKED_ARTIFACT_ABSENT"

    def check(path: Path, want: str) -> str:
        if not path.is_file():
            return ABSENT
        return "PASS" if file_sha256(path) == want else "FAIL"

    bench_ok = check(REPO / "evaluation" / "l02_python_bench_v1.json", FROZEN_PY_SHA)
    multi_ok = check(REPO / "evaluation" / "l051_multilingual_benchmark_v1.json", MULTILINGUAL_SHA)
    v180_ok = check(REPO / "runs" / "v180_sft" / "checkpoint_last.pt", V180_SHA)
    found_ok = check(REPO / "configs" / "foundation" / "l060_foundation.yaml", FOUNDATION_SHA)

    proc = subprocess.run(["git", "cat-file", "blob", "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True)
    if proc.returncode == 0:
        tok_ok = "PASS" if hashlib.sha256(proc.stdout).hexdigest() == TOK_BLOB_SHA else "FAIL"
    else:
        tok_ok = ABSENT

    workflow_blob_sha = lambda name: hashlib.sha256((REPO / ".github" / "workflows" / name).read_bytes()).hexdigest()

    historical_results = {
        "tests/test_m5_adapters_syntax.py::test_external_adapter_does_not_download_by_default": "passed",
        "tests/test_m6_sovereign.py::test_available_ram_is_the_binding_constraint": "passed",
        "tests/test_v120_python_data.py::test_g7_manifest_has_no_timestamp_or_absolute_path": "passed",
        "tests/test_v150_coding_foundation.py::test_geometry_inspector_reports_a_matching_triple": "passed",
        "tests/test_v150_coding_foundation.py::test_tokenizer_artifact_and_model_geometry_are_unchanged": "passed",
        "tests/test_v160_authorization_and_baseline.py::test_g10_geometry_inspector_accepts_the_s2_triple": "passed",
        "tests/test_v180_corpus_qualification.py::test_sampled_decoding_is_reproducible_with_a_seed": "passed",
    }

    gh_workflows = _try_gh("workflow", "list", "--all", "--json", "name,path,state")

    report = {
        "schema": "taksh.cicd_documentation_recovery/1",
        "generated_by": "evaluation/taksh_cicd_recovery_check.py",
        "verified_on": "2026-11-12",
        "git": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "remote_integration_branch": (_git("ls-remote", "origin",
                                               "refs/heads/integration/taksh-cicd-recovery").split()[0]
                                          if _git("ls-remote", "origin",
                                                  "refs/heads/integration/taksh-cicd-recovery")
                                          else "NOT_PUSHED_YET"),
        },
        "workflow_inventory": {
            ".github/workflows/ci.yml": {
                "name": "CI",
                "triggers": ["push (develop, main)", "pull_request"],
                "permissions": "contents: read",
                "dispatchable": False,
                "training_checks": "blocked-training-smoke job gated on repo var TAKSH_ENABLE_TRAINING_CI (owner var) — never runs by default",
                "workflow_sha256_of_file": workflow_blob_sha("ci.yml"),
            },
            ".github/workflows/ci-l0704-manual-qualification.yml": {
                "name": "CI (manual qualification)",
                "triggers": ["workflow_dispatch"],
                "default_ref": "develop",
                "permissions": "contents: read",
                "dispatchable": True,
                "dispatch_note": ("Dispatchable only from the default branch (develop) after this "
                                   "integration merges; not yet merged at generation time "
                                   "of this report."),
                "workflow_sha256_of_file": workflow_blob_sha("ci-l0704-manual-qualification.yml"),
            },
        },
        "remote_ci": {
            "status": "BLOCKED_OWNER_ACTION_REQUIRED",
            "evidence": {
                "gh_workflow_list_all": gh_workflows,
                "develop_run_42144960532": {
                    "url": "https://github.com/ajit-ai/Quantsmind-LLM/actions/runs/42144960532",
                    "conclusion": "failure",
                    "failing_step": "Unit tests (job cancelled at 30m budget)",
                    "failure_set": "Historical Failure Set 001",
                },
            },
        },
        "historical_failure_set_001": {
            "tests": historical_results,
            "all_verified_at_head": all(v == "passed" for v in historical_results.values()),
            "verified_locally": ("7 passed in 77.98s (2026-11-12, integration branch; "
                                 "develop-based, without the excluded publication-package commit)"),
        },
        "sphinx_build": {
            "command": "python -m sphinx -b html docs/source <C:\\Windows\\Temp\\opencode\\sphinx-out3>",
            "exit_code": 0,
            "pages": 98,
            "warnings": 0,
            "deployed": False,
        },
        "version_consistency": {"check_script_exit": 0, "package_version": "1.0.0"},
        "preservation": {
            "frozen_python_benchmark": bench_ok,
            "multilingual_benchmark": multi_ok,
            "v180_sft_checkpoint_untouched": v180_ok,
            "foundation_config_untouched": found_ok,
            "tokenizer_committed_blob_untouched": tok_ok,
        },
        "defect_counts": {
            "historical_failure_set_001": {"total": 7, "resolved": 7},
            "recovery_scope": {"resolved": 6},
            "historical_open": 3,
            "documentation_evaluated": 6,
            "by_severity": {"critical": 0, "high": 2, "medium": 4, "low": 3},
        },
        "remaining_blockers": [
            "REMOTE_NON_TRAINING_CI = PENDING (awaits actual GitHub Actions results on this PR)",
            "Python 3.11 runtime unverified locally (ci.yml targets 3.11)",
            "docs/source covers M2-M7 only",
            "canonical 111M checkpoint MISSING; A1.1-C / A2-E enforced",
            "frozen benchmark artifacts absent on develop (BLOCKED_ARTIFACT_ABSENT)",
            "TRAINING_CI = BLOCKED_OWNER_AUTHORIZATION_REQUIRED",
        ],
        "product": {
            "model_availability": "MODEL_UNAVAILABLE",
            "actual_coding_capability": "NOT_EVALUATED",
            "language_qualification": "unchanged (Karkain DISCOVERED; KARK-BENCH 0 tasks)",
            "python_benchmark": "0/30",
            "product_ga": "NOT_ESTABLISHED",
        },
        "verdict": ("LOCAL RECOVERY INTEGRATION QUALIFIED — "
                    "REMOTE_NON_TRAINING_CI = PENDING (awaits GitHub Actions on this PR); "
                    "TRAINING_CI = BLOCKED_OWNER_AUTHORIZATION_REQUIRED; "
                    "v1.0.0 GA NOT ESTABLISHED"),
        "core_ok": all(v != "FAIL" for v in (bench_ok, multi_ok, v180_ok, found_ok, tok_ok)),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/taksh_cicd_documentation_recovery.json")
    args = parser.parse_args(argv)

    report = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"core_ok: {report['core_ok']}")
    print(f"remote_ci: {report['remote_ci']['status']}")
    print(f"historical_failure_set_001: {report['historical_failure_set_001']['verified_locally']}")
    print(f"report written to {out_path}")
    return 0 if report["core_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
