"""Taksh release metadata.

The repository name and compatibility identifiers remain ``Quantsmind-LLM`` /
``quantsmind``; the product-facing name is Taksh.
"""

from __future__ import annotations

import argparse
import json
import sys

__title__ = "Taksh"
__version__ = "1.0.0"
__release_status__ = "final"


def _print_provider_table() -> int:
    from serving.provider_diagnostics import diagnostics

    table = diagnostics()
    for name, info in sorted(table.items()):
        print(
            f"{name:28} status={info.get('status'):24} "
            f"available={str(info.get('available')).lower():5} "
            f"test_only={str(info.get('test_only')).lower():5} "
            f"model={info.get('model_id') or '-'}"
        )
    return 0


def _confirm(plan) -> bool:
    if not plan.changes:
        print("No structured changes were proposed.")
        return False
    print(f"Proposed change plan: {plan.summary}")
    for change in plan.changes:
        print(f"  {change.operation:7} {change.path}  # {change.reason}")
    try:
        answer = input("Apply these changes? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _run_code(args) -> int:
    from serving import coding_providers  # noqa: F401 - registers providers
    from serving import native_provider  # noqa: F401 - registers native-taksh
    from serving.provider_diagnostics import diagnostic
    from serving.providers import get_provider
    from taksh.engineering import EngineeringError, SecurityError, run_coding_task

    if args.list_providers:
        return _print_provider_table()

    if args.diagnose:
        print(json.dumps(diagnostic(args.diagnose), indent=2, sort_keys=True))
        return 0

    if not args.task:
        print("error: --task is required unless --list-providers or --diagnose is used", file=sys.stderr)
        return 2

    try:
        provider = get_provider(args.provider)
    except Exception as exc:  # noqa: BLE001 - user-facing error
        print(f"error: provider {args.provider!r} is unavailable: {exc}", file=sys.stderr)
        return 2

    capabilities = provider.capabilities()
    if capabilities.test_only and not args.allow_test_provider:
        print(
            "error: this is a deterministic test provider and cannot be used silently; "
            "pass --allow-test-provider for a demonstration only",
            file=sys.stderr,
        )
        return 2
    if not capabilities.configured:
        print(
            f"error: provider {args.provider!r} is not configured; "
            f"status={capabilities.status}; run --diagnose {args.provider}",
            file=sys.stderr,
        )
        return 2

    reviewer = None
    if args.apply:
        if args.yes:
            reviewer = lambda plan: True
        elif sys.stdin.isatty():
            reviewer = _confirm
        else:
            print(
                "error: applying changes requires explicit approval; use --yes in non-interactive mode",
                file=sys.stderr,
            )
            return 2

    try:
        report = run_coding_task(
            args.project,
            args.task,
            provider,
            reviewer=reviewer,
            apply=args.apply,
            allow_file_contents=args.allow_file_contents,
            run_project_tests=args.run_tests,
            test_timeout=int(args.timeout),
        )
    except (SecurityError, EngineeringError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        workflow = report.get("workflow", {})
        task = report.get("task", {})
        print(f"provider: {args.provider}")
        print(f"workflow: {workflow.get('status')}")
        print(f"approved: {workflow.get('approved')}")
        print(f"state: {task.get('state')}")
        for change in (task.get("plan") or {}).get("changes", []):
            print(f"  {change.get('operation'):7} {change.get('path')}")
        validation = task.get("validation") or {}
        if validation:
            print(f"validation: {validation.get('status')} rolled_back={validation.get('rolled_back', False)}")
        tests = task.get("test_result") or {}
        if tests:
            print(f"tests: {tests.get('status')} rc={tests.get('returncode')}")
        if workflow.get("error"):
            print(f"error: {workflow['error']}", file=sys.stderr)
    return 0 if report.get("workflow", {}).get("status") in ("COMPLETED", "PLANNED", "VALIDATED") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="taksh", description="Taksh release utilities")
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__title__} {__version__} ({__release_status__})",
    )
    subparsers = parser.add_subparsers(dest="command")

    code = subparsers.add_parser("code", help="run the provider-independent coding workflow")
    code.add_argument("--project", default=".", help="project root (default: current directory)")
    code.add_argument("--task", help="coding objective")
    code.add_argument("--provider", default="openai-compatible", help="registered provider name")
    code.add_argument("--apply", action="store_true", help="apply the approved plan")
    code.add_argument("--yes", action="store_true", help="non-interactive explicit approval")
    code.add_argument("--allow-file-contents", action="store_true", help="allow bounded file contents to be sent to the provider")
    code.add_argument("--run-tests", action="store_true", help="run discovered project tests after applying")
    code.add_argument("--allow-test-provider", action="store_true", help="allow deterministic test-only providers")
    code.add_argument("--timeout", type=float, default=120, help="test command timeout in seconds")
    code.add_argument("--list-providers", action="store_true", help="list registered providers and status")
    code.add_argument("--diagnose", metavar="NAME", help="print one provider's non-secret diagnostic")
    code.add_argument("--json", action="store_true", help="print the full workflow report as JSON")

    args = parser.parse_args(argv)
    if args.command == "code":
        return _run_code(args)
    return 0
