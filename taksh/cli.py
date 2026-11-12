"""TAKSH 1.4.0 — Developer CLI extensions.

Non-destructive wrappers around existing package entry points. The `model`
subcommand now exposes inspect, providers, and validate; output never pretends
that a checkpoint exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taksh.project_understanding import project_summary
from taksh.specialists import SPECIALISTS
from taksh.coding_validation import full_validate


def _load_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("taksh")
    except Exception:
        from taksh import __release_status__, __version__

        return f"{__version__} ({__release_status__})"


def _cmd_inspect(args):
    root = Path(args.path)
    summary = project_summary(root)
    print(f"root: {summary['root']}")
    print(f"files: {summary['total_files']} ({summary['python_files']} Python)")
    print(f"bytes: {summary['total_bytes']:,}")
    print(f"symbols indexed: {summary['symbols_indexed']}")
    print(f"modules with imports: {summary['modules_with_imports']}")


def _cmd_providers(args):
    from serving import providers

    try:
        names = providers.list_providers()
    except Exception:
        names = ["taksh-s2"]
    print("registered providers:", ", ".join(names))


def _cmd_validate(args):
    source = Path(args.source).read_text(encoding="utf-8")
    root = Path(args.test_root) if args.test_root else None
    result = full_validate(source, test_root=root)
    print(json.dumps(result.to_dict(), indent=2))


def _cmd_specialists(args):
    print("Python specialists:")
    for name, cls in SPECIALISTS.items():
        print(f"  - {name}")


def _cmd_validate_task(args):
    print("inline coding task validation is available via tests/test_l130_five_version_components.py")


def _cmd_publication(args):
    from evaluation.publication_readiness import evaluate

    report = evaluate()
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"product: {report['product']['name']} {report['product']['version']}")
    print(f"model:   {report['capability']['model_availability']}")
    print(f"coding:  {report['capability']['actual_coding_capability']}")
    print("channels:")
    for channel in report["channels"]:
        print(f"  {channel['status']:<12} {channel['name']}")
    print(f"verdict: {report['verdict']['code']}")


def _cmd_capability(args):
    from evaluation.publication_readiness import evaluate

    capability = evaluate()["capability"]
    print(f"model_availability: {capability['model_availability']}")
    print(f"actual_coding_capability: {capability['actual_coding_capability']}")
    history = capability["historical_m7"]
    if history.get("exists"):
        print(f"historical M7: {history['locked_syntax']}; {history['functional']}; "
              f"evidence level {history['evidence_level']}")
    print(f"frozen Python benchmark: {capability['frozen_python_benchmark']['status']} "
          f"({capability['frozen_python_benchmark']['historical_result']})")
    print(f"production GA: {capability['production_ga']['status']}")


def _cmd_foundation(args):
    cfg = "configs/model/foundation_111m.yaml"
    from pathlib import Path

    exists = Path(cfg).is_file()
    print("native foundation config:", cfg)
    print("file present locally:", exists)
    if exists:
        import yaml

        doc = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        print(f"  id: {doc['id']} / params: {doc['model']['d_model']}x{doc['model']['n_layers']}L")


def _eng_provider(name, changes_file=None):
    from taksh.engineering import NativeTakshProvider, DeterministicTestProvider

    if name == "native":
        return NativeTakshProvider()
    if name == "deterministic":
        changes = []
        if changes_file:
            import json

            data = json.loads(Path(changes_file).read_text(encoding="utf-8"))
            from taksh.engineering import FileChange

            changes = [FileChange(**item) for item in data]
        return DeterministicTestProvider(changes)
    raise ValueError(f"unknown provider: {name}")


def _cmd_eng_inspect(args):
    from taksh.engineering import inspect_project

    info = inspect_project(args.path)
    summary = info["summary"]
    print(f"root: {info['root']}")
    print(f"files: {summary['total_files']} ({summary['python_files']} Python)")
    print(f"bytes: {summary['total_bytes']:,}")
    print(f"symbols indexed: {summary['symbols_indexed']}")
    print(f"modules with imports: {summary['modules_with_imports']}")
    print(f"tests discovered: {info['test_count']}")


def _cmd_eng_validate(args):
    from taksh.engineering import validate_project

    result = validate_project(args.path, args.files)
    print(json.dumps(result, indent=2))


def _cmd_eng_test(args):
    from taksh.engineering import run_tests

    result = run_tests(args.path, timeout=args.timeout)
    print(json.dumps(result, indent=2))


def _cmd_eng_plan(args):
    from taksh.engineering import create_task, plan_task, review_plan, inspect_project

    provider = _eng_provider(args.provider, args.changes_file)
    task = create_task(args.task_id, args.objective, args.root)
    context = inspect_project(args.root)
    task, plan = plan_task(task, provider, context)
    if plan is not None:
        review_plan(plan)
    if args.json:
        print(json.dumps({"task": task.to_dict(), "plan": plan.to_dict() if plan else None}, indent=2))
    else:
        print(f"state: {task.state.value}")
        if plan is None:
            print("plan: none (provider unavailable or failed)")
        else:
            print(f"provider: {plan.provider}")
            print(f"specialist: {plan.specialist}")
            print(f"changes: {len(plan.changes)}")
            for change in plan.changes:
                print(f"  {change.operation} {change.path}: {change.reason}")


def _cmd_eng_apply(args):
    import json as _json

    from taksh.engineering import apply_plan, ChangePlan, FileChange

    data = _json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
    plan = ChangePlan(
        task_id=data["task_id"],
        root=args.root,
        changes=[FileChange(**item) for item in data.get("changes", [])],
        approved=data.get("approved", False),
    )
    result = apply_plan(args.root, plan, approved=args.approve)
    print(_json.dumps(result.to_dict(), indent=2))


def _cmd_eng_run(args):
    import json as _json

    from taksh.engineering import (
        create_task,
        inspect_project,
        run_engineering_task,
        engineering_report,
    )

    provider = _eng_provider(args.provider, args.changes_file)
    task = create_task(args.task_id, args.objective, args.root, max_repair_iterations=args.max_repairs)
    task = run_engineering_task(
        task,
        provider,
        apply=args.apply,
        validate=args.validate,
        run_project_tests=args.test,
    )
    report = engineering_report(task)
    if args.json:
        print(_json.dumps(report, indent=2))
    else:
        print(f"state: {task.state.value}")
        print(f"validation: {task.validation.get('status') if task.validation else 'none'}")
        if task.test_result:
            print(f"tests: {task.test_result.get('status')}")
        print(f"capability_claim: {task.capability_claim}")


def build_parser():
    parser = argparse.ArgumentParser(prog="taksh", description="TAKSH developer CLI")
    parser.add_argument("--version", action="version", version=f"Taksh {_load_version()}")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("inspect", help="summarize a project root")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("providers", help="list registered inference providers")
    p.set_defaults(func=_cmd_providers)

    p = sub.add_parser("validate", help="validate a Python source file syntactically/semantically")
    p.add_argument("source")
    p.add_argument("--test-root", dest="test_root", default=None)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("specialists", help="list Python specialist interfaces")
    p.set_defaults(func=_cmd_specialists)

    p = sub.add_parser("foundation", help="report foundation configuration status")
    p.set_defaults(func=_cmd_foundation)

    p = sub.add_parser("task", help="report coding-task integration status")
    p.set_defaults(func=_cmd_validate_task)

    p = sub.add_parser("publication", help="report distribution and publication readiness")
    p.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    p.set_defaults(func=_cmd_publication)

    p = sub.add_parser("capability", help="report the honest capability state")
    p.set_defaults(func=_cmd_capability)

    # ---- engineering platform (DLCAE-integrated, model-independent) ----
    p = sub.add_parser("engineer", help="integrated multi-file engineering platform")
    eng_sub = p.add_subparsers(dest="eng_cmd")

    q = eng_sub.add_parser("inspect", help="inspect a project root")
    q.add_argument("path", nargs="?", default=".")
    q.set_defaults(func=_cmd_eng_inspect)

    q = eng_sub.add_parser("validate", help="validate Python files in a project")
    q.add_argument("path", nargs="?", default=".")
    q.add_argument("--files", nargs="*", default=None)
    q.set_defaults(func=_cmd_eng_validate)

    q = eng_sub.add_parser("test", help="discover and run project tests")
    q.add_argument("path", nargs="?", default=".")
    q.add_argument("--timeout", type=int, default=120)
    q.set_defaults(func=_cmd_eng_test)

    q = eng_sub.add_parser("plan", help="plan a coding task")
    q.add_argument("--task-id", required=True)
    q.add_argument("--objective", required=True)
    q.add_argument("--root", required=True)
    q.add_argument("--provider", choices=["native", "deterministic"], default="native")
    q.add_argument("--changes-file", default=None)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_eng_plan)

    q = eng_sub.add_parser("apply", help="apply an approved plan JSON file")
    q.add_argument("--plan-file", required=True)
    q.add_argument("--root", required=True)
    q.add_argument("--approve", action="store_true")
    q.set_defaults(func=_cmd_eng_apply)

    q = eng_sub.add_parser("run", help="run an engineering task end to end")
    q.add_argument("--task-id", required=True)
    q.add_argument("--objective", required=True)
    q.add_argument("--root", required=True)
    q.add_argument("--provider", choices=["native", "deterministic"], default="native")
    q.add_argument("--changes-file", default=None)
    q.add_argument("--apply", action="store_true")
    q.add_argument("--validate", action="store_true", default=True)
    q.add_argument("--test", action="store_true")
    q.add_argument("--max-repairs", type=int, default=3)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_eng_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
