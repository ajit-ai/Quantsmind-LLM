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
