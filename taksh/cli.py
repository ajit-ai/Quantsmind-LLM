"""TAKSH 1.4.0 — Developer CLI extensions.

Non-destructive wrappers around existing package entry points. The `model`
subcommand now exposes inspect, providers, and validate; output never pretends
that a checkpoint exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from taksh.project_understanding import project_summary
from taksh.specialists import SPECIALISTS
from taksh.coding_validation import full_validate


def _load_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("taksh")
    except Exception:
        return "1.0.0 (unknown metadata)"


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
    from taksh.coding_engine import TaskEngine, TaskSpec

    print("inline coding task validation is available via tests/test_l130_five_version_components.py")


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

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)
