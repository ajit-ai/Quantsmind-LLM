"""M6 command-line entry point.

Each M6 module already has its own ``python -m`` entry point. This aggregator
exists so a full M6 sweep can be driven from one place, and so the stage order
is explicit: inventory -> resources -> evaluate -> report.

    python -m evaluation.m6_cli inventory
    python -m evaluation.m6_cli resources
    python -m evaluation.m6_cli evaluate --candidate quantsmind/m3-base \\
        --checkpoint runs/m3_base/checkpoint_best.pt --out-dir runs/m6/m3_base
    python -m evaluation.m6_cli report

Nothing here retrains, downloads or uploads. Every subcommand delegates to the
module that owns the logic.
"""

from __future__ import annotations

import argparse
import sys

STAGES = ("inventory", "resources", "licensing", "evaluate", "sovereign",
          "report")


def _delegate(module_name, argv):
    module = __import__(module_name, fromlist=["main"])
    return module.main(argv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.m6_cli",
        description="M6: local capability assessment (no training, no upload)")
    parser.add_argument("stage", choices=STAGES)

    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    stage, rest = raw[0], raw[1:]
    if stage not in STAGES:
        parser.error(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")

    if stage == "inventory":
        return _delegate("evaluation.m6_inventory", rest)
    if stage == "resources":
        return _delegate("evaluation.m6_resources", rest)
    if stage == "licensing":
        from evaluation.m6_licensing import REGISTRY, unresolved_licensing
        unresolved = unresolved_licensing()
        for model_id, record in sorted(REGISTRY.items()):
            print(f"{model_id:45} {record.get('license_id') or 'UNRESOLVED'}")
        if unresolved:
            print(f"\nunresolved: {', '.join(sorted(unresolved))}")
        return 0
    if stage == "evaluate":
        return _delegate("evaluation.m6_evaluate", rest)
    if stage == "sovereign":
        return _delegate("evaluation.m6_sovereign", rest)
    return _delegate("evaluation.m6_report", rest)


if __name__ == "__main__":
    sys.exit(main())
