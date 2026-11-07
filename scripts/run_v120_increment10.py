"""Run the authorized v1.2.0 Increment 10 identifier-aware experiment.

Dry run by default. Training requires BOTH --train and --authorize and
reuses the existing trainer with the Increment 9 identifier-supervision
sidecar enabled (configurable, mean-normalized, single application).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.experiment_artifacts import RunArtifacts
from training.v120_experiment10 import (
    DEFAULT_CONFIG,
    combined_training_examples,
    combined_validation_examples,
    experiment_identity,
    identifier_config,
    load_config,
    run_preflight,
)


def _launch_command(config) -> str:
    return (
        "python scripts/run_v120_increment10.py --train --authorize "
        f"--root {config['experiment_root']} --run-id {config['run_id']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the authorized Increment 10 experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--root", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root:
        config = {**config, "experiment_root": args.root}
    if args.run_id:
        config = {**config, "run_id": args.run_id}

    if args.train and not args.authorize:
        print(
            "REFUSED: --train without --authorize. Training requires explicit "
            "authorization via --authorize.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    preflight = run_preflight(config, authorized=args.authorize, for_training=args.train)
    plan = {
        "mode": "train" if args.train else "dry_run",
        "experiment_identity": experiment_identity(config),
        "architecture": config["architecture"],
        "baseline_checkpoint_sha256": config["baseline_checkpoint_sha256"],
        "tokenizer": config["tokenizer"],
        "curricula": config["curricula"],
        "identifier_supervision": config["identifier_supervision"],
        "combined_train_examples": config["combined_train_examples"],
        "combined_validation_examples": config["combined_validation_examples"],
        "steps": config["steps"],
        "lr": config["lr"],
        "seed": config["seed"],
        "device": config["device"],
        "experiment_root": config["experiment_root"],
        "run_id": config["run_id"],
        "preflight": preflight,
        "launch_command_after_approval": _launch_command(config),
    }
    if not args.train:
        static = [g for g in preflight["gates"] if g["gate"] != "memory"]
        if not all(g["passed"] for g in static):
            print(json.dumps({**plan, "refused": True}, indent=2))
            raise SystemExit(2)
        print(json.dumps(plan, indent=2))
        return

    from training.continued_training import train_from_checkpoint

    import torch
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(config["tokenizer"]))
    training_examples = combined_training_examples(config)
    validation_examples = combined_validation_examples(config)
    artifacts = RunArtifacts(str(config["experiment_root"]), config["run_id"])
    artifacts.write_json(
        "integrity",
        "config.json",
        {
            "increment": "v1.2.0-increment-10",
            "experiment_identity": experiment_identity(config),
            "identifier_supervision": config["identifier_supervision"],
            "config": config,
            "command": _launch_command(config),
        },
    )
    started = time.time()
    result, _model, _config = train_from_checkpoint(
        checkpoint_path=str(config["baseline_checkpoint"]),
        checkpoint_hash=str(config["baseline_checkpoint_sha256"]),
        tokenizer=tokenizer,
        examples=training_examples,
        val_examples=validation_examples,
        lr=config["lr"],
        steps=config["steps"],
        max_length=config["max_length"],
        grad_accum=config["grad_accum"],
        warmup=config["warmup"],
        weight_decay=config["weight_decay"],
        grad_clip=config["grad_clip"],
        seed=config["seed"],
        identifier_config=identifier_config(config),
        artifacts=artifacts,
        device=str(config["device"]),
    )
    result["experiment"] = {**plan, "preflight": preflight}
    result["elapsed_seconds_total"] = round(time.time() - started, 2)
    artifacts.finalize({"stage": "train", "increment": "v1.2.0-increment-10", **plan})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
