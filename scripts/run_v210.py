"""Run the v2.1.0 instruction-to-operation experiment (immutable artifacts).

    python scripts/run_v210.py --run-root <dir> --run-id run-0001 --stage baseline
    python scripts/run_v210.py --run-root <dir> --run-id run-0001 --stage train
    python scripts/run_v210.py --run-root <dir> --run-id run-0001 --stage eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml

from training.experiment_artifacts import RunArtifacts, file_sha256
from training.operation_tasks import (
    build_operation_curriculum,
    contamination_report,
    curriculum_digest,
)
from training.supervised_train import evaluate_supervised, train_supervised

EVAL_PER_PARTITION = 4
BUDGET = 64
TINY_OPERATIONS = ("add", "subtract", "maximum", "minimum")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.1.0 operation-learning experiment")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--stage", default="baseline",
                        choices=["baseline", "train", "eval"])
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    model_cfg = yaml.safe_load(
        (REPO / "configs" / "model" / "code_m3_9m.yaml").read_text(encoding="utf-8")
    )["model"]
    parts = build_operation_curriculum(args.seed)
    digest = curriculum_digest(parts)
    report = contamination_report(parts)
    if report["contaminated"]:
        raise SystemExit(f"REFUSED: contaminated curriculum {report['violations'][:3]}")
    eval_sets = {
        name: parts[name][:EVAL_PER_PARTITION]
        for name in ("T", "V", "H1", "H2", "H3", "C")
    }
    tiny = [e for e in parts["T"] if e.operation in TINY_OPERATIONS][:6]

    if args.stage == "baseline":
        from training.determinism import set_determinism
        from training.model import QMModel

        artifacts = RunArtifacts(args.run_root, args.run_id)
        artifacts.write_json("integrity", "curriculum.json", {
            "digest": digest, "contamination": report,
            "partitions": {k: [e.to_dict() for e in v] for k, v in parts.items()},
        })
        artifacts.write_json("integrity", "config.json", {
            "lr": args.lr, "steps": args.steps, "seed": args.seed,
            "context_length": args.max_length, "grad_accum": args.grad_accum,
            "model_config": model_cfg, "curriculum_digest": digest,
            "evaluation_budget": BUDGET,
            "evaluation_per_partition": EVAL_PER_PARTITION,
            "supervision": "instruction+signature+body, response-only loss mask",
        })
        set_determinism(args.seed)
        model = QMModel(model_cfg)
        artifacts.write_checkpoint(
            "scratch_baseline",
            {"model": model.state_dict(), "config": {"model": model_cfg}},
        )
        started = time.time()
        result = evaluate_supervised(model, tokenizer, eval_sets, budget=BUDGET)
        print("baseline eval seconds", round(time.time() - started, 1))
        for name, metrics in result["metrics"].items():
            print(" ", name, json.dumps(metrics))
        checkpoint_hash = file_sha256(
            artifacts.path("checkpoints", "scratch_baseline.pt")
        )
        artifacts.write_evaluation("scratch_baseline.json", {
            "run_id": args.run_id,
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_suite_digest": digest,
            "generation": {"budget": BUDGET, "decoding": "greedy", "seed": args.seed},
            "metric_version": "v2.1.0",
            "per_partition": result["metrics"],
            "records": result["records"],
        })
        artifacts.finalize({
            "schema": "quantsmind.v210_experiment/1",
            "stage": "baseline",
            "curriculum_digest": digest,
            "scratch_checkpoint_sha256": checkpoint_hash,
            "per_partition": result["metrics"],
        })
        return 0

    run_dir = Path(args.run_root) / args.run_id
    artifacts = RunArtifacts(args.run_root, args.run_id, create=not run_dir.exists())

    if args.stage == "train":
        result, model = train_supervised(
            model_config=model_cfg, tokenizer=tokenizer,
            examples=parts["T"], val_examples=parts["V"][:8],
            lr=args.lr, steps=args.steps, max_length=args.max_length,
            grad_accum=args.grad_accum, seed=args.seed, artifacts=artifacts,
        )
        artifacts.write_checkpoint(
            "trained",
            {"model": model.state_dict(), "config": {"model": model_cfg}},
        )
        print("train steps", result["steps"], "loss", result["initial_loss"],
              "->", result["final_loss"], "val", result["validation_loss"],
              "tokens", result["tokens_consumed"], "sec", result["elapsed_seconds"])
        return 0

    from serving.local_inference import load_model

    checkpoint = artifacts.path("checkpoints", "trained.pt")
    model, _ = load_model(str(checkpoint), "cpu")
    started = time.time()
    result = evaluate_supervised(model, tokenizer, eval_sets, budget=BUDGET)
    tiny_result = evaluate_supervised(model, tokenizer, tiny, budget=BUDGET)
    print("eval seconds", round(time.time() - started, 1))
    for name, metrics in result["metrics"].items():
        print(" ", name, json.dumps(metrics))
    print("  tiny-set", json.dumps(tiny_result["metrics"]))
    checkpoint_hash = file_sha256(checkpoint)
    artifacts.write_evaluation("trained.json", {
        "run_id": args.run_id,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_suite_digest": digest,
        "generation": {"budget": BUDGET, "decoding": "greedy", "seed": args.seed},
        "metric_version": "v2.1.0",
        "per_partition": result["metrics"],
        "tiny_set": tiny_result["metrics"],
        "records": result["records"],
    })
    metrics = json.loads(
        artifacts.path("metrics", "training.json").read_text(encoding="utf-8")
    )
    artifacts.finalize({
        "schema": "quantsmind.v210_experiment/1",
        "stage": "eval",
        "curriculum_digest": digest,
        "trained_checkpoint_sha256": checkpoint_hash,
        "training": {
            "initial_loss": metrics.get("initial_loss"),
            "final_loss": metrics.get("final_loss"),
            "validation_loss": metrics.get("validation_loss"),
            "tokens_consumed": metrics.get("tokens_consumed"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
        },
        "per_partition": result["metrics"],
        "tiny_set": tiny_result["metrics"],
    })
    print("finalized", artifacts.path("manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
