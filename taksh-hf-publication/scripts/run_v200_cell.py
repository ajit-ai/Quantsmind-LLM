"""Run one v2.0.0 controlled training cell with immutable artifacts.

Usage:
    python scripts/run_v200_cell.py --run-root <dir> --run-id run-0001 \
        --lr 3e-4 --steps 8 --stage train
    python scripts/run_v200_cell.py --run-root <dir> --run-id run-0001 --stage eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml

from training.contrastive_tasks import (
    build_contrastive_curriculum,
    contamination_report,
    curriculum_digest,
)
from training.experiment_artifacts import RunArtifacts, file_sha256
from training.synthetic_eval import classify_task_conditioning, evaluate_cases_v2
from training.synthetic_train import build_corpus, train_cell

EVAL_CASES = 6
BUDGET = 64


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.0.0 controlled training cell")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--baseline", default="runs/m7_pretrain/checkpoint_last.pt")
    parser.add_argument("--stage", default="train", choices=["train", "eval"])
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    model_cfg = yaml.safe_load(
        (REPO / "configs" / "model" / "code_m3_9m.yaml").read_text(encoding="utf-8")
    )["model"]

    partitions = build_contrastive_curriculum(args.seed)
    contamination = contamination_report(partitions)
    digest = curriculum_digest(partitions)
    if contamination["contaminated"]:
        print("REFUSED: the contrastive curriculum is contaminated:", contamination["violations"][:3])
        return 2

    if args.stage == "train":
        artifacts = RunArtifacts(args.run_root, args.run_id)
        artifacts.write_json("integrity", "curriculum.json", {
            "digest": digest,
            "contamination": contamination,
            "partitions": {k: [c.to_dict() for c in v] for k, v in partitions.items()},
        })
        config = {
            "lr": args.lr, "steps": args.steps, "grad_accum": args.grad_accum,
            "context_length": args.context_length, "seed": args.seed,
            "init_checkpoint": args.init_checkpoint,
            "baseline_checkpoint": args.baseline,
            "model_config": model_cfg, "curriculum_digest": digest,
            "evaluation_budget": BUDGET, "evaluation_cases_per_partition": EVAL_CASES,
        }
        artifacts.write_json("integrity", "config.json", config)
        result, model = train_cell(
            model_config=model_cfg, tokenizer=tokenizer,
            train_records=build_corpus(partitions, "train"),
            val_records=build_corpus(partitions, "A"),
            lr=args.lr, steps=args.steps, context_length=args.context_length,
            grad_accum=args.grad_accum, seed=args.seed,
            init_checkpoint=args.init_checkpoint, artifacts=artifacts,
        )
        artifacts.write_checkpoint(
            "checkpoint_final",
            {"model": model.state_dict(), "config": {"model": model_cfg}},
        )
        print("train steps", result["steps"], "loss",
              result["initial_loss"], "->", result["final_loss"],
              "val", result["validation_loss"], "tokens", result["tokens_consumed"],
              "seconds", result["elapsed_seconds"])
        return 0

    artifacts = RunArtifacts(args.run_root, args.run_id, create=False)
    from serving.local_inference import load_model

    eval_sets = {
        name: partitions[name][:EVAL_CASES] for name in ("A", "B", "C", "D")
    }
    baseline_path = REPO / args.baseline
    baseline_model, _ = load_model(str(baseline_path), "cpu")
    started = time.time()
    baseline = evaluate_cases_v2(baseline_model, tokenizer, eval_sets, budget=BUDGET)
    trained_path = artifacts.path("checkpoints", "checkpoint_final.pt")
    trained_model, _ = load_model(str(trained_path), "cpu")
    trained = evaluate_cases_v2(trained_model, tokenizer, eval_sets, budget=BUDGET)
    print("eval seconds", round(time.time() - started, 1))
    for label, result in (("baseline", baseline), ("trained", trained)):
        print(label, json.dumps(result["metrics"]))
    metrics = json.loads(
        artifacts.path("metrics", "training.json").read_text(encoding="utf-8")
    )
    loss_delta = None
    if metrics.get("initial_loss") is not None and metrics.get("final_loss") is not None:
        loss_delta = metrics["final_loss"] - metrics["initial_loss"]
    verdict = classify_task_conditioning(trained["metrics"], loss_delta=loss_delta)
    baseline_verdict = classify_task_conditioning(baseline["metrics"])
    print("BASELINE:", baseline_verdict["verdict"], "|", baseline_verdict["reason"])
    print("TRAINED :", verdict["verdict"], "|", verdict["reason"])
    baseline_hash = file_sha256(baseline_path)
    trained_hash = file_sha256(trained_path)
    artifacts.write_evaluation("baseline.json", {
        "run_id": args.run_id,
        "checkpoint_sha256": baseline_hash,
        "checkpoint_path": str(baseline_path),
        "evaluation_suite_digest": digest,
        "generation": {"budget": BUDGET, "decoding": "greedy", "seed": args.seed},
        "metric_version": "v2.0.0",
        "metrics": baseline["metrics"],
        "records": baseline["records"],
    })
    artifacts.write_evaluation("post_train.json", {
        "run_id": args.run_id,
        "checkpoint_sha256": trained_hash,
        "checkpoint_path": str(trained_path),
        "evaluation_suite_digest": digest,
        "generation": {"budget": BUDGET, "decoding": "greedy", "seed": args.seed},
        "metric_version": "v2.0.0",
        "metrics": trained["metrics"],
        "classification": verdict,
        "baseline_classification": baseline_verdict,
        "records": trained["records"],
    })
    artifacts.finalize({
        "schema": "quantsmind.v200_experiment/1",
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "curriculum_digest": digest,
        "baseline_checkpoint_sha256": baseline_hash,
        "trained_checkpoint_sha256": trained_hash,
        "training": {
            "initial_loss": metrics.get("initial_loss"),
            "final_loss": metrics.get("final_loss"),
            "validation_loss": metrics.get("validation_loss"),
            "tokens_consumed": metrics.get("tokens_consumed"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
        },
        "task_conditioning": verdict,
        "baseline_task_conditioning": baseline_verdict,
    })
    print("finalized", artifacts.path("manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
