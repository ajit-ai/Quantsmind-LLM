"""Run the bounded v1.2.0 Increment 1 training experiment.

Default mode is a dry run. Real training requires --train and an explicit
isolated experiment root. The frozen S2 checkpoint is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.binding_curriculum import build_training_examples, build_validation_examples, curriculum_digest
from training.continued_training import train_from_checkpoint
from training.experiment_artifacts import RunArtifacts, file_sha256

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
BASELINE_SHA = "02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac"
TOKENIZER_PATH = REPO / "tokenizer" / "qm_code_bpe_v0.json"
AD1_SUITE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json")
DEFAULT_ROOT = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120")


def _ad1_pairs() -> set[tuple[str, str]]:
    if not AD1_SUITE.is_file():
        return set()
    suite = json.loads(AD1_SUITE.read_text(encoding="utf-8"))
    return {(case["instruction"], case["signature"]) for case in suite.get("cases", [])}


def _check_overlap(examples) -> None:
    ad1 = _ad1_pairs()
    overlap = [
        example.example_id
        for example in examples
        if (example.instruction, example.signature) in ad1
    ]
    if overlap:
        raise ValueError(f"training examples overlap held-out AD-1 cases: {overlap[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Taksh v1.2.0 Increment 1")
    parser.add_argument("--train", action="store_true", help="execute the bounded training run")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--run-id", default="increment1-binding-240")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--min-available-gib", type=float, default=0.5)
    args = parser.parse_args()

    if not (1 <= args.steps <= 300):
        raise SystemExit("REFUSED: steps must be between 1 and 300")
    if not BASELINE.is_file():
        raise SystemExit(f"REFUSED: baseline checkpoint missing: {BASELINE}")
    checkpoint_sha = file_sha256(BASELINE)
    if checkpoint_sha != BASELINE_SHA:
        raise SystemExit("REFUSED: baseline checkpoint hash does not match frozen S2")
    if not TOKENIZER_PATH.is_file():
        raise SystemExit(f"REFUSED: tokenizer missing: {TOKENIZER_PATH}")

    training_examples = build_training_examples(seed=args.seed)
    validation_examples = build_validation_examples(seed=args.seed + 1)
    _check_overlap(training_examples)
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        available_gib = psutil.virtual_memory().available / 1024**3
        if available_gib < args.min_available_gib:
            raise SystemExit(
                f"REFUSED: available RAM {available_gib:.2f} GiB is below {args.min_available_gib} GiB"
            )

    plan = {
        "mode": "train" if args.train else "dry_run",
        "baseline_checkpoint": str(BASELINE),
        "baseline_checkpoint_sha256": checkpoint_sha,
        "tokenizer": str(TOKENIZER_PATH),
        "curriculum_digest": curriculum_digest(),
        "train_examples": len(training_examples),
        "validation_examples": len(validation_examples),
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "experiment_root": str(args.root),
        "run_id": args.run_id,
    }
    if not args.train:
        print(json.dumps(plan, indent=2))
        return

    import torch
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    artifacts = RunArtifacts(str(args.root), args.run_id)
    artifacts.write_json(
        "integrity",
        "config.json",
        {
            "increment": "v1.2.0-increment-1",
            "baseline_checkpoint": str(BASELINE),
            "baseline_checkpoint_sha256": checkpoint_sha,
            "tokenizer": str(TOKENIZER_PATH),
            "curriculum_digest": curriculum_digest(),
            "train_examples": len(training_examples),
            "validation_examples": len(validation_examples),
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "max_length": 256,
            "grad_accum": 2,
            "heldout_policy": "AD-1 frozen suite is not used for training",
            "command": f"python scripts/run_v120_increment1.py --train --root {args.root} --run-id {args.run_id} --steps {args.steps} --lr {args.lr} --seed {args.seed}",
        },
    )
    started = time.time()
    result, _model, _config = train_from_checkpoint(
        checkpoint_path=str(BASELINE),
        checkpoint_hash=checkpoint_sha,
        tokenizer=tokenizer,
        examples=training_examples,
        val_examples=validation_examples,
        lr=args.lr,
        steps=args.steps,
        max_length=256,
        grad_accum=2,
        seed=args.seed,
        artifacts=artifacts,
        device="cpu",
    )
    result["experiment"] = plan
    result["elapsed_seconds_total"] = round(time.time() - started, 2)
    artifacts.finalize({"stage": "train", "increment": "v1.2.0-increment-1", **plan})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
