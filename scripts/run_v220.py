"""Run the v2.2.0 complete-function and generalization experiment.

    python scripts/run_v220.py --root <dir> --stage freeze
    python scripts/run_v220.py --root <dir> --stage baseline
    python scripts/run_v220.py --root <dir> --stage train --cell A
    python scripts/run_v220.py --root <dir> --stage all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.continued_training import train_from_checkpoint
from training.experiment_artifacts import RunArtifacts, canonical_digest, file_sha256
from training.generalization_eval import evaluate_cases
from training.generalization_tasks import (build_cells, build_eval_cases,
                                           eval_digest, validation_cases)
from training.supervised import build_example

TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"
DEFAULT_CHECKPOINT = Path(
    r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
EXPECTED_CHECKPOINT = (
    "02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac")
BUDGET = 64
SEED = 20261022


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def _examples(records, split):
    return [build_example(
        example_id=record["example_id"], instruction=record["instruction"],
        signature=record["signature"], body=record["body"],
        operation=record["operation"], family=record["family"],
        split=split, seed=SEED) for record in records]


def freeze_suite(root: Path):
    directory = root / "frozen"
    path = directory / "eval_suite.json"
    cases = build_eval_cases()
    digest = eval_digest(cases)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("digest") != digest:
            raise SystemExit("REFUSED: existing frozen suite has a different digest")
        print("frozen suite already present", digest)
        return cases, digest
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"version": "2.2.0", "seed": SEED, "budget": BUDGET,
               "digest": digest, "cases": cases}
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    (directory / "eval_suite.sha256").write_text(
        file_sha256(path) + "\n", encoding="utf-8")
    print("froze", len(cases), "cases", digest)
    return cases, digest


def _load_suite(root: Path):
    path = root / "frozen" / "eval_suite.json"
    if not path.is_file():
        raise SystemExit("REFUSED: run the freeze stage first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if payload["digest"] != eval_digest(cases):
        raise SystemExit("REFUSED: frozen suite digest does not verify")
    return cases, payload["digest"]


def _evaluation(artifacts, model, tokenizer, cases, digest, checkpoint_hash,
                limit=None):
    selected = cases if limit is None else cases[:limit]
    started = time.time()
    result = evaluate_cases(model, tokenizer, selected, budget=BUDGET,
                            device="cpu")
    elapsed = round(time.time() - started, 2)
    payload = {"version": "2.2.0", "checkpoint_sha256": checkpoint_hash,
               "evaluation_suite_digest": digest, "budget": BUDGET,
               "decoding": "greedy", "seed": SEED,
               "cases_requested": len(selected), "elapsed_seconds": elapsed,
               "metrics": result["metrics"], "records": result["records"]}
    artifacts.write_evaluation("generalization.json", payload)
    print("evaluated", len(selected), "cases in", elapsed, "s")
    for name, metrics in result["metrics"]["by_partition"].items():
        print(" ", name, metrics["cases"], "exact", metrics["exact"],
              "structural", metrics["structural"],
              "composition", metrics["composition_match"])
    return payload

def _new_run(root: Path, run_id: str):
    target = root / run_id
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"REFUSED: run directory already exists: {target}")
    return RunArtifacts(str(root), run_id)


def run_baseline(root: Path, checkpoint: Path, checkpoint_hash: str, limit=None):
    from serving.local_inference import load_model
    cases, digest = _load_suite(root)
    artifacts = _new_run(root, "baseline")
    artifacts.write_json("integrity", "config.json", {
        "version": "2.2.0", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_suite_digest": digest, "budget": BUDGET,
        "decoding": "greedy", "seed": SEED,
    })
    model, _config = load_model(str(checkpoint), "cpu")
    _evaluation(artifacts, model, _tokenizer(), cases, digest, checkpoint_hash,
                limit=limit)
    artifacts.finalize({"stage": "baseline", "version": "2.2.0",
                        "checkpoint_sha256": checkpoint_hash,
                        "evaluation_suite_digest": digest})
    print("baseline finalized")


def run_cell(root: Path, cell: str, checkpoint: Path, checkpoint_hash: str,
             *, lr: float, steps: int, grad_accum: int, limit=None):
    key = cell.replace("_repeat", "")
    cells = build_cells()
    records = cells[key]
    validation = validation_cases()
    cases, digest = _load_suite(root)
    artifacts = _new_run(root, f"cell{cell}")
    artifacts.write_json("integrity", "config.json", {
        "version": "2.2.0", "cell": key, "repeat": cell != key,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "evaluation_suite_digest": digest, "lr": lr, "steps": steps,
        "grad_accum": grad_accum, "seed": SEED, "budget": BUDGET,
        "train_examples": len(records), "validation_examples": len(validation),
    })
    result, model, _config = train_from_checkpoint(
        checkpoint_path=str(checkpoint), checkpoint_hash=checkpoint_hash,
        tokenizer=_tokenizer(), examples=_examples(records, "train"),
        val_examples=_examples(validation, "validation"), lr=lr, steps=steps,
        grad_accum=grad_accum, seed=SEED, artifacts=artifacts, device="cpu",
    )
    trained_hash = result["checkpoint_sha256"]
    _evaluation(artifacts, model, _tokenizer(), cases, digest, trained_hash,
                limit=limit)
    artifacts.finalize({"stage": "train", "version": "2.2.0", "cell": key,
                        "repeat": cell != key,
                        "base_checkpoint_sha256": checkpoint_hash,
                        "trained_checkpoint_sha256": trained_hash,
                        "evaluation_suite_digest": digest,
                        "train_examples": len(records),
                        "tokens_consumed": result["tokens_consumed"]})
    del model
    gc.collect()
    print(f"cell {cell} finalized")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.2.0 experiment")
    parser.add_argument("--root", default=r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v220")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--stage", default="all",
                        choices=["freeze", "baseline", "train", "all"])
    parser.add_argument("--cell", default=None, choices=["A", "B", "C", "C_repeat"])
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke-test only: evaluate the first N cases")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--allow-other-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(4)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint missing: {checkpoint}")
    checkpoint_hash = file_sha256(checkpoint)
    if not args.allow_other_checkpoint and checkpoint_hash != EXPECTED_CHECKPOINT:
        raise SystemExit("REFUSED: checkpoint hash differs from the frozen v2.1.0 checkpoint")

    if args.stage in ("freeze", "all"):
        _cases, digest = freeze_suite(root)
        print("suite digest", digest)
    if args.stage == "freeze":
        return 0
    if args.stage in ("baseline", "all"):
        run_baseline(root, checkpoint, checkpoint_hash, limit=args.limit)
    if args.stage in ("train", "all"):
        selected = [args.cell] if args.cell else ["A", "B", "C", "C_repeat"]
        for cell in selected:
            run_cell(root, cell, checkpoint, checkpoint_hash, lr=args.lr,
                     steps=args.steps, grad_accum=args.grad_accum,
                     limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

