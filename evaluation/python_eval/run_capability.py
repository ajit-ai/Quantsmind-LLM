"""Run the Python capability evaluation and write a report (v1.4.0).

Usage
-----
    python -m evaluation.python_eval.run_capability --untrained \\
        --model-id qm-code-m3-9m --class untrained --out reports/untrained.json

    python -m evaluation.python_eval.run_capability \\
        --checkpoint runs/m7_pretrain/checkpoint_last.pt \\
        --tokenizer tokenizer/qm_code_bpe_v0.json \\
        --model-id qm-code-m3-9m --class real-data-trained \\
        --out reports/m7.json

The report records the model identity, the sandbox capability, the metrics
with their denominators, and every per-case record. The ``class`` field
separates an untrained model from a synthetic-only model and from a model
trained on authorised real data; the classes are never merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODEL_CLASSES = ("untrained", "synthetic-only", "real-data-trained")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint, tokenizer_path, device, seed):
    from serving.local_inference import load_model
    from tokenizers import Tokenizer

    if checkpoint:
        model, model_cfg = load_model(str(checkpoint), device)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return model, tokenizer, model_cfg
    from training.determinism import set_determinism
    from training.model import QMModel
    from tokenizers import Tokenizer

    import yaml

    set_determinism(seed)
    model_cfg = yaml.safe_load(
        Path("configs/model/code_m3_9m.yaml").read_text(encoding="utf-8")
    )["model"]
    model = QMModel(model_cfg)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return model, tokenizer, model_cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Python capability evaluation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    parser.add_argument("--model-id", default="qm-code-m3-9m")
    parser.add_argument("--class", dest="model_class", default="untrained",
                        choices=list(MODEL_CLASSES))
    parser.add_argument("--trained-on", default=None)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from evaluation.python_eval.runner import evaluate
    from evaluation.python_eval.schema import load_cases
    from serving.python_codegen import PythonCodeGenerator

    cases_path = args.cases or str(
        Path(__file__).resolve().parent / "data" / "python_capability_v1.json"
    )
    cases = load_cases(cases_path)

    model, tokenizer, model_cfg = _load_model(
        args.checkpoint, args.tokenizer, args.device, args.seed
    )
    identity = {
        "model_id": args.model_id,
        "class": args.model_class,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": model_cfg.get("vocab_size"),
        "context_window": model_cfg.get("max_seq_len"),
        "trained_on": args.trained_on,
        "qualified_formats": ["code_completion", "chat_instruction"],
        "note": (
            "A base (pretraining) model is served raw text; the chat template "
            "is not applied and no chat qualification is claimed."
        ),
    }
    generator = PythonCodeGenerator(
        model,
        tokenizer,
        model_identity=identity,
        qualified_formats=("code_completion", "chat_instruction"),
        max_new_tokens_cap=max(64, args.max_new_tokens),
        device=args.device,
    )
    report = evaluate(
        cases,
        generator,
        model_identity=identity,
        repeats=max(1, args.repeat),
    )
    report["cases_file"] = cases_path
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metrics = report["metrics"]
    print(f"model            : {identity['model_id']} ({identity['class']})")
    print(f"cases            : {metrics['cases_total']}")
    print(
        "syntax valid     : "
        f"{metrics['syntax_valid_count']}/{metrics['syntax_valid_denominator']} "
        f"({metrics['syntax_valid_rate']})"
    )
    print(
        "functional pass  : "
        f"{metrics['functional_pass_count']}/{metrics['functional_pass_denominator']} "
        f"({metrics['functional_pass_rate']})"
    )
    print(
        "contract pass    : "
        f"{metrics['completion_success_count']}/{metrics['completion_success_denominator']} "
        f"({metrics['completion_success_rate']})"
    )
    print(f"failures         : {metrics['failure_categories']}")
    print(f"latency (mean s) : {metrics['mean_latency_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
