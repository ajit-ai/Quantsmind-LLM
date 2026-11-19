"""Increment 8 architecture/learning diagnosis (no training).

Runs deterministic diagnostics: tokenizer identifier fragmentation,
architecture/capacity inspection, supervision alignment/masking, checkpoint
load consistency, and synthetic copy-behaviour probes on the baseline,
Increment 5, and Increment 7 checkpoints (inference only). Writes a
machine-readable JSON report to the output path; never modifies a
checkpoint, tokenizer, or benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.diagnostics import (
    COPY_CASES,
    architectures_consistent,
    checkpoint_param_hash,
    copy_case_bindings,
    supervision_alignment,
    tokenizer_stats,
)

TOKENIZER_PATH = REPO / "tokenizer" / "qm_code_bpe_v0.json"
CHECKPOINTS = {
    "baseline": Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt"),
    "increment5": Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\checkpoints\trained.pt"),
    "increment7": Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment7\increment7-targeted-100\checkpoints\trained.pt"),
}
DIAGNOSTICS_ROOT = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment8")


def run_copy_probes(checkpoint: Path, *, tokenizer) -> list[dict]:
    import gc
    import torch
    from serving.local_inference import generate_ids, load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    records = []
    try:
        for case_id, kind, instruction, signature, fragment in COPY_CASES:
            prompt = instruction + "\n" + signature + "\n"
            result = generate_ids(
                model, tokenizer, prompt,
                max_new_tokens=48, temperature=0.0, top_k=1, top_p=1.0,
                device="cpu", seed=20261023, binding_repair=False,
            )
            completion = result["text"]
            bindings = copy_case_bindings(completion, signature)
            records.append({
                "case_id": case_id,
                "kind": kind,
                "instruction": instruction,
                "signature": signature,
                "completion": completion,
                "fragment_present": fragment in completion,
                **bindings,
            })
    finally:
        del model
        gc.collect()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Increment 8 diagnostics")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_stats = tokenizer_stats(tokenizer)

    arch = architectures_consistent(CHECKPOINTS)
    param_hashes = {label: checkpoint_param_hash(path) for label, path in CHECKPOINTS.items()}
    param_hashes_repeat = {label: checkpoint_param_hash(path) for label, path in CHECKPOINTS.items()}

    from training.binding_curriculum import build_training_examples as build_binding
    from training.functional_curriculum import build_training_examples as build_functional

    alignment = [
        supervision_alignment(example, tokenizer)
        for example in build_binding()[:10] + build_functional()[:10]
    ]

    probes = {
        label: run_copy_probes(path, tokenizer=tokenizer)
        for label, path in CHECKPOINTS.items()
    }
    probe_summary = {}
    for label, records in probes.items():
        novel = [r for r in records if "novel" in r["case_id"]]
        familiar = [r for r in records if "familiar" in r["case_id"]]
        probe_summary[label] = {
            "exact_copy": sum(r["exact_copy"] for r in records),
            "parses": sum(r["parses"] for r in records),
            "fragment_present": sum(r["fragment_present"] for r in records),
            "novel_exact_copy": sum(r["exact_copy"] for r in novel),
            "novel_cases": len(novel),
            "familiar_exact_copy": sum(r["exact_copy"] for r in familiar),
            "familiar_cases": len(familiar),
        }

    result = {
        "tokenizer": tok_stats,
        "architecture": {
            "config": arch["configs"]["baseline"],
            "all_same_architecture": arch["all_same_architecture"],
            "tensor_count": arch["configs"]["baseline"]["_tensors"],
            "param_hashes": param_hashes,
            "load_consistent": param_hashes == param_hashes_repeat,
        },
        "supervision_alignment": alignment,
        "copy_probes": probes,
        "copy_summary": probe_summary,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "tokenizer": {k: {"mean": v["mean_fragments"], "round_trip": v["all_round_trip"]} for k, v in tok_stats.items()},
        "architecture": arch["configs"]["baseline"],
        "load_consistent": param_hashes == param_hashes_repeat,
        "alignment_ok": all(
            item["prompt_all_masked"] and item["target_all_supervised"] and item["all_params_in_target"] and item["eos_in_target"]
            for item in alignment
        ),
        "copy_summary": probe_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
