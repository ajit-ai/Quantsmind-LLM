"""Evaluate a checkpoint on the fixed M4 evaluation set.

One command, used identically before and after instruction tuning, so the
comparison is apples to apples:

    python -m evaluation.m4_evaluate --ckpt <path> --label m3-base
    python -m evaluation.m4_evaluate --ckpt <path> --label m4-tuned

Everything is recorded: the raw output of every prompt, the extracted code, the
syntax and test status, and the compliance check. Nothing is filtered or
cherry-picked. Greedy decoding is the default so the comparison is not decided
by the sampler.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.code_check import SAFE_TESTS, evaluate_generation, summarise
from evaluation.m4_eval_set import SET_ID, load_set, prompt_hash


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def evaluate_checkpoint(ckpt_path, tokenizer_path, label, eval_set_path=None,
                        max_new_tokens=96, device="cpu", seed=11):
    from training.generate import (
        GenerationError, load_checkpoint_for_generation, load_tokenizer,
    )
    from training.instruction_format import render_prompt

    rows = load_set(eval_set_path or "evaluation/m4_eval_set.jsonl")
    model, info = load_checkpoint_for_generation(ckpt_path, device=device)
    tokenizer, vocab = load_tokenizer(
        tokenizer_path or (info.get("tokenizer_identity") or {}).get("path"),
        info["vocab_size"],
    )

    results = []
    started = time.perf_counter()
    for row in rows:
        prompt_text = render_prompt(row["prompt"])
        prompt_ids = tokenizer.encode(prompt_text).ids
        # Greedy: the sampler must not decide a before/after verdict.
        generated = _greedy_generate(
            model, tokenizer, prompt_ids, max_new_tokens, info["max_seq_len"],
            device,
        )
        test_source = SAFE_TESTS.get(row["test"]) if row["test"] else None
        result = evaluate_generation(
            row["id"], row["prompt"], generated, test_source
        )
        result.update({
            "category": row["category"],
            "task": row["task"],
            "expects": row["expects"],
            "prompt_rendered": prompt_text,
            "generated_tokens": generated,
        })
        results.append(result)

    elapsed = time.perf_counter() - started
    by_category = {}
    for category in sorted({r["category"] for r in results}):
        subset = [r for r in results if r["category"] == category]
        by_category[category] = summarise(subset)

    report = {
        "schema": "quantsmind.m4_eval/1",
        "label": label,
        "generated_at": utc_now_iso(),
        "eval_set": {
            "set_id": SET_ID,
            "path": str(eval_set_path or "evaluation/m4_eval_set.jsonl"),
            "prompt_hash": prompt_hash(),
            "prompts": len(rows),
        },
        "checkpoint": info,
        "tokenizer": {"path": str(tokenizer_path), "vocab_size": vocab},
        "decoding": {
            "strategy": "greedy", "max_new_tokens": max_new_tokens, "seed": seed,
            "note": "greedy so sampling noise cannot decide the comparison",
        },
        "metrics": summarise(results),
        "by_category": by_category,
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
        "provenance": (
            "All outputs generated locally from the named checkpoint. No hosted "
            "or external language model was involved."
        ),
    }
    return report


def _greedy_generate(model, tokenizer, prompt_ids, max_new_tokens, max_seq_len,
                     device):
    import torch

    eos_id = tokenizer.token_to_id("<eos>")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
            logits, _ = model(window)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token = int(next_id.item())
            if eos_id is not None and token == eos_id:
                break
            generated.append(token)
            x = torch.cat([x, next_id], dim=1)
    return tokenizer.decode(generated)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the fixed M4 evaluation set"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--eval-set", default="evaluation/m4_eval_set.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    try:
        report = evaluate_checkpoint(
            args.ckpt, args.tokenizer, args.label,
            eval_set_path=args.eval_set,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"evaluation failed: {type(exc).__name__}: {exc}")
        return 1

    out = args.out or f"reports/m4_eval_{args.label}.json"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(f"M4 evaluation: {args.label}")
    print(f"  checkpoint     : {report['checkpoint']['checkpoint']}")
    print(f"  step           : {report['checkpoint']['checkpoint_step']}")
    print(f"  eval set       : {SET_ID} ({report['eval_set']['prompts']} prompts)")
    print(f"  eval hash      : {report['eval_set']['prompt_hash'][:16]}...")
    print(f"  syntax valid   : {metrics['syntax_valid']}/{metrics['items']} "
          f"({metrics['syntax_valid_rate']:.1%})")
    if metrics["test_pass_rate"] is not None:
        print(f"  test pass      : {metrics['test_pass']}/{metrics['testable_items']} "
              f"({metrics['test_pass_rate']:.1%})")
    print(f"  compliance     : {metrics['instruction_compliance_rate']:.1%}")
    print(f"  mean out chars : {metrics['mean_output_chars']}")
    print(f"  elapsed        : {report['elapsed_seconds']}s")
    print(f"  written        : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())