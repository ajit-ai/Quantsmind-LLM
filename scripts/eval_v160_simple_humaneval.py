"""Taksh v1.6.0 — Evaluate SFT model on simple_humaneval.

Loads the v1.6.0 SFT checkpoint and evaluates pass@1 on the 20
simple_humaneval problems using the instruction-tuned format.

Usage:
    python scripts/eval_v160_simple_humaneval.py --ckpt runs/v160_sft/checkpoint_last.pt
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.model import QMModel
from training.instruction_format import render_prompt, render_completion
from evaluation.simple_humaneval import PROBLEMS, summarize

CKPT_PATH = Path("runs/v160_sft/checkpoint_last.pt")
MODEL_CONFIG = Path("configs/model/code_v160_15m.yaml")
TOKENIZER_PATH = Path("tokenizer/taksh_code_bpe_v1.json")


def load_model(ckpt: Path, model_cfg_path: Path, device: str = "cpu"):
    import yaml
    from training.checkpoint import load_model_weights

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))["model"]
    model = QMModel(cfg)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if state is None:
        raise SystemExit(f"{ckpt_path} has no 'model' entry")
    report = load_model_weights(model, state, source=str(ckpt_path))
    print(f"Loaded {len(report.loaded)} tensors from {ckpt_path}")
    return model.to(device).eval(), cfg


def generate_instruction(model, tokenizer, instruction_text: str,
                         max_new_tokens: int = 32, temperature: float = 0.2,
                         top_k: int = 0, device: str = "cpu"):
    """Generate a completion for an instruction-tuned prompt.

    Wraps the instruction in the M4 format and generates the response.
    """
    prompt_text = render_prompt(instruction_text)
    prompt_ids = tokenizer.encode(prompt_text).ids
    eos_id = tokenizer.token_to_id("<eos>")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= 2048 else x[:, -2048:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tok = int(next_id.item())

        if tok == eos_id:
            break

        generated.append(tok)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(generated)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate v1.6.0 on simple_humaneval")
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--model-config", default=str(MODEL_CONFIG))
    p.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=0)
    args = p.parse_args()

    print(f"Loading model from {args.ckpt}")
    model, model_cfg = load_model(Path(args.ckpt), Path(args.model_config), args.device)
    print(f"Model: {model.num_params():,} params, vocab={model_cfg['vocab_size']}")

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print(f"<eos> id: {tokenizer.token_to_id('<eos>')}")

    print(f"\nEvaluating on {len(PROBLEMS)} simple_humaneval problems...")

    results = []
    for problem in PROBLEMS:
        instruction = (
            "Complete the following Python function. Return only the code that "
            "replaces the ellipsis or completes the return statement. "
            "Do not repeat the function signature.\n\n" + problem.prompt
        )
        generated = generate_instruction(
            model, tokenizer, instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )

        # Check if expected tail is in the generated output
        expected_present = problem.expected_tail.strip() in generated
        # Also check syntax validity of prompt + generation
        full = problem.prompt + generated
        syntax_valid = False
        try:
            ast.parse(full)
            syntax_valid = True
        except SyntaxError:
            pass

        passed = syntax_valid and expected_present

        results.append({
            "id": problem.id,
            "category": problem.category,
            "passed": passed,
            "syntax_valid": syntax_valid,
            "expected_tail_present": expected_present,
            "generated": generated,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {problem.id} ({problem.category}): "
              f"syntax={syntax_valid}, expected={expected_present}")
        if not passed:
            print(f"      expected: {problem.expected_tail!r}")
            print(f"      got:      {generated[:80]!r}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    syntax_valid = sum(1 for r in results if r["syntax_valid"])
    expected_present = sum(1 for r in results if r["expected_tail_present"])

    print(f"\n{'='*60}")
    print(f"v1.6.0 simple_humaneval Results")
    print(f"{'='*60}")
    print(f"Total:        {total}")
    print(f"Passed:       {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Syntax valid: {syntax_valid}/{total} ({syntax_valid/total*100:.1f}%)")
    print(f"Expected tail: {expected_present}/{total} ({expected_present/total*100:.1f}%)")

    # Category breakdown
    from collections import Counter
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0}
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1
    for cat, v in sorted(by_cat.items()):
        print(f"  {cat}: {v['passed']}/{v['total']} ({v['passed']/v['total']*100:.0f}%)")

    # v1.6.0 success criterion: pass@1 > 0
    if passed > 0:
        print(f"\n[PASS] v1.6.0 SUCCESS CRITERION MET: pass@1 = {passed}/{total} > 0")
    else:
        print(f"[FAIL] v1.6.0 SUCCESS CRITERION NOT MET: pass@1 = {passed}/{total} (need > 0)")


if __name__ == "__main__":
    main()
