"""Taksh v1.8.0 — Full evaluation on expanded dataset.

Runs both syntax-only and execution-based evaluation on all 57 SFT problems.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.model import QMModel

CKPT_PATH = Path("runs/v180_sft/checkpoint_last.pt")
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
    from training.instruction_format import render_prompt
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


def extract_function_name(prompt: str) -> str:
    for line in prompt.split("\n"):
        line = line.strip()
        if line.startswith("def "):
            return line[4:].split("(")[0].strip()
    return "unknown_function"


def run_exec_test(full_code: str, func_name: str, test_args_list: list,
                  expected_outputs: list, timeout: int = 10) -> dict:
    """Run execution test cases against generated code.

    Builds a single wrapper script that runs all test cases and reports
    results as JSON.
    """
    tests_data = list(zip(test_args_list, expected_outputs))
    wrapper = f'''import json
{full_code}

results = []
for args, expected in {tests_data!r}:
    try:
        fn = globals()["{func_name}"]
        output = fn(*args)
        results.append({{
            "args": args,
            "passed": output == expected,
            "output": output,
            "expected": expected,
            "error": None,
        }})
    except Exception as exc:
        results.append({{
            "args": args,
            "passed": False,
            "output": None,
            "expected": expected,
            "error": f"{{type(exc).__name__}}: {{exc}}",
        }})

print(json.dumps(results))
'''

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapper)
            temp_path = f.name

        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        os.unlink(temp_path)

        if proc.returncode != 0:
            return {
                "all_passed": False,
                "results": [{"error": f"exit {proc.returncode}: {proc.stderr[:200]}"}],
            }

        try:
            test_results = json.loads(proc.stdout.strip())
            all_passed = all(tr.get("passed", False) for tr in test_results)
            return {"all_passed": all_passed, "results": test_results}
        except json.JSONDecodeError:
            return {
                "all_passed": False,
                "results": [{"error": "json parse error"}],
            }
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return {
            "all_passed": False,
            "results": [{"error": f"timeout after {timeout}s"}],
        }


def main() -> None:
    import argparse
    import os

    p = argparse.ArgumentParser(description="Evaluate v1.8.0 on expanded dataset")
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--model-config", default=str(MODEL_CONFIG))
    p.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    print(f"Loading model from {args.ckpt}")
    model, model_cfg = load_model(Path(args.ckpt), Path(args.model_config), args.device)
    print(f"Model: {model.num_params():,} params")

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # Load SFT data with test cases
    sft_path = Path("data/v180_sft/sft.jsonl")
    problems = []
    with sft_path.open("r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"\nEvaluating on {len(problems)} problems...")

    total = len(problems)
    syntax_valid = 0
    exec_passed = 0
    both_passed = 0

    for prob in problems:
        pid = prob["id"]
        prompt = prob["instruction"]
        expected_tail = prob["response"]
        test_args = prob.get("test_args", [])
        expected_outputs = prob.get("expected_outputs", [])

        generated = generate_instruction(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )

        full = prompt + generated
        syntax_ok = False
        try:
            ast.parse(full)
            syntax_ok = True
        except SyntaxError:
            pass

        exec_result = None
        if syntax_ok and test_args:
            func_name = extract_function_name(prompt)
            exec_result = run_exec_test(
                full, func_name, test_args, expected_outputs
            )

        is_syntax = syntax_ok
        is_exec = exec_result["all_passed"] if exec_result else False
        is_both = is_syntax and is_exec

        if is_syntax:
            syntax_valid += 1
        if is_exec:
            exec_passed += 1
        if is_both:
            both_passed += 1

        status = "PASS" if is_both else "FAIL"
        print(f"  [{status}] {pid} ({prob['task']}): "
              f"syntax={is_syntax}, exec={is_exec}")
        if not is_both:
            print(f"      generated: {generated[:60]!r}")
            if exec_result and not is_exec:
                for r in exec_result["results"]:
                    if not r["passed"]:
                        print(f"      failed test: {r}")

    print(f"\n{'='*60}")
    print(f"v1.8.0 Full Evaluation Results")
    print(f"{'='*60}")
    print(f"Total problems:     {total}")
    print(f"Syntax valid:       {syntax_valid}/{total} ({syntax_valid/total*100:.1f}%)")
    print(f"Execution passed:   {exec_passed}/{total} ({exec_passed/total*100:.1f}%)")
    print(f"Both passed:        {both_passed}/{total} ({both_passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()
