import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate_pilot import load_qm_checkpoint

HUMANEVAL_PROBLEMS = [
    {"task_id": "HumanEval/0", "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == False\n"},
    {"task_id": "HumanEval/1", "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your task is to separate those groups into separate strings and return the list of them.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n", "entry_point": "separate_paren_groups", "test": "def check(candidate):\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n"},
    {"task_id": "HumanEval/2", "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n", "entry_point": "truncate_number", "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert candidate(1.33) == 0.33\n    assert candidate(123.456) == 0.456\n"},
    {"task_id": "HumanEval/3", "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n", "entry_point": "below_zero", "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == False\n    assert candidate([1, 2, -4, 5]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n"},
    {"task_id": "HumanEval/4", "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n", "entry_point": "mean_absolute_deviation", "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) - 1.5) < 1e-6\n"},
]

MBPP_PROBLEMS = [
    {"task_id": "MBPP/0", "prompt": "Write a function to find the common elements in the given lists.\n", "test": "def test_common_elements():\n    assert sorted(common_elements([1, 2, 3, 4], [3, 4, 5, 6])) == [3, 4]\n    assert sorted(common_elements([1, 2, 3], [4, 5, 6])) == []\n", "entry_point": "common_elements"},
    {"task_id": "MBPP/1", "prompt": "Write a function to reverse a string.\n", "test": "def test_reverse_string():\n    assert reverse_string('hello') == 'olleh'\n    assert reverse_string('') == ''\n    assert reverse_string('a') == 'a'\n", "entry_point": "reverse_string"},
    {"task_id": "MBPP/2", "prompt": "Write a function to check if a string is a palindrome.\n", "test": "def test_is_palindrome():\n    assert is_palindrome('racecar') == True\n    assert is_palindrome('hello') == False\n    assert is_palindrome('') == True\n", "entry_point": "is_palindrome"},
    {"task_id": "MBPP/3", "prompt": "Write a function to find the factorial of a number.\n", "test": "def test_factorial():\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n    assert factorial(5) == 120\n    assert factorial(10) == 3628800\n", "entry_point": "factorial"},
    {"task_id": "MBPP/4", "prompt": "Write a function to find the largest number in a list.\n", "test": "def test_find_largest():\n    assert find_largest([1, 2, 3, 4, 5]) == 5\n    assert find_largest([-1, -2, -3]) == -1\n    assert find_largest([42]) == 42\n", "entry_point": "find_largest"},
]


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cpu") -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = x if x.size(1) <= model.cfg["max_seq_len"] else x[:, -model.cfg["max_seq_len"]:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.token_to_id("<eos>"):
                break
    generated = tokenizer.decode(x[0].tolist())
    return generated[len(prompt):] if generated.startswith(prompt) else generated


def run_humaneval(model, tokenizer, problems: list, device: str = "cpu", k: int = 1) -> dict:
    passed = 0
    total = len(problems)
    results = []
    for prob in problems:
        prompt = prob["prompt"]
        completions = []
        for _ in range(k):
            completion = generate_completion(model, tokenizer, prompt, device=device)
            completions.append(prompt + completion)
        prob_passed = False
        for code in completions:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code + "\n" + prob["test"] + "\n" + f"check({prob['entry_point']})\n")
                    f.flush()
                    result = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
                    if result.returncode == 0:
                        prob_passed = True
                        break
            except Exception:
                continue
            finally:
                Path(f.name).unlink(missing_ok=True)
        if prob_passed:
            passed += 1
        results.append({"task_id": prob["task_id"], "passed": prob_passed})
    return {"passed": passed, "total": total, "pass_at_k": passed / total if total > 0 else 0, "results": results}


def compute_code_perplexity(model, tokens_path: str, seq_len: int = 2048, max_windows: int = 32, device: str = "cpu") -> dict:
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    split = int(len(data) * 0.99)
    losses = []
    for i in range(max_windows):
        off = split + i * seq_len
        if off + seq_len + 1 > len(data):
            break
        x = torch.from_numpy(data[off:off + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[off + 1:off + 1 + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    avg = sum(losses) / max(1, len(losses))
    return {"nll": round(avg, 4), "perplexity": round(math.exp(min(20.0, avg)), 2), "windows": len(losses)}


def main():
    p = argparse.ArgumentParser(description="Evaluate code model on HumanEval + MBPP + perplexity")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    p.add_argument("--tokens", default="data/code_corpus_v0.tokens.bin")
    p.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp", "perplexity"])
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_qm_checkpoint(args.ckpt, device)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    report = {}

    if "perplexity" in args.benchmarks:
        ppl = compute_code_perplexity(model, args.tokens, device=device)
        report["perplexity"] = ppl
        print(f"perplexity: {ppl}")

    if "humaneval" in args.benchmarks:
        he = run_humaneval(model, tokenizer, HUMANEVAL_PROBLEMS, device=device, k=args.k)
        report["humaneval"] = he
        print(f"HumanEval pass@{args.k}: {he['passed']}/{he['total']} = {he['pass_at_k']:.2%}")

    if "mbpp" in args.benchmarks:
        mbpp = run_humaneval(model, tokenizer, MBPP_PROBLEMS, device=device, k=args.k)
        report["mbpp"] = mbpp
        print(f"MBPP pass@{args.k}: {mbpp['passed']}/{mbpp['total']} = {mbpp['pass_at_k']:.2%}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()