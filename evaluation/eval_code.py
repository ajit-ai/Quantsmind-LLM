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

# SCOPE OF THE TABLES ABOVE — read before quoting any number they produce.
#
# HUMANEVAL_PROBLEMS and MBPP_PROBLEMS are 5-problem hand-transcribed subsets that
# exist to prove the generation-and-execute harness runs end to end. They are NOT
# the published benchmarks: HumanEval has 164 problems, and the standard MBPP
# sanitised split has 500. A pass rate measured over 5 problems carries a
# granularity of 20 percentage points per problem and is not comparable to the
# >=20% (HumanEval) and >=25% (MBPP) thresholds in evaluation/promotion.yaml.
#
# Therefore every report built from these tables is stamped
# ``status: "unmeasured"`` and ``promotion_eligible: False``. Only a run against
# the full published suite may claim a benchmark score. Do not raise these
# constants to make a gate pass.
SUBSET_SIZE = {"humaneval": len(HUMANEVAL_PROBLEMS), "mbpp": len(MBPP_PROBLEMS)}
PUBLISHED_SIZE = {"humaneval": 164, "mbpp": 500}


def _unmeasured_subset(suite: str) -> dict:
    """Provenance stamp for a score computed over the 5-problem subset."""
    return {
        "suite": suite,
        "status": "unmeasured",
        "problems_evaluated": SUBSET_SIZE[suite],
        "published_problems": PUBLISHED_SIZE[suite],
        "promotion_eligible": False,
        "note": (
            f"Subset smoke check over {SUBSET_SIZE[suite]} of "
            f"{PUBLISHED_SIZE[suite]} published problems. Not a benchmark score; "
            "does not satisfy evaluation/promotion.yaml. No full-suite run has "
            "been executed in this repository."
        ),
    }


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


def compute_code_perplexity(
    model,
    tokens_path: str,
    seq_len: int = 2048,
    max_windows: int = 32,
    val_fraction: float = 0.01,
    device: str = "cpu",
) -> dict:
    """Held-out NLL / perplexity over non-overlapping windows of the tail split.

    Guards against the silent false pass this used to produce: when the held-out
    tail was shorter than one window the loop exited with zero losses, the mean
    stayed 0.0 and `exp(0)` reported a *perfect* perplexity of 1.0. An
    unmeasurable split now returns ``perplexity: None`` and ``ok: False`` so the
    promotion gate can never pass on missing evidence.
    """
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    split = int(len(data) * (1.0 - val_fraction))
    val_tokens = len(data) - split

    result = {
        "nll": None,
        "perplexity": None,
        "windows": 0,
        "val_tokens": int(val_tokens),
        "ok": False,
    }
    if val_tokens < seq_len + 1:
        result["error"] = (
            f"held-out split has {val_tokens} tokens, need >= {seq_len + 1} "
            f"for one {seq_len}-token window; corpus too small to score"
        )
        return result

    losses = []
    for i in range(max_windows):
        off = split + i * seq_len
        if off + seq_len + 1 > len(data):
            break
        x = torch.from_numpy(data[off:off + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[off + 1:off + 1 + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())

    if not losses:
        result["error"] = "no held-out window could be scored"
        return result

    avg = sum(losses) / len(losses)
    result["nll"] = round(avg, 4)
    result["perplexity"] = round(math.exp(min(20.0, avg)), 2)
    result["windows"] = len(losses)
    result["ok"] = True
    return result


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
        if not ppl.get("ok"):
            print("NOTE: perplexity NOT measurable; it is excluded from any gate.")

    if "humaneval" in args.benchmarks:
        he = run_humaneval(model, tokenizer, HUMANEVAL_PROBLEMS, device=device, k=args.k)
        he.update(_unmeasured_subset("humaneval"))
        report["humaneval"] = he
        print(
            f"HumanEval subset smoke check pass@{args.k}: {he['passed']}/{he['total']} "
            f"= {he['pass_at_k']:.2%} (UNMEASURED - {he['problems_evaluated']} of "
            f"{he['published_problems']} problems, not a benchmark score)"
        )

    if "mbpp" in args.benchmarks:
        mbpp = run_humaneval(model, tokenizer, MBPP_PROBLEMS, device=device, k=args.k)
        mbpp.update(_unmeasured_subset("mbpp"))
        report["mbpp"] = mbpp
        print(
            f"MBPP subset smoke check pass@{args.k}: {mbpp['passed']}/{mbpp['total']} "
            f"= {mbpp['pass_at_k']:.2%} (UNMEASURED - {mbpp['problems_evaluated']} of "
            f"{mbpp['published_problems']} problems, not a benchmark score)"
        )

    report["benchmark_status"] = {
        "full_suite_executed": False,
        "promotion_eligible": False,
        "note": "No full HumanEval (164) or MBPP (500) run exists in this repository.",
    }

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()