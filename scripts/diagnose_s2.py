"""Taksh v1.5.0 — S2 Neural Zero-Capability Diagnostic.

Loads the frozen S2 checkpoint (artifacts/taksh-s2-experimental/taksh-s2.pt),
runs 20 simple Python code-completion prompts, and logs exactly how each
generation fails: syntax error, wrong operation, hallucinated API, incomplete
output, repetition, or other.

This is Week 1 of v1.5.0. Output is written to docs/S2_FAILURE_DIAGNOSIS.md.

Usage:
    python scripts/diagnose_s2.py

Nothing in this script modifies checkpoints, tokenizer, or training data.
It only reads the frozen S2 artifact for diagnosis.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on path so we can import serving/local_inference.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer  # type: ignore

from serving.local_inference import load_model, generate_ids  # noqa: E402

CKPT_PATH = ROOT / "artifacts" / "taksh-s2-experimental" / "taksh-s2.pt"
TOKENIZER_PATH = ROOT / "tokenizer" / "qm_code_bpe_v0.json"
OUTPUT_PATH = ROOT / "docs" / "S2_FAILURE_DIAGNOSIS.md"

# 20 simple Python code-completion prompts (prompt ends mid-function)
PROMPTS: List[Tuple[str, str, str]] = [
    ("p01_add", "def add(a, b):\n    return", "a + b"),
    ("p02_mul", "def multiply(x, y):\n    return", "x * y"),
    ("p03_is_even", "def is_even(n):\n    return", "n % 2 == 0"),
    ("p04_max", "def maximum(a, b):\n    if a > b:\n        return a\n    else:\n        return", "b"),
    ("p05_factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return", "n * factorial(n-1)"),
    ("p06_reverse", "def reverse_string(s):\n    return", "s[::-1]"),
    ("p07_sum_list", "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return", "total"),
    ("p08_filter_pos", "def filter_positive(nums):\n    result = []\n    for n in nums:\n        if n > 0:\n            result.", "append(n)"),
    ("p09_len", "def string_length(s):\n    return", "len(s)"),
    ("p10_is_pal", "def is_palindrome(s):\n    return", "s == s[::-1]"),
    ("p11_count_vowels", "def count_vowels(s):\n    count = 0\n    for ch in s:\n        if ch in 'aeiou':\n            count += 1\n    return", "count"),
    ("p12_abs_val", "def absolute(x):\n    if x < 0:\n        return", "-x\n    else:\n        return x"),
    ("p13_caps", "def capitalize_words(s):\n    return", "s.title()"),
    ("p14_merge", "def merge_dicts(d1, d2):\n    result = {}\n    for k, v in d1.items():\n        result[k] = v\n    for k, v in d2.items():\n        result[k] = v\n    return", "result"),
    ("p15_flatten", "def flatten_list(nested):\n    flat = []\n    for sub in nested:\n        for item in sub:\n            flat.", "append(item)"),
    ("p16_dedup", "def remove_duplicates(items):\n    return", "list(set(items))"),
    ("p17_c2f", "def celsius_to_fahrenheit(c):\n    return", "c * 9/5 + 32"),
    ("p18_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return", "False\n    return True"),
    ("p19_sort", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] =", "arr[j+1], arr[j]\n    return arr"),
    ("p20_mod", "def mod(a, b):\n    return", "a % b"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def classify_failure(prompt: str, generated: str, expected_tail: str) -> Dict[str, Any]:
    """Classify why a generation failed."""
    result: Dict[str, Any] = {
        "failure_category": "other",
        "details": [],
    }

    full = prompt + generated

    # 1. Empty or trivial
    if not generated.strip() or len(generated.strip()) < 2:
        result["failure_category"] = "empty_or_trivial"
        result["details"].append("generation is empty or < 2 chars")
        return result

    # 2. Syntax check
    if not check_syntax(full):
        result["failure_category"] = "syntax_error"
        result["details"].append("full prompt+generation does not parse as Python")
        try:
            ast.parse(full)
        except SyntaxError as exc:
            result["details"].append(f"SyntaxError: {exc.msg} at line {exc.lineno}")
        return result

    # 3. Repeated tokens (degeneration)
    tokens = generated.split()
    if len(tokens) >= 3:
        seen = set()
        repeats = sum(1 for t in tokens if t in seen or seen.add(t))
        repeat_ratio = repeats / len(tokens) if tokens else 0.0
        if repeat_ratio > 0.3:
            result["failure_category"] = "excessive_repetition"
            result["details"].append(f"repeat ratio {repeat_ratio:.2f} > 0.3")

    # 4. Premature end / unclosed delimiters
    stripped = generated.rstrip()
    open_parens = stripped.count("(") - stripped.count(")")
    open_braces = stripped.count("{") - stripped.count("}")
    open_brackets = stripped.count("[") - stripped.count("]")
    if open_parens != 0 or open_braces != 0 or open_brackets != 0:
        if result["failure_category"] == "other":
            result["failure_category"] = "incomplete_output"
        result["details"].append(
            f"unclosed delimiters: parens={open_parens}, braces={open_braces}, brackets={open_brackets}"
        )

    # 5. Check if expected tail appears (semantic match)
    if expected_tail.strip() not in generated:
        # Look for common wrong operations
        lower = generated.lower()
        if "return" in lower:
            if result["failure_category"] == "other":
                result["failure_category"] = "wrong_operation"
            result["details"].append(f"expected tail '{expected_tail}' not found in generation")
        else:
            if result["failure_category"] == "other":
                result["failure_category"] = "no_return_statement"
            result["details"].append("no return statement in generation")

    # 6. Hallucinated API calls (calls to undefined functions)
    try:
        tree = ast.parse(full)
        func = tree.body[0]
        if isinstance(func, ast.FunctionDef):
            called = set()
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.add(node.func.id)
            # Basic builtins that are OK
            builtins = {"len", "max", "min", "sum", "abs", "sorted", "range",
                        "list", "dict", "set", "tuple", "str", "int", "float",
                        "bool", "print", "append", "items", "title"}
            params = {arg.arg for arg in func.args.args}
            assigned = set()
            for node in ast.walk(func):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
            hallucinated = called - params - assigned - builtins
            if hallucinated:
                if result["failure_category"] == "other":
                    result["failure_category"] = "hallucinated_api"
                result["details"].append(f"possible undefined calls: {sorted(hallucinated)}")
    except SyntaxError:
        pass

    return result


def main() -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] S2 Diagnostic starting")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Tokenizer:  {TOKENIZER_PATH}")

    if not CKPT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: tokenizer not found: {TOKENIZER_PATH}", file=sys.stderr)
        sys.exit(1)

    ckpt_hash = sha256_file(CKPT_PATH)
    tok_hash = sha256_file(TOKENIZER_PATH)
    print(f"Checkpoint SHA-256: {ckpt_hash}")
    print(f"Tokenizer SHA-256:  {tok_hash}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model (safe load, weights_only=True)
    model, model_cfg = load_model(str(CKPT_PATH), device, allow_unsafe_trusted=False)
    print(f"Model loaded: {model.num_params():,} params, cfg={model_cfg}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Check <eos> token exists
    eos_id = tokenizer.token_to_id("<eos>")
    print(f"<eos> token id: {eos_id}")

    results: List[Dict[str, Any]] = []
    pass_count = 0
    syntax_valid_count = 0

    for pid, prompt, expected_tail in PROMPTS:
        print(f"\n--- {pid}: {prompt[:60]}... ---")
        out = generate_ids(
            model,
            tokenizer,
            prompt,
            max_new_tokens=64,
            temperature=0.8,
            top_k=50,
            device=device,
        )
        generated = out["text"]
        full = prompt + generated

        syntax_ok = check_syntax(full)
        classification = classify_failure(prompt, generated, expected_tail)
        expected_present = expected_tail.strip() in generated

        if syntax_ok:
            syntax_valid_count += 1
        if expected_present and syntax_ok:
            pass_count += 1

        result = {
            "id": pid,
            "prompt": prompt,
            "expected_tail": expected_tail,
            "generated": generated,
            "full_code": full,
            "syntax_valid": syntax_ok,
            "expected_tail_present": expected_present,
            "finish_reason": out["finish_reason"],
            "completion_tokens": out["completion_tokens"],
            "failure_category": classification["failure_category"],
            "details": classification["details"],
        }
        results.append(result)

        status = "PASS" if (syntax_ok and expected_present) else "FAIL"
        print(f"  [{status}] syntax={syntax_ok}, expected={expected_present}, "
              f"category={classification['failure_category']}, "
              f"tokens={out['completion_tokens']}, finish={out['finish_reason']}")
        if classification["details"]:
            for d in classification["details"]:
                print(f"    - {d}")
        print(f"  Generated: {generated[:120]!r}")

    # Summary
    total = len(results)
    fail_rate = (total - pass_count) / total if total else 0.0
    syntax_rate = syntax_valid_count / total if total else 0.0

    # Category counts
    from collections import Counter
    cat_counts = Counter(r["failure_category"] for r in results)

    # Write diagnosis markdown
    lines: List[str] = []
    lines.append("# Taksh S2 Neural Failure Diagnosis — v1.5.0 Week 1")
    lines.append("")
    lines.append(f"- **Date:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- **Checkpoint:** `{CKPT_PATH}`")
    lines.append(f"- **Checkpoint SHA-256:** `{ckpt_hash}`")
    lines.append(f"- **Tokenizer:** `{TOKENIZER_PATH}`")
    lines.append(f"- **Tokenizer SHA-256:** `{tok_hash}`")
    lines.append(f"- **Model config:** d_model={model_cfg.get('d_model')}, n_layers={model_cfg.get('n_layers')}, "
                 f"n_heads={model_cfg.get('n_heads')}, n_kv_heads={model_cfg.get('n_kv_heads')}, "
                 f"ffn_dim={model_cfg.get('ffn_dim')}, vocab_size={model_cfg.get('vocab_size')}, "
                 f"max_seq_len={model_cfg.get('max_seq_len')}, tie_embeddings={model_cfg.get('tie_embeddings')}")
    lines.append(f"- **Device:** {device}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total prompts:** {total}")
    lines.append(f"- **Passed (syntax valid + expected tail present):** {pass_count}/{total} ({pass_count/total*100:.1f}%)")
    lines.append(f"- **Fail rate:** {fail_rate*100:.1f}%")
    lines.append(f"- **Syntax valid rate:** {syntax_rate*100:.1f}%")
    lines.append("")
    lines.append("### Failure Category Counts")
    lines.append("")
    for cat, count in cat_counts.most_common():
        lines.append(f"- **{cat}:** {count}")
    lines.append("")
    lines.append("## Per-Prompt Results")
    lines.append("")
    lines.append("| ID | Status | Syntax | Expected Tail | Category | Details |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        status = "PASS" if (r["syntax_valid"] and r["expected_tail_present"]) else "FAIL"
        detail_str = "; ".join(r["details"]) if r["details"] else "—"
        lines.append(
            f"| {r['id']} | {status} | {r['syntax_valid']} | {r['expected_tail_present']} | "
            f"{r['failure_category']} | {detail_str} |"
        )
    lines.append("")
    lines.append("## Generated Outputs (Raw)")
    lines.append("")
    for r in results:
        lines.append(f"### {r['id']}")
        lines.append(f"```python")
        lines.append(r["full_code"])
        lines.append("```")
        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Also write JSON for programmatic use
    json_path = ROOT / "docs" / "S2_FAILURE_DIAGNOSIS.json"
    json_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_sha256": ckpt_hash,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": tok_hash,
        "model_config": model_cfg,
        "device": device,
        "total": total,
        "passed": pass_count,
        "fail_rate": round(fail_rate, 4),
        "syntax_valid_rate": round(syntax_rate, 4),
        "failure_categories": dict(cat_counts),
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"Passed: {pass_count}/{total} ({pass_count/total*100:.1f}%)")
    print(f"Fail rate: {fail_rate*100:.1f}%")
    print(f"Syntax valid: {syntax_rate*100:.1f}%")
    print(f"Report: {OUTPUT_PATH}")
    print(f"JSON:   {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
