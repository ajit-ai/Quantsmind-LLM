"""Taksh v2.3.0 — Multimodal Evaluation.

Evaluates the v2.3.0 model on coding benchmarks (HumanEval, MBPP, BigCodeBench)
and multimodal tasks (image-to-code). Requires GPU for inference at 70B scale.

Usage:
    python scripts/eval_v230_multimodal.py \
        --ckpt runs/v230_70b_dpo/checkpoint_last.pt \
        --model-config configs/model/code_v230_70b.yaml \
        --tokenizer tokenizer/taksh_code_bpe_v3.json \
        --benchmark data/v230_eval/humaneval.jsonl \
        --device cuda
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

CKPT_PATH = Path("runs/v230_70b_dpo/checkpoint_last.pt")
MODEL_CONFIG = Path("configs/model/code_v230_70b.yaml")
TOKENIZER_PATH = Path("tokenizer/taksh_code_bpe_v3.json")


def load_model(ckpt: Path, model_cfg_path: Path, device: str = "cuda"):
    import yaml

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))["model"]
    model = QMModel(cfg)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if state is None:
        raise SystemExit(f"{ckpt_path} has no 'model' entry")
    from training.checkpoint import load_model_weights

    report = load_model_weights(model, state, source=str(ckpt_path))
    print(f"Loaded {len(report.loaded)} tensors from {ckpt_path}")
    return model.to(device).eval(), cfg


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = "cuda",
):
    """Generate code completion for a prompt."""
    prompt_ids = tokenizer.encode(prompt).ids
    eos_id = tokenizer.token_to_id("<eos>")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= 131072 else x[:, -131072:]
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


def run_exec_test(full_code: str, test_code: str, timeout: int = 10) -> bool:
    """Run execution test against generated code."""
    wrapper = f"""{full_code}

{test_code}
"""

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapper)
            temp_path = f.name

        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(temp_path)

        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return False
    except Exception:
        return False


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate v2.3.0 multimodal model")
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--model-config", default=str(MODEL_CONFIG))
    p.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    p.add_argument("--benchmark", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    print(f"Loading model from {args.ckpt}")
    model, model_cfg = load_model(Path(args.ckpt), Path(args.model_config), args.device)
    print(f"Model: {model.num_params():,} params, vocab={model_cfg['vocab_size']}")

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_path}")

    problems = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"\nEvaluating on {len(problems)} problems...")

    total = len(problems)
    passed = 0
    syntax_valid = 0

    for i, problem in enumerate(problems):
        instruction = problem.get("prompt", "")
        test_code = problem.get("test", "")
        expected = problem.get("canonical_solution", "")

        generated = generate_completion(
            model,
            tokenizer,
            instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device,
        )

        full = instruction + generated
        syntax_ok = False
        try:
            ast.parse(full)
            syntax_ok = True
        except SyntaxError:
            pass

        exec_ok = False
        if syntax_ok and test_code:
            exec_ok = run_exec_test(full, test_code)

        if syntax_ok:
            syntax_valid += 1
        if exec_ok:
            passed += 1

        status = "PASS" if exec_ok else "FAIL"
        print(f"  [{status}] {problem.get('task_id', i)}: syntax={syntax_ok}, exec={exec_ok}")

    print(f"\n{'='*60}")
    print(f"TAKSH v2.3.0 Evaluation Results")
    print(f"{'='*60}")
    print(f"Total problems:     {total}")
    print(f"Syntax valid:       {syntax_valid}/{total} ({syntax_valid/total*100:.1f}%)")
    print(f"Execution passed:   {passed}/{total} ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()
