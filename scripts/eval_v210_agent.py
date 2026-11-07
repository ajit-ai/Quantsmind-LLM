"""Taksh v2.1.0 — Agent Evaluation.

Evaluates the v2.1.0 agentic model on coding tasks using a sandboxed
environment. The agent is given a task and must use tools to complete it.

Usage:
    python scripts/eval_v210_agent.py --ckpt runs/v210_agent_dpo/checkpoint_last.pt \
        --model-config configs/model/code_v210_7b_agent.yaml \
        --tokenizer tokenizer/taksh_code_bpe_v2.json \
        --benchmark data/v210_agent/benchmark.jsonl \
        --device cuda
"""

from __future__ import annotations

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

CKPT_PATH = Path("runs/v210_agent_dpo/checkpoint_last.pt")
MODEL_CONFIG = Path("configs/model/code_v210_7b_agent.yaml")
TOKENIZER_PATH = Path("tokenizer/taksh_code_bpe_v2.json")


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


def generate_agent_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = "cuda",
):
    """Generate agent response (tool calls or text)."""
    prompt_ids = tokenizer.encode(prompt).ids
    eos_id = tokenizer.token_to_id("<eos>")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= 32768 else x[:, -32768:]
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


def run_tool_in_sandbox(
    tool_name: str,
    tool_args: dict,
    timeout: int = 30,
) -> dict:
    """Run a tool in a sandboxed environment.

    Returns dict with keys: success, output, error.
    """
    # Build a Python script that calls the tool
    wrapper = f"""
import json
import sys

# Tool implementations
def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    return True

def run_shell(command):
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return {{"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}

def run_python(code):
    import io
    from contextlib import redirect_stdout, redirect_stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, {{}})
        return {{"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "error": None}}
    except Exception as e:
        return {{"stdout": "", "stderr": str(e), "error": str(e)}

# Tool map
tools = {{
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "run_python": run_python,
}}

# Execute the tool
tool_name = {json.dumps(tool_name)}
tool_args = {json.dumps(tool_args)}

if tool_name not in tools:
    print(json.dumps({{"success": False, "error": f"Unknown tool: {{tool_name}}"}}))
else:
    try:
        result = tools[tool_name](**tool_args)
        print(json.dumps({{"success": True, "result": result}}))
    except Exception as e:
        print(json.dumps({{"success": False, "error": str(e)}}))
'''

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

        if proc.returncode != 0:
            return {"success": False, "output": None, "error": proc.stderr[:500]}

        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"success": False, "output": None, "error": "JSON parse error"}
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return {"success": False, "output": None, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}


def run_agent_task(
    model,
    tokenizer,
    task: str,
    max_steps: int = 10,
    device: str = "cuda",
):
    """Run an agent task end-to-end.

    The agent is given a task and can use tools to complete it.
    Returns dict with keys: success, steps, tool_calls, human_interventions.
    """
    steps = []
    tool_calls = 0
    human_interventions = 0
    success = False

    context = f"""You are a coding agent. Your task is:

{task}

You have access to the following tools:
- read_file(path): Read a file
- write_file(path, content): Write or overwrite a file
- run_shell(command): Run a shell command
- run_python(code): Execute Python code

When you need to use a tool, output a JSON object with keys: tool, arguments, reasoning.
When you are done, output a JSON object with key: "done": true.

Start by exploring the workspace and understanding the task.
"""

    for step in range(max_steps):
        # Generate agent response
        response = generate_agent_response(
            model,
            tokenizer,
            context,
            max_new_tokens=512,
            device=device,
        )

        # Try to parse as JSON
        try:
            action = json.loads(response.strip())
        except json.JSONDecodeError:
            # Not valid JSON, treat as text response
            context += f"\nAgent response: {response}\n"
            continue

        tool_name = action.get("tool")
        tool_args = action.get("arguments", {})
        reasoning = action.get("reasoning", "")

        if tool_name == "done":
            success = True
            break

        if tool_name:
            tool_calls += 1
            result = run_tool_in_sandbox(tool_name, tool_args)

            context += f"\nTool: {tool_name}\nArgs: {json.dumps(tool_args)}\nResult: {json.dumps(result)}\n"

            if not result.get("success"):
                human_interventions += 1
                context += f"\nError: {result.get('error')}\n"
        else:
            context += f"\nResponse: {response}\n"

        steps.append(
            {
                "step": step + 1,
                "tool": tool_name,
                "args": tool_args,
                "reasoning": reasoning,
            }
        )

    return {
        "success": success,
        "steps": steps,
        "tool_calls": tool_calls,
        "human_interventions": human_interventions,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate v2.1.0 agentic model")
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--model-config", default=str(MODEL_CONFIG))
    p.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    p.add_argument("--benchmark", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=1024)
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

    tasks = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    print(f"\nEvaluating agent on {len(tasks)} tasks...")

    total = len(tasks)
    solved = 0
    total_tool_calls = 0
    total_human_interventions = 0

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", i)
        task_desc = task.get("task", "")

        result = run_agent_task(
            model,
            tokenizer,
            task_desc,
            device=args.device,
        )

        solved += result["success"]
        total_tool_calls += result["tool_calls"]
        total_human_interventions += result["human_interventions"]

        status = "SOLVED" if result["success"] else "FAILED"
        print(
            f"  [{status}] {task_id}: solved={result['success']}, "
            f"steps={result['steps']}, tools={result['tool_calls']}, "
            f"interventions={result['human_interventions']}"
        )

    print(f"\n{'='*60}")
    print(f"TAKSH v2.1.0 Agent Evaluation Results")
    print(f"{'='*60}")
    print(f"Total tasks:     {total}")
    print(f"Tasks solved:    {solved}/{total} ({solved/total*100:.1f}%)")
    print(f"Avg tool calls:  {total_tool_calls/total:.1f}")
    print(f"Avg interventions: {total_human_interventions/total:.1f}")


if __name__ == "__main__":
    main()
