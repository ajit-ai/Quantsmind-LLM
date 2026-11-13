"""Taksh L0.2 Python Capability Benchmark -- executable runner.

Usage (from the repository root):

    python -m evaluation.l02_runner --ckpt runs/v180_sft/checkpoint_last.pt \
        --tokenizer tokenizer/taksh_code_bpe_v1.json --tag baseline_v180_sft

For each problem: the prompt is rendered exactly as the L0.1 application
renders it (chat template derived from the tokenizer), the checkpoint generates
a completion, the code is extracted, syntax-checked, AST-gated and executed in
a restricted subprocess against the problem's test cases.

Recording: per-problem results land in ``runs/l02_eval/<tag>.jsonl`` and a
summary in ``runs/l02_eval/<tag>.summary.json``. Failed generations are kept
in the JSONL for analysis. Nothing outside ``runs/l02_eval/`` is written.

Safety: generated code runs via ``python -I`` in a fresh temporary directory
with a scrubbed environment and a hard timeout. An AST gate rejects dangerous
operations before execution. This is a restricted runner, not a security
boundary for hostile multi-tenant input.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH_PATH = REPO / "evaluation" / "l02_python_bench_v1.json"
DEFAULT_BENCH_PATH = BENCH_PATH

ALLOWED_MODULES = frozenset({
    "math", "re", "json", "collections", "itertools", "functools", "string",
    "datetime", "csv", "io", "typing", "hashlib", "statistics", "textwrap",
    "heapq", "bisect", "operator",
})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "exit", "quit", "globals", "locals", "vars", "dir",
})
FORBIDDEN_DUNDERS = frozenset({
    "__builtins__", "__import__", "__globals__", "__subclasses__", "__mro__",
    "__code__", "__closure__", "__dict__", "__bases__",
})
FORBIDDEN_ROOTS = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "ctypes", "pickle",
    "importlib", "builtins", "pathlib", "glob", "tempfile",
})
ALLOWED_DUNDER_METHODS = frozenset({
    "__init__", "__str__", "__repr__", "__eq__", "__len__", "__iter__", "__next__",
})
MAX_NODES = 400
MAX_SOURCE_BYTES = 4000


# --------------------------------------------------------------------------- #
# Code extraction                                                               #
# --------------------------------------------------------------------------- #
def extract_code(completion: str) -> str:
    """Deterministically extract executable code from a raw completion.

    1. A fenced ``` block, if present, wins.
    2. Otherwise, from the first line that starts a definition or import.
    3. Trailing lines that look like instructions/examples/markup are dropped.
    """
    text = completion or ""
    if "```" in text:
        parts = text.split("```")
        body = parts[1] if len(parts) > 1 else ""
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in ("python", "py"):
            lines = lines[1:]
        return "\n".join(lines).strip("\n")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            start = i
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        stripped = line.strip().lower()
        if stripped.startswith(("instruction", "example", "<|", "q:", "response",
                                "task:", "note:", "write a", "your task")):
            break
        out.append(line)
    return "\n".join(out).strip("\n")


# --------------------------------------------------------------------------- #
# AST safety gate                                                               #
# --------------------------------------------------------------------------- #
def ast_gate(source: str) -> tuple[bool, str]:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return False, f"source exceeds {MAX_SOURCE_BYTES} bytes"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        return False, f"source exceeds {MAX_NODES} AST nodes"
    has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  for n in nodes)
    if not has_def:
        return False, "no function or class definition found"
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, f"import of {root!r} is not allowed"
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return False, f"use of {node.id!r} is not allowed"
            if node.id in FORBIDDEN_DUNDERS:
                return False, f"use of {node.id!r} is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDERS:
                return False, f"attribute {node.attr!r} is not allowed"
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDER_METHODS:
                return False, f"dunder attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                return False, f"call to {func.id!r} is not allowed"
            if isinstance(func, ast.Name) and func.id == "open":
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    path = first.value
                    if (".." in path or path.startswith(("/", "\\"))
                            or (len(path) >= 2 and path[1] == ":")):
                        return False, "open() path escapes the sandbox directory"
    # Reject explicitly dangerous attribute roots (import os.path is caught by
    # the import gate above; bare names like ``os`` cannot appear).
    for node in nodes:
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in FORBIDDEN_ROOTS:
                return False, f"reference to {root.id!r} is not allowed"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Restricted subprocess execution                                                #
# --------------------------------------------------------------------------- #
WRAPPER_TEMPLATE = '''
import json
import sys

SOURCE = {source!r}
TESTS = json.loads({tests_json!r})
KIND = {kind!r}
ENTRY = {entry!r}

ns = {{}}
exec(compile(SOURCE, "<generated>", "exec"), ns)

def emit(obj):
    print(json.dumps(obj))

try:
    results = []
    if KIND == "function":
        fn = ns[ENTRY]
        for t in TESTS:
            try:
                got = fn(*t["args"])
                if t.get("approx"):
                    ok = got is not None and abs(float(got) - float(t["expected"])) < 1e-6
                else:
                    ok = got == t["expected"]
                results.append({{"name": t["name"], "passed": bool(ok),
                                  "got": repr(got)[:200], "expected": repr(t["expected"])[:200], "error": None}})
            except Exception as exc:
                results.append({{"name": t["name"], "passed": False,
                                  "got": None, "expected": repr(t["expected"])[:200],
                                  "error": "%s: %s" % (type(exc).__name__, exc)}})
    elif KIND == "class":
        cls = ns[ENTRY]
        for t in TESTS:
            try:
                obj = cls(*t.get("init_args", []))
                ok = True
                detail = []
                for call in t["calls"]:
                    got = getattr(obj, call["method"])(*call["args"])
                    good = got == call["expected"]
                    detail.append("%s(%s) -> %r (want %r)" % (call["method"], call["args"], got, call["expected"]))
                    if not good:
                        ok = False
                results.append({{"name": t["name"], "passed": bool(ok),
                                  "got": "; ".join(detail)[:400], "expected": "stepwise match", "error": None}})
            except Exception as exc:
                results.append({{"name": t["name"], "passed": False, "got": None,
                                  "expected": "stepwise match",
                                  "error": "%s: %s" % (type(exc).__name__, exc)}})
    elif KIND == "file_read":
        fn = ns[ENTRY]
        for t in TESTS:
            try:
                path = "fixture_input.txt"
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(t["fixture_content"])
                got = fn(path)
                ok = got == t["expected"]
                results.append({{"name": t["name"], "passed": bool(ok),
                                  "got": repr(got)[:200], "expected": repr(t["expected"])[:200], "error": None}})
            except Exception as exc:
                results.append({{"name": t["name"], "passed": False, "got": None,
                                  "expected": repr(t["expected"])[:200],
                                  "error": "%s: %s" % (type(exc).__name__, exc)}})
    elif KIND == "file_transform":
        fn = ns[ENTRY]
        for t in TESTS:
            try:
                src, dst = "fixture_in.txt", "fixture_out.txt"
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write(t["fixture_content"])
                fn(src, dst)
                with open(dst, encoding="utf-8") as fh:
                    got = fh.read()
                ok = got == t["expected_output"]
                results.append({{"name": t["name"], "passed": bool(ok),
                                  "got": repr(got)[:200], "expected": repr(t["expected_output"])[:200], "error": None}})
            except Exception as exc:
                results.append({{"name": t["name"], "passed": False, "got": None,
                                  "expected": repr(t["expected_output"])[:200],
                                  "error": "%s: %s" % (type(exc).__name__, exc)}})
    emit({{"ok": True, "results": results}})
except Exception as exc:
    emit({{"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}})
'''


def run_in_sandbox(source: str, problem: dict, timeout: float = 8.0) -> dict:
    tests_json = json.dumps(problem["tests"])
    payload = WRAPPER_TEMPLATE.format(
        source=source,
        tests_json=tests_json,
        kind=problem["kind"],
        entry=problem["entry_point"],
    )
    with tempfile.TemporaryDirectory(prefix="l02_bench_") as tmpdir:
        script = Path(tmpdir) / "payload.py"
        script.write_text(payload, encoding="utf-8")
        env = {
            "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
            "TEMP": tmpdir,
            "TMP": tmpdir,
            "PATH": os.path.dirname(sys.executable),
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s", "results": []}
        out = (proc.stdout or "").strip().splitlines()
        if not out:
            return {"ok": False,
                    "error": f"no output (exit={proc.returncode}): {(proc.stderr or '')[:300]}",
                    "results": []}
        try:
            return json.loads(out[-1])
        except json.JSONDecodeError:
            return {"ok": False, "error": f"unparseable output: {out[-1][:300]}", "results": []}


# --------------------------------------------------------------------------- #
# Scoring                                                                       #
# --------------------------------------------------------------------------- #
def score_problem(problem: dict, completion_text: str) -> dict:
    record = {"id": problem["id"], "category": problem["category"],
              "name": problem["name"]}
    code = extract_code(completion_text)
    record["extracted_code"] = code
    if not code.strip():
        record["status"] = "degenerate"
        record["tests"] = []
        record["tests_passed"] = 0
        record["passed"] = False
        return record
    ok, reason = ast_gate(code)
    if not ok:
        record["status"] = "syntax_error" if reason.startswith("syntax") else "gate_rejected"
        record["gate_reason"] = reason
        record["tests"] = []
        record["tests_passed"] = 0
        record["passed"] = False
        return record
    sandbox = run_in_sandbox(code, problem)
    if not sandbox.get("ok", False):
        record["status"] = "runtime_error"
        record["gate_reason"] = sandbox.get("error")
        record["tests"] = sandbox.get("results", [])
        record["tests_passed"] = 0
        record["passed"] = False
        return record
    tests = sandbox.get("results", [])
    passed_n = sum(1 for t in tests if t.get("passed"))
    record["tests"] = tests
    record["tests_passed"] = passed_n
    record["tests_total"] = len(tests)
    if passed_n == len(tests) and tests:
        record["status"] = "correct"
        record["passed"] = True
    else:
        record["status"] = "wrong_answer"
        record["passed"] = False
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/taksh_code_bpe_v1.json")
    parser.add_argument("--tag", required=True, help="output tag (files: runs/l02_eval/<tag>.jsonl/.summary.json)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--binding-repair", action="store_true", default=True)
    parser.add_argument("--no-binding-repair", dest="binding_repair", action="store_false")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N problems")
    parser.add_argument("--bench", default=str(DEFAULT_BENCH_PATH),
                        help="benchmark JSON to use (default: evaluation/l02_python_bench_v1.json)")
    parser.add_argument("--bench-sha256", default=None,
                        help="if set, refuse to run unless the benchmark file's SHA-256 matches")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat the full benchmark N times; reports cross-run output agreement (determinism check)")
    args = parser.parse_args(argv)

    if any(sep in args.tag for sep in ("/", "\\", "..")):
        parser.error("--tag must be a simple file-name stem")

    bench_path = Path(args.bench)
    if not bench_path.is_absolute():
        bench_path = REPO / bench_path
    if not bench_path.is_file():
        parser.error(f"benchmark file not found: {bench_path}")
    if args.bench_sha256:
        import hashlib

        actual = hashlib.sha256(bench_path.read_bytes()).hexdigest()
        if actual.lower() != args.bench_sha256.lower():
            parser.error(
                f"benchmark hash mismatch: expected {args.bench_sha256}, got {actual}. "
                "Refusing to run on an unverified benchmark."
            )
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    problems = bench["problems"]
    if args.limit:
        problems = problems[: args.limit]

    from serving import local_assistant as la
    from serving.local_inference import generate_ids
    from tokenizers import Tokenizer

    device = "cpu"
    model, model_cfg = la.load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    step = None
    try:
        from serving.local_inference import torch_load_checkpoint

        step = torch_load_checkpoint(args.ckpt).get("step")
    except Exception:
        step = None

    out_dir = REPO / "runs" / "l02_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.tag}.jsonl"
    summary_path = out_dir / f"{args.tag}.summary.json"

    def run_once():
        records = []
        for i, problem in enumerate(problems, 1):
            prompt = la.build_prompt(tokenizer, problem["instruction"])
            t1 = time.time()
            gen = generate_ids(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                seed=args.seed,
                binding_repair=args.binding_repair,
            )
            t2 = time.time()
            scored = score_problem(problem, gen["text"])
            scored.update({
                "prompt_tokens": gen["prompt_tokens"],
                "completion_tokens": gen["completion_tokens"],
                "finish_reason": gen["finish_reason"],
                "completion_text": gen["text"],
                "raw_completion_text": gen["raw_text"],
                "repair_status": gen["repair"]["status"],
                "latency_s": round(t2 - t1, 2),
            })
            records.append(scored)
            print(f"[{i:2d}/{len(problems)}] {problem['id']:3s} {problem['name']:22s} "
                  f"status={scored['status']:12s} tokens={gen['completion_tokens']:4d} "
                  f"time={t2 - t1:5.1f}s", flush=True)
        return records

    t0 = time.time()
    repeats = [run_once() for _ in range(max(1, args.repeat))]
    results = repeats[0]
    if len(repeats) > 1:
        agreement = sum(
            1 for r0, r1 in zip(results, *(repeats[1:]))
            if r0["completion_text"] == r1["completion_text"]
        ) / len(results)
    else:
        agreement = None

    def executed(r):
        return r["status"] in ("correct", "wrong_answer", "runtime_error")

    def syntax_valid(r):
        return r["status"] not in ("syntax_error", "gate_rejected", "degenerate")

    def timeouts(r):
        return r["status"] == "runtime_error" and "timeout" in str(r.get("gate_reason", "")).lower()

    executed_n = sum(1 for r in results if executed(r))
    syntax_valid_n = sum(1 for r in results if syntax_valid(r))
    timeout_n = sum(1 for r in results if timeouts(r))
    tests_passed = 0
    tests_total = 0
    for r in results:
        for t in r.get("tests", []):
            tests_total += 1
            tests_passed += int(bool(t.get("passed")))

    by_cat: dict[str, dict] = {}
    for r in results:
        bucket = by_cat.setdefault(
            r["category"],
            {"total": 0, "correct": 0, "syntax_valid": 0, "executed": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(bool(r["passed"]))
        bucket["syntax_valid"] += int(syntax_valid(r))
        bucket["executed"] += int(executed(r))

    finish_reasons = collections.Counter(r["finish_reason"] for r in results)
    import hashlib
    bench_sha = hashlib.sha256(bench_path.read_bytes()).hexdigest()

    summary = {
        "schema": "taksh.l02_eval_summary/1",
        "tag": args.tag,
        "benchmark": str(bench_path.relative_to(REPO)) if bench_path.is_relative_to(REPO) else str(bench_path),
        "benchmark_version": bench["version"],
        "benchmark_sha256": bench_sha,
        "checkpoint": args.ckpt,
        "checkpoint_step": step,
        "tokenizer": args.tokenizer,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "binding_repair": args.binding_repair,
            "device": device,
        },
        "problems_total": len(results),
        "correct": sum(1 for r in results if r["passed"]),
        "degenerate": sum(1 for r in results if r["status"] == "degenerate"),
        "syntax_error": sum(1 for r in results if r["status"] == "syntax_error"),
        "gate_rejected": sum(1 for r in results if r["status"] == "gate_rejected"),
        "runtime_error": sum(1 for r in results if r["status"] == "runtime_error"),
        "wrong_answer": sum(1 for r in results if r["status"] == "wrong_answer"),
        "syntax_valid_rate": round(syntax_valid_n / len(results), 4),
        "execution_rate": round(executed_n / len(results), 4),
        "unit_test_pass_rate": round(tests_passed / max(1, tests_total), 4),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "timeouts": timeout_n,
        "gate_rejections": sum(1 for r in results if r["status"] == "gate_rejected"),
        "unsafe_rejections": sum(1 for r in results if r["status"] == "gate_rejected"),
        "finish_reason_distribution": dict(finish_reasons),
        "repeat_runs": len(repeats),
        "repeat_agreement": agreement,
        "by_category": by_cat,
        "wall_seconds": round(time.time() - t0, 1),
    }

    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
