"""Native v0.1 evaluation runner for the existing TAKSH checkpoint.

Runs bounded CPU inference with the local native model only (no mocks, no
external providers). Executes generated code in isolated temporary directories
with strict timeouts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = Path(r"F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt")
DEFAULT_TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
DEFAULT_CHECKPOINT_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_code(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # If model echoed prompt/instruction markers, keep from first def onward when present.
    m = re.search(r"(^|\n)(def\s+\w+\s*\()", s)
    if m:
        s = s[m.start(2):].strip()
    return s


def _repetition_stats(text: str) -> dict[str, Any]:
    tokens = re.findall(r"\S+", text or "")
    if not tokens:
        return {"repeated": True, "unique_ratio": 0.0, "max_run": 0, "token_count": 0}
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    max_run = 1
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    unique_ratio = len(counts) / max(1, len(tokens))
    repeated = (len(tokens) >= 6 and (most_common / len(tokens) >= 0.6 or max_run >= 6 or unique_ratio <= 0.25))
    return {
        "repeated": repeated,
        "unique_ratio": round(unique_ratio, 4),
        "max_run": max_run,
        "token_count": len(tokens),
    }


def evaluate_output(task, raw_text: str, raw_result: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_result = raw_result or {}
    code = extract_code(raw_text or "")
    rep = _repetition_stats(raw_text or "")
    syntax_ok = False
    syntax_error = None
    if code.strip():
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError as exc:
            syntax_error = f"{exc.__class__.__name__}: {exc.msg}"
    else:
        syntax_error = "empty_output"

    has_target = bool(task.entry_point and re.search(rf"\bdef\s+{re.escape(task.entry_point)}\s*\(", code))
    empty = not (raw_text or "").strip()
    truncated = raw_result.get("finish_reason") == "length"
    degenerate = bool(rep["repeated"] or (raw_text or "").strip() in {"()", ")", "("})
    instruction_followed = bool(has_target)  # conservative: produced requested function/target

    return {
        "task_id": task.task_id,
        "raw_text": raw_text,
        "code": code,
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "has_target_definition": has_target,
        "instruction_followed": instruction_followed,
        "empty_output": empty,
        "degenerate_output": degenerate,
        "truncated": truncated,
        "repetition": rep,
        "finish_reason": raw_result.get("finish_reason"),
        "prompt_tokens": raw_result.get("prompt_tokens"),
        "completion_tokens": raw_result.get("completion_tokens"),
        "repair": raw_result.get("repair"),
    }


def _build_harness(task, code: str) -> str:
    tests_json = json.dumps(list(task.tests))
    entry = json.dumps(task.entry_point)
    prefix = task.completion_prefix or ""
    return f'''import json, traceback, sys
CODE = {json.dumps(code)}
PREFIX = {json.dumps(prefix)}
ENTRY = {entry}
TESTS = json.loads({json.dumps(tests_json)})
result = {{"executable": False, "defined": False, "passed": 0, "total": len(TESTS), "correct": False, "error": None, "case_results": []}}
ns = {{}}
try:
    src = CODE
    if PREFIX and ENTRY and f"def {{ENTRY}}" not in src:
        src = PREFIX.rstrip() + "\\n" + src.lstrip("\\n")
    compile(src, "<candidate>", "exec")
    exec(src, ns, ns)
    result["executable"] = True
    fn = ns.get(ENTRY)
    result["defined"] = callable(fn)
    if callable(fn):
        for case in TESTS:
            try:
                got = fn(*case["args"])
                ok = got == case["expected"]
                result["case_results"].append({{"name": case["name"], "ok": ok, "got": repr(got), "expected": repr(case["expected"])}})
                if ok:
                    result["passed"] += 1
            except Exception as exc:
                result["case_results"].append({{"name": case["name"], "ok": False, "error": type(exc).__name__}})
        result["correct"] = result["passed"] == result["total"] and result["total"] > 0
    else:
        result["error"] = "entry_point_not_defined"
except Exception as exc:
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(result))
'''


def execute_task_tests(task, code: str, timeout_s: float = 5.0) -> dict[str, Any]:
    if not code.strip():
        return {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": "empty_output", "case_results": []}
    # Syntax gate before execution.
    try:
        compile(code if f"def {task.entry_point}" in code else ((task.completion_prefix or "") + "\n" + code), "<candidate>", "exec")
    except SyntaxError as exc:
        return {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": f"SyntaxError: {exc.msg}", "case_results": []}

    harness = _build_harness(task, code)
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="taksh_nv01_") as td:
            script = Path(td) / "run_case.py"
            script.write_text(harness, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        return {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": "timeout", "case_results": []}
    except Exception as exc:  # noqa: BLE001
        return {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": f"{type(exc).__name__}: {exc}", "case_results": []}

    stdout = (proc.stdout or "").strip().splitlines()
    if stdout:
        try:
            data = json.loads(stdout[-1])
            if proc.returncode != 0 and data.get("error") is None:
                data["error"] = f"exit_{proc.returncode}"
            return data
        except Exception:  # noqa: BLE001
            pass
    err = (proc.stderr or "").strip()[:300] or f"exit_{proc.returncode}"
    return {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": err, "case_results": []}


def score_task(task, raw_text: str, raw_result: dict[str, Any] | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    metrics = evaluate_output(task, raw_text, raw_result)
    exec_result = {"executable": False, "defined": False, "passed": 0, "total": len(task.tests), "correct": False, "error": "not_run", "case_results": []}
    inference_failed = raw_result is None or raw_result.get("error") is not None
    if metrics["syntax_ok"] and task.tests and not inference_failed:
        exec_result = execute_task_tests(task, metrics["code"], timeout_s=timeout_s)
    elif metrics["syntax_ok"] and not task.tests:
        exec_result = {"executable": True, "defined": metrics["has_target_definition"], "passed": 0, "total": 0, "correct": False, "error": "no_executable_tests", "case_results": []}
    else:
        exec_result["error"] = metrics.get("syntax_error") or exec_result["error"]

    metrics.update(
        {
            "category": task.category,
            "difficulty": task.difficulty,
            "kind": task.kind,
            "executable": bool(exec_result.get("executable")),
            "target_defined": bool(exec_result.get("defined")),
            "tests_passed": int(exec_result.get("passed", 0)),
            "tests_total": int(exec_result.get("total", len(task.tests))),
            "correct": bool(exec_result.get("correct")),
            "execution_error": exec_result.get("error"),
            "case_results": exec_result.get("case_results", []),
            "inference_failed": inference_failed,
        }
    )
    return metrics


def _load_native(checkpoint_path: Path, tokenizer_path: Path, device: str = "cpu"):
    sys.path.insert(0, str(REPO))
    from serving.foundation_inference import load_foundation  # noqa: WPS433

    return load_foundation(str(checkpoint_path), str(tokenizer_path), device)


def run_evaluation(
    checkpoint_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    tasks: list[Any] | None = None,
    *,
    device: str = "cpu",
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    seed: int = 1,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    from .tasks import get_tasks

    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    tokenizer = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER
    task_list = list(tasks) if tasks is not None else get_tasks()

    report: dict[str, Any] = {
        "schema": "taksh.native_v01_baseline/1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "tokenizer_path": str(tokenizer),
        "tokenizer_exists": tokenizer.is_file(),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
            "exec_timeout_s": timeout_s,
        },
        "tasks": [],
        "summary": {},
        "status": "NOT_RUN",
        "blocker": None,
    }

    if not checkpoint.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"checkpoint_missing: {checkpoint}"
        return report
    if not tokenizer.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"tokenizer_missing: {tokenizer}"
        return report

    report["checkpoint_sha256"] = _sha256_file(checkpoint)
    report["tokenizer_sha256"] = _sha256_file(tokenizer)

    t0 = time.perf_counter()
    try:
        model, tok, meta = _load_native(checkpoint, tokenizer, device=device)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "BLOCKED"
        report["blocker"] = f"model_load_failed: {type(exc).__name__}: {exc}"
        report["load_seconds"] = round(time.perf_counter() - t0, 3)
        return report
    load_s = time.perf_counter() - t0

    report["model_meta"] = meta
    report["load_seconds"] = round(load_s, 3)
    report["status"] = "RUN"

    from serving.foundation_inference import generate_bounded  # noqa: WPS433

    results = []
    total_gen_s = 0.0
    for task in task_list:
        started = time.perf_counter()
        raw = None
        error = None
        try:
            raw = generate_bounded(
                model,
                tok,
                task.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raw = {"text": "", "raw_text": "", "finish_reason": "error", "prompt_tokens": 0, "completion_tokens": 0, "error": error}
        gen_s = time.perf_counter() - started
        total_gen_s += gen_s
        scored = score_task(task, raw.get("text", "") if isinstance(raw, dict) else str(raw or ""), raw if isinstance(raw, dict) else None, timeout_s=timeout_s)
        scored["task_id"] = task.task_id
        scored["category"] = task.category
        scored["difficulty"] = task.difficulty
        scored["prompt"] = task.prompt
        scored["generation_seconds"] = round(gen_s, 4)
        scored["inference_error"] = error
        results.append(scored)

    report["tasks"] = results
    report["total_generation_seconds"] = round(total_gen_s, 3)
    report["summary"] = summarize(results)
    report["status"] = "COMPLETED"
    return report


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"task_count": 0}
    syntax = sum(1 for r in results if r.get("syntax_ok"))
    executable = sum(1 for r in results if r.get("executable"))
    correct = sum(1 for r in results if r.get("correct"))
    instruction = sum(1 for r in results if r.get("instruction_followed"))
    truncated = sum(1 for r in results if r.get("truncated"))
    empty = sum(1 for r in results if r.get("empty_output"))
    degenerate = sum(1 for r in results if r.get("degenerate_output"))
    inference_failed = sum(1 for r in results if r.get("inference_failed") or r.get("inference_error"))

    by_cat: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        c = by_cat.setdefault(cat, {"count": 0, "syntax_ok": 0, "executable": 0, "correct": 0, "instruction_followed": 0})
        c["count"] += 1
        c["syntax_ok"] += int(bool(r.get("syntax_ok")))
        c["executable"] += int(bool(r.get("executable")))
        c["correct"] += int(bool(r.get("correct")))
        c["instruction_followed"] += int(bool(r.get("instruction_followed")))
    for c in by_cat.values():
        cnt = c["count"] or 1
        c["syntax_rate"] = round(c["syntax_ok"] / cnt, 4)
        c["execution_rate"] = round(c["executable"] / cnt, 4)
        c["correctness_rate"] = round(c["correct"] / cnt, 4)
        c["instruction_rate"] = round(c["instruction_followed"] / cnt, 4)

    return {
        "task_count": n,
        "syntax_ok": syntax,
        "executable": executable,
        "correct": correct,
        "instruction_followed": instruction,
        "truncated": truncated,
        "empty_output": empty,
        "degenerate_output": degenerate,
        "inference_failed": inference_failed,
        "syntax_rate": round(syntax / n, 4),
        "execution_rate": round(executable / n, 4),
        "correctness_rate": round(correct / n, 4),
        "instruction_following_rate": round(instruction / n, 4),
        "truncation_rate": round(truncated / n, 4),
        "empty_output_rate": round(empty / n, 4),
        "degenerate_output_rate": round(degenerate / n, 4),
        "inference_failure_rate": round(inference_failed / n, 4),
        "by_category": by_cat,
    }


def write_results(report: dict[str, Any], out_json: str | Path, out_md: str | Path | None = None) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if out_md:
        out_md = Path(out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    lines = [
        "# TAKSH Native v0.1 — Baseline Capability Evaluation",
        "",
        f"- Generated (UTC): {report.get('generated_at_utc')}",
        f"- Status: **{report.get('status')}**",
        f"- Checkpoint: `{report.get('checkpoint_path')}`",
        f"- Checkpoint SHA-256: `{report.get('checkpoint_sha256')}`",
        f"- Tokenizer: `{report.get('tokenizer_path')}`",
        f"- Model meta: `{json.dumps(report.get('model_meta', {}), ensure_ascii=False)}`",
        f"- Load: {report.get('load_seconds')} s | Total generation: {report.get('total_generation_seconds')} s",
        f"- Generation config: `{json.dumps(report.get('generation', {}), ensure_ascii=False)}`",
        "",
        "## Overall results",
        "",
        f"- Tasks: **{s.get('task_count', 0)}**",
        f"- Syntax valid: **{s.get('syntax_ok', 0)}** ({s.get('syntax_rate', 0):.2%})",
        f"- Executable: **{s.get('executable', 0)}** ({s.get('execution_rate', 0):.2%})",
        f"- Correct: **{s.get('correct', 0)}** ({s.get('correctness_rate', 0):.2%})",
        f"- Instruction-followed (target produced): **{s.get('instruction_followed', 0)}** ({s.get('instruction_following_rate', 0):.2%})",
        f"- Truncated: **{s.get('truncated', 0)}** ({s.get('truncation_rate', 0):.2%})",
        f"- Empty: **{s.get('empty_output', 0)}** ({s.get('empty_output_rate', 0):.2%})",
        f"- Degenerate/repetitive: **{s.get('degenerate_output', 0)}** ({s.get('degenerate_output_rate', 0):.2%})",
        f"- Inference failures: **{s.get('inference_failed', 0)}** ({s.get('inference_failure_rate', 0):.2%})",
        "",
        "> Native inference execution ≠ coding capability. Syntax validity ≠ correctness.",
        "",
        "## Results by category",
        "",
        "| Category | Tasks | Syntax | Executable | Correct | Instruction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, c in (s.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {c['count']} | {c['syntax_ok']} ({c['syntax_rate']:.0%}) | {c['executable']} ({c['execution_rate']:.0%}) | {c['correct']} ({c['correctness_rate']:.0%}) | {c['instruction_followed']} ({c['instruction_rate']:.0%}) |"
        )
    if report.get("blocker"):
        lines += ["", "## Blocker", "", f"- {report['blocker']}"]
    lines += ["", "## Per-task results", "", "| Task | Category | Syntax | Exec | Correct | Instr | Trunc | Degenerate | Error |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report.get("tasks", []):
        err = (r.get("execution_error") or r.get("inference_error") or r.get("syntax_error") or "")
        err = str(err).replace("|", "/")[:60]
        lines.append(
            f"| {r.get('task_id')} | {r.get('category')} | {int(bool(r.get('syntax_ok')))} | {int(bool(r.get('executable')))} | {int(bool(r.get('correct')))} | {int(bool(r.get('instruction_followed')))} | {int(bool(r.get('truncated')))} | {int(bool(r.get('degenerate_output')))} | {err} |"
        )
    lines += ["", "## Representative outputs", ""]
    shown = 0
    for r in report.get("tasks", []):
        if shown >= 6:
            break
        lines += [
            f"### {r.get('task_id')} ({'correct' if r.get('correct') else 'not correct'})",
            "",
            "```text",
            (r.get("raw_text") or "")[:400],
            "```",
            "",
        ]
        shown += 1
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="TAKSH Native v0.1 baseline evaluation")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--out-json", default=str(REPO / "evaluation" / "native_v01" / "baseline_results.json"))
    p.add_argument("--out-md", default=str(REPO / "docs" / "audit" / "TAKSH-NATIVE-V0.1-BASELINE.md"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    report = run_evaluation(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        timeout_s=args.timeout,
    )
    write_results(report, args.out_json, args.out_md)
    print(json.dumps({"status": report.get("status"), "summary": report.get("summary"), "out_json": args.out_json, "out_md": args.out_md}, ensure_ascii=False))
    return 0 if report.get("status") in {"COMPLETED", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
