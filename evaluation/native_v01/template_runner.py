"""Native v0.1 evaluation under the product chat-template condition.

The immutable baseline (``baseline_results.json``) feeds each task prompt to the
model *raw*. The serving path (``serving/code_chat.py``) renders prompts through
``training.chat_template.render_for``. Those are different conditions.

This runner keeps the frozen task set and scoring logic untouched, but renders
each prompt exactly as the product does. It writes separate evidence and never
overwrites the baseline. A result here is *not* the native v0.1 baseline; it is
the same tasks under the serving-interface condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from evaluation.native_v01.runner import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKENIZER,
    score_task,
    summarize,
)
from evaluation.native_v01.tasks import get_tasks  # noqa: E402

CONDITION = "chat_template_serving"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(report: dict) -> str:
    s = report.get("summary", {})
    lines = [
        "# TAKSH Native v0.1 — Chat-Template Condition",
        "",
        f"- Generated (UTC): {report.get('generated_at_utc')}",
        f"- Condition: **{CONDITION}** (not the immutable raw-prompt baseline)",
        f"- Status: **{report.get('status')}**",
        f"- Checkpoint: `{report.get('checkpoint_path')}`",
        f"- Checkpoint SHA-256: `{report.get('checkpoint_sha256')}`",
        f"- Tokenizer SHA-256: `{report.get('tokenizer_sha256')}`",
        f"- Load: {report.get('load_seconds')} s | Total generation: {report.get('total_generation_seconds')} s",
        f"- Prompt rendering: `{report.get('prompt_rendering')}`",
        "",
        "## Overall results",
        "",
        f"- Tasks: **{s.get('task_count', 0)}**",
        f"- Syntax valid: **{s.get('syntax_ok', 0)}** ({s.get('syntax_rate', 0):.2%})",
        f"- Executable: **{s.get('executable', 0)}** ({s.get('execution_rate', 0):.2%})",
        f"- Correct: **{s.get('correct', 0)}** ({s.get('correctness_rate', 0):.2%})",
        f"- Instruction-followed: **{s.get('instruction_followed', 0)}** ({s.get('instruction_following_rate', 0):.2%})",
        f"- Truncated: **{s.get('truncated', 0)}** ({s.get('truncation_rate', 0):.2%})",
        f"- Empty: **{s.get('empty_output', 0)}** ({s.get('empty_output_rate', 0):.2%})",
        f"- Degenerate/repetitive: **{s.get('degenerate_output', 0)}** ({s.get('degenerate_output_rate', 0):.2%})",
        f"- Inference failures: **{s.get('inference_failed', 0)}** ({s.get('inference_failure_rate', 0):.2%})",
        "",
        "> This condition is comparable to the baseline in task content and scoring,",
        "> but not identical in prompt bytes. It measures the serving interface.",
        "",
        "## Per-task results",
        "",
        "| Task | Syntax | Exec | Correct | Empty | Degenerate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report.get("tasks", []):
        lines.append(
            f"| {r.get('task_id')} | {int(bool(r.get('syntax_ok')))} | "
            f"{int(bool(r.get('executable')))} | {int(bool(r.get('correct')))} | "
            f"{int(bool(r.get('empty_output')))} | {int(bool(r.get('degenerate_output')))} |"
        )
    if report.get("blocker"):
        lines += ["", "## Blocker", "", f"- {report['blocker']}"]
    lines += [
        "",
        "## Honesty note",
        "",
        "A passing result here would demonstrate that the *serving interface* can elicit",
        "code from the checkpoint. It would not erase the raw-prompt baseline, and it",
        "would not by itself establish general coding capability.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Native v0.1 chat-template condition")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--out-json", default=str(REPO / "evaluation" / "capability" / "template_results.json"))
    p.add_argument("--out-md", default=str(REPO / "docs" / "audit" / "TAKSH-NATIVE-V0.1-TEMPLATE.md"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    tokenizer = Path(args.tokenizer)
    report: dict = {
        "schema": "taksh.native_v01_template/1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "condition": CONDITION,
        "python": sys.version,
        "platform": platform.platform(),
        "device": args.device,
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "tokenizer_path": str(tokenizer),
        "tokenizer_exists": tokenizer.is_file(),
        "prompt_rendering": "training.chat_template.render_for(tokenizer, SYSTEM_PROMPT, task.prompt)",
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "exec_timeout_s": args.timeout,
        },
        "tasks": [],
        "summary": {},
        "status": "NOT_RUN",
        "blocker": None,
    }

    if not checkpoint.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"checkpoint_missing: {checkpoint}"
    elif not tokenizer.is_file():
        report["status"] = "BLOCKED"
        report["blocker"] = f"tokenizer_missing: {tokenizer}"
    else:
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        report["tokenizer_sha256"] = _sha256_file(tokenizer)
        t0 = time.perf_counter()
        try:
            from serving.foundation_inference import generate_bounded, load_foundation
            from training.chat_template import SYSTEM_PROMPT, render_for

            model, tok, meta = load_foundation(str(checkpoint), str(tokenizer), args.device)
            report["model_meta"] = meta
            report["status"] = "RUN"
        except Exception as exc:  # noqa: BLE001
            report["status"] = "BLOCKED"
            report["blocker"] = f"model_load_failed: {type(exc).__name__}: {exc}"
        else:
            report["load_seconds"] = round(time.perf_counter() - t0, 3)
            results = []
            total_gen_s = 0.0
            for task in get_tasks():
                prompt = render_for(tok, SYSTEM_PROMPT, task.prompt)
                started = time.perf_counter()
                error = None
                try:
                    raw = generate_bounded(
                        model,
                        tok,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        device=args.device,
                        seed=args.seed,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    raw = {
                        "text": "", "raw_text": "", "finish_reason": "error",
                        "prompt_tokens": 0, "completion_tokens": 0, "error": error,
                    }
                gen_s = time.perf_counter() - started
                total_gen_s += gen_s
                scored = score_task(task, raw.get("text", ""), raw, timeout_s=args.timeout)
                scored["task_id"] = task.task_id
                scored["category"] = task.category
                scored["difficulty"] = task.difficulty
                scored["prompt"] = task.prompt
                scored["rendered_prompt"] = prompt
                scored["generation_seconds"] = round(gen_s, 4)
                scored["inference_error"] = error
                results.append(scored)
            report["tasks"] = results
            report["total_generation_seconds"] = round(total_gen_s, 3)
            report["summary"] = summarize(results)
            report["status"] = "COMPLETED"

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "condition": CONDITION,
        "status": report.get("status"),
        "summary": report.get("summary"),
        "out_json": str(out_json),
        "out_md": args.out_md,
    }, ensure_ascii=False))
    return 0 if report.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
