"""M7 generation and evaluation on the development and locked partitions.

Scoring is deliberately split by what can be checked:

* **syntax** -- the output parses and contains executable content (Python).
* **functional** -- only for the eight held-out reference-execution probes.
* **format** -- deterministic constraints for documentation, explanations, and
  structured/instruction fixtures. It is not semantic correctness.
* **sql** -- candidate and reference queries both run against the same in-memory
  SQLite schema; results must agree.
* **semantic** -- there is no automatic oracle for explanation or docstring
  meaning, so those fields are reported ``NOT_EVALUABLE``, never as passes.

The model is loaded once per checkpoint and generation runs in this process,
but every oracle and SQL check executes in a separate process (or an in-memory
database). Every record names the checkpoint hash and generation settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import yaml

from evaluation.m5_scorers import (
    SYNTAX_VALID,
    extract_code,
)
from evaluation.m7_oracles import run_oracle

SCHEMA = "quantsmind.m7_eval/1"
SQL_SCHEMA = """
CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, email TEXT, status TEXT);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL,
                    created_at TEXT);
INSERT INTO users VALUES (1, 'Alice', 'a@example.com', 'active');
INSERT INTO users VALUES (2, 'Bob', 'b@example.com', 'inactive');
INSERT INTO users VALUES (3, 'Alice', 'a@example.com', 'active');
INSERT INTO orders VALUES (1, 1, 10.0, '2026-01-01');
INSERT INTO orders VALUES (2, 1, 5.0, '2026-02-01');
INSERT INTO orders VALUES (3, 2, 7.0, '2026-03-01');
"""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path="configs/eval/m7_protocol.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def load_oracles(path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["example_id"]: item for item in payload["oracles"]}


def make_untrained_checkpoint(path, model_config="configs/model/code_m3_9m.yaml",
                              tokenizer="tokenizer/qm_code_bpe_v0.json") -> dict:
    """Create the reproducible untrained control. Never used for training."""
    import torch

    from training.model import QMModel

    cfg = yaml.safe_load(Path(model_config).read_text(encoding="utf-8"))["model"]
    torch.manual_seed(0)
    model = QMModel(cfg)
    payload = {
        "schema": "quantsmind.checkpoint/1",
        "step": 0,
        "model": model.state_dict(),
        "config": {"model": cfg, "run": {"model_config": str(model_config)}},
        "seed": 0,
        "tokenizer_identity": {"path": str(tokenizer),
                               "sha256": sha256_file(tokenizer)},
        "notes": ("Intentional untrained control: random initialisation, seed 0, "
                  "zero optimizer steps. Used only as an M7 baseline."),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "sha256": sha256_file(path), "step": 0,
            "parameters": sum(p.numel() for p in model.parameters())}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", text, re.S)
    return match.group(1).strip() if match else text


def _real_body(node) -> bool:
    """A function body that is not only a docstring, pass, or ellipsis."""
    import ast

    for item in node.body:
        if isinstance(item, ast.Pass):
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            if item.value.value is Ellipsis or isinstance(item.value.value, str):
                continue
        return True
    return False


def _code_shape(output) -> tuple[bool, str]:
    """Whether parsed text looks like code, not a bare name or a string."""
    import ast

    code, extraction = extract_code(output)
    if code is None:
        return False, extraction
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return False, f"SYNTAX_INVALID: {exc}"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Import, ast.ImportFrom,
                             ast.Assign, ast.AnnAssign, ast.If, ast.For,
                             ast.While, ast.With, ast.Try)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return (_real_body(node), "FUNCTION_WITH_BODY"
                        if _real_body(node) else "EMPTY_FUNCTION")
            return True, type(node).__name__
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True, "CALL_EXPRESSION"
        if isinstance(node, ast.Return):
            return True, "RETURN"
    return False, "NO_CODE_SHAPE"


def score_generation_syntax(output, function_name=None) -> dict:
    """Task-aware syntax: the response must contain a real function definition."""
    import ast

    code, extraction = extract_code(output)
    if code is None:
        return {"metric": "syntax", "status": extraction, "score": 0.0,
                "denominator": 1, "numerator": 0, "detail": "no code"}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return {"metric": "syntax", "status": "SYNTAX_INVALID", "score": 0.0,
                "denominator": 1, "numerator": 0, "detail": str(exc)}
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not definitions:
        return {"metric": "syntax", "status": "NO_FUNCTION", "score": 0.0,
                "denominator": 1, "numerator": 0,
                "detail": "output contains no top-level function definition"}
    if function_name:
        definitions = [node for node in definitions if node.name == function_name]
        if not definitions:
            return {"metric": "syntax", "status": "WRONG_FUNCTION_NAME",
                    "score": 0.0, "denominator": 1, "numerator": 0,
                    "detail": f"expected def {function_name}"}
    if not any(_real_body(node) for node in definitions):
        return {"metric": "syntax", "status": "EMPTY_FUNCTION", "score": 0.0,
                "denominator": 1, "numerator": 0,
                "detail": "function body is only a docstring/pass/ellipsis"}
    return {"metric": "syntax", "status": SYNTAX_VALID, "score": 1.0,
            "denominator": 1, "numerator": 1,
            "detail": "function definition with a body"}


def score_completion_syntax(output) -> dict:
    """Completion syntax must be code-shaped, not merely parseable text."""
    shape, detail = _code_shape(output)
    if shape:
        return {"metric": "syntax", "status": SYNTAX_VALID, "score": 1.0,
                "denominator": 1, "numerator": 1, "detail": detail}
    status = detail if detail.startswith("SYNTAX_INVALID") else "NO_CODE_SHAPE"
    return {"metric": "syntax", "status": status, "score": 0.0,
            "denominator": 1, "numerator": 0, "detail": detail}


def _sql_rows(sql: str):
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(SQL_SCHEMA)
        cursor = connection.execute(sql)
        return [tuple(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def score_sql(record, output) -> dict:
    candidate = _strip_fence(output)
    if not candidate:
        return {"status": "NO_CODE", "pass": False, "detail": "empty output"}
    try:
        expected = _sql_rows(record["response"])
    except sqlite3.Error as exc:
        return {"status": "FIXTURE_ERROR", "pass": False,
                "detail": f"reference query failed: {exc}"}
    try:
        actual = _sql_rows(candidate)
    except sqlite3.Error as exc:
        return {"status": "SQL_ERROR", "pass": False, "detail": str(exc)}
    ordered = "order by" in record["response"].lower()
    if ordered:
        ok = actual == expected
    else:
        ok = sorted(actual) == sorted(expected)
    return {"status": "PASS" if ok else "RESULT_MISMATCH", "pass": ok,
            "expected": expected, "actual": actual,
            "detail": "exact order" if ordered else "order-insensitive"}


def score_json(record, output) -> dict:
    text = _strip_fence(output)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"status": "INVALID_JSON", "pass": False, "detail": str(exc)}
    required = {"name": str, "host": str, "port": int, "debug": bool}
    if not isinstance(value, dict):
        return {"status": "NOT_OBJECT", "pass": False, "detail": type(value).__name__}
    if set(value) != set(required):
        return {"status": "WRONG_KEYS", "pass": False,
                "detail": f"keys={sorted(value)}"}
    for key, kind in required.items():
        if not isinstance(value[key], kind) or (
                kind is int and isinstance(value[key], bool)):
            return {"status": "WRONG_TYPE", "pass": False,
                    "detail": f"{key} is {type(value[key]).__name__}"}
    return {"status": "PASS", "pass": True, "detail": "keys and types match"}


def score_table(record, output) -> dict:
    text = _strip_fence(output)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("|"):
        return {"status": "NOT_A_TABLE", "pass": False, "detail": "no table"}
    rows = [line for line in lines if line.startswith("|")]
    if len(rows) < 4:
        return {"status": "WRONG_ROWS", "pass": False,
                "detail": f"{len(rows)} table lines"}
    body = " ".join(rows)
    ok = ("qm-tiny" in body and "2.85M" in body
          and "qm-code" in body and "9.36M" in body)
    return {"status": "PASS" if ok else "WRONG_CONTENT", "pass": ok,
            "detail": "required rows present" if ok else "rows missing"}


def score_yaml(record, output) -> dict:
    text = _strip_fence(output)
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {"status": "INVALID_YAML", "pass": False, "detail": str(exc)}
    if not isinstance(value, dict):
        return {"status": "NOT_MAPPING", "pass": False, "detail": type(value).__name__}
    ok = (value.get("name") == "web" and value.get("replicas") == 3
          and str((value.get("resources") or {}).get("limits", {}).get("cpu"))
          == "2")
    return {"status": "PASS" if ok else "WRONG_CONTENT", "pass": ok,
            "detail": "name/replicas/cpu match" if ok else "content mismatch"}


def score_instruction(record, output) -> dict:
    instruction = record["instruction"].lower()
    text = output.strip()
    if "exactly three items" in instruction:
        bullets = [line for line in text.splitlines()
                   if line.strip().startswith("- ")]
        ok = len(bullets) == 3 and len(
            [line for line in text.splitlines() if line.strip()]) == 3
        return {"status": "PASS" if ok else "WRONG_COUNT", "pass": ok,
                "detail": f"{len(bullets)} bullets"}
    if "exactly four items" in instruction:
        numbered = [line for line in text.splitlines()
                    if re.match(r"^\s*\d+\.\s+", line)]
        ok = len(numbered) == 4 and len(
            [line for line in text.splitlines() if line.strip()]) == 4
        return {"status": "PASS" if ok else "WRONG_COUNT", "pass": ok,
                "detail": f"{len(numbered)} numbered items"}
    if "word ready" in instruction:
        ok = text == "READY"
        return {"status": "PASS" if ok else "WRONG_TEXT", "pass": ok,
                "detail": repr(text[:40])}
    if "sum of 17 and 25" in instruction:
        ok = text == "42"
        return {"status": "PASS" if ok else "WRONG_TEXT", "pass": ok,
                "detail": repr(text[:40])}
    if "uppercase" in instruction:
        ok = text == "QUANTSMIND"
        return {"status": "PASS" if ok else "WRONG_TEXT", "pass": ok,
                "detail": repr(text[:40])}
    return {"status": "NO_RULE", "pass": False, "detail": "unknown fixture"}


def score_documentation(output) -> dict:
    text = (output or "").strip()
    if not text:
        return {"status": "EMPTY", "format_pass": False,
                "semantic": "NOT_EVALUABLE"}
    fenced = "```" in text
    parses = False
    try:
        import ast

        tree = ast.parse(text)
        parses = (len(tree.body) == 1
                  and isinstance(tree.body[0], ast.Expr)
                  and isinstance(tree.body[0].value, ast.Constant)
                  and isinstance(tree.body[0].value.value, str))
    except (SyntaxError, ValueError):
        parses = False
    starts = text.startswith('"""') or text.startswith("'''")
    ok = not fenced and (starts or parses) and 40 <= len(text) <= 4000
    return {"status": "PASS" if ok else "WRONG_FORMAT", "format_pass": ok,
            "semantic": "NOT_EVALUABLE",
            "detail": "docstring-only output" if ok else "not a docstring-only response"}


def score_explanation(output) -> dict:
    text = (output or "").strip()
    ok = (20 <= len(text) <= 2000 and "```" not in text
          and not text.startswith(("def ", "class ")))
    return {"status": "PASS" if ok else "WRONG_FORMAT", "format_pass": ok,
            "semantic": "NOT_EVALUABLE",
            "detail": "plain paragraph" if ok else "not a plain explanation"}


def score_completion(record, output) -> dict:
    syntax = score_completion_syntax(output)
    context_ok = False
    if syntax["status"] == SYNTAX_VALID:
        import ast

        try:
            ast.parse(record["input"] + "\n" + output)
            context_ok = True
        except (SyntaxError, ValueError):
            context_ok = False
    return {"syntax": syntax, "context_parse": context_ok}


def score_record(record, output, oracle=None) -> dict:
    task = record["task"]
    scored = {"task": task, "output_chars": len(output or "")}
    if task == "code_generation":
        function_name = (oracle or {}).get("function")
        scored["syntax"] = score_generation_syntax(output, function_name)
        scored["functional"] = (run_oracle(output, oracle) if oracle else None)
    elif task == "code_completion":
        scored.update(score_completion(record, output))
    elif task == "documentation":
        scored["documentation"] = score_documentation(output)
    elif task == "code_explanation":
        scored["explanation"] = score_explanation(output)
    elif task == "sql":
        scored["sql"] = score_sql(record, output)
    elif task == "json_output":
        scored["structured"] = score_json(record, output)
    elif task == "config_output":
        scored["structured"] = (score_yaml(record, output)
                                if "YAML" in record["instruction"]
                                else score_table(record, output))
    elif task in ("format_following", "constrained_output"):
        scored["instruction"] = score_instruction(record, output)
    else:
        scored["status"] = "NO_SCORER"
    return scored


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def evaluate_model(name, checkpoint, records, oracles, out_dir,
                   max_new_tokens=160, seed=7, limit=None) -> dict:
    from training.generate import (
        generate_text,
        load_checkpoint_for_generation,
        load_tokenizer,
    )
    from training.instruction_format import render_prompt

    started = time.time()
    checkpoint_sha = sha256_file(checkpoint)
    model, info = load_checkpoint_for_generation(checkpoint)
    tokenizer_path = (info.get("tokenizer_identity") or {}).get("path") \
        or "tokenizer/qm_code_bpe_v0.json"
    tokenizer, _ = load_tokenizer(tokenizer_path, info["vocab_size"])
    records = records[:limit] if limit else records
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, record in enumerate(records, 1):
        prompt = render_prompt(record["instruction"], record.get("input", "") or "")
        item = {
            "id": record["id"],
            "task": record["task"],
            "partition": record.get("partition"),
            "source": record.get("source"),
            "prompt": prompt,
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
        try:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens, temperature=0.0,
                greedy=True, seed=seed, device="cpu",
            )
            item["output"] = generated["completion"]
            item["finish_reason"] = generated["finish_reason"]
            item["scored"] = score_record(record, item["output"],
                                          oracles.get(record["id"]))
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            item["output"] = ""
            item["generation_error"] = f"{type(exc).__name__}: {exc}"
            item["scored"] = {"task": record["task"], "error": "GENERATION_ERROR"}
        item["elapsed_seconds"] = round(time.time() - started, 2)
        results.append(item)
        if index % 20 == 0 or index == len(records):
            print(f"  {name}: {index}/{len(records)}")

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = aggregate(name, info, results, time.time() - started)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    return summary


def _rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def aggregate(name, info, results, elapsed) -> dict:
    by_partition = {}
    for partition in sorted({r.get("partition") for r in results}):
        subset = [r for r in results if r.get("partition") == partition]
        by_task = {}
        for task in sorted({r["task"] for r in subset}):
            items = [r for r in subset if r["task"] == task]
            scores = [r["scored"] for r in items]
            entry = {"attempted": len(items)}
            syntax = [s["syntax"] for s in scores if "syntax" in s]
            if syntax:
                valid = sum(1 for s in syntax if s["status"] == SYNTAX_VALID)
                entry["syntax_valid"] = valid
                entry["syntax_rate"] = _rate(valid, len(syntax))
            functional = [s["functional"] for s in scores
                          if s.get("functional")]
            if functional:
                entry["functional_attempted"] = len(functional)
                entry["functional_passed"] = sum(f["passed"] for f in functional)
                entry["functional_rate"] = _rate(
                    entry["functional_passed"], len(functional))
                entry["functional_cases_attempted"] = sum(
                    f["attempted"] for f in functional)
                entry["functional_cases_passed"] = sum(
                    len([c for c in f["cases"] if c.get("pass")])
                    for f in functional)
                entry["functional_statuses"] = dict(
                    Counter(f["status"] for f in functional))
            sql = [s["sql"] for s in scores if "sql" in s]
            if sql:
                entry["sql_attempted"] = len(sql)
                entry["sql_passed"] = sum(1 for s in sql if s["pass"])
                entry["sql_rate"] = _rate(entry["sql_passed"], len(sql))
            structured = [s["structured"] for s in scores if "structured" in s]
            if structured:
                entry["structured_attempted"] = len(structured)
                entry["structured_passed"] = sum(
                    1 for s in structured if s.get("pass"))
                entry["structured_rate"] = _rate(
                    entry["structured_passed"], len(structured))
            instruction = [s["instruction"] for s in scores
                           if "instruction" in s]
            if instruction:
                entry["instruction_attempted"] = len(instruction)
                entry["instruction_passed"] = sum(
                    1 for s in instruction if s.get("pass"))
                entry["instruction_rate"] = _rate(
                    entry["instruction_passed"], len(instruction))
            documentation = [s["documentation"] for s in scores
                             if "documentation" in s]
            if documentation:
                entry["documentation_format_passed"] = sum(
                    1 for s in documentation if s.get("format_pass"))
                entry["documentation_semantic"] = "NOT_EVALUABLE"
            explanation = [s["explanation"] for s in scores
                           if "explanation" in s]
            if explanation:
                entry["explanation_format_passed"] = sum(
                    1 for s in explanation if s.get("format_pass"))
                entry["explanation_semantic"] = "NOT_EVALUABLE"
            completion = [s for s in scores if "context_parse" in s]
            if completion:
                entry["completion_context_parse"] = sum(
                    1 for s in completion if s["context_parse"])
            entry["generation_errors"] = sum(
                1 for s in scores if s.get("error") == "GENERATION_ERROR")
            by_task[task] = entry
        by_partition[partition] = {"tasks": by_task, "records": len(subset)}
    return {
        "schema": SCHEMA,
        "model": name,
        "checkpoint": info,
        "elapsed_seconds": round(elapsed, 2),
        "records": len(results),
        "partitions": by_partition,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run M7 evaluation")
    parser.add_argument("--protocol", default="configs/eval/m7_protocol.yaml")
    parser.add_argument("--models", default="untrained,m3_base,m4_tuned,m7_candidate")
    parser.add_argument("--partitions", default="locked")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task filter, e.g. code_generation")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="override the protocol generation budget")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="runs/m7_eval")
    parser.add_argument("--no-create-untrained", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    dataset = protocol["dataset"]
    oracles = load_oracles(dataset["oracles"])
    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else None)

    records = []
    for partition in partitions:
        path = dataset[partition]
        subset = load_jsonl(path)
        for record in subset:
            record["partition"] = partition
        records.extend(subset)
    if tasks:
        records = [r for r in records if r["task"] in set(tasks)]

    checkpoints = dict(protocol["checkpoints"])
    if ("untrained" in models and not args.no_create_untrained
            and not Path(checkpoints["untrained"]).exists()):
        info = make_untrained_checkpoint(
            checkpoints["untrained"], "configs/model/code_m3_9m.yaml",
            dataset.get("tokenizer", "tokenizer/qm_code_bpe_v0.json"))
        print(f"created untrained control: {info}")

    summaries = []
    for name in models:
        checkpoint = checkpoints.get(name)
        if not checkpoint or not Path(checkpoint).exists():
            print(f"SKIP {name}: checkpoint not found: {checkpoint}")
            continue
        print(f"evaluating {name} on {partitions} ({len(records)} records)")
        summary = evaluate_model(
            name, checkpoint, records, oracles,
            Path(args.out_dir) / name,
            max_new_tokens=(args.max_new_tokens
                            or protocol["generation"]["max_new_tokens"]),
            seed=protocol["generation"]["seed"], limit=args.limit,
        )
        summaries.append(summary)

    combined = {
        "schema": SCHEMA,
        "protocol": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "dataset_manifest": dataset["manifest"],
        "dataset_manifest_sha256": sha256_file(dataset["manifest"]),
        "partitions": partitions,
        "tasks": tasks,
        "max_new_tokens": (args.max_new_tokens
                           or protocol["generation"]["max_new_tokens"]),
        "models": summaries,
    }
    out = Path(args.out_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"summary: {out}")
    for summary in summaries:
        for partition, data in summary["partitions"].items():
            tasks = data["tasks"]
            generation = tasks.get("code_generation", {})
            print(f"  {summary['model']:12} {partition:7} "
                  f"syntax={generation.get('syntax_rate')} "
                  f"functional={generation.get('functional_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
