# TAKSH v2.0.0 — Evaluation Data Directory

This directory contains evaluation benchmarks for qm-code-v200-7b.

## Structure

| File | Description |
|---|---|
| `humaneval.jsonl` | HumanEval problems (164) |
| `mbpp.jsonl` | MBPP problems (974) |

## Format

HumanEval:
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def add(a, b):\\n    \"\"\"Return sum.\"\"\"\\n    ",
  "canonical_solution": "return a + b\\n",
  "test": "def test_add():\\n    assert add(2, 3) == 5\\n"
}
```

MBPP:
```json
{
  "task_id": "mbpp/0",
  "text": "Write a function to...",
  "code": "def my_function():\\n    ...",
  "test_list": ["assert my_function(...) == ..."]
}
```

## Provenance

Evaluation benchmarks are for research use only. See LICENSE_PROVENANCE.md.
