# TAKSH v2.3.0 — Evaluation Data Directory

## Structure

| File | Description |
|---|---|
| `humaneval.jsonl` | HumanEval problems (164) |
| `mbpp.jsonl` | MBPP problems (974) |
| `bigcodebench.jsonl` | BigCodeBench problems (1000) |
| `image_to_code.jsonl` | Image-to-code problems (500) |

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

Image-to-Code:
```json
{
  "task_id": "img2code-001",
  "image_path": "screenshots/login_page.png",
  "instruction": "Convert this UI screenshot to React code",
  "expected_structure": {
    "framework": "React",
    "language": "TypeScript",
    "components": ["LoginForm", "Input", "Button"]
  }
}
```

## Provenance

Evaluation benchmarks are for research use only. See LICENSE_PROVENANCE.md.
