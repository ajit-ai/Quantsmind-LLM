# TAKSH v2.0.0 — SFT Data Directory

This directory contains instruction→code fine-tuning pairs for qm-code-v200-7b.

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Training instruction→code pairs (10K–100K) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Format

Each line is a JSON object:
```json
{
  "id": "task-001",
  "task": "code_generation",
  "instruction": "Write a Python function to...",
  "input": "",
  "response": "def my_function():\\n    ...",
  "test_args": [[1, 2]],
  "expected_outputs": [3]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
