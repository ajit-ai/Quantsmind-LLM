# TAKSH v3.0.0 — SFT Data Directory

This directory contains instruction→code fine-tuning pairs for qm-code-v300-70b.

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Instruction→code pairs (100K–1M) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Categories

- Code generation
- Debugging & refactoring
- Code explanation
- Multi-turn coding conversation
- Image-to-code
- Document-to-code

## Format

```json
{
  "id": "sft-001",
  "task": "code_generation",
  "instruction": "Create a Python function to...",
  "input": "",
  "response": "def my_function():\\n    ...",
  "test_args": [[1, 2]],
  "expected_outputs": [3]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
