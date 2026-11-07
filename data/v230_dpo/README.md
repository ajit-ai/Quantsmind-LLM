# TAKSH v2.3.0 — DPO Data Directory

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Preference pairs (100K–1M) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Format

Each line is a JSON object:
```json
{
  "id": "dpo-001",
  "prompt": "Write a Python function to...",
  "chosen": "def good_function():\\n    return result",
  "rejected": "def bad_function():\\n    return None",
  "category": "code_quality"
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
