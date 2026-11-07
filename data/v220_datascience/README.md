# TAKSH v2.2.0 — Data Science Domain Data

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Data science instruction→code pairs (10K–50K) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Categories

- Pandas data manipulation
- PyTorch model training
- SQL queries
- Data visualization
- EDA pipelines
- Model evaluation

## Format

```json
{
  "id": "datascience-001",
  "task": "datascience",
  "instruction": "Create a pandas function to calculate moving average...",
  "input": "",
  "response": "import pandas as pd\\n\\ndef moving_average(df, column, window):\\n    return df[column].rolling(window=window).mean()",
  "test_args": [[pd.DataFrame({"a": [1, 2, 3, 4, 5]}), "a", 3]],
  "expected_outputs": [[1.0, 1.5, 2.0, 3.0, 4.0]]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
