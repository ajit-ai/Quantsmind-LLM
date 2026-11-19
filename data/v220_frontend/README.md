# TAKSH v2.2.0 — Frontend Domain Data

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Frontend instruction→code pairs (10K–50K) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Categories

- React component generation
- TypeScript type definitions
- CSS styling and layout
- UI state management
- Accessibility (ARIA, keyboard navigation)
- UI testing

## Format

```json
{
  "id": "frontend-001",
  "task": "frontend",
  "instruction": "Create a React button component with...",
  "input": "",
  "response": "import React from 'react';\\n\\nconst Button = ({...}) => {",
  "test_args": [["<button>Click</button>"]],
  "expected_outputs": ["<button>Click</button>"]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
