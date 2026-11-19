# TAKSH v2.2.0 — Backend Domain Data

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Backend instruction→code pairs (10K–50K) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Categories

- REST API design
- Database schemas
- Authentication & authorization
- DevOps & CI/CD
- Error handling
- Performance optimization

## Format

```json
{
  "id": "backend-001",
  "task": "backend",
  "instruction": "Create a REST API endpoint for...",
  "input": "",
  "response": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get(\"/users/{user_id}\")\\nasync def read_user(user_id: int):\\n    return {\"user_id\": user_id}",
  "test_args": [[1]],
  "expected_outputs": [{"user_id": 1}]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
