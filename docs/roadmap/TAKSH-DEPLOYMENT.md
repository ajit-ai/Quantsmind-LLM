# TAKSH Deployment and Operations

**Status:** Local/self-hosted deployment helpers implemented. No external deployment performed.

## Startup

```bash
python -m taksh.deployment  # not a CLI entry point; use the helpers below
```

Use `taksh.deployment.validate_environment()`, `health_check()` and
`readiness_check()` from Python or the REST `/health` endpoint.

## Requirements

- Python >= 3.10
- `configs/model/foundation_111m.yaml`
- `tokenizer/qm_code_bpe_v0.json`

## Health and readiness

- `GET /v1/health` returns service status.
- `readiness_check()` validates environment and project.

## Operational notes

- No cloud account is required.
- No paid service is used.
- Container support is optional and not required for the local workflow.
- Process supervision and graceful shutdown are documented but not automated.
- Performance baselines are recorded in `reports/taksh_performance.json`.
