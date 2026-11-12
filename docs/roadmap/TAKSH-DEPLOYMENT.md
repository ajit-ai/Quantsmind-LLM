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

## Environment configuration

Protected engineering endpoints require `TAKSH_ENGINEERING_TOKEN`. Optional scopes are configured with `TAKSH_TOKEN_SCOPES` (default `read`). Available scopes: `read`, `execute`, `mutate`, `memory`, `audit`. `/v1/health` is public; every other endpoint is protected. Without a configured token, protected endpoints deny access.

Example:
```env
TAKSH_ENGINEERING_TOKEN=<strong-random-token>
TAKSH_TOKEN_SCOPES=read,execute
TAKSH_BIND_HOST=127.0.0.1
TAKSH_BIND_PORT=8000
```

Never commit real tokens. Rotate any token that was ever logged or shared.

## Operational notes

- No cloud account is required.
- No paid service is used.
- Container support is optional and not required for the local workflow.
- Process supervision and graceful shutdown are documented but not automated.
- Performance baselines are recorded in `reports/taksh_performance.json`.
