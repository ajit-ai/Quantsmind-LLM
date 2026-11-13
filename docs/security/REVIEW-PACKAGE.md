# TAKSH Security Review Package

**Audience:** Independent security reviewers.
**Status:** Review-ready package prepared by the development team; NOT independently reviewed or certified.

## 1. System architecture and trust boundaries

- Local-first architecture. No cloud account required for the local engineering service.
- Loopback default binding (`127.0.0.1`). Remote binding requires explicit configuration.
- Trust boundaries:
  - Untrusted: project source files, retrieved content, model-generated text.
  - Trusted: bearer token store, local tool registry after authentication, workspace path resolution.
  - Untrusted runtime input: user-provided objectives, tool arguments, patches.

## 2. API endpoint inventory

| Endpoint | Protection | Required scope |
|---|---|---|
| `/v1/health` | public | none |
| `/v1/tools` | protected | `read` |
| `/v1/projects/inspect` | protected | `read` |
| `/v1/tasks` (POST/GET) | protected | `read` |
| `/v1/tasks/{task_id}/validate` | protected | `read` |
| `/v1/tasks/{task_id}/tests` | protected | `execute` |
| `/v1/tasks/{task_id}/apply` | protected | `mutate` |
| `/v1/tasks/{task_id}/report` | protected | `read` |
| `/v1/memory/{project_id}` | protected | `memory` |
| `/v1/audit` | protected | `audit` |

## 3. Authentication and authorization design

- Bearer token from `TAKSH_ENGINEERING_TOKEN`.
- Optional `TAKSH_TOKEN_SCOPES` selects `read`, `execute`, `mutate`, `memory`, or `audit`.
- Deny-by-default when token is unconfigured.
- Single shared identity; no rotation, expiry or external IdP implemented.
- Audit log records identity, operation, decision and outcome.

## 4. Token and secret handling

- Tokens supplied via environment or untracked config; never committed.
- Placeholder tokens rejected by `validate_deployment_config`.
- Memory refuses secret-like stored values.
- Tokens are not returned by endpoints or logged in structured logs.

## 5. Tool registry and execution flow

- Tools are registered in `taksh/tools.py`.
- Every invocation validates schema, permission and execution limits.
- Mutation tools require `mutate` scope and explicit approval.
- Tool calls are audited and bounded.

## 6. Sandbox architecture

- `taksh/sandbox.py` provides application-level safeguards:
  - workspace canonicalization,
  - command allowlisting,
  - timeouts,
  - bounded output,
  - environment filtering,
  - working-directory restriction.
- `taksh/isolation.py` defines an isolation adapter boundary.
- Strict mode fails closed; no OS-level isolation adapter is implemented for this platform.
- No Docker/container dependency is required.

## 7. Filesystem and process boundaries

- Patch application is restricted to the configured workspace root.
- Path traversal and symlink escapes are rejected.
- Snapshots and rollback are applied on multi-file failures.
- Test execution runs in controlled subprocess calls with timeouts.

## 8. Persistent memory handling

- Project-scoped, keyed by project identity.
- Provenance, confidence, verification status, expiration and invalidation are tracked.
- Secret-like values are rejected.
- Clear/list/invalidate/delete APIs exist.

## 9. Audit and logging design

- Structured JSON logging via `taksh/logging.py`.
- REST audit log records allow/deny decisions, identity and operation.
- Audit log is local and in-memory; rotation and retention policy are not implemented.

## 10. Dependency inventory

- Python: `pyproject.toml`, `requirements.txt`
- Node: `ide/vscode/package.json`, `ide/vscode/package-lock.json`
- Inventory generator: `evaluation/dependency_inventory.py`
- Scanner workflow: `evaluation/dependency_scan.py`

## 11. Vulnerability scan results

- Python `pip_audit`: `PASS` (advisory source reachable at scan time)
- Node `npm audit`: `PASS` after explicit `overrides` for `glob` and `minimatch`
- Reports: `reports/taksh_dependency_scan.json`, `reports/taksh_pip_audit.json`, `reports/taksh_npm_audit.json`

## 12. Existing security tests

- `tests/test_engineering_api.py`
- `tests/test_sandbox.py`
- `tests/test_project_memory.py`
- `tests/test_engineering_service.py`
- `tests/test_tools_runtime.py`
- `tests/test_engineering_platform.py`

## 13. Known limitations and outstanding risks

- No independent security review has been performed.
- No OS/container isolation.
- Single shared bearer token; no rotation/IdP.
- Audit log is local and finite.
- No dependency vulnerability scan of the full development environment beyond declared dependencies.
