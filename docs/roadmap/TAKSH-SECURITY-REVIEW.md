# TAKSH Security Review and Qualification

**Status:** Engineering security review, not an independent penetration test or certification.

## Implemented controls
- REST bearer authentication via `TAKSH_ENGINEERING_TOKEN`
- Token scopes via `TAKSH_TOKEN_SCOPES` (`read`, `execute`, `mutate`, `memory`, `audit`)
- Deny-by-default protected endpoints
- `/v1/health` public only
- Request-size limits and authorization audit log
- Workspace canonicalization, path-traversal and symlink-escape protection
- Command allowlisting, timeouts, bounded output, environment filtering
- Explicit approval required for mutations
- Project memory refuses secret-like values

## Tested attack cases
- Missing credentials (401)
- Wrong/invalid credentials (401)
- Missing scopes (403)
- Unconfigured token (503)
- Malformed authorization header (401)
- Protected endpoint without authorization (401/403)
- Workspace escape (SecurityError)
- Symlink escape (SecurityError)
- Unauthorized patch application is not possible without approval

## Dependency vulnerability scan
- Python: `pip_audit` requirements scan: `PASS` (advisory source reachable)
- Node/IDE: `npm audit` reports `found 0 vulnerabilities` after explicit `overrides` for `glob` and `minimatch`
- Workflow: `python -m evaluation.dependency_scan`
- Reports: `reports/taksh_dependency_scan.json`, `reports/taksh_npm_audit.json`

## OS/container isolation status
Not implemented. The sandbox is application-level only. No OS namespace, seccomp,
container or process isolation is claimed.

## Outstanding independent review requirements
- Independent security review of REST auth trust boundary
- Independent review of workspace isolation and file-write paths
- Production secret-management and token-rotation design
- Container/OS isolation if stronger guarantees are required

## Threat boundaries
- Local-only, loopback default
- Token-authenticated local service
- Project files are untrusted input
- Model-generated content is untrusted input
- No external network access from tools
- No Marketplace publication; extension is local client
