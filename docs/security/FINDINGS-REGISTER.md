# TAKSH Security Findings Register

| ID | Severity | Component | Evidence | Exploitability | Mitigation | Status | Independent review required |
|---|---|---|---|---|---|---|---|
| TAK-SEC-001 | High | Sandbox | Application-level safeguards only; strict mode fails closed | Confined to host user account; no OS isolation guarantee | Require independent review; implement OS adapter before any untrusted execution | OPEN | Yes |
| TAK-SEC-002 | Medium | REST auth | Single shared bearer token; no rotation/expiry/IdP | Token theft grants full configured scope | Document as local-only; add IdP integration and rotation before remote deployment | OPEN | Yes |
| TAK-SEC-003 | Low | Audit | Audit log is in-memory and finite | Log loss on restart; tampering by local user | Add persistent tamper-evident audit storage | OPEN | Yes |
| TAK-SEC-004 | Medium | Dependencies | Node glob/minimatch vulnerabilities in dev tooling | Development-time only; not shipped | Resolved with npm `overrides`; lock file retained | CLOSED | No |
| TAK-SEC-005 | Low | Memory | Secret-like value rejection is pattern-based | False negatives possible | Add formal secret scanner and redaction tests | OPEN | Yes |
| TAK-SEC-006 | Low | Deployment | Upgrade/rollback not implemented for state schema | Failed upgrade may lose or corrupt state | Implement migration and backup/restore drill | OPEN | Yes |
