# TAKSH Security Findings Register

| ID | Severity | Component | Evidence | Exploitability | Mitigation | Status | Independent review required |
|---|---|---|---|---|---|---|---|
| TAK-RC2-001 | Critical | VS Code extension | `taksh.apiKey` ordinary configuration; secret in workspace settings | Secret persistence in repository/settings | Migrated to VS Code SecretStorage; legacy global migration implemented; host and unit tests pass | CLOSED | Yes |
| TAK-SEC-001 | High | Sandbox | Application-level safeguards only; strict mode fails closed | Confined to host user account; no OS isolation guarantee | Require independent review; implement OS adapter before any untrusted execution | OPEN | Yes |
| TAK-SEC-002 | High | REST auth | Single shared bearer token; no expiry/rotation/IdP | Token theft grants full configured scope | Expiry and revocation-hash environment controls added for service token; external IdP not claimed | PARTIAL | Yes |
| TAK-SEC-003 | Medium | Audit | In-memory finite log; no redaction/persistence | Log loss on restart; secret leakage | Durable JSONL audit with rotation and required redaction; storage failure fails closed for authorization | PARTIAL | Yes |
| TAK-SEC-004 | Medium | Dependencies | Node glob/minimatch vulnerabilities in dev tooling | Development-time only; not shipped | Resolved with npm overrides; lock file retained; full pip_audit coverage recorded; Python coverage now includes pyproject groups | CLOSED / PARTIAL | No |
| TAK-SEC-005 | Low | Memory | Secret-like value rejection is pattern-based | False negatives possible | Add formal secret scanner and redaction tests | OPEN | Yes |
| TAK-SEC-006 | Low | Deployment | Upgrade/rollback not implemented for state schema | Failed upgrade may lose or corrupt state | Schema validation, migration, atomic backup/restore, rollback drill and tests added; production migration still not qualified | PARTIAL | Yes |
