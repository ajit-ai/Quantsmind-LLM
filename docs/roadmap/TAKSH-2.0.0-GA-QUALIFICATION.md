# TAKSH 2.0.0 — Production GA Qualification Checklist

**Status:** NOT STARTED / REQUIRED — nothing here proves GA readiness by itself.

---

## Security and privacy

- [ ] Workspace sandboxing for generated code (network-disabled, filesystem-jailed or explicit allowlist)
- [ ] Permission model documented (who can apply patches, run tests, execute commands)
- [ ] Secret scanner configuration (no credentials in repo, logs, or artifacts)
- [ ] Dependency audit (licenses, known CVEs)
- [ ] Supply-chain provenance for published wheels (if any are ever produced)

## Deployment

- [ ] Documented local CPU deployment profile
- [ ] Optional local GPU profile (if a compatible artifact exists and owner authorizes)
- [ ] Container definition (if containerized distribution is authorized)
- [ ] Health endpoints documented (`/health`, `/v1/models`)

## Compatibility and quality

- [ ] Supported Python version matrix (local host is 3.14.7; 3.11 never executed)
- [ ] Provider compatibility matrix
- [ ] API version policy (`/v1/*`)
- [ ] CLI exit-code policy
- [ ] SDK surface documented (`taksh.coding_engine`, `taksh.coding_validation`, `taksh.project_understanding`, `taksh.specialists`, `taksh.cli`)

## Qualification evidence

- [ ] Frozen Python benchmark remains 0/30 (preserved, never re-scored against a model without owner authorization)
- [ ] Multilingual benchmark artifacts absent/available with SHA-256 verification
- [ ] Every release decision signed by owner
- [ ] Release notes distinguish model readiness from package readiness

## Blocked / not ready

- [ ] Canonical `taksh-foundation-111m` artifact: **MISSING**
- [ ] Training authorization: **NOT GRANTED**
- [ ] Compute authorization: **NOT GRANTED**
- [ ] Capability evaluation: **NOT EVALUATED**
- [ ] Karkain: **DISCOVERED, KCC blocked (`0xC0000135`), KARK-BENCH 0 tasks**

## Verdict

TAKSH is **not GA-ready** as of this milestone implementation. It is a reproducible research/engineering platform with preserved evidence and explicit owner boundaries.
