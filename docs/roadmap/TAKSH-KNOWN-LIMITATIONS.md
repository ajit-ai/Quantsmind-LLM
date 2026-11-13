# TAKSH Known Limitations Register

**Status:** Current as of 2026-11-13.

| Area | Limitation | Status | Blocker |
|---|---|---|---|
| Native model capability | 111M checkpoint absent; no authorized evaluation | NOT EVALUATED | Canonical checkpoint |
| 1.2.0 model capability | Training/compute not authorized | BLOCKED | Training/compute/corpus authorization |
| 1.3.0 coding intelligence | No measured coding capability | PARTIAL | Checkpoint + benchmark artifacts |
| 1.4.0 autonomous engineering | Tool runtime implemented; not product-qualified | PARTIAL | Capability evidence + approval |
| 2.0.0 GA | Security review, deployment automation, licensing and release authorization absent | NOT STARTED | Owner decisions |
| IDE integration | VS Code Extension Development Host verified; not Marketplace-published | PARTIAL | Marketplace publication not authorized |
| Sandbox | Application-level only; no OS/container isolation | PARTIAL | Independent security review |
| Dependency scan | Python pip-audit PASS; Node has 3 moderate dev-dependency findings | PARTIAL | Independent audit |
| IDE | No Marketplace publication | NOT STARTED | Owner publication authorization |
| Deployment | Local/self-hosted helpers only; no cloud deployment | PARTIAL | Deployment automation |
| Benchmarks | Frozen Python benchmark remains 0/30; multilingual unmeasured | PRESERVED | Owner authorization to re-measure |
| Performance | Benchmarks recorded for this machine; no general claims | RECORDED | No target set |
| Dependency scan | Inventory only; no vulnerability scan | NOT EVALUATED | Audit execution |
