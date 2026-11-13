# TAKSH Consolidated Release-Gate Matrix

**Date:** 2026-11-13
**Package version:** `1.0.0`
**Program labels are execution labels, not package versions.**

| Gate | Requirement | Implementation | Evidence | Unresolved defects | External prerequisite | Blocks GA |
|---|---|---|---|---|---|---|
| 1.1.0 Native Foundation | Config/tokenizer/model/inference contracts engineering-qualified | `configs/model/foundation_111m.yaml`, `evaluation/l110_native_foundation.py`, `reports/l110_native_model_foundation.json` | `tests/test_l110_native_foundation.py` passes; hash reconciliation recorded | None known | None | No |
| 1.1.1 Native Inference Mechanics | Generation contract mechanics | `evaluation/l111_native_inference.py`, `serving/local_inference.py` | `tests/test_l111_native_inference.py` passes | None known | None | No |
| 1.2.0 Native Model Capability | Canonical checkpoint and authorized evaluation | `evaluation/l120_training_authorization.py`, `reports/l120_training_authorization.json` | Readiness report; benchmarks manifest | No canonical checkpoint; training not authorized | Training/compute/corpus/checkpoint authorization | Yes |
| 1.3.0 Coding Intelligence | Measured coding capability against frozen benchmarks | `taksh/coding_engine.py`, `taksh/coding_validation.py` | Focused tests pass; capability `NOT_EVALUATED` | No checkpoint/benchmark; capability unevaluated | Canonical artifact + benchmark authorization | Yes |
| 1.4.0 Autonomous Engineering | Multi-file engineering workflow, tools, memory, approvals | `taksh/engineering.py`, `taksh/tools.py`, `taksh/memory.py`, `taksh/agent.py`, `taksh/service.py`, IDE client | Full Python suite passes; VS Code host tests pass; service tests pass | Product qualification missing; only infrastructure verified | Owner authorization for capability/tool use | Yes |
| 2.0.0 Production GA | Security, deployment, model capability, owner release authorization | Security review package prepared; deployment helpers; recovery; structured logging | `docs/security/REVIEW-PACKAGE.md`; deployment tests | Independent security review outstanding; model capability unevaluated; deployment not proven under production conditions; 2.0.0 `NOT STARTED` | Independent review + owner release authorization + model artifact | Yes |

Historical Python benchmark remains `0/30` and is preserved. Multilingual suite remains unmeasured.
