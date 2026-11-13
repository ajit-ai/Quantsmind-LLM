# TAKSH L0.6.1 — Integration Closure

**Status:** `PRODUCT WORKFLOW QUALIFIED WITH LIMITATIONS`

Date: 2026-11-13. Branch `feature/l060-1-integration-closure`.

The provider-independent coding workflow is integrated, deterministic-test-backed,
approval-gated, and fail-closed. Native model capability is **not evaluated**
(`BLOCKED_MISSING_CHECKPOINT`). Real external-provider operation is **not tested**
(`BLOCKED_EXTERNAL_DEPENDENCY`). Production deployment is **not qualified**.

No training, checkpoint creation/modification/substitution, dataset or weight
download, paid API call, publication, tag, release, force-push, or merge to
`develop`/`main` occurred in this milestone.

## T01–T18 closure ledger

| ID | Closed issue | Status | Evidence |
|---|---|---|---|
| T01 | Canonical native checkpoint compatibility and capability gating | `BLOCKED_MISSING_CHECKPOINT` | Read-only inspection of every candidate checkpoint under the primary repo; none matches the canonical 111M configuration. Native adapter is selectable but fails closed; no fallback checkpoint. See "T01 checkpoint inspection". |
| T02 | Provider contract error taxonomy and capability metadata | `FIXED_AND_TESTED` | `serving/providers.py`: `ProviderAuthenticationError`, `ProviderRateLimitError`, `ProviderServerError`, `ProviderResponseError`, `redact_secrets`, additive `configured`/`status`/`test_only` capability fields. `tests/test_provider_contract.py`, `tests/test_coding_providers.py`. |
| T03 | OpenAI-compatible HTTP mapping and malformed-response rejection | `FIXED_AND_TESTED` | `serving/coding_providers.py`: 401/403 → auth, 429 → rate limit, 5xx → server, other 4xx → response; timeout/connection mapped; empty/non-JSON/non-object/no-choice/empty-content rejected; error detail redacted. Error-matrix tests. |
| T04 | Ollama reachability/model probe and missing-model diagnostics | `FIXED_AND_TESTED` | Explicit `probe()` against `/api/tags`; actionable missing-model state; malformed/empty responses rejected. Probe is never invoked implicitly. Mock-transport tests. |
| T05 | Diagnostics credential exposure and implicit network I/O | `FIXED_AND_TESTED` | `serving/provider_diagnostics.py` returns non-secret fields only; probe requires an explicit flag; construction/capabilities/diagnostics perform no network I/O. Network-forbidden test. |
| T06 | Repository-content transmission without opt-in | `FIXED_AND_TESTED` | `serving/model_change_provider.py`: file contents only with `allow_file_contents=True`; bounded (5 files × 4,000 chars); secret-name/suffix filtering; provenance records `file_contents_transmitted`; deterministic fixture labelled `test_only`. Opt-in/opt-out workflow tests. |
| T07 | Optional-checkpoint preservation tests hard-failing without the ignored artifact | `FIXED_AND_TESTED` | `evaluation/artifact_integrity.py` makes absence explicit (`ABSENT`/`MATCH`/`MISMATCH`) with operator overrides; applied to `tests/test_l060_foundation.py`, `test_l062_inference.py`, `test_l100eng_dlcae_runtime.py`, `test_l073_readiness.py`, `test_l04_pipeline.py`. |
| T08 | `scripts/eval_v210_agent.py` f-string wrapper syntax defect | `FIXED_AND_TESTED` | Corrected unbalanced brace and triple-quote terminator; `tests/test_eval_script_syntax.py` parses, compiles, and asserts the `__main__` guard. `compileall` passes. |
| T09 | Apply path lacked an explicit reviewer approval gate | `FIXED_AND_TESTED` | `taksh/engineering.py::run_engineering_task` requires a callable `reviewer` when `apply=True`; unapproved plans stay `PLANNED`. CLI requires `--yes` or interactive confirmation. Regression tests. |
| T10 | Validation failure left applied changes in place | `FIXED_AND_TESTED` | All workflow applies go through `PatchTransaction`; validation failure rolls back to pre-task bytes. Regression tests assert rollback. |
| T11 | Patch path traversal / project-root escape | `VERIFIED_NO_DEFECT` | `_resolve` and `PatchTransaction` already confined writes; regression test confirms `../escaped.py` is rejected and nothing escapes the root. |
| T12 | Automatic execution of model-generated shell commands | `VERIFIED_NO_DEFECT` | The workflow consumes only structured `FileChange` proposals and never executes shell strings; a monkeypatched-subprocess test proves the apply path does not spawn commands. |
| T13 | No unified CLI for provider inventory, diagnosis, and workflow | `FIXED_AND_TESTED` | `python -m taksh code` with `--list-providers`, `--diagnose`, `--apply`, `--yes`, `--allow-file-contents`, `--run-tests`, `--allow-test-provider`, `--json`. `tests/test_coding_cli.py` plus the end-to-end demonstration below. |
| T14 | Serving UI bound all interfaces by default | `FIXED_AND_TESTED` | `serving/app.py` resolves a loopback bind host by default via `serving/server_security.resolve_bind_host`; non-loopback binding requires `TAKSH_UI_ALLOW_REMOTE=1`. |
| T15 | Deterministic providers could be mistaken for model capability | `FIXED_AND_TESTED` | Deterministic providers report `test_only=True`, `status="test_only"`, and `source="test-double"`; the CLI refuses them without `--allow-test-provider`. Reports separate mock, real, native, and production qualification. |
| T16 | Real external-provider operation untested and no opt-in harness | `BLOCKED_EXTERNAL_DEPENDENCY` | `tests/test_live_providers.py` adds `TAKSH_LIVE_OPENAI=1` / `TAKSH_LIVE_OLLAMA=1` opt-in minimal-prompt tests. No credentials or endpoint were configured; tests skip; no paid calls were made. |
| T17 | Closure documentation and qualification report stale | `FIXED_AND_TESTED` | This document and `reports/l061_integration_qualification.json` updated; historical L0.6.0 report carries a banner without altering its results. |
| T18 | DLCAE runtime verification after syntax repair | `VERIFIED_NO_DEFECT` | `tests/test_l100eng_dlcae_runtime.py` passes 70 runtime tests after the checkpoint-preservation skip fix. |

## Provider architecture

Canonical public contract: `serving/providers.py`. All providers implement
`ModelProvider` and return `GenerationResult`; failures raise the structured
`ProviderError` hierarchy. Capability reporting is additive and carries
`configured`, `status`, and `test_only`.

`serving/native_provider.py` routes only through
`serving/foundation_inference.load_foundation` / `generate_bounded`.
`serving/inference_engine.py` remains the separate L0.6.2 HTTP serving engine and
is **not** a provider. `taksh.engineering.ChangeProvider` is an internal planning
adapter seam; `native_change_provider()` builds it from the canonical native
adapter.

Provider selection is independent of the coding workflow: `repository → task →
provider → structured change proposal → explicit review → PatchTransaction →
validation → tests → report`.

| Provider | Registry name | Availability | Labelled test-only | Notes |
|---|---|---|---|---|
| OpenAI-compatible | `openai-compatible` | env-configured | no | Credentials never embedded or logged. |
| Ollama | `ollama` | env-configured; local only | no | Reachability/model presence require explicit probe. |
| Native TAKSH | `native-taksh` | fail-closed | no | Requires compatible checkpoint + tokenizer; `checkpoint_missing`/`checkpoint_incompatible` states. |
| Deterministic mock | `deterministic-mock` | always | yes | Conformance test double; not capability evidence. |
| Deterministic patch | `deterministic-patch` | test fixture | yes | Structured-patch fixture used by tests and demonstrations. |

## Security controls

- Repository contents are never transmitted to an external provider unless
  `--allow-file-contents` is explicitly set; the default prompt carries only
  paths, symbols, and test lists.
- Model-generated shell commands are never executed; only bounded structured
  file changes are applied.
- Applying any change requires an explicit reviewer (`--yes` or interactive
  confirmation); plan-only is the default.
- Failed validation rolls the transaction back to the pre-task bytes.
- Patch paths are confined to the resolved project root.
- Provider diagnostics expose no credentials or authorization headers and make
  no network calls unless an explicit probe is requested.
- External endpoints bind loopback by default; non-loopback requires an explicit
  opt-in.
- Missing, incompatible, or absent native checkpoints fail closed with
  `BLOCKED`; there is no fallback checkpoint and no hosted-provider fallback.

## Verification evidence

### Focused regression (authoritative)

Command (Python 3.14, repository root):

```
python -m pytest \
  tests/test_eval_script_syntax.py tests/test_dlcae_runtime_syntax.py \
  tests/test_provider_contract.py tests/test_coding_providers.py \
  tests/test_native_provider.py tests/test_unified_provider_diagnostics.py \
  tests/test_coding_cli.py tests/test_live_providers.py \
  tests/test_model_change_provider.py tests/test_coding_agent_scenarios.py \
  tests/test_engineering_platform.py tests/test_l060_foundation.py \
  tests/test_l062_inference.py tests/test_l100eng_dlcae_runtime.py \
  tests/test_provider_architecture.py tests/test_seed_coder_provider.py \
  tests/test_app_ui.py tests/test_cli_hardening.py \
  tests/test_local_assistant_l01.py tests/test_legacy_serving_security.py \
  tests/test_openai_server.py tests/test_binding_repair_integration.py \
  tests/test_kv_cache.py tests/test_l04_pipeline.py \
  tests/test_l073_readiness.py tests/test_learning_signal.py \
  tests/test_reliability_v122.py tests/test_tokenizer_replacement.py \
  tests/test_v122_reliability.py tests/test_v141_safe_checkpoint.py \
  tests/test_v180_corpus_qualification.py tests/test_v190_readiness.py \
  tests/test_v200_readiness.py tests/test_v210_readiness.py \
  tests/test_v220_readiness.py tests/test_v230_readiness.py \
  tests/test_v300_readiness.py -q
```

Result: **674 passed, 13 skipped, exit code 0** (180.84 s). All skips are
explicit optional-artifact or opt-in-live-provider skips, not silent passes.

Focused subsets previously recorded:

| Subset | Result |
|---|---|
| Provider/native/diagnostics/CLI/live | 41 passed, 2 skipped |
| Engineering/workflow | 33 passed, 1 skipped |
| Foundation/L0.6.2/DLCAE runtime | 168 passed, 3 skipped |

Compilation: `python -m compileall -q serving taksh tests scripts` → exit code 0
(Python 3.11).

### T01 checkpoint inspection (read-only)

Every `*.pt` artifact under the primary repository's `runs/` tree was inspected
with `torch.load(..., weights_only=True, mmap=True)`. NumPy-RNG-containing
artifacts were retried only under the repository's reviewed
`safe_globals(_numpy_checkpoint_allowlist())` context, still with
`weights_only=True`; no unsafe deserialization was attempted and no artifact was
modified.

Canonical target (`configs/foundation/l060_foundation.yaml`): vocab 5,822,
d_model 768, 12 layers, 12 heads, 6 KV heads, FFN 3,072, max sequence 2,048,
tied embeddings, 110,658,816 parameters.

| Artifact family | Parameters | Recorded configuration | Verdict |
|---|---|---|---|
| v160_pretrain, v160_sft, v170_sft_demo, v180_sft, l02_sft, l02_sft_r2, l03_pretrain, l04_e1/e2/e3 | 31,022,976 | `code_v160_15m.yaml` (d_model 384) | `NOT_CANONICAL` |
| m7_pretrain, m7_sft, m7_untrained | 10,849,536 | `code_m3_9m.yaml` (d_model 256) | `NOT_CANONICAL` |
| mm_fit_check | 15,805,184 | `local_15m_mm.yaml` | `NOT_CANONICAL` |
| mm_m1_local, mm_m1_smoke | not established | `weights_only=True` refused (`torch.torch_version.TorchVersion`) | `NOT_PROMOTED` |
| code_pretrain, code_smoke | 24,645,888 | `code_30m.yaml` | `NOT_CANONICAL` |
| code_pretrain_tiny, code_sft, code_sft_smoke, corporate_pretrain, corporate_sft | 868,416 | `code_tiny.yaml` | `NOT_CANONICAL` |
| m3_base, m4_sft | 3,784,800 | `code_m3_3m.yaml` | `NOT_CANONICAL` |
| local_15m | 14,059,136 | `local_15m.yaml` | `NOT_CANONICAL` |
| smoke | 158,016 | vocab override 512 | `NOT_CANONICAL` |

The canonical tokenizer `tokenizer/taksh_code_bpe_v1.json` has vocab 5,822 and
matches the canonical vocabulary size only. Several artifacts share that
vocabulary but differ in width/depth; vocabulary equality alone is not
compatibility. **No compatible canonical checkpoint exists**, so native
capability remains unevaluated. The two mm_m1 artifacts remain
`NOT_PROMOTED`: safe loading is blocked by an unsupported global, and promoting
them would require either unsafe deserialization or a new checkpoint, neither of
which this milestone authorizes.

### Deterministic end-to-end demonstration

A demonstration ran in `C:\Windows\Temp\opencode\taksh-e2e-demo` (outside the
repository) against a throwaway project, exercising the real CLI and workflow
code paths with a registered `demo-patch` test double (30 checks):

- provider inventory and non-secret diagnostics;
- refusal to use a test-only provider without `--allow-test-provider`;
- plan-only run writes nothing and sends no file contents;
- non-interactive `--apply` without `--yes` refuses with no partial write;
- approved `--apply --yes --run-tests` completes, fixes the file, creates the
  second file, and reports project tests `PASS`;
- reviewer rejection leaves the workspace unchanged;
- simulated provider timeout yields `BLOCKED` with no fallback and no write;
- `allow_file_contents=True` is required to send bounded file context;
- scope containment: no file escapes the project root.

Result: **30/30 checks passed, exit code 0**. This is conformance evidence for
the workflow and safety controls, not language-model capability evidence.

## Qualification by surface

| Surface | Status | Basis |
|---|---|---|
| Mock/deterministic conformance | `QUALIFIED_FOR_CONFORMANCE_ONLY` | Contract, error-matrix, workflow, approval, rollback, path-safety, and CLI tests pass. Deterministic fixtures are labelled test doubles. |
| Native TAKSH model | `NATIVE_CAPABILITY_NOT_EVALUATED` | No compatible canonical checkpoint exists (`BLOCKED_MISSING_CHECKPOINT`). Adapter is fail-closed with no fallback. |
| Real external provider | `REAL_PROVIDER_NOT_TESTED` | No credentials/endpoint configured; opt-in live harness exists but was not executed (`BLOCKED_EXTERNAL_DEPENDENCY`). |
| Production deployment | `PRODUCTION_NOT_QUALIFIED` | Security, deployment, operational, and supply-chain qualification remain outstanding. |

**Final decision:** `PRODUCT WORKFLOW QUALIFIED WITH LIMITATIONS`.

## Limitations and required owner action

- P1: provide a canonical `taksh-foundation-111m` checkpoint and matching
  tokenizer (or authorize an artifact-handling decision) to evaluate native
  capability.
- P2: provide an explicit external endpoint/credential environment (or authorize
  a paid test) to qualify the OpenAI-compatible provider; install and select an
  Ollama model to qualify the local provider.
- P2: authorize remote CI on the feature branch or accept local-only evidence;
  feature-branch pushes do not trigger CI.
- P3: choose which foundation artifacts are versioned versus regenerated; the
  optional checkpoint remains intentionally uncommitted.
- P3: DLCAE runtime functionality is verified only through its runtime test
  suite; no independent DLCAE model qualification is claimed.

## Preservation evidence

- Untracked copied data directories were never staged and remain present with
  these SHA-256 tree fingerprints (path + per-file digest manifest):
  - `data/l02_overfit` → `8f710805a56787e1f9e0acdcf645ad07bfad3d9bcbeac18a57b6f78b8358db8b`
  - `data/l02_sft` → `08984e5cfcf765ee58237350fb7da20f4d52b3f4b3d52fc358620c8ed5e872d8`
  - `data/l03_corpus` → `234ae92b5802744f27da2abfff7a3e23edae3f5d5aa2ab68a5bc360f80ab95d7`
  - `data/l04_code_corpus` → `6f736c23c70a1ff48ab82b89e9948c55a3afcf0d7aa4cc40c15a7adb5c26f651`
- The primary repository working tree (`F:/Codes/Git/Quantsmind-LLM`) was not
  modified; its branch, HEAD, and dirty working-tree entries were preserved.
