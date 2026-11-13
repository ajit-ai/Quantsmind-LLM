# TAKSH Unified Provider Integration

**Status:** Integrated, deterministic-test-backed, approval-gated.
**Limitation:** native model capability is not evaluated and real external-provider
operation is not tested. No model weights, training, or checkpoints were modified.

## Architecture

Provider selection uses one canonical public contract in `serving/providers.py`:

- `GenerationRequest`, `GenerationConfig`, `ProviderModelIdentity`,
  `ProviderCapabilities`, `GenerationResult`
- `ModelProvider` protocol
- Structured errors: `ProviderError`, `ProviderNotFoundError`,
  `ProviderUnavailableError`, `ProviderTimeoutError`,
  `ProviderAuthenticationError`, `ProviderRateLimitError`, `ProviderServerError`,
  `ProviderResponseError`
- `redact_secrets()` for diagnostics and error detail
- Additive capability fields: `configured`, `status`, `test_only`

The committed local-inference path remains unchanged. `serving/native_provider.py`
routes only through `serving/foundation_inference.py`; it never falls back to an
unrelated checkpoint or a hosted provider. `serving/inference_engine.py` remains
the separate L0.6.2 HTTP serving engine and is **not** a provider.
`taksh.engineering.ChangeProvider` is an internal planning adapter seam;
`native_change_provider()` builds it from the canonical native adapter.

Workflow:

`repository -> task -> provider -> structured change proposal -> explicit review
-> PatchTransaction -> validation -> tests -> report`

## Providers

| Provider | Registry name | Status | Evidence |
|---|---|---|---|
| OpenAI-compatible | `openai-compatible` | Implemented; env-configured; no embedded credentials; error matrix tested | `tests/test_coding_providers.py`, `tests/test_provider_contract.py` |
| Ollama | `ollama` | Implemented; local-only; explicit `probe()`; not assumed installed | `tests/test_coding_providers.py` |
| Native TAKSH | `native-taksh` | Adapter over `foundation_inference`; fail-closed without compatible checkpoint | `tests/test_native_provider.py` |
| Deterministic mock | `deterministic-mock` | Test-only conformance double | `tests/test_provider_contract.py` |
| Deterministic patch | `deterministic-patch` | Test fixture for structured patch parsing | `tests/test_model_change_provider.py` |

Deterministic providers are labelled `test_only=True` / `status="test_only"` and
are refused by the CLI unless `--allow-test-provider` is passed.

## Configuration

- External: `TAKSH_OPENAI_API_KEY`, `TAKSH_OPENAI_MODEL`, `TAKSH_OPENAI_BASE_URL`
- Local Ollama-compatible: `TAKSH_OLLAMA_MODEL`, `TAKSH_OLLAMA_BASE_URL`,
  `TAKSH_OLLAMA_CONTEXT_WINDOW`
- Native: `TAKSH_NATIVE_CHECKPOINT`, `TAKSH_NATIVE_TOKENIZER`
- UI bind: loopback by default; `TAKSH_UI_ALLOW_REMOTE=1` required to bind remotely
- Optional artifact overrides: `TAKSH_V180_SFT_CHECKPOINT`,
  `TAKSH_L03_PRETRAIN_CHECKPOINT`, `TAKSH_L02_OVERFIT_CHECKPOINT`

No credentials are committed. Diagnostics never return API keys or
Authorization headers and make no network calls unless an explicit probe is
requested.

## Native boundary

- No checkpoint is marked native-qualified.
- Missing or incompatible checkpoint/tokenizer fails closed (`BLOCKED`); there is
  no fallback checkpoint and no hosted-provider fallback.
- Context length is enforced before generation; generation is capped to remaining
  context and a hard token limit.
- Read-only inspection of every candidate artifact under the primary repository
  found none matching the canonical 111M configuration
  (see `docs/L060-1-INTEGRATION-CLOSURE.md` § "T01 checkpoint inspection").
- Native capability remains `NATIVE_CAPABILITY_NOT_EVALUATED`.

## Security

- Repository contents are sent to a provider only with the explicit
  `--allow-file-contents` opt-in, bounded to five safe text files × 4,000 chars,
  excluding secret-like filenames; provenance records that transmission occurred.
- Applying changes requires an explicit reviewer (`--yes` or interactive
  confirmation). Plan-only is the default.
- Validation failure rolls back the applied transaction.
- Patch paths are confined to the project root.
- Model-generated shell commands are never executed; only structured file
  changes are applied.
- External endpoints default to loopback binding.
- Secrets are redacted from diagnostics and error details.

## Qualification

| Surface | Status |
|---|---|
| Mock/deterministic conformance | Tested: contract, error mapping, redaction, diagnostics, workflow, approval, rollback, path safety, CLI |
| Real external provider | `REAL_PROVIDER_NOT_TESTED` — no credentials/endpoint configured; opt-in live harness implemented (`tests/test_live_providers.py`) but skipped |
| Native TAKSH model | `NATIVE_CAPABILITY_NOT_EVALUATED` — no compatible canonical checkpoint (`BLOCKED_MISSING_CHECKPOINT`) |
| Production deployment | `PRODUCTION_NOT_QUALIFIED` — security/deployment/operational qualification outstanding |

Authoritative focused regression over all touched modules:
**674 passed, 13 skipped, exit code 0**. The full evidence ledger is in
`reports/l061_integration_qualification.json`.
