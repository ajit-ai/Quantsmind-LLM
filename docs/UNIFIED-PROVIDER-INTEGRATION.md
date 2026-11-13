# TAKSH Unified Provider Integration

**Status:** Engineering integration only. No model weights, training or checkpoints were modified.

## Architecture

The committed local-inference path remains unchanged. Provider selection uses a single `ModelProvider` contract in `serving/providers.py`. Coding/provider bridges reuse the existing workflow:

`repository -> task -> provider -> structured change proposal -> PatchTransaction/validation -> tests -> report`

Provider code reused from the 2.0.0 worktree is imported, not re-implemented:
- `serving/coding_providers.py` — OpenAI-compatible, Ollama and deterministic test providers
- `serving/model_change_provider.py` — model provider to change-plan bridge
- `serving/provider_diagnostics.py` — non-secret diagnostics
- `serving/native_provider.py` — native TAKSH adapter over `serving/inference_engine.py`
- `taksh/engineering.py`, `project_understanding.py`, `coding_validation.py`, `specialists.py` — shared engineering workflow

## Providers

| Provider | Status | Evidence |
|---|---|---|
| taksh-s2 | Existing local adapter; unchanged; checkpoint-dependent | Existing tests |
| seed-coder-8b-instruct | Existing local adapter; unchanged | Existing tests |
| OpenAI-compatible | Implemented; env configured; mock-transport tests | `tests/test_coding_providers.py` |
| Ollama | Implemented; not assumed installed; mock-transport tests | `tests/test_coding_providers.py` |
| deterministic-mock | Test provider only | `tests/test_coding_providers.py` |
| deterministic-patch | Test fixture for patch parsing | `tests/test_model_change_provider.py` |
| native-taksh | Adapter implemented over `ModelLoader`; unqualified without canonical checkpoint | `tests/test_native_provider.py` |

## Configuration

- External: `TAKSH_OPENAI_API_KEY`, `TAKSH_OPENAI_MODEL`, `TAKSH_OPENAI_BASE_URL`
- Local Ollama-compatible: `TAKSH_OLLAMA_MODEL`, `TAKSH_OLLAMA_BASE_URL` (Ollama optional; not assumed installed)
- Native: `TAKSH_NATIVE_CHECKPOINT`, `TAKSH_NATIVE_TOKENIZER`
- No credentials are committed; diagnostics do not return API keys or Authorization headers

## Native boundary

- No checkpoint is marked native-qualified
- No fallback to incompatible checkpoint exists
- Strict missing checkpoint/tokenizer fails with unavailable status
- Native model capability remains `NOT EVALUATED`

## Security

- External providers require explicit configuration
- Provider requests use explicit transport and bounded generation
- Mocked tests prove contract behavior only; no external service was contacted automatically
- Secrets are not written to workspace configuration or logs
- Deterministic providers are labelled test doubles and are not capability evidence
- No automatic command execution is introduced

## Qualification

- Mock OpenAI/Ollama/native contract behavior: tested
- Real external provider operation: NOT TESTED (no credentials/configured endpoint)
- Real Ollama operation: NOT TESTED (not assumed installed)
- Native TAKSH qualification: NOT EVALUATED (canonical checkpoint missing/unverified)
