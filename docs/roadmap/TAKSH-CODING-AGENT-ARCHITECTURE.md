# TAKSH Functional Coding Intelligence

**Status:** Functional provider-backed coding workflow implemented and tested using deterministic conformance fixtures. Native model capability remains `NOT EVALUATED`.

## Provider matrix

| Provider | Implemented | Configured | Tested | Usable now |
|---|---|---|---|---|
| seed-coder-8b-instruct | Existing local adapter | Local checkpoint required | Existing adapter tests | Only with approved local checkpoint |
| taksh-s2 | Existing local adapter | Local checkpoint required | Existing tests | Only with explicit checkpoint |
| qwen2.5-coder-7b-instruct | Placeholder/unavailable | No | N/A | No |
| openai-compatible | Added | `TAKSH_OPENAI_API_KEY`, `TAKSH_OPENAI_MODEL`, `TAKSH_OPENAI_BASE_URL` | Mock-transport tests | With configured local/external endpoint |
| ollama | Added | `TAKSH_OLLAMA_MODEL`, `TAKSH_OLLAMA_BASE_URL` | Mock-transport tests | With local Ollama model |
| deterministic-mock | Added | Built-in/test only | Conformance tests | Test/dev only |
| deterministic-patch | Added | Built-in/test only | Bridge tests | Test/dev only |
| native TAKSH | Skeleton boundary exists | No canonical checkpoint | Native tests limited | Not evaluated |

## Architecture

`User/VS Code/CLI -> TAKSH service -> EngineeringService -> Task orchestrator -> repository context -> configured provider -> structured response -> parse -> validate -> apply through PatchTransaction -> test -> report`.

Provider backends are interchangeable. Parsing failures do not fall back to a different model silently. Mock providers are labelled test fixtures.

## Safe workflow essentials

- File-verbosity limits: prompt includes project summary, not repository contents, unless provider bridge explicitly enables file contents.
- Secrets are not uploaded in the default prompt.
- Changes are applied only through the existing `PatchTransaction`.
- Approvals and scopes are enforced by the existing REST/CLI/permission layer.
- Failure output is captured and fed back through bounded repair.

## Diagnostics

`serving.provider_diagnostics.diagnostics()` reports each registered provider's public identity, capabilities and availability. Credentials are not returned.

## Data-transmission controls

External providers require explicit configuration. The default model bridge does not transmit file contents. `TAKSH_OPENAI_API_KEY` and `TAKSH_OLLAMA_BASE_URL` are read from environment/secrets, never from repository configuration.

## Native model status

`NOT EVALUATED`. No checkpoint was modified. Seed-Coder is not native TAKSH performance.

## Version gates

- 1.1.0 PASS
- 1.1.1 PASS
- 1.2.0 BLOCKED
- 1.3.0 PARTIAL
- 1.4.0 PARTIAL
- 2.0.0 NOT QUALIFIED
