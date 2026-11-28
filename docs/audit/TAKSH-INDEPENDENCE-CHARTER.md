# Taksh Independence Charter

**Status:** active engineering constraint
**Scope:** the local coding backend in `serving/`
**Owner:** repository owner

## 1. What "independent" means

Taksh is intended to be usable as a coding model without a second model, a
hosted API, downloaded foundation weights, external training data, or money.

| Layer | Taksh-owned requirement | Current state |
|---|---|---|
| Tokenizer | `tokenizer/taksh_code_bpe_v1.json` (5,822 vocab) | present |
| Training | `training/pretrain.py`, `training/sft.py` | present |
| Checkpoint | local `runs/*/checkpoint_last.pt`, loaded at process start | present, local only |
| Inference | `serving/local_inference.py` (own sampling loop, own KV cache) | present |
| API | `serving/openai_server.py` | present, loopback by default |
| Evaluation | frozen `evaluation/native_v01` baseline | present, immutable |
| Client | an external editor/agent is a *client*, never another model | by design |

"The OpenAI-compatible wire format" is a **local JSON contract**. It does not
call OpenAI, and it does not imply any hosted dependency. A fully proprietary
wire format can be added later without changing the model path.

## 2. What independence does not mean

- It does not mean capability parity with Qwen, DeepSeek or Claude.
- It does not mean the current model can code. The frozen Native v0.1 baseline
  remains **0/20 correct**, and the frozen Python benchmark remains **0/30**.
- It does not authorize publication, release, tags, uploads or merges.
- It does not authorize downloading weights or corpora.

## 3. Structural controls

The following are enforced by code and tests, not by intention:

1. **No provider fallback in the coding API.** `serving/openai_server.py` does
   not import `serving.providers`, so no external adapter can answer a request.
2. **No network client in the local API path.** `openai`, `httpx`, `requests`,
   `aiohttp`, `urllib` and `socket` are absent from the coding API modules.
   Enforced by `tests/test_taksh_independence.py`.
3. **No request-driven weight loading.** Checkpoints are named once on the
   command line. `/load`, `/v1/load`, `/reload` and proxy routes do not exist.
4. **Loopback by default.** Remote binding requires `--allow-remote`; an API key
   is optional but recommended for any non-loopback use.
5. **Declared external providers cannot generate.** Qwen placeholders report
   `available=False`, `offline=True` and raise `ProviderUnavailableError`.
6. **Tool and FIM protocols are text.** They add no vocabulary, no weights and
   no network. Their limitations are stated in the API profile and docs.

## 4. Evidence

| Claim | Evidence |
|---|---|
| Local API imports no network client | `tests/test_taksh_independence.py::test_local_api_path_imports_no_network_client` |
| No provider fallback | `tests/test_taksh_independence.py::test_local_api_does_not_import_the_provider_registry` |
| No checkpoint-load route | `tests/test_taksh_independence.py::test_coding_api_exposes_only_local_routes`; `tests/test_openai_server.py::test_no_endpoint_loads_a_checkpoint_from_a_request` |
| External placeholders cannot answer | `tests/test_taksh_independence.py::test_declared_external_providers_cannot_generate` |
| Streaming is real generation | `tests/test_local_streaming.py::test_stream_and_buffered_generation_are_identical` |
| Tool calls are never invented | `tests/test_openai_server.py::test_undeclared_tool_block_is_never_invented` |
| FIM is a documented text protocol | `tests/test_taksh_api_protocols.py` |

## 5. Change control

Any of the following is an **independence change** and requires explicit owner
authorization recorded before implementation:

- importing a hosted-model SDK into the coding API path;
- adding a base URL or API-key setting that points at a non-local model;
- downloading weights or tokenizers at runtime;
- adding a proxy or fallback route to another provider;
- adding external data to training or evaluation;
- publishing or releasing artifacts.

Until such authorization exists, the tests above are the contract.
