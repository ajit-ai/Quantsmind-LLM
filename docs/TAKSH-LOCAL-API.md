# Taksh Local API — client integration guide

Taksh is a **local, open-source coding model** with an OpenAI-compatible HTTP
surface. Use it as the only model behind your VSCode coder or agent. The editor
is the client; Taksh is the backend.

> **Capability status (22 Nov 2026).** The current checkpoint scores **0/20
> correct** on the frozen Native v0.1 baseline and **0/30** on the frozen Python
> benchmark. This API makes the model *reachable*; it does not make it *good*.
> Do not present it as a replacement for Qwen/DeepSeek/Claude until frozen
> benchmark evidence says otherwise.

## 1. Start the server

Windows:

```powershell
.\scripts\run_taksh_api.ps1
```

Linux/macOS:

```bash
./scripts/run_taksh_api.sh
```

Defaults:

| Setting | Value |
|---|---|
| Checkpoint | `runs/i4_aligned_sft_long/checkpoint_last.pt` |
| Tokenizer | `tokenizer/taksh_code_bpe_v1.json` |
| Bind | `127.0.0.1:8002` |
| Multi-turn compatibility | `--flatten-history` enabled by the launcher |
| Auth | off unless `QM_API_KEY` is set |

The server loads weights once at startup. No HTTP request can name a checkpoint.

## 2. Discover the model

```bash
curl http://127.0.0.1:8002/v1/models
```

Use the returned `data[0].id` as the `model` in requests. Sending an unknown id
is a `400 model_not_found`, not an echo.

## 3. Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | model id, context, vocab, capabilities |
| `GET /health` | liveness, contract, capability flags |
| `POST /v1/chat/completions` | chat; SSE when `stream=true` |
| `POST /v1/completions` | legacy text completion; `suffix` selects FIM |
| `POST /v1/fim/completions` | Taksh-native fill-in-the-middle |

### Chat (buffered)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="local")
response = client.chat.completions.create(
    model="qm-code-v160-15m",   # use the id from /v1/models
    messages=[{"role": "user", "content": "Write a Python add function."}],
    max_tokens=128,
    temperature=0.2,
)
print(response.choices[0].message.content)
```

### Chat (streaming)

```python
stream = client.chat.completions.create(
    model="qm-code-v160-15m",
    messages=[{"role": "user", "content": "Write a Python add function."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

Streaming deltas come from the shared token loop. **Binding repair is not
applied to streamed text**, because repair is a whole-completion transform. The
buffered endpoint is the repaired endpoint.

### Fill-in-the-middle

```python
response = client.completions.create(
    model="qm-code-v160-15m",
    prompt="def add(a, b):\n    ",
    suffix="\n",
    max_tokens=64,
    temperature=0.0,
)
print(response.choices[0].text)
```

The FIM sentinels (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) are
**literal text**, not tokenizer special tokens. The checkpoint has never been
FIM-trained, so no completion quality is claimed.

### Tools (protocol only)

```json
{
  "model": "qm-code-v160-15m",
  "messages": [{"role": "user", "content": "read a.py"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read a file",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    }
  }],
  "tool_choice": "auto"
}
```

The server renders the schemas into the prompt and parses exactly one
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` block. If the model
does not emit a valid block, the response is ordinary text with
`quantsmind.tool_call_error`; no call is invented.

`tool_choice` accepts only `auto` and `none`. `required` and forced-function
choices are refused because the model cannot guarantee a call. `stream=true`
with `tools` is also refused: a streamed text protocol cannot be turned into a
`tool_calls` message without buffering the whole completion.

**The checkpoint has no tool-use training.** Do not enable autonomous tool loops
on the expectation that calls will appear.

## 4. Multi-turn chat

The shipped tokenizers register only `<pad>`, `<bos>`, `<eos>` and `<unk>`, so
the v1 chat contract cannot express roles. By default a multi-turn request is
refused with `invalid_messages`.

For real editor clients the launcher enables `--flatten-history`, which
serialises the conversation as plain labelled text:

```text
User: first
Assistant: second
User: third
```

This is a compatibility shim, **not** the training template. Responses report
`quantsmind.history_serialization = "plain_text_transcript/1"`. Start the server
without `--flatten-history` if you need the strict contract.

## 5. Wiring a VSCode coder

1. Set the provider base URL to `http://127.0.0.1:8002/v1`.
2. Set the model id to the value from `GET /v1/models`.
3. If `QM_API_KEY` is set, send `Authorization: Bearer <key>`.
4. Keep approval on every edit. There is no autonomous execution endpoint.
5. Prefer buffered requests for patch generation; use streaming for chat UX.

No cloud key is needed. The `openai` Python package is only a convenient client
shape; it is not used by the server.

## 6. Honest limitations

| Limitation | Consequence |
|---|---|
| 0/20 on Native v0.1 | Outputs are not reliable code |
| 2,048-token context | Inject only small, targeted repository context |
| No role tokens | Multi-turn needs `--flatten-history` |
| No native tool training | Tool calls are unlikely; never unattended |
| No FIM training | Inline completion is experimental |
| CPU inference | Long generations are slow; keep `max_tokens` small |
| No structured output | `response_format` is rejected |

## 7. Security

- Loopback only by default; `--allow-remote` is an explicit opt-in.
- Optional bearer auth via `QM_API_KEY`.
- No request can load a checkpoint; weights come from the command line.
- The server never executes generated code.

See `docs/audit/TAKSH-INDEPENDENCE-CHARTER.md` for the independence contract and
`docs/OPEN_ISSUES.md` for known defects.
