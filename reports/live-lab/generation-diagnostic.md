# Taksh Live Lab Generation Diagnostic

**Started:** 2026-10-21T09:06:13+05:30  
**Status:** IN_PROGRESS  
**Probe budget:** 0 of 6 new generation calls used

This diagnostic is bounded. No UI, retraining, checkpoint change, tag, release,
push, or commit is in scope. It tests why the M7 checkpoint returns unrelated
Python fragments through the OpenAI-compatible server.

## Baseline

The prior hang diagnostic sent one request to the OpenAI route:

- prompt: What is 2 + 2? Reply with only the number.
- max_tokens=16, temperature=0.0, top_p=1.0, seed=7
- HTTP 200, 2,100 ms, 103 prompt tokens, 16 completion tokens, finish length
- content began with: `    :param _pytest.abc.`

The answer was unrelated code-like text. This establishes the symptom but not
the layer that causes it.

## Offline prompt finding

M7 was trained with `qm.plain_instruction/1`, rendered by
`training/instruction_format.py`:

```text
<bos>### Instruction
What is 2 + 2? Reply with only the number.

### Response
```

That prompt is 27 tokens. The OpenAI server does not use this renderer. A
single user turn with no system message goes through
`serving/openai_server.py::build_prompt` and
`serving/code_chat.py::format_prompt`, which use
`training/chat_template.py::render_prompt` with `CODE_SYSTEM_PROMPT`. That
prompt is 103 tokens and begins with the 419-character Taksh Coder system
preamble; it omits both instruction and response headers.

The two prompts are not prefixes of one another. The tokenizer has no role
markers, while the server multi-turn branch still constructs them.

Historical evidence from the same checkpoint: all 40 locked code-generation
prompts in `runs/m7_eval/m7_candidate/records_combined.jsonl` scored 0
syntax-valid, and their outputs contain the same `:param` and `_pytest`
fragments. Format-only explanation passes were not semantic passes. The model
therefore also degenerates under the plain training format.

## Probe plan

At most six new generation calls, sequential, no retries, no streaming, 16-32
new tokens, 60-second deadline. P1 uses HTTP; P2-P5 use one in-process
`training.generate` load. P6 is reserved for ambiguity only.

| Probe | Path | Purpose |
|---|---|---|
| P1 | HTTP chat template | Reproduce the live symptom |
| P2 | Plain training prompt | Same question under the trained format |
| P3 | Plain training prompt | Known M7 explanation or instruction task |
| P4 | Chat template | Same task through the serving renderer |
| P5 | Plain training prompt | Locked code-generation task |
| P6 | Reserved | Only if attribution remains ambiguous |

## Stop conditions

Stop before a seventh call, or on timeout, artifact mismatch, startup failure,
or an unresponsive process. No automatic retry.

## Files

- `reports/live-lab/generation-diagnostic.md`
- `reports/live-lab/generation-diagnostic.json`
- `reports/live-lab/generation-probes-raw.json`
- `reports/live-lab/generation-response-p1-raw.json`

No tracked file has been modified. No Git operation has been performed.

## 5. Probes

### P1 - HTTP serving path (2026-10-21T09:09:38+05:30)

- Route: `POST /v1/chat/completions`; server PID 13320; startup 19.6 s; health 200.
- Request: user message `What is 2 + 2? Reply with only the number.`; `max_tokens=16`, `temperature=0.0`, `top_p=1.0`, `seed=7`, `stream=false`.
- Result: HTTP 200 in 952 ms; 103 prompt tokens; 16 completion tokens; `finish_reason=length`.
- Exact content: `    :param _pytest.abc.\n    """\n    :`
- Correct answer: no. Expected `4`.
- Raw response: `reports/live-lab/generation-response-p1-raw.json` (SHA-256 `4492A838F30227FCADC8D36908016E42B7BAA9A01F1D5FEEBBE688C6A560DB63`).
- Interpretation: the live symptom is exactly reproducible on the serving path. The process was stopped and port 8012 is clear.

### P2 - Exact training prompt (in-process)

- Renderer: `training.instruction_format.render_prompt`, prefixed with `<bos>`; 27 tokens.
- Request: same question, 16 new tokens, greedy, seed 7.
- Result: 16 tokens in 7.5 s; `finish_reason=length`.
- Exact completion: `from _pytest.abc.abc.abc.abc. Anyp`
- Correct answer: no. Expected `4`; the model emitted an import-style fragment and a repeated `abc` loop.
- Raw result: `reports/live-lab/generation-p2-raw.json` (SHA-256 `652EA8923B6A85337BD59C190B219565FC24E88EB450FB1DB5FD7AEF3D8E6B18`).
- Interpretation: the serving format is not the only cause. With the training prompt and correct BOS, the checkpoint still fails this simple request and emits code-like fragments.
