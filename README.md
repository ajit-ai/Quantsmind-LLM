# Taksh - Corporate Coding LLM

> **Your own from-scratch coding LLM - trained on your code, owned by you.**

Taksh (formerly Quantsmind-LLM) is a sovereign-weights coding LLM project: own architecture, own tokenizer, own training
engine, own data pipeline. Same Transformer family as Llama (RMSNorm/RoPE/GQA/SwiGLU),
trained from scratch on Python + JavaScript/TypeScript code.

**Status (2026-10-20):** local prototype. The pipeline runs end to end on CPU; **no model
capability is demonstrated**. The M7 sovereign capability experiment completed and
reported a negative result: 0/40 locked syntax-valid generations, 0/8 functional
probes, evidence level 0. The current release target is **v1.0.0**; release
status and the publication decision are recorded in
[releases/M7_RELEASE_READINESS.rst](docs/source/releases/M7_RELEASE_READINESS.rst).
Read [Verified status](#verified-status) before quoting any number from this repo.

---

## Verified status

This repo separates four things that are easy to conflate. As of 2026-10-14:

| Layer | State | Evidence |
|---|---|---|
| **Implemented code** | Transformer, training loops, corpus/filters, tokenizer trainer, eval harness, 3 serving apps, CI | source tree; `python -m pytest tests -q` |
| **Smoke-tested execution** | tiny CPU pretrain + resume, SFT loop, multimodal fit-check | `.github/workflows/ci.yml`; `runs/mm_m1_local/log.txt` |
| **Completed experiments** | 200-step code pretrain (1 epoch, 16.3M params); 200-step SFT on a 495,808-param model; multimodal fit-check to step 450 | checkpoint `step` fields in `runs/*/checkpoint_last.pt` |
| **Demonstrated model capability** | **none** | no HumanEval/MBPP run exists; held-out perplexity is not measurable on the current corpus |

Correction log — values previously stated in this README that were wrong, and what is
actually true:

| Claimed | Actual | Source of truth |
|---|---|---|
| `code_30m` ≈ 30M params | **16,257,280** | computed from `configs/model/code_30m.yaml`; matches `state_dict` in `runs/code_pretrain/checkpoint_last.pt` |
| `local_15m` ≈ 11M params | **7,767,680** | computed from `configs/model/local_15m.yaml` |
| Code tokenizer "32k vocab" | **5,822** | `data/code_corpus_v0.tokens.meta.json` |
| Pretrain "10,000 steps" | **200 steps**, `seq_len` 512, ≈1 epoch | `runs/code_pretrain/checkpoint_last.pt` |
| Code corpus "public GitHub" | **77 chunks, all `source: local:*` — this repo's own files**, 68 py / 9 js / **0 ts** | `data/code_corpus_v0.jsonl` |
| Pretrain checkpoint feeds SFT | SFT checkpoint is a **different, 495,808-param `code_tiny` model**, not initialised from the 16.3M pretrain | `runs/code_sft/checkpoint_last.pt` |

The general-text tokenizer claim of **49,152 BPE is correct** (`tokenizer/qm_bpe_v0.json`),
as is the general corpus size of **9,993,983 tokens**.

Model config filenames are also misleading: `code_30m.yaml` is 16.3M, `local_15m.yaml` is
7.8M, `pilot_300m.yaml` is 113.1M. Files were left unrenamed to preserve experiment
reproducibility.

---

## Quick start — train your own coding LLM

```bash
git clone https://github.com/ajit-ai/Quantsmind-LLM.git
cd Quantsmind-LLM
pip install -r requirements.txt

# Or install the code as a package (the wheel does not contain weights or data):
pip install .
taksh --version

# 1. Build code corpus (public GitHub, permissive licenses)
python -m data.build_code_corpus --out data/code_corpus_v0.jsonl
# NOTE: the corpus currently committed in data/ was NOT produced by this command.
# It was produced by scripts/scale_corpus.py, which walks this repository's own
# source tree. See "Verified status" above.

# 2. Train code tokenizer
python -m tokenizer.train_tokenizer --input data/code_corpus_v0.jsonl \
  --out tokenizer/qm_code_bpe_v0.json --vocab-size 32768
# 32768 is a *target*. Realised vocab depends on corpus size; on the 77-chunk
# corpus this yields 5,822. A 32,768-wide embedding over a 5,822-token
# vocabulary leaves ~82% of the embedding table as dead weight.

# 3. Encode corpus to tokens
python -m data.encode_dataset --corpus data/code_corpus_v0.jsonl \
  --tokenizer tokenizer/qm_code_bpe_v0.json --out data/code_corpus_v0.tokens.bin

# 4. Pretrain (16.3M params; CPU-feasible)
python -m training.pretrain --config configs/train/code_local.yaml --out-dir runs/code_30m
# The config targets 10,000 steps x 65,536 tok = 655M tokens. The committed
# checkpoint is 200 steps at seq_len 512 (~1 epoch). The 10k-step figure is a
# configured target that has never been executed, and 100,244 tokens of unique
# corpus cannot support it without heavy repetition.

# 5. SFT on code tasks  -- BLOCKED: the shipped SFT corpus is quarantined
#    See data/code_sft_v0.quarantine.json. The trainer refuses to run on it.
#    423 of 442 pairs are fixed boilerplate; the 19 real pairs are malformed.
python -m training.sft --config configs/train/code_local.yaml \
  --sft-data data/code_sft_v0.jsonl --out-dir runs/code_sft \
  --init-ckpt runs/code_30m/checkpoint_last.pt
#   -> exits with a quarantine error. Pass --ignore-sft-quarantine to override
#      for diagnostics only; such a checkpoint supports no capability claim.

# 6. Chat with your coding assistant (generation works; output is not usable)
python -m serving.code_chat --ckpt runs/code_30m/checkpoint_last.pt
# Open http://localhost:7860
```

## Package and release preparation

The `taksh` package is the **v1.0.0 Apache-2.0 research release**, not a
model-capability claim. Model weights and training data are not included.

```bash
pip install .
taksh --version
```

Prepare and verify a release without publishing:

```bash
python scripts/check_version_consistency.py
python scripts/prepare_release.py
```

The full procedure builds and inspects the wheel and source distribution, runs
`twine check`, installs the wheel in isolated environments, checks the CLI,
serves the actual M7 checkpoint when available, runs the regression tests and
validates the Sphinx docs. Evidence is written to `reports/release-evidence/`;
artifacts go to `dist/`. The preparation script itself does not tag, push, merge,
upload or create a GitHub release. The authorized v1.0.0 source release is
published separately only after these gates pass; identifiers and hashes are in
the release-readiness report. See [RELEASING.md](docs/RELEASING.md),
[VERSIONING.md](docs/VERSIONING.md) and [CHANGELOG.md](CHANGELOG.md).

The M7 experiment demonstrated **no useful code generation**: 0/40 locked
syntax-valid generations, 0/8 functional tasks and evidence level 0. Serving
HTTP success does not establish output correctness.

## What this repo contains

| Path | Purpose |
|---|---|
| `training/model.py` | From-scratch Transformer: RMSNorm / RoPE / GQA / SwiGLU |
| `training/pretrain.py` | Training engine: cosine LR, checkpoint/resume, NaN watchdog |
| `training/sft.py` | SFT training loop with loss masking on assistant tokens |
| `training/data.py` | Token dataset loader (memmap, uint16) |
| `tokenizer/` | BPE trainer + code tokenizer. Wraps HuggingFace `tokenizers`; realised code vocab is **5,822** |
| `data/build_code_corpus.py` | Code corpus: GitHub download, license filter, dedup, encode |
| `data/build_code_sft.py` | SFT pair generator (code gen, explanation, review, refactor) |
| `data/ingest_internal.py` | Internal corporate repo ingestion with secret scanning |
| `evaluation/eval_code.py` | HumanEval + MBPP + code perplexity evaluation |
| `evaluation/promotion.yaml` | Promotion gate rules for code benchmarks |
| `serving/code_chat.py` | Code chat UI (Gradio, syntax highlighting) |
| `serving/completion_server.py` | REST API for IDE code completion (FastAPI) |
| `serving/review_server.py` | REST API for automated code review (FastAPI) |
| `serving/local_inference.py` | Local model loading + text generation |
| `configs/model/` | Model configs: code_30m, local_15m, pilot_300m |
| `configs/train/` | Training configs: code_local, corporate_ft, local_15m |
| `tests/` | Unit tests (model, tokenizer, SFT, eval, serving) |
| `.github/workflows/` | CI on every push |

## Architecture

```
Input tokens → [Token Embedding + RoPE] → 8× Transformer Blocks → [RMSNorm] → [LM Head] → Output logits

Each Transformer Block:
  ├── RMSNorm → Multi-Head Attention (GQA) → Residual
  └── RMSNorm → SwiGLU FFN → Residual
```

| Config | Filename says | **Actual params** | Vocab | Seq Len | Use Case |
|---|---|---|---|---|---|
| `code_tiny` | tiny | **495,808** | 5,822 | 512 | Smoke / CI. Used by every shipped checkpoint |
| `code_30m` | 30M | **16,257,280** | 32,768 | 2048 | Only 200 steps ever run |
| `local_15m` | 15M | **7,767,680** | 49,152 | 512 | General text; stopped at step 10 of 5,000 |
| `pilot_300m` | 300M | **113,145,408** | 49,152 | 2048 | Config only; never trained |
| `prod_1p5b` | 1.5B | **1,315,014,656** | 49,152 | 8192 | Config only; never trained |
| `scale_7b` | 7B | **6,073,618,432** | 49,152 | 8192 | Config only; never trained |

Counts are computed from the config dicts and cross-checked against the `state_dict`
of the corresponding checkpoint (which double-counts the tied LM head).

## OpenAI-Compatible Local API

`serving/openai_server.py` exposes the minimum OpenAI surface needed by an OpenAI SDK
client, so a separate agent project can use this model as its intelligence provider.

```bash
# Loopback only by default. No API key required for local use.
python -m serving.openai_server \
  --ckpt runs/code_pretrain_tiny/checkpoint_last.pt \
  --tokenizer tokenizer/qm_code_bpe_v0.json \
  --port 8002
# -> http://127.0.0.1:8002
```

**Model identifier:** `qm-code-tiny` (read from `configs/model/code_tiny.yaml` via the
checkpoint's embedded config path). Context window **512** tokens.

```bash
curl http://127.0.0.1:8002/v1/models
curl -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qm-code-tiny",
       "messages":[{"role":"system","content":"You are terse."},
                   {"role":"user","content":"Write a Python function that adds two numbers."}],
       "max_tokens":64,"temperature":0.2,"stream":false}'
```

### Endpoints

| Route | Notes |
|---|---|
| `GET /v1/models` | OpenAI model list. `id` is the real config name; a `quantsmind` block adds checkpoint step, context window, vocab and param count. The block name and `qm-*` ids are compatibility/schema identifiers and are intentionally unchanged |
| `POST /v1/chat/completions` | Non-streaming. Returns `id`, `object`, `created`, `model`, `choices[].message.{role,content}`, `finish_reason`, and real `usage` token counts |
| `GET /health` | Load state, model id, context window |

### Supported vs rejected

Supported: `model`, `messages` (`system`/`user`/`assistant`), `max_tokens`,
`temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`,
`stream: false`, `user`.

**Rejected with a structured 400** rather than silently ignored: `stream: true`,
`tools`, `functions`, `tool_choice`, `n`, `logprobs`, `top_logprobs`, `logit_bias`,
`response_format`, `stream_options`. Ignoring these would let a caller believe a tool
loop or `n>1` sampling was running when it was not.

Errors use the OpenAI envelope `{"error":{"message","type","param","code"}}` and never
expose filesystem paths, credentials or tracebacks.

### Security posture

- **Binds `127.0.0.1` by default.** Remote binding requires an explicit `--allow-remote`.
- **No endpoint loads a checkpoint.** Weights load once at process start from
  `--ckpt`. This deliberately does *not* reproduce the `POST /load` +
  `torch.load(weights_only=False)` arbitrary-code-execution path that was removed from
  `completion_server` and `review_server` on 2026-10-14
  (see [docs/OPEN_ISSUES.md](docs/OPEN_ISSUES.md) A1).
- **Optional bearer auth:** set `QM_API_KEY` and send `Authorization: Bearer <key>`.
  Unset means keyless, which is only safe because the listener is loopback.
- CPU inference runs in a worker thread so the asyncio event loop is never blocked.

### Honest limitations

This is API-shaped plumbing over an experimental checkpoint. It does **not** make the
model capable:

- `qm-code-tiny` is **495,808 parameters**, trained **200 steps** (~1 epoch) on 100,244
  tokens of this repository's own source. Measured behaviour: predictive entropy
  5.32 nats against a uniform baseline of 8.67, so it has genuinely learned Python
  *unigram* statistics — but greedy decoding collapses into a repetition loop and
  `finish_reason` is always `length`. It is undertrained, not broken.
- **No KV cache.** Each generated token re-runs a full forward pass over the window, so
  latency grows with context. Measured: ~0.24 s for a 24-token completion.
- `<|system|>` / `<|user|>` / `<|assistant|>` are **not in the tokenizer vocabulary**
  (only `<pad> <bos> <eos> <unk>`), so the chat template is byte-level shredded into
  many tokens and was never trained as a control format.
- Prompts longer than the 512-token window are rejected with `context_length_exceeded`.
- Checkpoints vary in context and vocab. Verified-compatible pairs:

  | Checkpoint | Model id | Params | Context | Tokenizer |
  |---|---|---|---|---|
  | `runs/code_pretrain_tiny` | `qm-code-tiny` | 495,808 | 512 | `qm_code_bpe_v0.json` (5,822) |
  | `runs/code_sft` | `qm-code-tiny` | 495,808 | 512 | `qm_code_bpe_v0.json` (5,822) |
  | `runs/local_15m` | `qm-local-15m` | 7,767,680 | 512 | `qm_bpe_v0.json` (49,152) |

  `runs/code_pretrain_tiny` is the recommended default: pretrain-only, so it is not
  affected by the quarantined SFT corpus. `runs/code_sft_smoke` and all
  `runs/mm_*` checkpoints **cannot** be served (`code_sft_smoke` has a 5,817-row
  embedding against a 5,822 config; `mm_*` are multimodal state dicts).

## Serving

All three API servers load a real checkpoint and run real sampling. **None of them serves a
useful model today** — see the limitations below and
[docs/OPEN_ISSUES.md](docs/OPEN_ISSUES.md).

Shared startup flags (`serving/server_security.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | **required** | Checkpoint path. Loaded once at startup; no request can change it |
| `--tokenizer` | `tokenizer/qm_code_bpe_v0.json` | Must not exceed the model vocab |
| `--host` | `127.0.0.1` | Non-loopback raises `SystemExit` unless `--allow-remote` is passed |
| `--port` | per server | |
| `--allow-remote` | off | Explicit opt-in to binding a routable interface |
| `QM_API_KEY` | unset | When set, `Authorization: Bearer <key>` is required on every request |

> **Security change (2026-10-14).** `POST /load` was **removed** from
> `completion_server` and `review_server`. It accepted a `ckpt` query parameter and ran
> `torch.load(weights_only=False)` on it with no authentication while binding `0.0.0.0` —
> unauthenticated remote code execution. Pass `--ckpt` on the command line instead.
> `torch_load_checkpoint` also attempts `weights_only=True` before falling back.
> Covered by `tests/test_legacy_serving_security.py`.

### Completion API (IDE integration)
```bash
python -m serving.completion_server \
  --ckpt runs/code_pretrain_tiny/checkpoint_last.pt --port 8000
# POST /complete {prefix, suffix, language, max_tokens, temperature, top_k}
# GET  /health
```
- Inline code suggestions
- FIM plumbing present but untrained — `<|fim_prefix|>` is not in the tokenizer vocabulary
- `top_k` is interpreted as a *count of completions*, not a sampling cutoff (known issue A6)

### Review API (CI/CD integration)
```bash
python -m serving.review_server \
  --ckpt runs/code_pretrain_tiny/checkpoint_last.pt --port 8001
# POST /review {code, language, context}
# GET  /health
```
- Regex lint heuristics plus optional model commentary
- The static findings are keyword matching; the summary text is a fixed template

### Code Chat UI
```bash
python -m serving.code_chat --ckpt runs/code_pretrain_tiny/checkpoint_last.pt --port 7860
```
- Conversational UI, Python + JS specialist
- Note: the Gradio UIs still launch with `server_name="0.0.0.0"` — use `--host localhost`
  or firewall the port if the machine is on a shared network.

## Evaluation status

| Metric | State |
|---|---|
| Held-out code perplexity | **Measured, with a caveat.** At the checkpoint's real context (`seq_len=512`, the value `code_tiny.yaml` declares) the split *is* scorable: **nll 4.6942, ppl 109.31** over 1 window of the 1,003-token tail. At `seq_len=2048` — the value `code_local.yaml` advertises — it is genuinely unmeasurable (1,003 < 2,049). **This is not a capability number:** the split is drawn from the same 100,244-token corpus the model was trained on, and it is a single window, so it carries very high variance. |
| HumanEval | **unmeasured.** `HUMANEVAL_PROBLEMS` is a 5-problem smoke subset of 164. Reports stamp `status: "unmeasured"`, `promotion_eligible: false`. |
| MBPP | **unmeasured.** `MBPP_PROBLEMS` is a 5-problem smoke subset of 500. Same stamping. |
| Promotion gate | `evaluation/promotion.yaml` is a declaration only; no code enforces it. |
| Generated-code syntax validity | **0 of 4** smoke prompts produced syntactically valid Python. Recorded in `docs/OPEN_ISSUES.md`; not a benchmark, a sanity signal. |

Two previously committed reports recorded a **false perfect perplexity of 1.0** produced
by an empty held-out split. Both JSON files are preserved unchanged and marked retracted
in `runs/code_pretrain/eval_report.RETRACTED.md` and
`runs/code_smoke/eval_report.RETRACTED.md`.

## Corporate Fine-Tuning

```bash
# Ingest internal codebase
python -m data.ingest_internal --repo /path/to/internal/repo \
  --repo-name myproject --out data/internal_corpus_v0.jsonl --scan-secrets

# Fine-tune on internal code
# NOTE: --tokens is required. configs/train/corporate_ft.yaml declares
# dataset_tag "corporate-internal-v0", which resolves to
# data/corporate_internal_v0.tokens.bin — a file that does not exist. The shipped
# internal corpus is data/internal_corpus_v0.tokens.bin.
python -m training.pretrain --config configs/train/corporate_ft.yaml \
  --out-dir runs/corporate_ft --tokens data/internal_corpus_v0.tokens.bin
```

The committed `data/internal_corpus_v0.tokens.bin` is **2,126 tokens from 5 synthetic
demo files** in `data/internal_demo/`. The shipped `runs/corporate_*` checkpoints are
495,808-param `code_tiny` models; the "1–2B tokens" figure in
`docs/CODE_LLM_STRATEGY.md` is a target, not an executed run.

## Data Sources

| Source | License | Languages | Actually ingested? |
|---|---|---|---|
| GitHub public repos | MIT, Apache-2.0, BSD | Python, JS, TS | **No** — zero GitHub files in `data/code_corpus_v0.jsonl` |
| Python stdlib | PSF | Python | No |
| npm packages | MIT, ISC | JS, TS | No |
| This repository's own source | Project Apache-2.0 | Python, JS | **Yes** — all 77 chunks, `source: local:*` |
| Internal repos | Corporate | All | 5 synthetic demo files only |

No model-weights licence and no training-data licence exist. `LICENSE` covers source
code only.

## Docs

- [CODE_LLM_STRATEGY](docs/CODE_LLM_STRATEGY.md) — corporate coding LLM strategy
- [ROADMAP](docs/ROADMAP.md) — development roadmap
- [STATUS](docs/STATUS.md) — execution status tracker
- [DESIGN_ARCHITECTURE](docs/DESIGN_ARCHITECTURE.md) — full design doc

## License

Taksh source code is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

Model weights and training datasets are not part of this repository release and
carry no licence. Third-party dependencies remain under their own licences.
