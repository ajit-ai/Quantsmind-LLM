# TAKSH Distribution and Publication Plan

**Status:** PLAN ONLY. No upload, tag, release, listing, model card or dataset
card has been created. No Hugging Face, PyPI, OpenRouter, Cline or OpenCode
service has been contacted.

**Branch:** `feature/taksh-five-version-ga-program`
**Evidence:** `reports/taksh_publication_readiness.json`
**Generator:** `evaluation/publication_readiness.py`
**Date:** 2026-11-12
**Authority:** A1.1-C (preparation only) · A2-E (no compute) · no publication,
merge or release authorization for this branch.

---

## 1. Direct answers

| Question | Answer |
|---|---|
| Can Taksh be published on Hugging Face as a model today? | **No — BLOCKED.** There is no canonical `taksh-foundation-111m` checkpoint. The only legacy checkpoint, `v180_sft`, is `GEOMETRY_INCOMPATIBLE` and is never substituted. |
| Can the dataset be published on Hugging Face? | **No — BLOCKED.** Licence, provenance, PII/secret and benchmark-contamination review are incomplete. The corpus is research-only. |
| Can the Python package go to PyPI? | **Not yet — PARTIAL.** Package metadata, Apache-2.0 licensing and release tooling exist. Upload needs registry credentials and explicit owner authorization; the historical v1.0.0 upload was blocked on credentials. |
| Can Cline use Taksh? | **No — BLOCKED.** Cline needs an OpenAI-compatible endpoint and tool calling. Taksh's server rejects tool calls and has no canonical checkpoint. |
| Can OpenCode use Taksh? | **No — BLOCKED.** Same reason. A custom OpenAI-compatible provider can be configured, but the agentic coding loop cannot run without tool calling and a model. |
| Can Taksh be listed on OpenRouter? | **No — NOT STARTED.** OpenRouter lists models served by hosting providers. Taksh has no hosted endpoint, pricing/terms decision or capability evidence. |
| Can Taksh run locally as a chat-completions API? | **PARTIAL.** The FastAPI surface exists and fails closed, but `--ckpt` is required and no canonical checkpoint exists. Streaming and tool calling are rejected. |
| What is the actual coding capability? | **NOT EVALUATED.** Historical M7 result: 0/40 syntax-valid, 0/8 functional, evidence level 0. Frozen Python benchmark: 0/30 preserved, not re-scored. |

The short version: **Taksh currently qualifies as an engineering source
package, not as a usable coding model, model artifact, dataset or hosted
service.**

---

## 2. Channel matrix

Statuses use only: `PASS`, `PARTIAL`, `BLOCKED`, `NOT STARTED`, `NOT EVALUATED`.

| Channel | What it distributes | Status | Blocking item |
|---|---|---|---|
| GitHub source release | Source, tests, configs, docs | `PARTIAL` | v1.0.0 source release exists; a new release from this branch is not authorized or prepared. |
| PyPI | Wheel and source distribution | `PARTIAL` | Registry credentials, owner upload authorization and a fresh build for this branch. |
| Hugging Face model repository | Weights, config, tokenizer, model card | `BLOCKED` | No canonical checkpoint; no HF-native export; no capability evidence; no weight-licence decision. |
| Hugging Face dataset repository | Corpus and dataset card | `BLOCKED` | Licence/provenance, PII/secret, contamination and owner-release decisions. |
| Local OpenAI-compatible server | `/v1/models`, `/v1/chat/completions` | `PARTIAL` | Chat-only; `--ckpt` required; no canonical checkpoint; tool calls and streaming rejected. |
| Cline | Agentic VS Code client | `BLOCKED` | No canonical checkpoint and no tool-calling support. |
| OpenCode | Agentic terminal client | `BLOCKED` | No canonical checkpoint and no tool-calling support. |
| OpenRouter | Hosted model marketplace | `NOT STARTED` | No hosted endpoint, pricing/terms decision or qualified capability evidence. |

A `PASS` on any channel would still require an explicit owner decision. This
plan does not convert engineering readiness into publication authorization.

---

## 3. Recommended publication sequence

The order is deliberate: lowest-risk, least-claiming channels first; model and
commercial channels last.

### Stage 0 — Local developer preview (current state: PARTIAL)

**Goal:** let a developer run the package and inspect the API surface without
any publication.

- Install from source: `pip install .`
- Build from source: `python scripts/prepare_release.py`
- Run tests: `python -m pytest tests -q`
- Start the OpenAI-compatible server only when a checkpoint exists:
  `taksh-openai --ckpt <canonical-checkpoint> --host 127.0.0.1 --port 8002`

**Gate:** no canonical checkpoint, so native generation cannot be previewed.
The server is chat-only: streaming and tool calls are rejected with HTTP 400.

### Stage 1 — GitHub source release (current state: PARTIAL)

**Goal:** publish the source package, tests, configs and evidence. This is what
v1.0.0 already did; the release notes explicitly said "not a model release".

Prerequisites:

- owner authorization for a new tag/release;
- verified release commit on the intended branch;
- `scripts/prepare_release.py` evidence;
- no model-capability or production claim in the release notes.

**Do not** tag or release from this branch without a separate decision.

### Stage 2 — PyPI source package (current state: PARTIAL)

**Goal:** `pip install taksh` for the code-only package.

Prerequisites:

- registry credentials (historical upload blocked on them);
- owner upload authorization;
- fresh wheel/sdist build, `twine check`, isolated-install evidence;
- confirmation that the wheel contains no weights, data or checkpoints.

The package is Apache-2.0. Model weights and datasets would need their own
licence decision; they are not part of the package.

### Stage 3 — Hugging Face model repository (current state: BLOCKED)

**Goal:** host a canonical checkpoint with an honest model card.

Minimum artifacts:

- canonical `taksh-foundation-111m` checkpoint (110,658,816 parameters,
  vocab 5,822, context 2,048);
- `config.json` or the custom Taksh loading code;
- `tokenizer/qm_code_bpe_v0.json` and its SHA-256;
- model card covering architecture, data, training, evaluation, intended use
  and limitations;
- evaluation card including the historical negative results and the current
  `NOT EVALUATED` status;
- weight-licence decision;
- owner upload authorization.

**Important qualification detail:** Taksh's `QMModel` is a custom PyTorch
implementation, not a `transformers` architecture. Therefore:

- HF Hub as a **file host** is possible once weights exist;
- `transformers.from_pretrained(...)` is **NOT IMPLEMENTED**;
- a local `safetensors` exporter is **IMPLEMENTED but BLOCKED**:
  `evaluation/hf_export.py` is dry-run by default and refuses a missing or
  geometry-incompatible checkpoint. No canonical checkpoint exists, so no export
  has been written and no real weights have been exercised;
- the exported directory can be loaded with
  `evaluation.hf_export.load_export`, not `AutoModelForCausalLM`;
- HF Inference Endpoints / Inference Providers are **NOT READY** unless a
  supported architecture export or a custom container is built and tested.

A raw `.pt` checkpoint is not a usable Hugging Face model by itself. The
exporter plans these files only after a compatible checkpoint exists:

```
model.safetensors
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
README.md
SHA256SUMS
taksh_export_manifest.json
```

The manifest records the checkpoint and tokenizer hashes, the strict parameter
names/shapes, the model-card status and an explicit
`"publication": {"uploaded": false, "network_access": "none"}` block. Writing an
export is a local packaging action, not publication authorization.

The tool is dry-run by default:

```
python -m evaluation.hf_export --checkpoint <canonical-checkpoint> --out <dir>
python -m evaluation.hf_export --checkpoint <canonical-checkpoint> --out <dir> --execute
python -m evaluation.hf_export --verify <dir>
```

Because the canonical checkpoint is missing, every planning run currently
returns `BLOCKED` and writes nothing.

### Stage 4 — Hugging Face dataset repository (current state: BLOCKED)

**Goal:** publish a reproducible, legally clean corpus. This is optional and may
never be appropriate for a corporate/private corpus.

Prerequisites:

- per-source licence and provenance clearance;
- PII and secret review;
- published-benchmark decontamination evidence;
- dataset card with collection, filtering and known limitations;
- owner data-release decision.

The frozen research corpus is **research-only** and is not present on this ref.
It must not be described as a product training corpus.

### Stage 5 — Hosted API / OpenRouter (current state: NOT STARTED)

**Goal:** make Taksh available as a hosted model. This is a commercial and
operational decision, not a packaging step.

Prerequisites:

- a serving provider or owner-operated inference capacity;
- pricing, terms of service and data-handling policy;
- capability and safety evidence suitable for a public listing;
- model metadata: context window, tokenizer, licence, limits;
- owner authorization to submit or publish.

OpenRouter does not run a local checkpoint. A local-only Taksh cannot be listed.

---

## 4. Tool integration details

### 4.1 Local OpenAI-compatible server

Entry point: `taksh-openai = serving.openai_server:main`
Default bind: `127.0.0.1:8002`
Required argument: `--ckpt`
Tokenizer default: `tokenizer/qm_code_bpe_v0.json`

Implemented:

| Surface | Status |
|---|---|
| `GET /health` | Implemented |
| `GET /v1/models` | Implemented; returns an empty list with `unavailable_reason` when no model is loaded |
| `POST /v1/chat/completions` | Implemented, non-streaming only |
| Bearer auth | Optional via `QM_API_KEY`; loopback-only by default |
| Streaming | Rejected with HTTP 400 (`streaming_unsupported`) |
| Tool calling (`tools`, `functions`, `tool_choice`) | Rejected with HTTP 400 (`unsupported_parameter`) |
| `n > 1`, logprobs, logit bias, structured output | Rejected with HTTP 400 |

Because `--ckpt` is required, the server cannot start at all without a
checkpoint. There is no canonical checkpoint, so native inference remains
blocked.

### 4.2 Cline

Cline supports an "OpenAI Compatible" provider. The intended configuration is:

- Base URL: `http://127.0.0.1:8002/v1`
- API key: the same value as `QM_API_KEY` when auth is enabled; otherwise a
  placeholder is sufficient because auth is disabled.
- Model ID: the exact `id` returned by `GET /v1/models`. For the canonical
  config this is **`qm-foundation-111m`** (the `model.name` field), not the
  product/config id `taksh-foundation-111m`. Verify before configuring a client.

**Qualification: BLOCKED.** Cline's coding workflow is agentic and requires
tool calling. Taksh rejects tool calls with HTTP 400. Even if a checkpoint were
loaded, Cline could not edit files, run commands or complete its normal loop.
A chat-only mode is not a substitute for Cline's agentic workflow.

### 4.3 OpenCode

OpenCode V2 uses `providers` (plural). A custom OpenAI-compatible provider can be
declared in `opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "taksh/taksh-foundation-111m",
  "providers": {
    "taksh": {
      "name": "Taksh (local)",
      "env": ["TAKSH_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8002/v1"
      },
      "models": {
        "taksh-foundation-111m": {
          "name": "Taksh Foundation 111M",
          "modelID": "qm-foundation-111m",
          "limit": { "context": 2048, "output": 512 }
        }
      }
    }
  }
}
```

Set `TAKSH_API_KEY` to the same value as `QM_API_KEY` when auth is enabled. If
the server has no `QM_API_KEY`, omit `env` and use no credential. Verify the
model ID with `GET /v1/models`; the config key (`taksh-foundation-111m`) is the
OpenCode alias, while `modelID` is what is sent to the server.

**Qualification: BLOCKED.** OpenCode's coding loop also requires tool calling.
Taksh rejects `tools` and `tool_choice`, so OpenCode can only be wired for
non-agentic chat experiments, not for real coding work.

### 4.4 OpenRouter

OpenRouter is a marketplace for models served by hosting providers. It is not a
local runtime and does not host a checkpoint by itself. A Taksh listing would
require:

- a hosted endpoint or provider partnership;
- model metadata, pricing and terms;
- capability and safety evaluation suitable for public listing;
- owner authorization to submit.

**Qualification: NOT STARTED.** No submission has been made and none should be
made until the model and commercial decisions exist.

---

## 5. Model card requirements (when a checkpoint exists)

`evaluation/hf_export.py` contains the model-card template used by the local
exporter. It states the current capability status and preserves the historical
negative results; it does not invent a score. No card has been published because
no canonical checkpoint exists.

A future Hugging Face model card must state, without exaggeration:

1. **Identity:** Taksh, version, canonical checkpoint SHA-256, parameter count
   (110,658,816 for the 111M configuration), context window (2,048), tokenizer
   (`qm_code_bpe_v0.json`, vocab 5,822).
2. **Architecture:** RMSNorm, RoPE, GQA (12 heads / 6 KV heads), SwiGLU, tied
   embeddings; custom PyTorch implementation.
3. **Training data:** provenance, licence status, filtering, known gaps. Do not
   claim data that was not authorized or reviewed.
4. **Evaluation:** frozen benchmark protocol, raw outputs, historical 0/30,
   historical M7 0/40 and 0/8, and the current `NOT EVALUATED` status where
   applicable.
5. **Intended use and limitations:** no production, security, legal, medical or
   financial claims; no tool-calling claim; no agentic-coding claim unless
   measured.
6. **Licence:** code licence and weight licence are separate decisions.
7. **Safety:** prompt injection, code execution, secret leakage and sandboxing
   limitations.

No model card may imply that engineering tests are capability evidence.

---

## 6. Publication gates

Every channel must pass all applicable gates:

- [ ] canonical checkpoint exists and its SHA-256 is recorded;
- [ ] geometry/tokenizer compatibility check passes;
- [ ] capability evaluation exists, with raw outputs and a stated protocol;
- [ ] negative and inconclusive results are published, not hidden;
- [ ] code licence and weight licence are resolved separately;
- [ ] data licence, provenance, PII and contamination review are complete;
- [ ] security review covers prompt injection, code execution and secret leakage;
- [ ] tool-calling/streaming limitations are documented;
- [ ] owner signs the per-channel release decision;
- [ ] rollback/withdrawal plan exists;
- [ ] no tag, upload or listing happens before the decision.

Current gate state: **none of the model/dataset/agentic-client gates pass.**
Source-package gates are partially prepared.

---

## 7. Explicit non-actions in this slice

- No Hugging Face repository or model card was created. The local export tool and
  model-card template exist, but no export or card was written because the
  canonical checkpoint is missing.
- No dataset card or dataset upload was created.
- No PyPI upload or credential use occurred.
- No OpenRouter submission occurred.
- No Cline or OpenCode end-to-end test was run.
- No tag, GitHub release, merge to `develop`/`main`, or publication occurred.
- No training, optimizer update, checkpoint creation, dataset acquisition,
  GPU/cloud spend or paid API use occurred.
- `TAKSH_ENABLE_TRAINING_CI` remains unset.

---

## 8. Capability summary

See [`TAKSH-CAPABILITY-STATEMENT.md`](TAKSH-CAPABILITY-STATEMENT.md) for the
full statement. The essential facts:

- Model availability: **MODEL_UNAVAILABLE**
- Actual coding capability: **NOT EVALUATED**
- Historical M7: **0/40** syntax-valid, **0/8** functional, evidence level **0**
- Frozen Python benchmark: **0/30** preserved, not re-scored
- Karkain: **DISCOVERED**; KCC `0xC0000135`; KARK-BENCH 0 tasks
- Production GA: **NOT STARTED**

Taksh is an engineering platform with a preserved negative research result, not
a demonstrated coding model.
