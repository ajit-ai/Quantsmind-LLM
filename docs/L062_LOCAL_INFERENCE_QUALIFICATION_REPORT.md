# TAKSH L0.6.2 — Local Inference Qualification Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `1fef89d179f498ee2ce1205b193a13c6dbd1970c` (L0.6.1)
**Milestone type:** Inference engineering qualification only — **NOT a capability claim, NOT a training authorization.**
**Package version:** remains `1.0.0` (`taksh==1.0.0`) — not bumped, not a release.

---

## 1. Verdict

**CONDITIONAL PASS — inference infrastructure qualified; full/existing-checkpoint inference blocked by host resources.**

- Local inference infrastructure (loader, tokenizer adapter, context manager, sampling/generation
  controllers, streaming, cancellation, timeouts, API, observability) is **implemented and tested**
  on synthetic tiny models on CPU — 70/70 focused tests pass.
- The designated existing checkpoint (`runs/v180_sft/checkpoint_last.pt`) **could not be loaded on
  this host**: available memory collapsed below the requirement during the run (see §6). The
  load-time guard correctly **failed closed (BLOCKED)**. No load was forced.
- The canonical `taksh-foundation-111m` (110,658,816 params) **has no weights artifact anywhere in
  this repository** and cannot be executed on any host; nothing here claims otherwise.
- Frozen Python benchmark **not re-run**; baseline **0/30** stands. Zero capability is claimed.
- L0.6.0/R1/L0.6.1 suites remain green (61 + 48 tests verified), preservation anchors byte-identical.

Infrastructure qualification ≠ coding-capability qualification. No TAKSH language is capable.

---

## 2. Architecture and component responsibilities

Single coherent engine (`serving/inference_engine.py`); logical components, no microservices:

| # | Component | Implementation |
|---|---|---|
| 1 | Model Loader | `ModelLoader` — explicit state machine `NOT_LOADED → PREFLIGHT → LOADING → READY | BLOCKED | FAILED | UNLOADED`; safe `weights_only=True` load via `serving.local_inference.torch_load_checkpoint`; config resolution via `resolve_model_config`; state-dict validation via `training.checkpoint.load_model_weights` (missing/unexpected/mismatch are fatal); architecture fingerprint (SHA-256 of canonical JSON of the resolved model cfg, pinned for the canonical config at `92da3497…`); checkpoint/tokenizer SHA-256 fingerprints; memory preflight; clean unload. READY is set only after real load + validation succeed. |
| 2 | Tokenizer Adapter | `TokenizerAdapter` — file fingerprint, vocab-parity check vs model config; load failure is a `CheckpointIncompatibleError`. |
| 3 | Inference Engine | `ModelLoader.generate()` / `.stream()` — cooperative decode loop (no-cache; identical math to the existing no-cache path, O(n²) over the context window; acceptable for qualification scale). |
| 4 | Generation Controller | `InferenceRequest` + parameter validation: max-new-tokens ≤ 256, non-empty prompt ≤ 200k chars ≤ context, stop sequences (≤8, ≤128 chars each, non-empty), EOS handling, prompt+completion accounting; `prompt_tokens + generated_tokens ≤ context_limit` **enforced**; explicit truncation policy (`reject` default, `left` opt-in) — never silent. |
| 5 | Context Manager | `ContextManager.plan()` — tokenizer-aware accounting, utilization reporting, `ContextOverflowError` on violation; `max_new_tokens` may never consume the whole window. |
| 6 | Sampling Controller | `SamplingParams` + `SamplingController` — temperature/top-k/top-p/presence & frequency penalties, or greedy at temperature=0; **dedicated per-request `torch.Generator`** (concurrent requests never share random state). `repetition_penalty` is **not** supported by the architecture and is rejected; only the architecture-backed `presence_penalty`/`frequency_penalty` exist. |
| 7 | Streaming Interface | `ModelLoader.stream()` — per-token chunks with prefix-consistent decoded deltas; final terminal chunk carries the full `InferenceResponse`; SSE adapter in `serving/local_api.py`. |
| 8 | Cancellation Controller | `CancellationController` — `threading.Event` per in-flight request; `cancel(request_id)` → cooperative stop, `finish_reason="cancelled"`, partial output returned, engine remains READY; `cleanup()` always in `finally`. Deadline check per step → `finish_reason="timeout"`. |
| 9 | Resource Guard | Preflight (device allowlist, memory measured via psutil, checkpoint-size + scaled headroom = max(32 MiB, estimate/4)); fp32 upper-bound re-check at load time before any allocation; unload via `gc.collect()`; no network/GPU/payment paths anywhere; CUDA requires the L0.6.0 authorization phrase. |
| 10 | Inference Evaluation Adapter | `InferenceTrace.to_eval_record()` — task/language/sampling/token record for the frozen evaluation harness; `capability_claim` is always `null`; scoring only in the frozen harness, never here. |
| 11 | Provider/API Adapter | `serving/local_api.py` — FastAPI app factory (engine injected; app creation never loads weights) implementing an OpenAI-*subset*: `GET /health`, `GET /v1/resources`, `GET /v1/models`, `POST /v1/chat/completions` (stream/non-stream), `POST /v1/generate`, `POST /v1/cancel`. Unsupported params (reuses the `UNSUPPORTED_PARAMS` registry from `serving/openai_server.py`) → structured 400; validation errors → 422 in the same envelope; not-READY → 503 `resource_blocked`. External providers: **disabled by construction** (`provider: "local"` everywhere, no opt-in knobs implemented). |
| 12 | Observability & Diagnostics | `evaluation/l062_observability.py` + per-tool reports; every number labeled measured vs estimated; `host_resource_state` explicit. |

Existing modules reused, not duplicated: `serving/local_inference.py` (safe checkpoint load, config
resolution, sampling helpers), `training/checkpoint.py` (state-dict report), `evaluation/resource_guard.py`,
`serving/openai_server.py` (UNSUPPORTED_PARAMS registry), `evaluation/l060_inference_qualification.py`
(synthetic artifact builder).

---

## 3. Artifact reality check (measured)

Direct read-only inspection of `runs/v180_sft/checkpoint_last.pt` (weights not instantiated for this probe):

- `checkpoint["config"]["run"]` → `base_model_config = configs/model/code_v160_15m.yaml`,
  `name = qm-code-v180-sft`.
- Resolved model config: **vocab 5822, d_model 384, 12 layers, 8 heads, 4 KV heads, FFN 1536,
  context 2048, RoPE θ=10000, tied embeddings** — i.e. the **v160 15M-lineage chat model**, **not** the canonical 111M foundation.
- fp32 weight estimate: **115,149,312 B (28,787,328 params)**; file is 345,596,421 B ≈ **12 B/param** (fp32 weights + Adam m/v state).
- SHA-256: `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` (**unchanged**, §13).
- **No 110,658,816-param `taksh-foundation-111m` artifact exists anywhere in `runs/`;** the
  canonical foundation model has never been trained or saved. Canonical-inference claims are impossible today.

This does not change the milestone: the designated existing checkpoint is what was to be attempted
under preflight, and it failed closed on memory.

---

## 4. Run history and host-memory timeline (all measured)

| Time (UTC) | Event | Available B | Result |
|---|---|---|---|
| 2026-11-11T04:32Z | milestone start baseline | 156,569,600 (96.3% used) | preflight baseline recorded |
| 2026-11-11T04:55Z | first load-decision during tool run | 541,741,056 (87.1%) | PROCEED at decision (file 345,596,421 + headroom 86,399,105) |
| 2026-11-11T04:55Z | load-time re-check | **146,243,584** | **BLOCKED** — resolved fp32 upper bound 115,149,312 + headroom 33,554,432 > available; weights never instantiated |
| (subsequent) | observability re-runs | ~143–150 MB | BLOCKED at decision; attempt refused pre-allocation |
| 2026-11-11T04:5xZ | final report run | 143,458,304 (87.1→ high pressure) | BLOCKED outright; load refused |

The host loses hundreds of MB within seconds (other processes); free headroom is volatile. The
engine's two-stage guard (file-size decision + fp32 upper-bound check at load time) is what
converted that volatility into a safe refusal instead of a host swap-thrash/OOM. No quantization,
swap tricks, or offload were applied — any such strategy needs its own qualification first.

**Actual full/existing-checkpoint load outcome: BLOCKED (not attempted to allocation; guard held).**

---

## 5. CPU inference results (synthetic tiny artifact)

Measured on an in-tmp synthetic artifact (SMOKE_CFG: vocab 47, d_model 32, 1 layer, 4 heads, 2 KV,
FFN 64, context 24):

- State machine observed: `NOT_LOADED → PREFLIGHT → (BLOCKED on memory-unverified path) → LOADING → READY`; `unload()` → `UNLOADED`.
- `load()` without `allow_full_model=True` refuses (BLOCKED) by design.
- Tiny model **loaded successfully: READY**, 10,816 params; load fingerprints: checkpoint SHA-256 (64 hex), model-config fingerprint (64 hex), tokenizer SHA-256 (64 hex) — all recorded and verified in tests.
- Generation: greedy/seeded deterministic (`same seed ⇒ identical token_ids`, verified across two loaders and across runs).
- Streaming: per-token chunks; concatenated deltas == final `InferenceResponse.text` (verified).
- Cancellation: mid-stream cancel → `finish_reason="cancelled"`; engine remains READY and serves the next request normally.
- Unknown-request cancel → `False`; inflight registry cleaned in `finally`.
- Timeout: `timeout_seconds=1e-6` → `finish_reason="timeout"`, 0 tokens, engine READY.
- Stop sequences: greedy completion re-generated with its first word as stop string → `finish_reason="stop"`, stop string excluded.
- Successful API flows qualified via FastAPI `TestClient` (health, models, chat completion, SSE stream, native generate, cancel, resource-blocked 503).

---

## 6. Resource measurements (labeled)

| Quantity | Value | Type |
|---|---|---|
| Host total memory | 4,199,780,352 B (4.2 GB) | measured (psutil) |
| Available (final run) | 143,458,304 B (~137 MiB) | measured |
| Designated checkpoint file | 345,596,421 B | measured |
| fp32 weight estimate (v160-15m) | 115,149,312 B | **estimated** from config (validated formula reproduces 110,658,816 exactly for the canonical config) |
| Canonical foundation fp32 estimate | 442,635,264 B (= 110,658,816 × 4) | estimated; formula verified exact |
| Preflight headroom | max(32 MiB, estimate/4); 86,399,105 B for designated ckpt | policy |
| Tiny artifact working set | trivial (< 1 MB weights) | measured-ish (tiny) |
| GPU allocation | none | none |
| Network/external APIs | none (no invocation path exists) | none |
| Throughput/latency | **not published** — designated artifact never loaded on this host; tiny-model timings are not representative and are omitted deliberately | — |

No throughput, latency, or training-duration claims are made anywhere in this milestone.

---

## 7. Generation controls (enforced, with rejection policy)

`SamplingParams.validate()` rejects: `temperature ∉ [0, 2]` (and bools), `top_k < 0`/non-int (0 disables),
`top_p ∉ (0, 1]`, `presence_penalty`/`frequency_penalty ∉ [-2, 2]`, non-int/bool `seed`,
`max_new_tokens ∉ [1, 256]` (bool rejected), `max_new_tokens ≥ context_limit`.
`GenerationController` rejects: empty/whitespace prompts, prompts > 200k chars, > 8 stop sequences,
empty/overlong stop strings, `timeout_seconds ≤ 0`, unknown `mode`.
Unsupported sampling concepts (**repetition_penalty**, logit_bias, n>1, logprobs, tools/function
calling, response_format, stream_options) are **not silently accepted**: API layer rejects via the
same `UNSUPPORTED_PARAMS` registry as the existing server (structured 400) — never ignored.
`prompt_tokens + completion_tokens ≤ 2048` (canonical) enforced in `ContextManager`; the canonical
context is untouched; overflow is rejected or explicitly left-truncated (`truncation_policy="left"`)
with the fact recorded in the response/trace (`truncated=true`).

---

## 8. Determinism and reproducibility

- Deterministic mode: `temperature=0` → argmax; bitwise-identical for same model+prompt+params on this host/backend.
- Seeded sampling: dedicated per-request `torch.Generator(device="cpu")` — no global RNG sharing; replay of (model, tokenizer, prompt, params incl. seed) reproduces identical token ids (tested; incl. 4-thread interleaved contention test).
- Cross-hardware/cross-backend bitwise identity is **not claimed** — different kernels, thread counts or math libraries may diverge.
- `InferenceTrace` + `SamplingParams.recorded()` capture the full replay payload (prompt ids, seed, temperature, top-k/p, penalties, finish reason, token counts).
- Every request record includes `capability_claim: null` — replay ≠ scoring.

---

## 9. Streaming and cancellation

- `stream()` yields `InferenceChunk`: `index`, `token_id`, decoded **delta** (prefix-consistent), request id; terminal chunk sets `final=true` and carries `InferenceResponse`.
- Cancellation: `cancel(request_id)` sets an `Event`; the loop observes it before the next step → stops, `finish_reason="cancelled"`, partial tokens preserved, `cleanup()` in `finally`, engine remains READY.
- Timeout: monotonic deadline per request; same stop semantics with `finish_reason="timeout"`.
- No corrupted state: cancellation between steps cannot touch weights; engine passes `test_streaming_cancel_mid_generation` and subsequent `generate()` works.
- API: `POST /v1/cancel {request_id}` returns `{cancelled: bool}`; SSE stream emits `data: [DONE]` terminator; `X-Request-Id` on streamed chat completions.
- Concurrency isolation: per-request generator + per-request cancellation events; 4-thread contention test passes with identical per-seed outputs.

---

## 10. Context management

`ContextManager(context_limit=max_seq_len)`; canonical 2048 preserved (from resolved model cfg — for
synthetic smoke artifacts, 24). `plan()` returns accounting (prompt_tokens, remaining budgets,
utilization, truncated flag). Long prompts: **rejected** by default (ContextOverflowError → 400
`context_overflow` at API; UsageError otherwise); `truncation_policy="left"` keeps the newest
`limit − max_new_tokens` tokens and reports `truncated=true` — the policy is explicit, logged and
surfaced; there is no silent truncation anywhere on this path.

---

## 11. API qualification (subset only — no full-OpenAI claim)

`create_app(engine)` never loads at import or app-creation time. Verified via FastAPI `TestClient`
(httpx 0.28.1, fastapi 0.141.1). Loopback posture preserved by convention (no `--allow-remote`
here); no auth claims beyond the existing server conventions (optional bearer `QM_API_KEY` is not
re-implemented here; unauthenticated loopback remains the development posture).

- `GET /health` → `{status, provider:"local", model_loaded, load_state, external_providers:"disabled"}`
- `GET /v1/resources` → measured memory snapshot + inflight ids; labels measured vs estimate
- `GET /v1/models` → empty list + `unavailable_reason` when not loaded; single entry with quantsmind fingerprints when READY
- `POST /v1/chat/completions` → subset (messages, max_tokens, temperature, top_p, top_k, presence/frequency penalties, seed, stop, stream); `stream=true` → SSE
- `POST /v1/generate` → native structured response incl. trace record
- `POST /v1/cancel` → cooperative cancel
- Errors: OpenAI-shaped `{"error": {message, type, param, code}}` — 400 configuration, 400 context_overflow, 422 validation (custom handler), 503 resource_blocked, never a traceback.
- Unsupported params → 400 (registry reused from `serving/openai_server.py`).
- Local/external distinction: `provider: "local"` in every response; no external provider path exists; capability always `null`.
- **This is not OpenAI API conformance** — only the listed subset is exercised and qualified.

---

## 12. DLCAE, multilingual and Karkain integration

- **Multilingual**: inference accepts any registered language; `language_id` recorded in trace; metadata exposes registry status; nothing about registration implies capability. 53 languages + Karkain (54th) preserved byte-identical; registry untouched.
- **DLCAE ↔ inference** (`evaluation/l062_dlcae_inference_interface.py`): `identify_language` (declared/filename/unknown), `registry_lookup` (UNREGISTERED for `not_a_language_x`; karkain → DISCOVERED via extension registry), `comprehension_state_lookup` (NOT_STARTED for REGISTERED python), `NullKnowledgeSource` (honest empty: NO_KNOWLEDGE_BASE), `build_prompt` (diagnostics), `record_inference_feedback` (never scored; capability_verdict null), `propose_registry_update` (always `PENDING_OWNER_REVIEW`, `written: false`; inference never writes the registry).
- **Karkain** (`evaluation/l062_karkain_inference.py`): `.kark`/hint/content-marker identification; spec pinned **1.1.0**, compiler **1.1.0 (windows/amd64 Stable Build)**, KIR v1 SSA; TAKSH-KARK-BENCH v1.0.0 (0 tasks); `model_only` vs `compiler_assisted` modes kept strictly separate; `compiler_assisted` without evidence returns `evidence: "none"` with the host blocker note — **never fabricated**; task classification heuristic labeled as such (confidence 0.2–1.0). **Karkain remains DISCOVERED / NOT ELIGIBLE.**
- KCC host blocker **unchanged and unbypassed**: `kcc.exe` → `0xC0000135`; compile-pipeline commands → exit 3, zero diagnostics; freshly built reference compiler behaves identically (recorded in `configs/foundation/l0601_karkain_learning_contract.yaml`).

---

## 13. Preservation evidence (before and after)

| Artifact | Method | Result |
|---|---|---|
| `evaluation/l02_python_bench_v1.json` | SHA-256 (working tree) | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| `runs/v180_sft/checkpoint_last.pt` | streaming SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged; never modified/loaded-to-allocation |
| Tokenizer committed blob (`git cat-file blob HEAD:tokenizer/taksh_code_bpe_v1.json`) | SHA-256 of committed LF artifact | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| Tokenizer working-tree (CRLF) | SHA-256 | `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` — documented CRLF variant |
| `configs/foundation/l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| Multilingual benchmark (64 tasks), 53-language registry, 54th-language Karkain entry, L0–L0.6.1 reports & tests | untouched | green via L0.6.1/R1 suites |
| `taksh-hf-publication/` | only pre-existing user CRLF modification in `upload_manifest.json` | preserved, not staged |
| L0.6.1 milestone files | untouched by this milestone | preserved |
| `develop`, `main` | not touched (this branch only; no merges/tags/releases) | unchanged |

Committed-blob vs working-tree CRLF hashes are reported separately and never mixed (same method as L0.6.0/R1/L0.6.1).

---

## 14. Focused test commands and results

```
python -m pytest tests/test_l062_inference.py -q
  70 passed (18 requirement areas: loading, config compat, tokenizer, context,
  generation params, determinism, sampling, streaming, cancellation, timeout,
  concurrency isolation, resource preflight, API request/response, structured
  errors, multilingual metadata, DLCAE integration, Karkain contract,
  observability/replay)  · plus preservation-anchor tests
python -m pytest tests/test_l0601_pipeline.py tests/test_l060_foundation.py \
  tests/test_l060r1_architecture_alignment.py tests/test_m6_resources.py -q
  109 passed
python -m pytest (L0.6.2 ∪ L0.6.1 ∪ L0.6.0 ∪ R1 ∪ m6) -q
  179 passed
python -m evaluation.l062_inference_qualification   # exit 0; designated checkpoint BLOCKED
python -m evaluation.l062_dlcae_inference_interface  # exit 0
python -m evaluation.l062_karkain_inference          # exit 0
python -m evaluation.l062_observability               # exit 0
```
Interpreter: `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe` (3.14.7, CPU-only).

---

## 15. New file inventory (all L0.6.2-owned, all new)

| Path | Purpose |
|---|---|
| `serving/inference_engine.py` | core engine (all logical components) |
| `serving/local_api.py` | local OpenAI-subset FastAPI adapter |
| `evaluation/l062_inference_qualification.py` | synthetic probes + designated-checkpoint decision |
| `evaluation/l062_dlcae_inference_interface.py` | DLCAE↔inference interfaces |
| `evaluation/l062_karkain_inference.py` | Karkain inference contract |
| `evaluation/l062_observability.py` | observability aggregate |
| `tests/test_l062_inference.py` | 70 focused tests |
| `reports/l062_inference_qualification.json` | probe + decision evidence |
| `reports/l062_dlcae_inference_interface.json` | DLCAE probe evidence |
| `reports/l062_karkain_inference.json` | Karkain contract evidence |
| `reports/l062_observability.json` | aggregate evidence |
| `docs/L062_LOCAL_INFERENCE_QUALIFICATION_REPORT.md` | this report |

No pre-existing file was modified. Pre-existing user CRLF modifications (serving/code_chat.py,
serving/local_inference.py, serving/openai_server.py is **not** modified — untracked or modified
state unchanged, training/manifest.py, training/pretrain.py, training/sft.py,
taksh-hf-publication/upload_manifest.json) are preserved, unstaged.

---

## 16. Remaining blockers and owner decisions

1. **Host memory** — the host literally cannot hold the designated artifact this session
   (available ≈ 137–156 MB at attempt time vs ~432 MB required). Owner decision required:
   bigger host, memory reservation, or (only after its own qualification) an offload/quantization
   strategy. Re-running preflight is mandatory before any future load attempt.
2. **Canonical foundation-111m artifact does not exist** — inference of the canonical model is
   impossible until (a) training is explicitly authorized, and (b) a checkpoint exists. L0.6.2
   changes nothing about training authorization.
3. **KCC toolchain** (`0xC0000135`) — Karkain inference/evaluation cannot proceed until the
   compiler starts on some host; owner-owned Karkain repository is untouched.
4. **KARK-BENCH** — 0 tasks instantiated; no qualification bar set; no Karkain capability claim possible.
5. **Frozen benchmark** — 0/30 stands; any future evaluation must use it unchanged.
6. **External providers / public serving** — intentionally absent; enabling any external provider
   requires explicit owner authorization and a separate security qualification.
7. **Streaming via the full existing `serving/openai_server.py`** — its SSE/streaming remains a
   follow-up outside this milestone; the L0.6.2 adapter qualifies the subset instead.

---

## 17. Attestation

No training, fine-tuning, data acquisition, GPU allocation, GPU spend, paid API use, external
model download, Hugging Face/PyPI publication, tag, release, merge to `develop`/`main`, Karkain
repository modification, or destructive cleanup occurred in this milestone.
`develop` and `main` were not touched. The canonical foundation configuration, the 53-language
catalogue, Karkain's first-class status, and the L0–L3 roadmap are preserved unchanged.
**TAKSH is not a proven coding intelligence; 0/30 frozen-benchmark ceiling stands; no capability is claimed.**

L0.6.2 is stopped here. L0.7.0 is not started.
