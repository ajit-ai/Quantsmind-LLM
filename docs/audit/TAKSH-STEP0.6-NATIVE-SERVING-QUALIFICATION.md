# TAKSH I5 — Native Serving Qualification (v180_sft)

**Date:** 2026-11-19
**Branch:** `feature/coding-genius-i5-serving-qualification`
**Status:** `IMPLEMENTED_NOT_QUALIFIED` as a *product* — serving qualification only, no capability, no release.

---

## 1. Objective

Qualify the native serving surface for the existing experiment checkpoint (`runs/v180_sft/checkpoint_last.pt`): safe checkpoint loading, architecture/tokenizer identity, template validation, context budget limits, deterministic bounded generation, honest failure modes, and provider integration with the Gradio UI (I6).

## 2. Benchmarks and pools excluded

- `evaluation/native_v01` (immutable 20-task native benchmark) — **not modified**.
- Frozen Python benchmark — **not modified** (0/30 admitted).
- v180_sft checkpoint, tokenizer v1 — **never modified**.
- No training, fine-tuning, acquisition, publication, merge, tag, or release performed.

## 3. Implemented / verified

| # | Finding | Evidence |
|---|---|---|
| Q1 | Checkpoint loader refuses malformed/truncated files; `weights_only=True`; narrowly-scoped NumPy retry only (`serving.local_inference.torch_load_checkpoint`) | tests/test_i5_native_serving.py::test_truncated_checkpoint_fails_closed |
| Q2 | Tokenizer contract enforced (`<bos>` and `<eos>` single-token registration); model rejects incompatible tokenizers (vocab-size check plus chat-template validation) | tests/test_i5_native_serving.py::test_bad_tokenizer_file_fails_closed; `validate_chat_template` gate |
| Q3 | Checkpoint and tokenizer SHA-256 recorded at load; meta exposes hashes, model name, total parameter count, template markers | test_i5_native_serving.py::test_load_foundation_returns_identity_hashes |
| Q4 | Deterministic greedy generation confirmed (seed held constant → identical token ids and text) | tests/test_i5_native_serving.py::test_generate_bounded_deterministic |
| Q5 | Prompt/context budget enforced; overflow raises `ConfigurationError`; `max_new_tokens` bounds hard | tests/test_i5_native_serving.py::test_generate_bounded_rejects_* |
| Q6 | Provider identity/capabilities surface in the Gradio UI (I6, PR #6) with no silent fallback | tests/test_code_chat_provider.py |
| Q7 | Errors are structured (`InferenceError` subclasses) — never bare tracebacks | full suite green |

## 4. Measured on the dev machine (CPU, Python 3.14, task constrained by RAM)

| Metric | Value |
|---|---|
| Checkpoint sha256 | `9B54103695054ECB30D8521FFCEB62F443D2D1FD8D064993EDC2D5070CB7F666` |
| Tokenizer sha256 | `554C8E7572D58923A0766F114B5F419B14E68BDFF3A738803B30728EC3FFCD92` |
| Params / tensors | 28,787,328 / 111 |
| Model identifier | `qm-code-v160-15m`, context 2,048 |
| Load time (cold, CPU) | ~42.1 s |
| Generation (5–10-token greedy) | ~2.9 s |
| Decoding output (prompt "Write a Python function named add…") | `()` — 2 tokens, degenerate as expected |

These numbers are the host's; they are **not** a throughput/latency claim for any other machine.

## 5. Known limitations (honest)

1. **No capability.** Native v0.1 0/20: the model emits fragments, not functions. This is a serving/integrity qualification, not a capability result.
2. **Streaming is not implemented.** The OpenAI-compatible path returns the full text; incremental SSE is not shipped in this milestone.
3. **FIM sentinels are absent** from the tokenizer vocabulary — FIM cannot be meaningfully trained until the tokenizer is extended (v2, owner decision).
4. **Parameter total doubles `model.num_params()` by embedding tie**, so `model.num_params()` reports 28,787,328; see the memory note for `total_parameters` (both equal for tied models).
5. **Peak memory is uninstrumented** — not claimed.
6. **Threads**: CPU run is single-device; concurrency/timeout are covered at the provider layer (`serving/providers.py::generate_with_timeout`) but not re-qualified here.
7. **Checkpoint licensing**: the native weights are experimental and carry no redistribution rights; nothing here authorizes distribution.

## 6. Verification gate

- `python -m pytest tests/test_i5_native_serving.py -q` → **10 passed**.
- `python -m pytest tests/test_native_v160_inference.py tests/test_architect_package.py tests/test_provider_architecture.py -q` → **32 passed**.
- Python 3.11 `compileall` on the changed module and tests → clean.
- `import serving.app` OK; `serving.code_chat` imports providers including `native-taksh` OK.

## 7. Explicit admissions

- This qualification confers no release, no capability claim, no geography of languages, no peer comparison.
- The frozen Python benchmark remains **0/30** and Native v0.1 remains **0/20**. TAKSH remains **NOT_EVALUATED** for coding capability.
- The owner retains sole authority to release, publish or merge; no gate here upgrades those rights.
