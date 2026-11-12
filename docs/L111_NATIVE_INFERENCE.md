# TAKSH L1.1.1 — Native Inference Mechanics and Contract Qualification

**Baseline:** L1.1.0 @ `c7443f91c7537d8a46d67afaa0679a50f2504f53` (== pushed remote).
**Branch:** `feature/l111-native-inference-mechanics`.
**Authorization:** source-code preparation + non-training qualification only; A1.1-C / A2-E in force; **no training, no checkpoint creation, no data acquisition, no compute/GPU/cloud, no publication, no merge, no tag/release, no `TAKSH_ENABLE_TRAINING_CI`.**

---

## 1. Verdict

**Native inference mechanics — engineering-qualified.** The decode loop is exercised end-to-end on small deterministic in-memory models and on the native architecture (QMModel) without a production checkpoint. Every rejection path fails closed. **No capability claim is made.** The canonical 111M artifact remains MISSING; v180_sft remains GEOMETRY_INCOMPATIBLE; frozen Python benchmark 0/30; Karkain DISCOVERED (KCC 0xC0000135).

## 2. What changed

| Area | Change | Risk |
|---|---|---|
| `serving/generation_contract.py` | **new** — shared fail-closed generation-parameter validator (prompt, max_new_tokens, temperature bounds, top_k/top_p bounds, penalties, seed/type, stop sequences, NaN/Inf guards). | none (additive) |
| `training/tokenizer_fingerprint.py` | **new** — `file_fingerprint` (raw bytes), `vocabulary_digest` (canonical vocab JSON), `assert_special_token_ids` (pad=0, bos=1, eos=2, unk=3), `verify_against` | none (additive) |
| `serving/local_inference.py` | `generate_ids` and legacy `generate()` validate inputs via `generation_contract` (replacing silent permissive behavior) | interface-preserving |
| `evaluation/l111_native_inference.py` | **new** — runtime qualification runner → `reports/l111_native_inference.json` | read-only |
| `tests/test_l111_native_inference.py` | **new** — 47 tests, 1 skip-with-reason (feature-branch-only benchmark) | additive |
| `docs/L111_NATIVE_INFERENCE.md` | **new** — this file | docs |

Everything else untouched: pretrain loop, datasets, providers (TakshS2Provider contract already enforced), openai_server.py (API surface unchanged — only unloaded-state assertion exercised), binding_repair, Karkain adapter, all L0.x experimental artifacts.

## 3. Mechanics verified

| # | Behaviour | Evidence |
|---|---|---|
| 1 | Prompt prefill produces `(B, T, vocab_size)` logits; forward accepts pad-aware lengths | `test_prefill_shapes` |
| 2 | Greedy next token = argmax over final-position logits | `test_next_token_is_argmax_of_final_position_greedy` |
| 3 | Multi-token loop feeds generated tokens back | `test_multi_token_generation_feeds_back` |
| 4 | EOS stops at 2 tokens when script hits `<eos>` — **before** consuming max budget | `test_eos_terminates_before_budget` |
| 5 | max_new_tokens hard cap respected | `test_max_new_tokens_stops_generation` |
| 6 | Greedy determinism | `test_greedy_is_deterministic` |
| 7 | Seeded sampling reproducibility (same seed → same ids) | `test_seeded_sampling_reproduces` |
| 8 | Temperature semantics (0.0 = greedy, via argmax; >0 through `_filter_logits`) | `test_argmax_greedy_equals_temperature_zero` |
| 9 | top-k validation and behavior (bounds, vocab-size clamp; `top_k=99999` safe) | `test_top_k_is_bounded_by_vocab_size` |
| 10 | top-p keeps at least one candidate | `test_top_p_filtering_keeps_at_least_one_candidate` |
| 11 | Context-window left-truncation — long prompts don't crash, generation completes | `test_context_window_truncation_survives_long_prompts` |
| 12 | Padding/attention-mask correctness: masked rows == unmasked prefix (within fp32 tolerance); causal + padding masks are distinct | `test_padded_attention_matches_unpadded_prefix`, `test_causal_vs_padding_mask_are_distinct` |
| 13 | Multi-prompt batch NOT offered at this layer (by design; `list[str]` rejected) | `test_multi_prompt_batch_is_not_an_api_here` |
| 14 | Special-token contract pinned: pad=0 bos=1 eos=2 unk=3 | `test_special_token_ids_are_pinned` |
| 15 | Malformed inputs rejected: empty/whitespace/non-str prompts, NaN temperature, `top_p` outside (0,1], `temperature=2.5`, `top_k=-1`, `max_new_tokens=0` | parametrized `test_invalid_generation_parameters_rejected`; `test_malformed_prompt_rejected` |
| 16 | Checkpoint geometry rejection: missing file → failed closed; wrong shape → rejected; vocab mismatch (5822 vs 64) → fails closed | `test_geometry_missing_checkpoint_reports`, `test_geometry_wrong_shape_rejected`, `test_geometry_tokenizer_vocab_must_match_config` |
| 17 | Tokenizer fingerprint mismatch: single-byte reformat detected; vocab addition detected; special-token id change rejected | `test_fingerprint_detects_any_byte_change`, `test_vocab_digest_detects_vocab_change`, `test_special_token_mismatch_rejected` |
| 18 | Provider request/result compatibility (TakshS2Provider identity, capabilities, offline, generate, validation errors, empty prompt rejection) | `TestProviderAndApiContracts` |
| 19 | API contract: `/v1/models` empty when unloaded; `/health` 200; empty `messages` → 422; `require_loaded` → 503 HTTPException | `TestProviderAndApiContracts` |
| 20 | Missing production checkpoint: explicit BLOCKED behaviour is preserved; no substituted anywhere | `test_missing_production_checkpoint_is_explicit_blocked` |

**Batch behavior:** attends to batches at the attention-mask level (`test_padded_attention_matches_unpadded_prefix`); API surface remains single-prompt by design. The batch boundary is documented here in writing.

## 4. Checkpoint and tokenizer contract enforcement

- Tokenizer fingerprints (`file_fingerprint`, `vocabulary_digest`) are computed from raw bytes; mismatch paths raise `TokenizerFingerprintMismatch`; no path silently accepts.
- `training/geometry.py` fails closed on missing/truncated/mis-shaped checkpoints.
- **`v180_sft` remains GEOMETRY_INCOMPATIBLE — this milestone never loads it, substitutes it, or touches it; `9b541036…` preserved.**
- Canonical foundation artifact remains **MISSING**.
- No checkpoint artifacts are created by this milestone, under any fixture name. Tmp_path fixtures in tests create candidate checkpoints only to demo rejection and cleanup.

## 5. API and provider contract evidence

- `serving/providers.py` — unchanged; `TakshS2Provider` capabilities contract holds: `available=true`, `offline=true`, modes include `binding_repair`, `max_context_tokens` reads `model.cfg["max_seq_len"]` (32 for the fixture).
- `serving/openai_server.py` — unchanged; `/v1/models` data `[]` when unloaded; `/health` 200; malformed payload 422; unloaded `require_loaded` → 503 `HTTPException`.
- OpenAI-compatible surface offers deterministic mechanics; no Python-coding claim made anywhere.

## 6. Executed tests (exact commands and results)

```
python -m pytest tests/test_l111_native_inference.py -q
    47 passed, 1 skipped in 11.64s

python -m pytest tests/test_l110_native_foundation.py \
                 tests/test_l111_native_inference.py \
                 tests/test_model.py \
                 tests/test_training_correctness.py \
                 tests/test_sft.py \
                 tests/test_configs.py \
                 tests/test_m3_training_core.py \
                 tests/test_m3_end_to_end.py \
                 tests/test_token_contract_and_chat.py \
                 tests/test_tokenizer.py \
                 tests/test_openai_server.py \
                 tests/test_provider_architecture.py \
                 tests/test_code_serving.py -q
    254 passed, 6 skipped in 94.48s

python -m evaluation.l111_native_inference
    loop mechanics ok: True
```

Skips: frozen benchmark (feature-branch only, absent), multilingual benchmark (same), `tests/test_provider_architecture.py` skips are pre-existing by recorded design — not failures.

## 7. Not executed (deliberate)

- Full `pytest tests` (memory/age), GPU paths (no CUDA build), training/optimizer steps (PROHIBITED BY AUTHORIZATION), v180_sft capability measurement (never authorized), Python 3.11 qualification (3.14 host only), KARK-BENCH / KCC (`0xC0000135`), frozen benchmark re-scoring (benchmark artifacts are feature-branch protected; the 0/30 stands), L0.x milestone chains (different branches; not re-executed).
- No code path in this milestone references, calls, or imports anything tagged training-related for execution.

## 8. Limitations explicitly narrow this milestone

- Batch-style multi-prompt inference is **not offered** by the local interface; the attention-mask batch path is correct; the API path is single-prompt.
- KV cache is absent — every decode step re-runs a full forward pass. Correct, slow, documented.
- Random-weight models produce token soup; results here certify *mechanics*, never capability.
- `max_seq_len` left-truncation is the current documented policy; no alternate policy is implemented.
- `binding_repair` remained untouched; v180_sft binding contract remains untouched.

## 9. Preservation evidence

| Artifact | Status |
|---|---|
| `runs/v180_sft/checkpoint_last.pt` | untouched; hash `9b541036…` (referenced by prior milestones); **never loaded** by this milestone |
| `tokenizer/qm_code_bpe_v0.json` | unchanged bytes relative to L1.1.0 HEAD |
| `configs/model/foundation_111m.yaml` | unchanged bytes relative to L1.1.0 HEAD |
| Frozen benchmarks (hashes `854c7ca8…`, `7a839060…`) | untouched everywhere; absent on develop/main |
| `taksh-hf-publication/`, `taksh/dlcae/*`, Karkain, MicroQuantum | untouched |
| Owner CRLF files (6) | preserved untouched; never staged |

## 10. Final statuses

- **SOURCE IMPLEMENTATION: PASS** · **INFERENCE MECHANICS: PASS** · **CHECKPOINT CONTRACT: PASS (enforced; fixture-based rejection tests green)**
- **PROVIDER/API CONTRACT: PASS** · **PRODUCTION CHECKPOINT AVAILABILITY: UNAVAILABLE** · **NATIVE PRODUCTION INFERENCE: BLOCKED (by design; no artifact exists)**
- **PYTHON CODING CAPABILITY: NOT EVALUATED** · **TRAINING AUTHORIZATION: NOT GRANTED (A1.1-C/A2-E)** · **PUBLICATION AUTHORIZATION: NOT GRANTED** · **DEVELOP/MAIN MERGE AUTHORIZATION: NOT GRANTED**
