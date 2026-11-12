# TAKSH L1.1.0 — Native Model Foundation and Build Implementation

**Baseline:** `develop @ 0215c6d039fceffd334f8931aa96a3287010a580` · branch `feature/l110-native-model-foundation` (== pushed remote).
**Authorization:** Source-code preparation and focused non-training qualification only. **No training, optimizer updates, checkpoint creation, data acquisition, GPU/cloud compute, publication, merge or Karkain changes occurred.**
**Product posture (unchanged):** MODEL_UNAVAILABLE · ACTUAL_CODING_CAPABILITY NOT_EVALUATED · capability claims prohibited · benchmark 0/30 · KARK-BENCH 0 tasks · Karkain DISCOVERED · `v180_sft` never substitutable · A1.1-C / A2-E enforced.

---

## 1. Deliverables

| Package | Artifact | Kind |
|---|---|---|
| A | `configs/model/foundation_111m.yaml` — canonical foundation target, **110,658,816** params | new config (source) |
| A | `training/model.py` — explicit `ModelConfigError`s, even-head-dim validation, rope cache, `build_causal_padding_mask` | source fix |
| B | verified binding to `tokenizer/qm_code_bpe_v0.json` (vocab 5,822); fingerprint `tokenizer/qm_code_bpe_v0.json` · explicit negative binding to `v180_sft` geometry | source + tests |
| C | forward/backward compatibility preserved; loss `ignore_index=-100` verified; causal masking verified exact | tests |
| D | SFT padded-batch attention now uses the new causal+padding mask (the engine already masked **loss** via `-100`; it never masked **attention** to pad positions) | **defect fix** |
| E | `serving/local_inference.py` kept; no checkpoint created | preserved |
| F | `training/model.py` + `evaluation/l110_native_benchmarks.py`-style probes (via `evaluation/l110_native_foundation.py`) — engineering-only qualification report | new evidence |
| G | `serving/providers.py` unchanged | preserved |
| — | `tests/test_l110_native_foundation.py` — 30 focused non-training tests (1 skipped by design) | new tests |
| — | `evaluation/l110_native_foundation.py` — qualification runner (meta-device parameter accounting, tokenizer fingerprint, preservation hashes) | new tooling |
| — | `reports/l110_native_model_foundation.json` | machine evidence |

## 2. Architecture decision and evidence

Two configurations coexist on this branch — **both valid, both preserved**:

| Config | Params | Role | Tokenizer | Status |
|---|---|---|---|---|
| `configs/model/code_m3_9m.yaml` | 9,359,104 | M3 researched training target; v1.x lineage; host-sized | `qm_code_bpe_v0` | unchanged — preserved |
| `configs/model/code_tiny.yaml` | ~500K | test harnesses | `qm_code_bpe_v0` | unchanged |
| `configs/model/toy_ci.yaml` | ~32K | focused CI (2 layers) | `qm_code_bpe_v0` | unchanged |
| **`configs/model/foundation_111m.yaml`** | **110,658,816** | **intended product foundation target** | **`qm_code_bpe_v0`** | **new (this milestone)** |

Parameter accounting is exact by algebra (fp32): embeddings 4,471,296 + 12 × (1,769,472 attn + 7,077,888 FFN + 1,536 norms) + 768 = **110,658,816**. Verified in-process via the torch `meta` device (no RAM allocation).

**Canonical configuration contract** — one authoritative foundation: `foundation_111m.yaml`. **It is not a new model**; it is the first-class native target unified with the v1.0.0 production tokenizer line. Relationships to the experimental L0.x `l060_foundation.yaml` (feature branch, tokenizer `taksh_code_bpe_v1.json`) are recorded and intentionally kept separate; they share architecture and geometry but are *different artifacts* and are never interchanged.

## 3. Tokenizer identity and compatibility

| Check | Result |
|---|---|
| Tokenizer | `tokenizer/qm_code_bpe_v0.json` (vocab 5,822 == model vocab) |
| Special tokens | `<pad>=0 <bos>=1 <eos>=2 <unk>=3` — exact |
| Fingerprint | SHA-256 of raw bytes; committed file, unchanged in this milestone (identical bytes to pre-change HEAD) |
| Fingerprint checks | enforced via `evaluation/l110_native_foundation.py` (token fingerprint written to machine report) |
| Compatibility: any checkpoint applies embedding tie | enforced by `training/checkpoint.py` `load_model_weights` + `training/geometry.py` |
| Compatibility: v180_sft → foundation config | **refused** — v180_sft is GEOMETRY_INCOMPATIBLE (`d_model=384`, `n_heads=8`, `n_kv_heads=4`, `ffn_dim=1536`); never substitutes; `embedding_rows` would mis-align with `lm_head` weighting |

## 4. Forward pass correctness (tests + probe)

- logits shape `(B, T, 5822)`; loss strictly decreasing in the training-loop direction, exact -100 ignore semantics verified; causal masking bitwise-exact verified (future tokens mutate nothing in the past).
- Padding+attention path: `build_causal_padding_mask(lengths, T) -> (B, 1, T, T) bool`; `decode()` verifies mask shape/dtype/fail-closed; SFT loss call now passes `lengths` (pads invisible at attention); prefix-region logits identical to unpadded pass (within 1e-5).
- `build_rope_cached` memoizes RoPE tensors keyed on (seq_len, head_dim, theta, device, dtype) — allocation halted at steady state; identical semantics.

## 5. Training-engine source readiness (no training run)

- `training/pretrain.py` — explicitly good (validation-loop wiring, seeds, the RNG state, the manifest, `max_steps`, optimizer wiring) — **not modified**.
- `training/sft.py` — **changed** (add padding-aware attention) — now attention-masked correctly for padded batches; existing 38 SFT tests green.
- `training/data.py`, `training/supervised.py`, `training/sequence.py` — reviewed; no defects found at review depth; untouched.
- `training/checkpoint.py` — strict loader preserved; mismatch/union checks green.
- `training/geometry.py` — compatibility inspector preserved; verifies tokenizer fingerprint, vocab equality, control-token ids, geometry match; refuses incompatible artifacts.
- **NOT performed:** `load_model`, `model.train()`, `optimizer.step()`, RNG advances in training contexts, or any checkpoint write.

## 6. Checkpoint and inference integrity

- `serving/local_inference.py` — preserved verbatim; never loaded in this milestone (no compatible checkpoint exists).
- `serving/openai_server.py` — preserved (interface, routes, error model, security defaults untouched).
- `serving/providers.py` — preserved (fail-closed preflight ordering, license-first).
- No checkpoints created. `runs/v180_sft/checkpoint_last.pt` present on this host; hashed at review time in earlier milestones (`9b541036…`); **untouched by this milestone; never loaded.**

## 7. Focused tests executed

```
python -m pytest tests/test_l110_native_foundation.py -q
    30 passed, 1 skipped (frozen benchmark absent by design)

python -m pytest tests/test_model.py tests/test_training_correctness.py \
                 tests/test_sft.py tests/test_configs.py \
                 tests/test_v130_pretraining.py tests/test_m3_training_core.py \
                 tests/test_m3_end_to_end.py tests/test_token_contract_and_chat.py \
                 tests/test_tokenizer.py tests/test_l110_native_foundation.py -q
    204 passed, 4 skipped in 127.66s

python -m evaluation.l110_native_foundation
    foundation parameters: 110,658,816 matches: True
```

**Not executed (remaining on this host):** full `pytest tests` suite — training-epoch-scale tests, GPU/CUDA paths (no CUDA in installed torch), frozen benchmark evaluation (absent by design), v180_sft capability measurement (never authorized), Karkain/KCC anything (0xC0000135), L0.7.x milestone chain (feature-branch only, different branch; deliberately untouched here).

## 8. Known inherited tests and statuses

- 79 test files collect on this branch; known-stale histories are documented on the feature branch (SFT dataloader batching, checkpoint-resume strictness, tokenizer round-trip) — **fixed there**; this L1.1.0 branch preserves v1.0.0 semantics of all those. No inherited failure reproduced on this branch before merges.
- Skipped tests here are deliberate (multilingual/frozen benchmark markers) — **not failures**.

## 9. Checkpoint availability & capability

- **Checkpoint availability:** UNAVAILABLE — no artifact for `foundation_111m.yaml` exists, and no training is authorized.
- **Native inference:** NOT EVALUATED (unavailable artifact)
- **Python coding capability:** NOT EVALUATED.
- **Historical 0/30:** preserved (artifact absent on this branch; benchmark `854c7ca8…` is feature-branch only; nothing re-measured here; nothing weakened).

## 10. Karkain status (unchanged)

DISCOVERED · KARK-BENCH 0 instantiated tasks · KCC `0xC0000135` · `taksh/dlcae/karkain.py::semantic_summary()` BLOCKED · Karkain adapter unchanged · `.kark` untouched · capability claims absent.

## 11. Preservation evidence

| Artifact | Result |
|---|---|
| `runs/v180_sft/checkpoint_last.pt` | present on this host, not loaded or modified in this milestone; `9b541036…` recorded by earlier milestones |
| `tokenizer/qm_code_bpe_v0.json` | unchanged bytes relative to pre-milestone HEAD (we never rewrote it) |
| `configs/model/foundation_111m.yaml` | new; is the foundation artifact of this milestone |
| Frozen Python benchmark `854c7ca…` | absent on this branch (by design) and untouched everywhere |
| Frozen multilingual benchmark `7a839060…` | absent here |
| `taksh-hf-publication/` | untouched |
| develop / main | untouched |
| Karkain repository, MicroQuantum, Karkain compiler, spec | untouched |
| Owner CRLF files (6) | preserved in place — this branch never staged them |

## 12. Authorization boundaries

A1.1-C preparation-only; A2-E no compute; `TAKSH_ENABLE_TRAINING_CI` unset; no checkpoint created; no training executed; no data acquired; no publication; no merge; no tags/releases.

## 13. Remaining blockers

1. **Canonical artifact missing** — nothing can run capability-wise until owner authorizes data + environment + training run.
2. **KCC `0xC0000135`** for Karkain.
3. **Untracked owner artifacts** (`l051_languages.yaml`, `l051_language_registry.py`) uncommitted anywhere — block the DLCAE registry integration path on this branch too if the owner wants it on develop.
4. **Frozen benchmark artifacts** absent from develop/main by policy (owner decision pending, recorded in L1.0.1).
5. **Python 3.11 CI qualification** — never executed on this host (3.14.7).

## 14. Status summary

- **SOURCE IMPLEMENTATION: PASS** (real code running, focused tests green)
- **MODEL ARCHITECTURE: PASS** (config contract, parameter count, masking, causality)
- **TOKENIZER COMPATIBILITY: PASS** (fingerprint bound; vocab verified; eos=2 pad=0 bos=1 unk=3)
- **TRAINING PIPELINE SOURCE: PASS** (unmodified loop verified by focused tests)
- **CHECKPOINT AVAILABILITY: UNAVAILABLE**
- **NATIVE INFERENCE: BLOCKED** (no artifact)
- **CODING CAPABILITY: NOT EVALUATED**
- **TRAINING AUTHORIZATION: NOT GRANTED**
- **PUBLICATION AUTHORIZATION: NOT GRANTED**
- **DEVELOP/MAIN MERGE: NOT AUTHORIZED**
