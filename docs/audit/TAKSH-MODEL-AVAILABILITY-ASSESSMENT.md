# TAKSH — Canonical Model Availability Assessment (L1.0.0, 2026-11-12)

**Authorization:** A1.1-C (preparation-only) / A2-E (no compute). **No training, conversion, fine-tuning, downloads, or checkpoint creation is authorized or performed by this assessment.**
**Baseline:** assessment performed against feature-branch HEAD `78b6cdf…` (evidence) and develop `0215c6d` (no canonical artifacts present there either).

---

## 1. Canonical target identity

| Item | Value |
|---|---|
| Model id | `taksh-foundation-111m` |
| Config | `configs/foundation/l060_foundation.yaml` (feature branch; SHA-256 `01a90464e6f2…`) |
| Schema | `taksh.foundation_config/1` |
| Architecture | Llama-style decoder-only: RMSNorm + RoPE (θ = 10000) + GQA + SwiGLU, tied embeddings |
| vocab_size | 5,822 (fixed: `tokenizer/taksh_code_bpe_v1.json`) |
| d_model | 768 |
| n_layers | 12 |
| n_heads | 12 |
| n_kv_heads | 6 (GQA 2:1) |
| ffn_dim | 3,072 |
| max_seq_len | 2,048 |
| norm_eps | 1.0e-05 |
| tie_embeddings | true |
| Parameters | **110,658,816** (formulaic: embeddings 4,470,144; per-layer attn 768²+2×768×384+768² ≈ 1.77M + FFN 3×768×3072 ≈ 7.08M ≈ 8.85M ×12 ≈ 106.1M; final norm ≈ 768 — exact matches v180_sft geometry receipts from L0.6.0) |
| Tokenizer | `taksh_code_bpe_v1.json` — committed blob SHA-256 `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` (feature branch; vocab 5,822; LF-normalized; CRLF working copy `554c8e75…` is intentionally different and is **never** used for hashes) |
| Init | torch.nn.init.normal_(std=0.02) for Linear/Embedding; norm weights initialised to ones |
| Seed policy | deterministic; training code required for any real run |

## 2. Canonical artifact inventory

| Artifact | Present? | Notes |
|---|---|---|
| Canonical config YAML | **YES** (feature branch; untracked in develop's tree; present in owner working tree) | hash-stable |
| Tokenizer artifact | **YES** (feature branch; LF blob committed) | vocab 5,822 == model |
| **Canonical checkpoint (any format)** | **NO** | Never created; never trained; not on any branch, tag, release, or artifact store examined |
| Intermediate/resume checkpoints | none for canonical | `runs/` is gitignored; nothing for taksh-foundation-111m |
| Optimizer state (AdamW m/v) | n/a — never trained | required for any real run |
| `v180_sft/checkpoint_last.pt` | **YES** (gitignored, owner-held) | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — **GEOMETRY_INCOMPATIBLE** vs canonical: d_model 384 vs 768; heads 8 vs 12; n_kv_heads 4 vs 6; FFN 1536 vs 3072; embedding_rows 5,822 but geometry is 28.8M-class. **Never substitutable** (L0.6.0/L0.6.1 demonstrated, L0.8.0-PRE enforced, L1.0.0-ENG3 refuses). |
| v180_sft tokenizer (qm_code_bpe_v0.json, 5,822) | YES (or committed in QM configs) | belongs to the 28.8M lineage, not the canonical |

## 3. Compatibility matrix (verified or measured by this program)

| Check | Result |
|---|---|
| Tokenizer vocab == 5,822 | ✓ (matches model config) |
| Config params recomputed | 110,658,816 — exact match vs `l060_foundation.yaml` |
| `v180_sft` → canonical geometry swap | **refused everywhere** (geometry mismatch recorded in L0.6.1/L0.6.2/L0.6.4.0-PRE/L1.0.0-ENG3 reports with same numbers) |
| `v180_sft` → canonical substitution attempt (any milestone) | **refused** — v180_sft is GEOMETRY_INCOMPATIBLE; L1.0.0-ENG3 refuses at assess() with explicit `geometry_mismatch` in the report |
| Canonical → any serving engine | blocked until an artifact exists |
| Frozen benchmarks present on develop | **absent** (on feature branch only) |
| Benchmark hashes verified | `854c7ca8…` (Python) · `7a839060…` (multilingual) |

## 4. Missing artifact list

1. Canonical model weights (`taksh-foundation-111m`, any format).
2. Canonical optimizer/scheduler state for resume.
3. Training-proven calibration artifacts (init stats after a qualified training run).
4. Any evidence that the canonical token distribution is correct for the `(corpus × tokenizer × 5,822-vocab)` binding beyond fixture checks.
5. Any real-world inference measurement of the canonical model.
6. Any qualified checkpoint-load-fingerprint evidence on any external host.

## 5. Inference loading prerequisites (for a future artifact)

Assuming a canonical artifact existed and a qualified toolchain loaded it:

- fp32 weights ≈ **422 MiB** · fp16/bf16 ≈ **211 MiB** (+ rope tables, norms, ≈ +1 MiB)
- KV cache @ context 2048/12 layers/6 KV/64d ≈ 2048×12×6×64×2 (k+v) × 2 B ≈ **35.8 MB** per sequence (fp16) — or ~72 MB in fp32
- peak per request ≈ **230–500 MB** depending on precision/impl

**Inference host requirements (estimate, arithmetic on config, unmeasured):**
- RAM ≥ 1 GB usable for weights; Python + torch ≥ 2–4 GB practical
- CPU ok for single-request inference (8 CPU host like this one, ~4 GiB RAM — feasible for a quantized 110M model; PyTorch CPU build works)
- GPU optional; MX150/4 GiB VRAM sufficient for single-request fp16 inference once artifact exists; does not change A2-E

## 6. Hardware & feasibility estimates (measured vs estimated)

| Quantity | Value | Confidence |
|---|---|---|
| Host CPUs | 8 (measured, psutil) | measured |
| Host RAM | total 4,199,780,352 B; available measured as low as ~97 MB–4.2 GB over the session | measured, volatile |
| Host disk | ~170 GiB free (measured L0.7.x, this host) | measured at an earlier point |
| MX150 VRAM | 4,096 MiB; torch.cuda.is_available() == False (installed build is CPU-only; CUDA 11.1 driver) | measured |
| fp32 + Adam(fp32 m+v) state | **1,770,541,056 B ≈ 1.65 GiB** for params+grads+m+v (≈4×4 bytes×110,658,816) | calculated |
| bf16 mixed (weights+grads bf16 + fp32 m+v) | ~ (2+2+8)×110,658,816 ≈ 1.33 GiB | calculated estimate |
| Activation @ context 2048, fp32 attention | several GiB or FLOP-limited metadata | unmeasured, unresolvable here |
| Minimum practical training host | **GPU ≥ 16 GiB VRAM** (the L0.7.4 recommendation), RAM ≥ 16 GB, disk ≥ 80 GB, owner-authorized | estimated |

## 7. Feasible execution options (clearly separated from authorization)

| Option | Feasibility here | Authorization |
|---|---|---|
| Train canonical 111M on local CPU only | Not practical (hours–days, insufficient RAM, no training CI) | **not authorized** (owner decision would be required) |
| Fine-tune a canonical artifact | n/a — no artifact exists | would require A1.1-A + A2 authorization |
| Rent GPU (A2-D) | Unavailable without a concrete owner action + approved plan/provider/purchase | **not authorized** |
| Use MX150 | Not viable (CUDA mismatch + CPU-only torch + insufficient VRAM) | n/a |
| Use an existing v180_sft artifact | Architecturally incompatible with canonical config | **never** |
| Pre-train a 28.8M→111M Warm start on this host | Requires a canonical run; prohibited by A1.1-C/A2-E | blocked |
| Use the canonical artifact on CPU for inference after owner-authorization + construction | Feasible (see §5; estimates only) | owner permission required |

## 8. Path to a canonical artifact (no telescoping allowed)

| Stage | Required evidence | Owner authorization |
|---|---|---|
| 1. Training eligible data | L0.8.0 entry contract + new acquisition decisions | owner decisions (A1.1, A2) |
| 2. Training run configuration (§8 of L0.8.0-PRE) | Legal/eligibility review on the frozen corpus; all 18 L0.8.0 gates pass | A1.1-A, A2 decision, compute approved |
| 3. Materialize 111M artifact | Deterministic seed; full manifest; provenance record; 0 substitution | A2-D-class authorization + environment |
| 4. Checkpoint qualification | Atomically written, integrity-checked, verified against canonical config fingerprint + tokenizer fingerprint | automatic post-run |
| 5. Inference qualification | Reproduce loss curves; deterministic decodes; outputs sane (no claim yet) | free |
| 6. Coding-capability evaluation | see `TAKSH-CODING-CAPABILITY-EVALUATION-PLAN.md` | A1.1-R-like decision |

Every gate currently fails closed; the fastest honest announcement of a canonical artifact today is: **MISSING — nothing exists; do not substitute.**

## 9. Bottom line

**Canonical model does not exist. There is nothing to run, measure, or evaluate. Any workflow that implies a working TAKSH model is unverified speculation today.**
