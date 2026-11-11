# TAKSH L0.7.4 — Training Resource Strategy Decision Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Milestone type:** resource strategy qualification — **nothing is provisioned, purchased, rented or executed.**

---

## 1. Bottom line

**The canonical `taksh-foundation-111m` cannot be trained on this host, and cannot be practically
trained here under any CPU-only strategy. A qualifying environment is a single GPU with ≥ 16 GiB
VRAM (bf16-capable) plus ~16 GiB system RAM and ~80 GiB disk — and it does not exist here.**
No environment was provisioned, paid for, downloaded or configured. No training measurement exists.
Every number below is labeled **measured** (point-in-time, 2026-11-11T16:03Z, psutil) or **estimated**
(algorithmic derivation) — never blurred.

---

## 2. Host measurements (measured)

| Measure | Value | Type |
|---|---|---|
| Logical CPUs | 8 | measured |
| RAM total | 4,199,780,352 B (3.91 GiB) | measured |
| RAM available at 16:03Z | 207,020,032 B (~198 MiB), 95.1% used | measured |
| Swap | 8,589,934,592 B total; 6,379,991,040 B free | measured |
| Free disk | 182,920,556,544 B (~170.4 GiB) | measured |
| GPU | NVIDIA GeForce MX150, 4,096 MiB | measured |
| Driver / CUDA capability | 457.49 / 11.1 | measured |
| torch runtime | 2.13.0+cpu — **CUDA unsupported by the build**; `torch.cuda.is_available() == False` | measured |
| MPS | not applicable (non-Apple) | n/a |

The MX150 exists **but cannot run the installed torch** (CPU build, and the driver is far older than
any current CUDA wheels). Even with a compatible toolchain, MX150 (Pascal-class GP108) lacks bf16
support and has only 4 GiB VRAM — marginal for the 111M working set even in fp16.

---

## 3. Fit mathematics for `taksh-foundation-111m` (110,658,816 params, d_model 768, 12 layers, 12 heads/6 KV, FFN 3072, context 2048, GQA, SwiGLU, tied embeddings, RoPE θ=10000, RMSNorm)

| State | Size | Label |
|---|---:|---|
| fp32 weights (4 B/param) | 442,635,264 B (~0.41 GiB) | estimate |
| fp32 grads | +442,635,264 B | estimate |
| Adam m+v fp32 | +885,270,528 B | estimate |
| **fp32 + Adam total** | **1,770,541,056 B (~1.65 GiB)** | estimate |
| bf16 weights | ~221 MB | estimate |
| bf16 weights + bf16 grads + fp32 Adam | ~1.33 GB | estimate |
| bf16 weights + Adafactor-factored state | ~0.5–0.7 GB | estimate |
| Activations (context 2048, fp32, batch 4) | several GiB without checkpointing; ~2–4× less with checkpointing | estimate, unverified |

These are upper bounds derived from parameter count — **not** measured end-to-end, and not a
guarantee. No measured end-to-end training figure exists for this model.

---

## 4. Strategy comparison

### A. Current local CPU host — **REJECTED**
- Theoretical floor (fp32+Adam) ≈ 1.77 GiB; real requirement with activations, data loader, runtime ≈ 2+ GiB.
- Available RAM ≈ 0.2 GiB; total 4.2 GiB shared with a 95–97% occupied OS. **Does not fit.**
- Swap exists (6.38 GiB free) but training on swap is practically unusable (thrash risk, no measured throughput).

### B. Memory-efficient CPU configuration — **NOT VIABLE (theoretical only; unmeasured)**
bf16/fp16 + factorized optimizer (Adafactor) + gradient checkpointing + micro-batch 1 + 64× gradient accumulation + canonical context (any context reduction would need separate qualification). Estimated floor ~0.5–0.7 GB — **might fit** in RAM/swap, but: throughput on 8 CPU threads for a 111M bf16 model is unverified and effectively impractical for any real token budget; 8-bit optimizer availability on Windows CPU is unverified (bitsandbytes); and the machine is already 95% full. Not a qualification path. Deliberately **not** pursued as an escape hatch.

### C. Owner-provided GPU — **NOT AVAILABLE**
Requirements: bf16-capable GPU, ≥ 8 GiB useful VRAM (floor, estimate), recommended ≥ 16 GiB, ≥ 16 GiB system RAM, ≥ 80 GiB disk. This host's MX150 (4 GiB, CUDA 11.1, no CUDA-capable torch) does not qualify and is not usable with the installed stack.

### D. External compute subject to owner authorization — **POSSIBLE ONLY WITH WRITTEN AUTHORIZATION**
Minimum viable environment (estimate): ≥ 16 GiB VRAM (bf16), ≥ 16 GiB system RAM, ≥ 80 GiB disk, current CUDA toolchain. Recommended (estimate): ≥ 24 GiB VRAM, ≥ 32 GiB RAM, ≥ 160 GiB disk. **No prices, vendors, GPU models, throughput or durations are invented here.** No provisioning performed. Owner must authorize in writing (budget, security model, data residency, retention, checkpoint policy, external-vs-internal distribution of derived artifacts) before any spend, download, or execution.

---

## 5. Minimum viable environment (recommendation)

**Single GPU with ≥ 16 GiB bf16-capable VRAM (e.g., current high-memory consumer/datacenter part), ≥ 16 GiB system RAM, ≥ 80 GiB free SSD, current CUDA toolchain, and owner's written authorization for compute + corpus handling + training policy.**

Until then, the L0.7.3 readiness verdict remains **BLOCKED**; nothing about this report changes that. Any future GPU strategy must pass its own measured qualification (VRAM headroom, checkpoint atomicity, resume exactly-once semantics) before training.

## 6. What is *not* done

No GPU queried beyond the host's own adapter (informational `nvidia-smi`), no models or datasets downloaded, no cloud resources created, no prices or performance figures invented, no training launched or dry-run on any hardware. The MX150 is recorded as present-but-unusable — not as an available training resource.
