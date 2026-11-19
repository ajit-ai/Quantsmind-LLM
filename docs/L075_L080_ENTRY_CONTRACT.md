# TAKSH L0.8.0 — Entry Contract (Pretraining Authorization Preconditions)

**Purpose:** the complete, checkable set of preconditions for L0.8.0 Foundation
Code Pretraining. If ANY mandatory condition is unmet, **training must not begin —
no exceptions, no partial credit, no "just this once".** This contract exists so
that no future session (human or agent) can drift into training informally.

L0.8.0 itself does **not** start in L0.7.5; this document authorizes nothing.

---

## 1. Data (all mandatory)

- [ ] **Frozen eligible dataset manifest** — `taksh.l072_l04_freeze` snapshot `l072-l04-fdb1fee2beaae9c2` or a successor freeze; manifest sha verified at launch.
- [ ] **Legal permission** — a current, signed decision record covering the *exact* corpus to be trained on. Internal research pretraining must be explicitly in force at training time (not merely historically documented); distribution scope declared separately (internal research vs external release).
- [ ] **Owner authorization** — written grant by the owner: snapshot id, scope, model config, environment, budget/ceiling, distribution class, expiry/revocation. A missing or expired authorization is an absolute blocker.
- [ ] **Source provenance** — every sample traceable: source id, version, license, revision, fingerprint chain to the corpus manifest; exclusion audit complete.
- [ ] **Deduplication and contamination status** — exact + near-duplicate handling verified on the training manifest; frozen Python benchmark, 64-task multilingual benchmark and TAKSH-KARK-BENCH checked for overlap; any hit blocks or clears with documented resolution.
- [ ] **Language distribution** — actual measured token distribution across the catalogue; Python-only today; no fabricated multilingual claims.
- [ ] **Split integrity** — group-aware duplicates never span train/val/test; leakage checks pass; holdout benchmarks remain unused for training decisions.

## 2. Model (all mandatory)

- [ ] **Canonical configuration** — `configs/foundation/l060_foundation.yaml` byte-identical: vocab 5822, d_model 768, 12 layers, 12 heads, 6 KV heads, FFN 3072, context 2048, tied embeddings, RoPE θ=10,000, RMSNorm; 110,658,816 params; fingerprint recorded.
- [ ] **Tokenizer fingerprint** — committed LF blob `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` or a successor authorized tokenizer; tokenizer identity binds every checkpoint produced.
- [ ] **Vocabulary compatibility** — tokenizer vocab == 5822 == embedding rows == output projection rows.
- [ ] **Parameter count** — verified at initialization; any mismatch aborts before step 1.
- [ ] **Initialization provenance** — initialization script, seed, and weight fingerprints recorded; run id derives from dataset fingerprint + code revision.

## 3. Environment (all mandatory)

- [ ] **Authorized hardware** — owner authorization in writing; measured device (model, VRAM, driver/CUDA) recorded at run start; availability verified at run start.
- [ ] **RAM/VRAM/disk headroom** — measured with ≥25% headroom vs the *actual* training working set; disk capacity including checkpoint lineage and exports.
- [ ] **Compatible stack** — Python/PyTorch/CUDA combination verified live on host (not just declared); CPU fallback decided in writing.
- [ ] **Reproducibility settings** — fixed seed; deterministic kernels documented; environment snapshot (OS, versions, driver, GPU) recorded with the run manifest.

## 4. Training (all mandatory)

- [ ] **Authorized objective and scope** — written scope: "pretraining taksh-foundation-111m on the L0.4-derived frozen manifest" (or its authorized successor); out-of-scope items named and excluded.
- [ ] **Optimizer and scheduler** — AdamW (β1=0.9, β2=0.95, wd=0.1), cosine schedule, 200-step warmup, min_lr_ratio 0.1, grad clip 1.0 (as in L0.6.1 canonical configuration; any deviation requires new owner authorization).
- [ ] **Precision** — declared (e.g., bf16 if the device supports it); memory and throughput expectations recorded.
- [ ] **Batch and accumulation** — micro-batch and grad-accum chosen for the authorized hardware; effective batch ≤ hardware ceiling; no silent increase.
- [ ] **Context length** — 2048 (canonical); over-limit samples rejected, never truncated silently.
- [ ] **Checkpoint cadence** — every 500 steps; atomic writes; interruption detection; resume compatibility gates enforced.
- [ ] **Recovery procedure** — documented restart path; last-good checkpoint identified; resume validation gates run before resuming.
- [ ] **Stop conditions** — step ceiling, time ceiling, loss/grad nonfinite, evaluation regression triggers — all pre-registered.
- [ ] **Evaluation and acceptance criteria** — what "this run succeeded" means is written before step 1; frozen benchmark never altered; 0/30 stands until genuinely displaced by fresh, held-out measurement.

## 5. Evaluation (all mandatory)

- [ ] **Frozen benchmark identity** — `evaluation/l02_python_bench_v1.json` (sha256 `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6`) used unchanged; scoring thresholds, tasks and expected outputs untouched.
- [ ] **No training/evaluation contamination** — verified at data ingestion; any overlap quarantines the run before it produces a checkpoint.
- [ ] **Per-language measurement** — results reported per language with the frozen multilingual benchmark (64 tasks; ≥16 executed + ≥50% pass ratio denominator); Python-only reality reported honestly.
- [ ] **Reproducibility and safety checks** — deterministic decoding verified on the final checkpoint; the checkpoint carries its provenance (config hash, tokenizer hash, dataset snapshot, step, list of excluded samples) and integrity fingerprint.

## 6. Prohibitions (never, regardless of scope)

- No modification of frozen benchmark content, thresholds or scoring to help a model pass.
- No training on synthetic fixtures.
- No silent truncation of corpus content to fit a device; no dataset expansion without owner review; no license circumvention of any kind.
- No use of `runs/v180_sft/checkpoint_last.pt` or any existing artifact as a "foundation" substitute.
- No replacing the canonical 111M with a smaller or different configuration without a new, separately-approved qualification.
- No external publication or redistribution of any artifact without an explicit signed distribution decision.
- No silent resume from an unverified checkpoint; no manual patching of checkpoint bytes.
- No claiming "trained" or "capable" (by marketing, docs, model cards or SLA) without the acceptance criteria in §5 being met on fresh held-out measurement.

## 7. Go / no-go

**GO for L0.8.0 execution requires every box above checked, in writing, before any GPU is powered on for training.** Any missing box → **NO-GO**, logged in the L0.8.0 kickoff record, training does not begin.

*L0.7.5 does not check any of these boxes. The expected outcome of the preflight is NO-GO.*
