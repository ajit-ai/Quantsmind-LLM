# TAKSH L0.5 — Feasibility Gate Report

**Decision: CONDITIONAL GO** — a technically credible route exists; capability is not yet demonstrated anywhere (0/30 everywhere); L0.5 authorizes nothing.

## 1. Preservation evidence (verified 2026-11-11)

- Branch `main` @ `c601a73716d5`
- `main` = `c601a73716d5`; `develop` = `8c553465216d`
- Working tree entries: 55 (all pre-existing modifications/untracked files preserved)
- Tokenizer: vocab 5,822 (`fbedf003...` committed artifact); benchmark SHA-256 `854c7ca833c224eb...` matches frozen value
- Checkpoints byte-identical:
  - `0ff8ccb96cee3594…` `runs/v160_pretrain/checkpoint_last.pt`
  - `9b54103695054ecb…` `runs/v180_sft/checkpoint_last.pt`
  - `66ab9fa88092cf6c…` `runs/l02_sft/checkpoint_last.pt`
  - `5182f82ce4a4b790…` `runs/l02_sft_hp/checkpoint_last.pt`
  - `2ce5a48f8c54fb28…` `runs/l02_overfit/checkpoint_last.pt`
  - `90c3a72e4d439ee1…` `runs/l03_pretrain/checkpoint_last.pt`
  - `8a4364c0c5e46533…` `runs/l03_pretrain/checkpoint_best.pt`
  - `d01aa82010c3a661…` `runs/l04_e1/checkpoint_last.pt`
  - `07e270526e6dacad…` `runs/l04_e2/checkpoint_last.pt`
  - `c30066d3833c908e…` `runs/l04_e3/checkpoint_last.pt`

## 2. Architecture audit (WP1)

- Baseline params: **28,787,328** (analytic == measured)
- Status: EXACT MATCH
- Family: decoder-only causal transformer (RMSNorm + RoPE + GQA + SwiGLU); GQA n_heads:n_kv_heads = 8:4; tied embeddings; context 2048

| Scenario | d_model | Layers | Heads (KV) | FFN | Params | vs baseline |
|---|---|---|---|---|---:|---:|
| candidate_A_100M | 768 | 12 | 12 (6) | 3072 | **110,658,816** | 3.844x |
| candidate_B_300M | 1024 | 18 | 16 (8) | 4096 | **289,115,136** | 10.043x |
| candidate_C_1B | 2048 | 16 | 16 (8) | 8192 | **1,018,624,000** | 35.384x |

Checkpoint compatibility: existing checkpoints warmstart **none** of A/B/C (all shapes change). Tokenizer fully reusable (vocab unchanged).

## 3. Training-data feasibility (WP2)

- Qualified in-repo code supply: **1,590,703 tokens** (L0.4, measured)
- L0.3 general supply (CC-BY-SA / PD): **19,629,683 tokens** (not code)
- L0.4 total train: 3,182,359 tokens; code share **0.4999**

Required tokens (unique) vs qualified code supply:

| Model | @20/p | @50/p | @100/p | @200/p | supply coverage @100/p | epochs @100/p |
|---|---:|---:|---:|---:|---:|---:|
| current (28.8M) | 0.58B | 1.44B | 2.88B | 5.76B | 0.0553% | 1810x |
| A (110.7M) | 2.21B | 5.53B | 11.07B | 22.13B | 0.0144% | 6957x |
| B (289.1M) | 5.78B | 14.46B | 28.91B | 57.82B | 0.0055% | 18175x |
| C (1018.6M) | 20.37B | 50.93B | 101.86B | 203.72B | 0.0016% | 64036x |

- Register: 10 candidate corpora assessed; 6 require legal review; **none ingested**; benchmark sets excluded by policy.

## 4. Training-strategy comparison (WP3)

- **Strategy A — Continued pretraining (code-dense)**: CONTINUE ONLY WITH NEW LICENSED DATA AT SCALE; not a fix in isolation
- **Strategy B — Code-focused supervised fine-tuning only**: NECESSARY BUT NOT SUFFICIENT; without a strengthened base it produces memorization, not capability
- **Strategy C — Combined: continued pretraining + code-focused SFT (two-phase)**: THE VIABLE ROUTE; both phases required; both blocked on licensed data + compute authorization
- **Strategy D — Architecture redesign (larger model vs. changes)**: DO NOT REDESIGN; scale with data. Consider A/B only after data authorization

Recommended sequence: Sequence: (1) owner authorizes data acquisition + legal review (candidate corpora in the register); (2) A100-class compute; (3) strategy C: continued pretraining at 100M, then code-focused SFT; (4) re-run the frozen L0.2 benchmark with the identical protocol; (5) keep the L0.1 default checkpoint until a candidate demonstrably beats it on the frozen benchmark.

## 5. Compute and memory (WP4)

- Method: training FLOPs = 6 x params x tokens; fp32/bf16 working set ~= 16 bytes/param (weights+grads+Adam).
- CPU anchor measured in-repo: ~215 tok/s (range [180, 260])
- existing_28.8M: fits training on 6/7 hardware classes
- candidate_A_110M: fits training on 6/7 hardware classes
- candidate_B_289M: fits training on 5/7 hardware classes
- candidate_C_1019M: fits training on 3/7 hardware classes

GPU durations deliberately NOT invented; owner supplies price + target stack for pricing.

## 6. Benchmark audit (WP5)

- Frozen hash matches: **True**
- Structural verdict: **PASS** (33 checks)
- Raw vs serving equivalence: **1.0** (414 records; binding repair never changed an output)
- Acceptance bar for claiming capability: >= 15/30 frozen problems correct, >= 0.80 syntax-valid, >= 0.60 execution, all categories >= 30%, 100% repeated-run consistency, zero safety violations.

## 7. Final decision

### CONDITIONAL GO

A credible route exists; nothing is authorized by this milestone. Owner decisions outstanding:
- **Data acquisition budget + license selection** — consequence: no training beyond the current 1.59M-token pool; no capability recovery possible
- **GPU budget (A100/H100-class, 80 GB preferred)** — consequence: continued pretraining of even a 110M-class model for one epoch of a 20M-token slice is ~28 CPU-days; GPU is mandatory for anything real
- **Legal review sign-off on chosen corpora (OpenRAIL terms, per-file license obligations)** — consequence: candidate corpora cannot be ingested; benchmark-leakage and secret scans must be re-run on whatever is chosen
- **Decision on strategy C vs A and target size (110M vs 289M)** — consequence: blocks compute sizing and data-volume commitment

What would flip to GO:
- owner-authorized acquisition of >= 20B tokens of license-verified Python (or equivalent) with a clean legal review
- A100-class (or better) compute for >= 2 weeks, or equivalent acceptance of a > 1-month CPU-only schedule for a reduced experiment
- a pre-registered benchmark protocol in which a candidate passes >= 15/30 frozen problems with zero leakage

What would flip to NO-GO:
- owner declines data acquisition or GPU spend on this line of work
- legal review rejects all candidate corpora
- L0.6 experiment at authorized scale still yields 0/30 with a clean benchmark


*This document is generated by `python -m evaluation.l05_feasibility` from the same artifacts it summarizes. Nothing here authorizes purchasing, downloading, training, publishing, or changing the L0.1 default checkpoint.*
