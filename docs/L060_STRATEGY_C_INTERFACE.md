# TAKSH — Strategy C Interface (L0.6.0)

**Status: PREFERRED HYPOTHESIS, NOT EXPERIMENTALLY PROVEN.**
Strategy C = two phases: (1) foundation code pretraining, (2) supervised
coding-instruction fine-tuning. Strategy A (pretraining only) remains the
documented alternative from L0.5. Nothing in L0.6.0 authorizes running either
phase, or acquiring data, or training anything.

This document specifies the exact technical interfaces between the two phases
against the artefacts that actually exist in this repository.

## Phase 1 — Foundation code pretraining

| Interface | Artefact | Contract |
|---|---|---|
| Config | `configs/foundation/l060_foundation.yaml` | `taksh.foundation_config/1`; schema-validated by `evaluation/l060_config_qualification.py`; deterministic config `sha256`; vocab fixed at 5,822; params exactly 110,658,816 |
| Model | `training/model.py::QMModel` | RMSNorm + RoPE + GQA + SwiGLU, tied embeddings; causal SDPA; right-pad + `ignore_index=-100` loss |
| Warmstart | `training/pretrain.py --init-from <ckpt>` | safe-load only (`weights_only=True` + narrow NumPy allowlist retry); 111 tensors expected |
| Data | token stream + `quantsmind.token_stream/1` meta | `data.TokenDataset` checks artifact vocab ≤ model vocab, `<eos>` doc separator, no missing meta |
| Optimizer | AdamW cosine | peak 1e-3 (warmstart) / 3e-3 (scratch), warmup 20, min ratio 0.1, betas (0.9, 0.95), wd 0.1, grad clip 1.0 |
| Loss / objective | next-token prediction | cross entropy with document boundaries; validation via `training/validation.py::evaluate` (fixed windows, val seed) |
| Checkpoint schema | `quantsmind.checkpoint/1` | `model`/`optimizer`/`step`/`config`/`schema`/`tokens_processed`/`rng_state`/`val_loss`/`provenance` |
| Provenance | `build_run_manifest` | config, tokenizer, corpus manifests, seeds, initialization note, git HEAD, package version |

**Checkpoint compatibility between Phase 1 and Phase 2:** Phase 2 must consume
the Phase-1 `.pt` unchanged through `training.checkpoint::init_from_checkpoint`
(strict by default) with `allow_partial=False`. The only parameter shapes that
matter are `vocab_size=5822` and the architecture fields; any change aborts.

## Phase 2 — Supervised coding-instruction fine-tuning

| Interface | Artefact | Contract |
|---|---|---|
| SFT data | `data/l02_sft/sft.jsonl` (260 pairs) / `data/l02_overfit/sft.jsonl` (24) | `qm.plain_instruction/1` template via `training/chat_template.py::render_for`; no role tokens; `<bos>`/`<eos>` single tokens; v1 contract verified |
| Prompt contract | `training/tokenizer_contract.py::detect_contract` | returns `qm.plain_instruction/1` for taksh_code_bpe_v1; refuses others without single-token markers |
| Supervision layout | `training/sft.py::SFTPairDataset` | labels `-100` on prompt tokens; supervision starts at the response's first token; `<eos>` always supervised |
| Tokenizer | `tokenizer/taksh_code_bpe_v1.json` | vocab 5,822; bos=1 eos=2; piece count identical for prompt parts (prefix property) |
| Label shifting | `labels = ids[1:]`, `inputs = ids[:-1]` | standard 1-token shift; verified by `tests/test_l02_pipeline.py` (exact mask/boundary assertions) |
| Checkpoint | same schema as Phase 1 | consumed via `--init-ckpt` (strict); never overwritten |

## Tokenizer identity

- File: `tokenizer/taksh_code_bpe_v1.json`
- Committed artifact SHA-256: `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7`
- Working-tree CRLF copy: `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` (content identical; line endings differ, core.autocrlf=true)
- 5,822 vocab; `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3

## Data manifests

- `data/l03_corpus/dataset_manifest.json` (20,009,717 tokens; Wikipedia/Gutenberg CC-BY-SA-3.0 / Public Domain; m3-verified MIT/BSD code; Apache-2.0 project source) — passes L0.6.0 manifest validator
- `data/l04_code_corpus/dataset_manifest.json` (3,182,359 tokens; 50% code; dedup + secret scan + benchmark-leakage scan; 4 benchmark-colliding files quarantined) — passes validator

## Evaluation requirements (never lowered)

- Python: frozen `evaluation/l02_python_bench_v1.json`, SHA-256 `854c7ca8...`, ≥ 15/30 pass@1, ≥ 0.80 syntax-valid, ≥ 0.60 execution, all categories ≥ 30%, 100% greedy determinism, 0 safety violations.
- Multilingual: `evaluation/l051_multilingual_benchmark_v1.json` (64 starter tasks, 16 languages), threshold ≥ 50% of executed tasks per language with ≥ 16 tasks per language required for qualification (L0.6.0 correction).

## What this document does NOT claim

- Strategy C is not experimentally proven. No multi-phase run has been executed.
- Phase 2 on a weak Phase-1 base yields memorization, not capability (measured, L0.2).
- Phase 1 on the current qualified 1.59M-token pool cannot deliver capability (measured, L0.3/L0.4).
- Strategy C is viable only with owner-authorized data acquisition and compute.
