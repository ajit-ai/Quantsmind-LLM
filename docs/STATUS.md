# Quantsmind-LLM — Execution Status Tracker

> **Direction update (2026-09-19):** project is now a **general multimodal (image ↔ text)
> model family** — finance is the first vertical, not the whole product. Track against the new
> [ROADMAP](ROADMAP.md) (Phases A–G) and [SCOPE_AND_PRINCIPLES](SCOPE_AND_PRINCIPLES.md).
> Phases 0–6 below are the legacy text-only plan, kept for continuity until it is re-mapped.

Living document. Update at every phase end; phase completion requires develop→main merge.
Legend: [x] done · [~] partially done / verify pending · [ ] not started

---

## PHASE 0 — Foundations (Weeks 1–2) — ~90%

- [x] 0.1 Repo scaffold, configs, .gitignore, branch flow
- [x] 0.2 Strategy locked: **from-scratch sovereign Indian LLM** (ADR-0003 FINAL; supersedes interim open-base ADR-000). Ladder: 300M pilot → 1.5B Quantsmind-Bharat → 7B when funded. Open models = benchmark rivals only
- [x] 0.3 Transformer implementation (RMSNorm/RoPE/GQA/SwiGLU)
- [x] 0.4 Pretrain loop: cosine LR, checkpoint/resume, NaN watchdog
- [x] 0.5 Tokenizer trainer + round-trip sanity tests
- [x] 0.6 Data tools: corpus generator, encode-to-tokens pipeline
- [x] 0.7 Unit tests + CPU smoke pipeline scripts
- [~] 0.8 GitHub Actions CI pushed — verify first run green on GitHub
- [x] 0.9 Promotion gate rules (evaluation/promotion.yaml)
- [ ] 0.10 Lint config (ruff) in CI
- [ ] 0.11 Formalize docs/adr/0001 files

### Preview track (pulled forward from Phase 3)

- [x] P0.a Finance chat UI (Gradio) — prompt → financial knowledge, disclaimer enforced
- [x] P0.b Pluggable OpenAI-compatible backend (works with Ollama/vLLM/OpenRouter today)
- [x] P0.c UI E2E verified on user machine (Gradio 6.25; demo mode confirmed 2026-08-24)
- [ ] P0.d Optional: connect real backend (Ollama qwen2.5) and verify grounded answers
- [x] P0.e India data-residency guard + SEBI-aware prompt (ADR-0002)
- [x] P0.f Competitive analysis vs India market (docs/COMPETITIVE_ANALYSIS.md) — strategy: sell engine, not app; RA partnership required; ~2-quarter window

Exit when: CI green on GitHub + 0.10/0.11 closed.

## PHASE 1 — Data v0 + 300M Pilot (Weeks 2–6) — ~55%

- [~] 1.1 Corpus pipeline built & verified (downloaders, chunking, quality, PII, dedup, stats)
- [~] 1.1b Corpus v0 built: 9,311 chunks / ~10M tokens (wiki-en 2000 docs, wiki-hi 500, Gutenberg econ) — scale-up to 12B tokens pending
- [x] 1.2 Tokenizer v0 trained: 49,152 BPE, round-trip OK, ~10M tokens encoded (tokenizer/qm_bpe_v0.json)
- [~] 1.3 Data card v0 published (data/DATA_CARD_v0.md); fuzzy dedup (MinHash-LSH) + eval-decontamination now live in pipeline — re-verify before 12B build
- [~] 1.4 India-finance eval set: 26/500 seed items, schema + loader tested
- [~] 1.5 LR probes + 1.6 pilot: FIXED notebook re-imported, **Run All executed on Kaggle** (experiments/pilot_run_001.md) — awaiting completion + artifacts
- [ ] 1.7 Eval battery on pilot + benchmark rivals (harness ready: evaluation/evaluate_pilot.py)

**Rs.0 spent to date.**

## PHASE 2 — Finance Adaptation (Weeks 4–8) — 0%

- [ ] 2.1 Real data ingestion + license/provenance ledger
- [ ] 2.2 CPT mix v1 (~10B tokens) + decontamination audit
- [ ] 2.3 LR probes → CPT run (full-FT or LoRA ladder)
- [ ] 2.4 SFT mixture build (50–150k pairs incl. calculator traces) + train
- [ ] 2.5 DPO pair generation + train
- [ ] 2.6 Safety pass: advisory-boundary refusals + red-team round 1
- [ ] 2.7 Exact-match arithmetic eval suite (hard-fail gate)
- [ ] 2.8 Pass promotion.yaml gate vs base

Exit when: finance scores up, arithmetic 100% via-tool, refusal >=95%, forgetting <=2%.

## PHASE 3 — B2B2C Product Layer (Weeks 8–10) — 0%

- [ ] 3.1 API gateway: tenant router, partner keys, quotas, billing hooks, SSE
- [ ] 3.2 RAG service: hybrid retrieval + market-snapshot context packer
- [ ] 3.3 Guardrails: injection filter, PII redaction, disclaimer injector
- [ ] 3.4 vLLM serving pool (AWQ-int4) + K8s manifests
- [ ] 3.5 Speculative decoding with self-trained ~150M draft model
- [ ] 3.6 Immutable per-message audit log + OTel dashboards

Exit when: partner-key request -> grounded disclaimed tool-verified answer E2E; load-tested 2x peak.

## PHASE 4 — Pilot Launch (Weeks 10–12) — 0%

- [ ] 4.1 Sign pilot B2B partner + contracted success criteria
- [ ] 4.2 Shadow-mode week -> canary 5% -> ramp
- [ ] 4.3 Weekly eval reports; feedback capture loop live
- [ ] 4.4 Model card + eval report; go/no-go review

Exit when: production SLOs met; partner sign-off.

## PHASE 5 — Compound & Wave-2 Decision (Q2) — 0%

- [ ] 5.1 Data flywheel -> monthly SFT/DPO refresh cadence
- [ ] 5.2 Hindi/Hinglish strengthening
- [ ] 5.3 Draft-model scale-up; context extension experiments
- [ ] 5.4 Wave-2 gate: corpus + eval suite + anchor customer all present

---

## M1.1 — Multimodal Architecture Fit-Check (synthetic, hermetic, CPU)

ROADMAP Phase A. **M1 NOT promoted — this is architecture evidence only.**

- [x] A1 Vision branch archive + design review (docs/ROADMAP.md · docs/DESIGN_ARCHITECTURE.md)
- [x] A2 VisionEncoder fit — from-scratch ViT (patch16 · 112×112 → 50 tokens · learned pos_emb) — unit tests green (`tests/test_vision.py`)
- [x] A3 Text↔Vision MM architecture fit — ViT → projector → QM decoder, MM attention mask, 49156-token vocab (4 specials appended), M0 text bit-identical after warm-start — unit tests green (`tests/test_multimodal.py` · `tests/test_m0_compat.py`)
- [x] A3 final report: 500-step CPU fit-check on real `local_15m` checkpoint — initial 10.747 ≈ ln(49156), final 0.2821 (< 8.0 target), grad norms non-zero/finite, conditioning probe 0.578, elapsed 1411 s @ ~345 tok/s (1 thread)
- [x] M1.1 committed: `feat(mm): complete M1.1 multimodal fit-check` (`4da502e`)

Reserved MM ids: 49152=IMG_START, 49153=IMG_END, 49154=CAPTION_START, 49155=TEXT_EOS; image-token vocab boundary 51200 (documented only, never allocated).

## M1.2 — Real-Data Preparation (infrastructure only, no training run)

- [x] 1 Data structure `data/mm_m1/{train,val,test}` + `MANIFEST.json` schema template + `DATA_CARD.rst`; image files gitignored, nothing auto-downloaded
- [x] 2 Provenance-aware manifest parser (`training/mm_manifest.py`): required source, honest null for unknown license/attribution/url, split/field validation
- [x] 3 Real image loader (`RealImageDataset` in `training/image_data.py`): MANIFEST → image file → RGB → deterministic resize/crop → M1 tensor → caption tokens; synthetic `ImageTextDataset` path preserved
- [x] 4 Deterministic preprocessing (no augmentation): RGB, cover-resize, center crop, [0,1] tensor, optional normalization
- [x] 5 Hermetic dataset tests (`tests/test_mm_manifest.py` · `tests/test_mm_real_data.py`): parsing, splits, loading, RGB, dims, captions, determinism, missing image, malformed manifest; tiny tmp fixtures only
- [x] 6 `configs/train/mm_m1_local.yaml` (explicit real-data config; `mm_fit_check.yaml` untouched)
- [x] 7 M0 checkpoint consumers preserved (training/pretrain.py · serving/local_inference.py · evaluation/evaluate_pilot.py); `metadata` is additive only
- [x] 8 Reproducibility metadata in checkpoints: config, seed, manifest hash, step, optimizer/model state, git commit (no network)
- [x] 9 Sphinx/RST docs added (`docs/source/index.rst` · `mm_m1.rst` · `conf.py`); M1 not promoted stated explicitly
- [x] 10 M1 promotion gate untouched (evaluation/promotion.yaml · ROADMAP); synthetic 10.747→0.2821 is NOT M1 evidence
- [ ] M1.2 commit pending (proposed boundary: the M1.2 delta above)

---

## Milestone Board

| Milestone | Target | Status |
|---|---|---|
| M0 Pipeline proven | wk 2 | IN PROGRESS (CI verification left) |
| M1 Champion base chosen | wk 4 | PENDING |
| M2 Finance-adapted model passes gate | wk 8 | PENDING |
| M3 Product demo E2E | wk 10 | PENDING |
| M4 Production pilot live | wk 12 | PENDING |
| M5 Refresh cadence + Wave-2 gate | Q2 | PENDING |
