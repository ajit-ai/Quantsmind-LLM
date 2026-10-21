# Taksh - Versioning & Phase Matrix

> **M7 release target (2026-10-20).** Current repository/package metadata is
> `1.0.0`, the Apache-2.0 release for the completed M7 experiment. Publication
> is a separately authorised step recorded in the release-readiness report. The
> experiment was originally documented under the interim target `0.7.0`; that
> label is historical and was never released. The model and serving tracks below
> are unchanged; M7 is an experiment, not a Track 1 model promotion. See
> `docs/source/releases/M7_RELEASE_READINESS.rst`.

## Repository/package version policy

The authoritative version is `taksh.__version__` in
[`taksh/__init__.py`](../taksh/__init__.py). `pyproject.toml` reads it
dynamically; Sphinx and the OpenAI-compatible server import the same value. Do
not duplicate the version in package metadata or public version declarations.

| Range | Policy |
|---|---|
| `0.x.y` | Pre-1.0 development releases, only when deliberately authorised. Historical labels remain historical. |
| `1.0.0` | Intended first public stable package version, subject to explicit release approval. It is not a model-promotion claim. |
| `1.x.y` after `1.0.0` | Semantic versioning applies: patch releases contain backward-compatible fixes; minor releases add backward-compatible functionality. |
| `2.0.0` and later | Breaking changes require a major version and a documented migration path. |

The compatibility surface is the installed distribution metadata, the `taksh`
and `taksh-openai` entry points, the documented OpenAI-compatible HTTP
endpoints, and documented public Python entry points. Internal modules and
experiment artifacts are not promised as a stable API.

A real release requires the Git tag `vX.Y.Z`, the package version `X.Y.Z`, and
the GitHub release version to match. Publication is performed only after
explicit owner authorisation and a passing release-gate run. Historical
milestone labels such as `0.7.0` must not be retrospectively tagged or
published. The full preparation procedure is in
[RELEASING.md](RELEASING.md); it never publishes automatically.

> **Correction notice (2026-10-14).** Several claims in this file were wrong. Verified
> values are in the table below and the evidence column of the M0 table. Parameter
> counts are computed from the config dicts and cross-checked against the matching
> checkpoint `state_dict`. Steps are read from the checkpoint `step` field. Loss figures
> are marked unverified because no training log was retained for any code run — see
> [OPEN_ISSUES.md](OPEN_ISSUES.md) F2. Model config filenames were left unchanged to
> preserve experiment reproducibility; they do not describe parameter counts.

## Verified model sizes

| Config file | Filename implies | **Actual params** | Trained? |
|---|---|---|---|
| `configs/model/code_tiny.yaml` | tiny | **495,808** | yes — all shipped code checkpoints |
| `configs/model/code_30m.yaml` | 30M | **16,257,280** | 200 steps only |
| `configs/model/local_15m.yaml` | 15M | **7,767,680** | 10 steps only |
| `configs/model/pilot_300m.yaml` | 300M | **113,145,408** | no |
| `configs/model/prod_1p5b.yaml` | 1.5B | **1,315,014,656** | no |
| `configs/model/scale_7b.yaml` | 7B | **6,073,618,432** | no |

---

## Version Scheme

The project has **two independent tracks**. They release separately, and only one of
them is currently blocked. Treating them as one ladder (as this file originally did)
hides that most of the work is not gated on training compute at all.

### Track 1 — Model weights

**Format:** `qm-code-{size}-v{major}.{patch}`

| Tag | Meaning | Exists? |
|---|---|---|
| `qm-code-tiny-v0.1.0` | **495,808 params**, 200-step smoke checkpoint. Not a usable coder | **yes** (`runs/code_pretrain_tiny`) |
| `qm-code-tiny-v0.2.0` | Same size, trained on a real corpus with real held-out PPL | no — blocked on M1 |
| `qm-code-30m-v0.1.0` | 30M-named config, **actually 16,257,280 params** | no — 200-step smoke only |
| `qm-code-30m-v1.0.0` | First model that **passes the promotion gate** | no |
| `qm-code-100m-v1.0.0` | 100M params, single-GPU | no |
| `qm-code-pilot-300m-v1.0.0` | 300M-named config, **actually 113,145,408 params**, 8×A100 | no |

**A model tag `qm-code-*-v1.0.0` may only be issued when `evaluation/promotion.yaml`
is green and enforced.** Today it is neither: no benchmark has ever been run, and no
code reads the `rules:` block (Open Issue D4). No HumanEval or MBPP score exists in
this repository. The `taksh` repository/package release `1.0.0` is a
separate software release and is **not** a model promotion.

### Track 2 — Serving & integration (unblocked, CPU-only)

**Format:** `qm-serving-v{major}.{patch}`

| Tag | Contents | Status |
|---|---|---|
| `qm-serving-v0.1.0` | `GET /v1/models`, `POST /v1/chat/completions` (non-streaming), `GET /health`; loopback-by-default binding; optional `QM_API_KEY` bearer auth; no request-triggered checkpoint loading. **Real-checkpoint inference verified** (`tests/test_openai_real_checkpoint.py`, 20 tests, no mocks) | **ready to tag** |
| `qm-serving-v0.2.0` | SSE streaming (`data:` chunks + `data: [DONE]`) — Open Issue N8 | not started |
| `qm-serving-v0.3.0` | KV cache so generation cost stops scaling with context — Open Issue N6; plus removal of the blocking-`async` handlers (A5) | not started |
| `qm-serving-v1.0.0` | Serves a model that passes the Track 1 gate, with rate limiting and request timeouts (A8) | not started |

### Track 3 — Evaluation integrity (unblocked, CPU-only)

**Format:** `qm-eval-v{major}.{patch}`

| Tag | Contents | Status |
|---|---|---|
| `qm-eval-v0.1.0` | Empty-split and unavailable-benchmark guards; retracted false-perplexity artifacts; `status: "unmeasured"` / `promotion_eligible: false` stamping | **ready to tag** — `tests/test_eval_guards.py` green |
| `qm-eval-v0.2.0` | Full published HumanEval (164) and MBPP (500) suites replace the 5-problem smoke subsets | not started |
| `qm-eval-v1.0.0` | `promotion.yaml` enforced in CI; a promotion can pass or fail programmatically | not started |

### The single blocking dependency

Every Track 1 tag beyond `v0.1.0` is blocked on one thing: **scale**. The corpus
pipeline shipped on 2026-10-15 with verified provenance, but 2.28M tokens cannot train
a code model.
The shipped corpus is 100,244 tokens of this repository's own source, 0 TypeScript
(Open Issue E2). Perplexity is currently *not measurable* because the 1% held-out tail
is 1,003 tokens, shorter than one 2,048-token window. Track 2 and Track 3 do not depend
on it and can ship today.

---

## Phase Matrix

> **Numbering note (2026-10-18).** This matrix is the **model/version release
> ledger**. Its milestone letters are a legacy track numbering and do **not** match
> the authoritative execution roadmap in [ROADMAP.md](ROADMAP.md), whose M2 is
> *Dataset Expansion and Training Readiness* with slices M2.1–M2.12. Where the two
> disagree, ROADMAP.md governs. M2 acceptance governance:
> `docs/source/roadmap/M2_EXECUTION.rst`.

| Phase | Name | Track | Status | Gate |
|---|---|---|---|---|
| **M0** | Pipeline proven | model | COMPLETE | smoke train + CI green. **165 collected nodes at audit time; 256 now** |
| **M1** | Real code corpus | model | 🟡 **PIPELINE DONE, SCALE SHORT** — 2,277,473 tokens, 9 licence-verified repos, provenance per chunk. Target ≥100M, so M2/M3 stay blocked |
| **M2** | Code base model | model | PENDING — gated on roadmap **M2.12** (training-readiness release gate) | Measurable held-out PPL < 15. **Not measurable today** |
| **M3** | SFT-aligned coder | model | 🔴 BLOCKED (needs M1 + corpus rewrite) | HumanEval pass@1 ≥ 10%. SFT corpus quarantined: 423/442 pairs are boilerplate |
| **M4** | Serving & OpenAI API | serving | 🟢 **MOSTLY COMPLETE** | ✅ `GET /v1/models`, ✅ `POST /v1/chat/completions`, ✅ loopback default, ✅ bearer auth, ✅ no request-driven checkpoint loading. ❌ streaming, ❌ KV cache, ❌ rate limits |
| **M5** | Evaluation integrity | eval | 🟢 **MOSTLY COMPLETE** | ✅ false-pass guards, ✅ retracted artifacts, ✅ unmeasured stamping, ✅ CI regression tests. ❌ full 164/500 suites, ❌ enforced promotion gate |
| **M6** | Corporate fine-tune | model | PENDING | Real internal corpus. Currently 2,126 tokens from 5 demo files |
| **M7** | Scale to 100M | model | PENDING | Freezes Track 1 behind compute |
| **M8** | 300M-named pilot | model | PENDING, paid | 8×A100 rental; `pretrain_pilot.yaml` now resolves to a real token artifact (B4 fixed); its input artifact is **legacy-separated** and must be regenerated before use |

> No HumanEval or MBPP score exists in this repository. `evaluation/eval_code.py` carries
> 5-problem smoke subsets of 164 and 500 and stamps every report `status: "unmeasured"`,
> `promotion_eligible: false`. Any phase gate naming these benchmarks is unmet by default.

**Reading the matrix:** M4 and M5 are green and cost nothing but CPU time. M1, M2, M3
are all downstream of the same missing data. M7 and M8 additionally require money.

---

## Completed Milestones

### M0 — Pipeline Proven (2026-09-27)

| Component | Claimed here previously | Verified |
|---|---|---|
| Code corpus pipeline | "77 local files, ~100k tokens" | **77 kept chunks** (144 seen, 67 dedup-rejected), **100,244 tokens**. All `source: local:*` — this repo's own files. 68 py / 9 js / **0 ts** |
| Code BPE tokenizer | "5,817 vocab" | **5,822 vocab**, round-trip OK |
| Pretrain smoke test | "50 steps, loss 10.39 -> 7.42" | **step 200** of a 50-step smoke config (launched with `--max-steps 200`), `seq_len` **512**, ≈1 epoch. Loss figures **unverified** — no log retained |
| SFT pipeline | "439 pairs generated, 30 steps, loss 8.67 -> 8.55" | **442 pairs** in `data/code_sft_v0.jsonl`; checkpoint at **step 200**; loss figures **unverified**. Corpus now **quarantined** |
| SFT → base linkage | implied continuous | **Broken.** The SFT checkpoint is a 495,808-param `code_tiny` model, not the 16,257,280-param pretrain. `strict=False` hid it (Open Issue B3) |
| Evaluation harness | "HumanEval + MBPP + perplexity; promotion gate configured" | Harness exists. **No benchmark has been run.** Perplexity not measurable. `promotion.yaml` has no enforcing code (Open Issue D4) |
| Serving modules | "all import-verified" | Correct — import-verified only. See Open Issues C1–C5, A1–A6 |
| CI/CD | "165 tests green" | **165 collected nodes** at audit time; **256** after the guard, OpenAI-API and legacy-serving-security suites (2026-10-14) |
| Git | "All commits under Ajit Kumar; develop + main synced" | Single-author history; **not verified** in this pass |

---

## Milestone Detail

Numbering matches the Phase Matrix above.

### M1 — Real Code Corpus · PARTIAL — pipeline delivered, scale still short

Nothing in Track 1 moves until this is done. It needs network and disk, not compute.

As of 2026-10-15 a real, licence-verified external Python corpus exists:
**447 records / 2,277,473 tokens** (own code BPE), 9 pinned repositories, MIT +
Apache-2.0 + BSD-3-Clause, per-record attribution, deterministic content-hash splits
with 0 measured cross-split leakage. Built by `data/build_code_corpus_v1.py`; see
`data/DATA_CARD_code_v1.md` and `data/code_corpus_v1.manifest.json`.

| Task | Status | Blocker |
|---|---|---|
| Ingest real GitHub repos (permissive licences) | **DONE** | `data/build_code_corpus_v1.py`; allowlist in `configs/data/approved_code_sources.yaml`; live licence re-verified each run |
| Reach ≥100M unique tokens | **PENDING** | 2.28M delivered — 44× short. Needs many more approved sources; the allowlist cannot be widened without explicit approval |
| Add TypeScript | **PENDING** | 0 TS chunks. This phase is Python-only by specification |
| Record licence provenance per chunk | **DONE** | `repo`/`commit`/`spdx`/`attribution`/`retrieved_at`/`content_sha1` on every record; data card written |
| Populate `by_source` in the stats file | **DONE** | Manifest carries `counts.by_source` and per-source records |
| Verify the licence FILE, not just metadata | **DONE, WITH FINDINGS** | `data/verify_licenses.json`: 7/9 verified from file text. `psf/black` (GPL-2.0 subtree) and `pypa/setuptools` (AGPL-3.0 text) need a human decision — see OPEN_ISSUES H2/H3 |
| Confirm held-out split exceeds one window | **NOT MET** | Source-group splitting gives 1-2 repositories per holdout. Deterministic, but not statistically sufficient |
| Decontaminate against eval sets | **PARTIAL** | Audit found no overlap against the 10 problems in-repo. Published HumanEval/MBPP are **not available and were not checked** |
| Confirm 0 files are executed during ingest | **DONE** | Validation is `compile()` only; nothing is executed |

**Exit gate (unchanged):** ≥100M unique tokens, TypeScript present, licence provenance
recorded, held-out tail comfortably larger than `seq_len`. Only the provenance limb is met.

### M2 — Code Base Model

| Task | Status | Blocker |
|---|---|---|
| Full pretrain (10k steps) | PENDING | 655M tokens of budget against 100k tokens of data |
| Held-out PPL < 15 | **NOT MEASURABLE** | M1 |
| HumanEval pass@1 ≥ 10% | **UNMEASURED** | Needs the full 164-problem suite (M5) |
| Promotion gate pass | PENDING | Gate is declarative only (D4) |

### M3 — SFT-Aligned Coder

| Task | Status |
|---|---|
| Rewrite SFT extraction with `ast`, delete the three boilerplate generators | **prerequisite — E1** |
| SFT batching fix: only pairs `0..micro_bs-1` are sampled | **prerequisite — B1/B2** |
| `strict=True` on `--init-ckpt` so architecture mismatches surface | **prerequisite — B3** |
| SFT on 50k+ genuine code pairs | target: HumanEval pass@1 ≥ 10% |
| Refactoring targets must not assert edits absent from the response | hard requirement |

### M4 — Serving & OpenAI API · MOSTLY COMPLETE

| Task | Status |
|---|---|
| `GET /v1/models` reporting the real config name | done |
| `POST /v1/chat/completions` with roles, usage, `finish_reason` | done |
| Structured OpenAI error envelope | done |
| Loopback by default, `--allow-remote` opt-in | done |
| Optional `QM_API_KEY` bearer auth | done |
| No request-triggered checkpoint loading (A1) | fixed, 37 regression tests |
| Non-blocking CPU inference | done in `openai_server`; legacy servers still block (A5) |
| SSE streaming | not started (N8) |
| KV cache — generation cost stops scaling with context | not started (N6) |
| Rate limiting and request timeouts | not started (A8) |
| Latency SLOs < 200ms / < 2s | unreachable until N6 is done |

### M5 — Evaluation Integrity · MOSTLY COMPLETE

| Task | Status |
|---|---|
| Empty held-out split fails loudly instead of reporting PPL 1.0 | done, 14 tests |
| Seed summary cannot report a perfect 0.0 mean NLL | done |
| HumanEval/MBPP stamped `status: "unmeasured"`, `promotion_eligible: false` | done |
| False-perplexity artifacts retracted in place, originals preserved | done |
| Full published HumanEval (164) and MBPP (500) suites | not started |
| `promotion.yaml` enforced in CI (D4) | not started |

### M6 — Corporate Fine-Tune

| Task | Target |
|---|---|
| Internal repo ingestion | All internal Python/JS/TS repos |
| Continued pretrain on internal | 1–2B tokens, loss decrease |
| Internal SFT | Code review/QA pairs from internal patterns |
| Security audit | Zero secrets in training data |

> Shipped internal corpus is **2,126 tokens from 5 synthetic demo files**. Shipped
> `runs/corporate_*` checkpoints are 495,808-param models. The 1–2B token figure is a
> target, not an executed run. Secret scanning is 4 regexes (A7).

### M7 / M8 — Scale (paid)

| Task | Target |
|---|---|
| 100M model, single-GPU trainable | needs ≥10B tokens; freezes on compute |
| 300M-named pilot | 8×A100 rental, 100B+ tokens, MFU ≥ 40% |
| Rivals scored | DeepSeek-Coder, CodeLlama |

> `configs/train/pretrain_pilot.yaml` **cannot execute today** (B4). The Kaggle notebook
> is a fork of the architecture, not this repo's training code (E5).

---
## Version History

| Version | Track | Date | Description |
|---|---|---|---|
| `qm-code-tiny-v0.1.0` | model | 2026-09-27 | 495,808-param smoke checkpoint, step 200. Loss figures unverified. Superseded in practice — never was a usable coder |
| — | all | 2026-10-14 | **Truthfulness remediation.** Docs reconciled to artifacts; eval false-pass guards; SFT corpus quarantined; two false-perplexity reports retracted |
| `qm-eval-v0.1.0` | eval | 2026-10-14 | Empty-split and unavailable-benchmark guards, unmeasured stamping, CI regression tests. **Candidate for tagging** |
| `qm-serving-v0.1.0` | serving | 2026-10-14 | `GET /v1/models` + `POST /v1/chat/completions`; loopback default; bearer auth; `POST /load` removed from both legacy servers (A1 closed). **Candidate for tagging** |
| `0.7.0` | repository/docs | 2026-10-20 | Interim M7 documentation target. **Never tagged or released.** Historical only |
| `1.0.0` | repository/package | 2026-10-20 | `taksh` Apache-2.0 release target for the completed M7 experiment. Published only after explicit authorisation and passing release gates. **Not a model promotion and not a weight/data release** |

### Release readiness

The M7 release target is `1.0.0`. Packaging, clean-install, serving and
negative-capability evidence is in
`docs/source/releases/M7_RELEASE_READINESS.rst`. Publication is performed only
after explicit owner authorisation and a passing release-gate run; model weights
and training data remain private.

Two component tags remain immediately releasable because they depend only on
CPU-cheap work already in the tree:

```bash
# after review; both are test-covered and cost nothing to ship
git tag qm-eval-v0.1.0
git tag qm-serving-v0.1.0
```

The next model tag, `qm-code-tiny-v0.2.0`, is **not** reachable without first completing
M1 (a real corpus). Do not cut a model version that merely re-runs a 200-step smoke.
