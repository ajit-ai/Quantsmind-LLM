# TAKSH L0.7.2 — Training Corpus Freeze & Eligibility Qualification Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `27c7b3a6317101e08884742a4b95732b8986c9cf` (L0.7.1)
**Milestone type:** corpus-freeze qualification — **NOT training, NOT a capability claim, NOT acquisition.**
**Package version:** remains `1.0.0` — not bumped.

---

## 1. Verdict

**FROZEN_RESEARCH_ONLY — the derived Python corpus is frozen as a quality-qualified,
reproducible engineering artifact. It is NOT training-eligible: owner authorization for a
training run is MISSING, and no authorization is granted by this milestone.**

- Technical processing: 1,050 of 1,052 derived samples are **TECHNICALLY_ELIGIBLE**; 2 were
  excluded as exact duplicates (canonical kept).
- Legal training eligibility: **CONDITIONAL** (internal research pretraining only, per the L0.4
  record — external/commercial checkpoint release requires fresh counsel review).
- Owner authorization for a training run: **MISSING** → aggregate
  `TRAINING_ELIGIBILITY_BLOCKED_OWNER_AUTHORIZATION`. **Zero training-eligible tokens exist
  today and none is fabricated here.**
- Every emitted sample satisfies the canonical **2,048-token** context limit; nothing was
  silently truncated; oversized material was subdivided structurally or quarantined (204,424 tokens).

---

## 2. Reconciliation of the L0.7.1 baseline (as demanded)

The milestone flagged the 344 vs 349 "train count" discrepancy as not obviously harmless.
It is accounted for deterministically — no error:

- L0.7.1's split report (`reports/l071_splits.json`) counted **all 382 records** including the
  4 synthetic fixtures and the 1 upstream `QUALITY_INVALID` record (truncated docstring),
  which landed in the train split: **349 = 344 real eligible + 4 synthetic + 1 INVALID**.
- The "eligible subset" report (manifest adapter) excluded synthetic + the INVALID record:
  **344 = 345 real train − 1 INVALID** (train tokens 1,468,951).

The L0.7.2 freeze recomputes everything after re-slicing and deduplication and preserves the L0.7.1
snapshot byte-identically (`artifacts/l071_l04_snapshot/` untouched; see §10).

Corpus reality discovered during re-slicing:
- `m3-train` slice: 189 single-file pytest files, no markers, parseable (except the truncated one).
- `taksh-repo` slice: 189 bundles of TAKSH's own repo; **every** record ends with standalone
  `<|doc|>` separator lines (188 records × 1 + 1 record × 3); whole-record parse fails by
  construction; **189 whole-parse successes vs 188 bundle repairs** checked by the quality adapter.

---

## 3. Phase A — source-aware re-slicing (deterministic)

Config: `configs/data/l072_reslice_config.yaml`; implementation: `evaluation/l072_reslice.py`.

Extraction hierarchy (actually applied):
1. **Standalone `<|doc|>` lines** (verified to exist as file delimiters) — marker-based split;
   191 separators across 190 into-3 record → 190 files per the hierarchy's first rung.
2. **Single records** kept whole (m3-train; 74 records).
3. **Oversized files** → CPython `ast` top-level units packed greedily into ≤2,048-token
   fragments (complete units only; order preserved; module header lands in fragment 0, never
   duplicated elsewhere; class bodies subdivided at **method boundaries** for oversized classes).
4. **Monolithic units >2,048** (giant functions/data literals) → quarantine. Never truncated.

**Monolith recovery:** 77 oversized *classes* (e.g. FastAPI's `APIRouter`, `FastAPI`; pytest's
`Pytester`) were subdivided at method boundaries — recovered ~434k tokens of content that
naive quarantine would have lost. 27 oversized methods, 32 monolithic units/functions,
5 giant literal files, 3 unparseable files remain quarantined (**204,424 tokens, 62 units,
all with reasons + parent references preserved**). One record was quarantined upstream
(L0.7.1 `QUALITY_INVALID`) and remains quarantined with reason `excluded_upstream`.

Lineage: every child carries `parent_sample_id`, `parent_original_fingerprint`, `file_index`,
`fragment_index`, `extraction_method`, segment line span, and an append-only
`transformation_history`. **No file boundary is invented.**

### Results

| Metric | Value |
|---|---|
| Parent records processed | 378 (377 usable + 1 upstream-quarantined) |
| Derived child samples | **1,052** |
| Eligible child samples (after exact-dup canonicalization) | **1,050** |
| Derived tokens (all children) | **1,380,638** |
| Eligible tokens | **1,380,611** (27 exact-dup tokens excluded: 12 + 15) |
| Quarantined tokens | 204,424 |
| Max child length | **2,043 tokens** |
| All within 2,048 context | **yes** |
| Extraction methods | structural_subdivision_from_single_record 638 · structural_subdivision_from_marker_split 247 · marker_split 93 · single_record 74 |
| Wall clock | ~1–2 min per rebuild on this host |

Token delta (1,590,326 → 1,380,638): separators removed + cut-boundary BPE shifts (≈377) +
quarantined monolithic content (204,424 ≈ 12.9%). **No tokens were imputed or hidden.**

---

## 4. Phase B — quality requalification

`VALIDATED_WITH_WARNINGS` 1,041 (mostly `no_license_metadata_in_file` — fragment-level absence
is expected and is **not** treated as a legal defect; license authority for the corpus is the
L0.4 `dataset_manifest.json` decision, preserved by reference in every child) +
`VALIDATED` 11 + **0** INVALID (all quarantined upstream or at re-slice).
**`no_license_metadata_in_file` is classified as a heuristic warning — never as a legal
conclusion, in either direction.** Encoding verified at ingest (UTF-8). Python fragments parse.
Builders keep: encoding / syntax / truncation / structural / secret / contamination / dedup /
provenance / language / extraction integrity. 2 exact-duplicate children excluded;
7 near-duplicates kept-and-flagged (group-locked into one split).

No record was deleted; exclusions preserve parent identity (see `artifacts/l072_l04_freeze/quarantine.jsonl`).

---

## 5. Phase C — dedup and leakage-safe splits

Exact: full-content fingerprints (children with identical text merged; canonical = smallest
token count then smallest fingerprint, deterministically) · near-dup: word 5-gram shingle
Jaccard ≥ 0.7 (11 groups flagged, none silently removed) · cross-language: n/a (Python-only).
**Split policy (child→parent→group union):** every child of one parent co-locates ALWAYS;
every child of one duplicate group co-locates ALWAYS; assignment seeded by
sha256(component-root) with 0.9/0.05/0.05 ratios — reproducible.

**Counts (real eligible corpus):**

| Split | Samples | Tokens |
|---|---:|---:|
| train | 952 | 1,249,311 |
| val | 51 | 68,863 |
| test | 47 | 62,437 |

No duplicate group spans a split (verified in `tests/test_l072_freeze.py`). Ratios were **not**
tuned to inflate any split. Language-stratified reporting emitted; **the only real language
is Python** — no multilingual split qualification is claimed. The L0.7.1 snapshot's split
assignments remain byte-identical (§10).

---

## 6. Phase D — canonical 2,048-token candidates

Every child satisfies the limit (max 2,043). Tokenizer: `taksh_code_bpe_v1.json`, committed
fingerprint `fbedf003…` preserved; no tokenizer change, no architecture change, no context
increase, no vocabulary retraining. Per-candidate metadata: token count, fingerprint, parent
id, source/provenance refs, extraction method, line span, split, quality class, dup group,
contamination, eligibility. **Context duplication introduced: zero** (no shared headers or
imported context copied across fragments — the class-statement duplicate for member-subdivided
fragments is the single documented exception and is bounded and counted in the method histogram).
Counting is per-record measured; nothing estimated.

## 7. Phase E — split qualification

Deterministic, reproducible, group-aware (see §5). Duplicate groups: **never span splits**.
No synthetic in real splits (freeze contains 0 synthetic records). Insufficient-coverage
languages: **all other 52 + Karkain report 0** (see §9).

## 8. Phase F — eligibility engine

16 dimensions, all recorded per sample (`artifacts/l072_l04_freeze/eligibility.json`).
**Training rights trace to the L0.7.0 decision record, not to this tool:**

| Dimension | Result | Basis |
|---|---|---|
| source_identity_verified | PASS | dataset_manifest.json audited L0.4 |
| acquisition_status_valid | PASS | ACQUIRED (internal l04) |
| license_decision_resolved | PASS | APPROVED_WITH_CONDITIONS (internal research) |
| training_use_permission | **CONDITIONAL** | "internal SFT/pretraining … previously authorized"; **no external/commercial release right** |
| provenance_complete | PASS | audited |
| owner_authorization_for_run | **MISSING** | this milestone and all others grant nothing |
| (all other 11 dims) | PASS | measured in-run |

Aggregate: **1050 TECHNICALLY_ELIGIBLE · 2 EXCLUDED · 0 TRAINING_ELIGIBLE · verdict
`TRAINING_ELIGIBILITY_BLOCKED_OWNER_AUTHORIZATION`** — nothing is training-ready; no
synthetic owner ID created; nothing bypassed.

## 9. Phase H + I — multilingual coverage and pipeline compatibility

Per-language table: `reports/l072_coverage.json` — **python 1,380,611 tokens / 1,050 samples;
all 52 other languages + Karkain 53/53+0 with zero tokens**. Karkain preserved: `.kark`,
SPEC/compiler pinned 1.1.0, kcc `0xC0000135` intact, DISCOVERED, **zero corpus**, repository
untouched, qualification never upgraded.
L0.6.1 pipeline integration: **manifest compatibility pass** with authorization gate
**CLOSED** (`reports/l072_training_manifest_check.json`): schema valid; distinct per-split
fingerprints; contamination CLEAN; max child 2,043; frozen Python benchmark intact; **no
training job started by this milestone or ever**.

## 10. Immutable snapshot & preservation evidence

**Snapshot `l072-l04-fdb1fee2beaae9c2`** at `artifacts/l072_l04_freeze/`
(`snapshot.json`, `records_meta.jsonl`, `quarantine.jsonl`, `eligibility.json`).
Identity derives from content fingerprints only (byte-reproducible rebuilds — re-runs match).
Parent lineage recorded (`l071-l04-8de91c22931af682` + its records_sha256).

| Artifact | Method | SHA-256 |
|---|---|---|
| Frozen Python benchmark | streaming hash | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` ✓ |
| Multilingual benchmark | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` ✓ |
| `v180_sft` checkpoint | streaming hash | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` ✓ |
| Tokenizer committed blob | git blob hash | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` ✓ |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` ✓ |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` ✓ |
| train.jsonl (freeze input) | SHA-256 | `cf89b8bc803b7363d2643e337a561524644e4924318c90332399f1625dfa1d5f` ✓ |
| L0.7.1 snapshot & records | SHA-256 (in l072 snapshot `parent_snapshot`) | verified unchanged ✓ |
| taksh-hf-publication/ | only pre-existing CRLF change (`upload_manifest.json`) | preserved, not staged |

Toolset note: one earlier full-chain test run rewrote the *volatile* committed
`reports/l070_resource_readiness.json` (live disk/memory numbers) via L0.7.0's aggregation
test; it was restored byte-identical from git. L0.7.0 artifacts remain committed-pristine.

## 11. Focused tests and regression

```
tests/test_l072_freeze.py            36 passed   (30 required areas)
full chain L0.6.0→L0.7.2             314 passed
python -m evaluation.l072_observability   5/5 tools exit 0
python -m evaluation.l072_snapshot        deterministic (identical snapshot id on re-run)
```

(Areas: re-slicing, lineage, structural preservation, oversized handling, 2048 enforcement,
no truncation, quarantine, tokenizer compat, exact/near dedup, parent- and group-level leakage,
split determinism/integrity, license conditions, owner fail-closed, technical-vs-legal,
synthetic exclusion, contamination, manifest compat, reproducibility, integrity, parent
immutability, registry preservation, Karkain preservation, resource limits, interruption
recovery, L0.7.1 preservation, benchmark/checkpoint preservation.)

## 12. Freeze verdict — the honest answer

| Layer | Status |
|---|---|
| Technically processable | 1,052 derived samples (1,050 eligible) |
| Quality-qualified | 1,041 VALIDATED_WITH_WARNINGS + 11 VALIDATED |
| Provenance-qualified | yes (all children) |
| Legally training-eligible (internal research) | **CONDITIONAL** — internal pretraining only, per L0.4 record |
| Owner-authorized (any run) | **MISSING** |
| Training-ready | **ZERO tokens** |

**Frozen as research candidate: YES. Training-eligible: NO. Both: NO. Neither alone: the
correct answer.** `FROZEN_RESEARCH_ONLY` is therefore the literal, defensible freeze verdict.
A future milestone with (a) owner authorization for a training run, (b) counsel sign-off if
the checkpoint's distribution scope is ever external, and (c) optionally a repaired KCC for
Karkain, could upgrade this verdict without re-running any code — re-freeze only for new data.

## 13. Blockers and owner decisions

1. Owner sign-off for any training use of the freeze (internal research pretraining, or explicitly beyond it).
2. External-distribution posture of any checkpoints trained on this corpus (needs fresh counsel review — L0.7.0 record).
3. Karkain `0xC0000135` toolchain — still blocking all Karkain corpus work.
4. External corpus candidates — all still `HUMAN_LEGAL_REVIEW_REQUIRED` (L0.7.0).
5. Frozen benchmark — 0/30 stands; not executed; must be used unchanged for any future evaluation.
6. M3 slice licensing nuance — the 189 pytest files carry per-record MIT/BSD-3/Apache-2.0; external release of any trained artifact requires mapping each used file to its license (attribution SBOM).

## 14. Attestation

No external data was acquired, downloaded, cloned, or scraped; no network data path exists in
any L0.7.2 tool; no training or fine-tuning occurred; no GPU was used; no paid APIs or storage;
no publication, tag, release, or merge to `develop`/`main`; frozen Python benchmark not executed
(0/30 stands); tokenizer and checkpoints byte-identical; Karkain repository untouched;
`develop` and `main` unchanged; `package.json` remains `1.0.0`.

**L0.7.2 stops here. L0.8.0 is not started. No training is started or requested.**
