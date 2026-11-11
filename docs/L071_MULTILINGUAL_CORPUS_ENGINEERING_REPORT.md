# TAKSH L0.7.1 — Multilingual Corpus Engineering Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `da16051f810fe6a99e7564ad9f4c118a4e1340dc` (L0.7.0)
**Milestone type:** corpus-engineering qualification — **NOT training, NOT external acquisition, NOT a capability claim.**
**Package version:** remains `1.0.0` — not bumped.

---

## 1. Verdict

**L0.7.1 QUALIFIED (engineering).** The complete technical pipeline for future multilingual
corpus construction now exists, is deterministic, fail-closed, and proven end-to-end on the
one authorized real corpus in this repository. **External acquisition remains disabled; no
training-eligible corpus for training exists yet (owner authorization outstanding); the frozen
0/30 benchmark ceiling stands; zero capability is claimed.**

- Real corpus **processed and improved**: TAKSH internal l04 Python corpus (378 code records) — integrity-verified at ingest (train.jsonl SHA-256 matches `dataset_manifest.json`), parsed, measured, deduped (378/378 unique), contamination-screened (**CLEAN**), deterministically split (dup-group-aware), snapshotted.
- **1 INVALID record** (truncated at extraction: its docstring was cut mid-way by the corpus builder) was detected, quarantined (`QUALITY_INVALID`) and excluded — 377 parseable records are the clean pool.
- Synthetic fixtures remained explicitly synthetic throughout; nothing was counted as corpus.

---

## 2. Result summary

| Quantity | Measured value | Label |
|---|---|---|
| Real code records (train split of `data/l04_code_corpus/train.jsonl`, `domain=code`) | 378 (377 parseable + 1 INVALID) | REAL_AUTHORIZED_CORPUS |
| Per-record real tokens (canonical tokenizer, this run) | **1,590,326** | REAL, measured |
| avg tokens/record | 4,207.21 | measured |
| max sequence length | **75,326 tokens (211/378 samples exceed the 2,048-token context)** | measured — see §8 |
| Whole-corpus-stream tokens (dataset_manifest.json, includes doc separators) | 1,590,703 | measured upstream (L0.4) — differs by exactly 377 separator tokens (378 docs − 1) |
| Unique content fingerprints | 378/378 | measured |
| Near/exact duplicates | **none found** | measured (exact + 5-gram shingle Jaccard ≥ 0.7) |
| Benchmark contamination | **CLEAN** (0 exact, 0 near) | measured; frozen bench `854c7ca8…` intact |
| Quality | VALIDATED 2 · VALIDATED_WITH_WARNINGS 375 · INVALID 1 (excluded) · NO_AUTOMATED_FLAGS 1 synthetic | measured |
| Splits (382 records incl. 4 synthetic) | train 349 · val 17 · test 16 (dup-group-aware, seeded) | measured; no duplicate group spans splits |
| Eligible training subset (real, unexcluded) | **344 train samples / 1,468,951 tokens** · val 17 / 57,604 · test 16 / 61,346 (sum 1,587,901) | measured |
| Synthetic fixtures | 4 (python/karkain/javascript/c) — never eligible | SYNTHETIC_TEST_FIXTURE |
| Languages with real data | python only (52 + karkain: 0) | measured |

**Snapshot:** `l071-l04-8de91c22931af682` (content-derived id) — `artifacts/l071_l04_snapshot/snapshot.json`
+ `records_meta.jsonl` (metadata-only records; content never duplicated; verifiable references retained).
`frozen_for_training=false` (acquisition + owner gates outstanding — by design).

---

## 3. What was implemented (new L0.7.1-owned files)

**Configs:** `l071_corpus_record_schema.yaml` · `l071_normalization_policies.yaml` · `l071_pipeline_config.yaml`
**Tools:** `l071_corpus_record.py` (canonical record, l071-corpus-record/1) · `l071_ingest.py`
(adapters: authorized internal Python, synthetic fixtures, unapproved-source rejection before
any content read) · `l071_quality.py` (parser adapters: Python `ast`; all other languages
honestly NOT_CHECKED; secret/malware flags → HUMAN_REVIEW_REQUIRED) · `l071_dedup.py` (exact +
normalized + 5-gram shingle near-dup, groups, canonical selection, audit, contamination import
from L0.7.0) · `l071_stats.py` (tokenizer-integrated stats + language-family distribution;
synthetic separated) · `l071_splits.py` (seeded, duplicate-group-aware) · `l071_packing.py`
(context-2,048 packer; oversized rejected never truncated; traceable boundaries) ·
`l071_snapshot.py` (atomic writes; integrity hashes; eligibility gates) · `l071_manifest_adapter.py`
(L0.6.1 training-manifest compatible; fail-closed) · `l071_dlcae.py` (10-step flow audit) ·
`l071_karkain.py` (.kark adapter schema, synthetic fixtures only) · `l071_resource_integrity.py`
(host resources + pinned-hash verification) · `l071_observability.py` (aggregate; all 10 tools
exit 0)
**Tests:** `tests/test_l071_corpus.py` — 49 passed (30 required areas)
**Artifacts:** `artifacts/l071_l04_snapshot/{snapshot.json,records_meta.jsonl}`
**Reports:** `reports/l071_*.json` (observability, ingest, quality, dedup, stats, splits, packing,
manifest check, DLCAE, Karkain, resource/integrity)
**Doc:** this file

**Reuse:** L0.6.1 (`l0601_training_manifest.validate_manifest`, `_tokenizer` pattern),
L0.7.0 (`l070_contamination`, `l070_license_qualification`, `l070_source_registry`,
`l070_provenance`, `l070_corpus_planner`), `l051_language_registry` — imported, never duplicated.
No pre-existing file was modified.

---

## 4. Canonical corpus record

Schema `taksh.corpus_record/1` (`configs/data/l071_corpus_record_schema.yaml`): 33 fields —
sample_id (uuid5 source+fingerprint+index), source identity/version/snapshot, original
fingerprint (raw bytes, immutable), normalized fingerprint, language + family + technology
class, license id + decision ref (L0.7.0 records), provenance ref, tokenizer
id/version/fingerprint, token_count (measured), quality/duplicate/contamination/eligibility/
split status, exclusion reason, pipeline + normalizer version, transformation history
(append-only). **Provenance is never rewritten by transformation**; content is never
duplicated into the registry — `original_ref` + fingerprint = the verifiable reference.

## 5. Corpus quality and the bundle finding

The internal corpus's `taksh-repo` slice is a **concatenation of entire repository files glued
with standalone `<|doc|>` separator lines**. The quality engine handles this honestly:
whole-file `ast.parse` → if it fails, split at exact trimmed separator lines → parse each
segment. Result: 2 VALIDATED + 375 VALIDATED_WITH_WARNINGS (nearly all are the
`no_license_metadata_in_file` heuristic, since per-file markers are absent in that slice —
legal state is tracked at the corpus level via L0.4's manifest, see §10) + 1 INVALID
(truncated docstring; **excluded**) → **quality gate passes on the remaining pool**.
Two records contained interior U+FEFF (encoding artifact; removed for the parse probe;
logged in transformation history). TAKSH's own slice also explains records with a 75k-token
length: entire repos are single records.

## 6. Deduplication and contamination

Exact fingerprint: 378/378 unique. Near-duplicates (5-gram shingle Jaccard ≥ 0.7): none
merged (corpus builder already removed 41 exact + 1 near per its manifest). Cross-language:
n/a (Python-only real corpus). Every exclusion is audited (`exclusion_audit`); benchmark
fingerprint screen (L0.7.0 registry: l02 bench + 64-task multilingual + KARK schema, SHA-256
re-hashed at run time) → **CLEAN, 0 collisions**. Frozen benchmarks untouched (§9).

## 7. Tokenization and statistics

Canonical tokenizer only (`taksh_code_bpe_v1.json`; committed blob
`fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7`). Measured this run:
**1,590,326 per-record tokens** (avg 4,207; max 75,326; median bucket 257–2048). The upstream
1,590,703 (dataset_manifest.json) measures the concatenated stream including doc separators
(377 delta = exactly the 377 inter-document boundaries). Both numbers are real; **the
per-record figure is the training-sample accounting**; both are recorded. No counts were
inferred from file sizes; nothing was extrapolated. 211/378 records exceed the 2,048 context —
they are honestly oversized and are **not compatible** with the canonical window until the
corpus slices at `<|doc|>` boundaries (a future data task, documented in §13).

## 8. Splits and packing

Seeded assignment (sha256 of duplicate group; no RNG state) with dup-group integrity:
train 349 / val 17 / test 16; no group spans splits; language-stratified reporting emitted.
Sequence packer (greedy, order-preserving, EOS-delimited, 2,048 limit): tested on synthetic
fixtures; oversized samples **rejected and accounted, never truncated**;
`pack_manifest_sha256` deterministic. **No model training ran.**

## 9. Preservation evidence (before → after)

| Artifact | Method | Result |
|---|---|---|
| Frozen Python benchmark | SHA-256 | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| v180_sft checkpoint | streaming SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged |
| Tokenizer committed blob (LF) | git cat-file | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` — unchanged |
| Karkain extension registry, L0.6.x/L0.7.0 reports, taksh-hf-publication/ | untouched/not staged | preserved |
| `develop`, `main` | untouched | ✓ |

No prior milestone report was modified; new paths only. `package.json`/version = `1.0.0`.

## 10. Real vs synthetic vs candidate accounting

`REAL_AUTHORIZED_CORPUS` = 378 records / 1,590,326 tokens (internal l04, `APPROVED_WITH_CONDITIONS`,
**internal research use only**; external distribution prohibited) · `SYNTHETIC_TEST_FIXTURE` = 4
records / 65 tokens (python 16, karkain 13, javascript 17, c 19 — all INELIGIBLE) ·
`CANDIDATE_SOURCE` = BigCode v1/v2, CodeSearchNet, PyPI (**not ingested, zero bytes read**) ·
`UNVERIFIED_SOURCE` = all other languages · `BLOCKED_SOURCE` = Karkain corpus (zero; `0xC0000135`) ·
`BLOCKED_SOURCE`/training = everything until owner authorizes.

## 11. DLCAE and Karkain

DLCAE: all ten flow steps wired and audited (`reports/l071_dlcae.json`): registry lookup →
source discovery → qualification lookup (L0.7.0) → comprehension metadata (lookup only) →
normalization (l071-norm/1) → sample generation (378 records) → provenance/quality validation →
**eligibility decision: NOT_DETERMINED (owner authorization outstanding)** → stats feedback →
evaluation feedback (**NOT RUN; frozen bench unchanged**). Distinct states preserved; registry
never auto-upgraded; nothing functionally or engineering qualified.

Karkain: `.kark` adapter schema implemented and qualified (pins: SPEC 1.1.0, compiler 1.1.0
windows/amd64 Stable, KIR v1 SSA; TAKSH-KARK-BENCH 0 tasks; status DISCOVERED); 3 synthetic
fixtures label SYNTHETIC_TEST_FIXTURE; `kcc.exe` blocker (`0xC0000135`, exit-3 pipeline)
unchanged and **unbypassed**; zero counts claimed; no Karkain repository changes.

## 12. Training-manifest compatibility

`evaluation/l071_manifest_adapter.py` derives a `taksh.training_manifest/1`-compatible manifest
from the snapshot: entries `python_verified_l04_{train,val,test}` with **distinct**
split-derived fingerprints (an earlier draft shared the corpus hash and was correctly rejected
by L0.6.1's conflict check — fail-closed working as intended), distinct token contexts (train
1,468,951 / val 57,604 / test 61,346), all L0.6.1 validators pass, contamination flag false,
fingerprints re-hashed, acquisition internal_verified. Nothing is training-eligible; no training
was launched; no training is authorized.

## 13. Resource and integrity

Host measured: 170.4 GiB free disk; 3.91 GB RAM; processing floor 128 MiB; observed available
~140–150 MB under load (tight but sufficient for the 378-doc corpus; a memory-starved
`_FakeVM` fixture proves the pipeline fails closed instead of processing). Atomic writes
(mkstemp+fsync+os.replace) for both artifacts; no temp leaks; no network path exists; no
paid storage; no GPU. Integrity tool re-verifies all pinned hashes on every run
(`reports/l071_resource_integrity.json`). QA: **278 tests pass** across L0.6.0/R1/L0.6.1/L0.6.2/
L0.7.0/L0.7.1; L0.7.1 suite **49/49**.

## 14. Remaining blockers and owner decisions

1. **Corpus shape** — l04 bundles contain whole repos per record; 211/378 exceed the canonical
   2,048-token window. Future milestone: split at `<|doc|>` into per-file records (needs its
   own qualification) before any training use.
2. **Acquisition authorization** — all external candidates remain `HUMAN_LEGAL_REVIEW_REQUIRED`
   (L0.7.0); without counsel sign-off and owner authorization, acquisition stays impossible.
3. **Training eligibility** — snapshot `frozen_for_training=false` by design; owner must (a)
   authorize acquisition/internal expansion, (b) authorize the training run, (c) set the
   KARK-BENCH bar + repair KCC before any Karkain corpus work.
4. **Python breadth** — internal l04 train split is 1.47M tokens; owner decision needed on
   expanding it vs acquiring external Python (pre-review).
5. **Frozen benchmark** — 0/30 stands; never executed here; must be used unchanged for any
   future evaluation.

## 15. Attestation

No external dataset was acquired or downloaded; no network data access, scraping, cloning, or
PyPI/gated access occurred at any point; no training, fine-tuning, GPU use or spend; no paid
APIs/storage; no publication, tag, release, or merge to `develop`/`main`; no Karkain-repository
modification; no benchmark or tokenizer or checkpoint alteration. `develop` and `main` remain
untouched. Package version `1.0.0`.

**L0.7.1 stops here. L0.7.2 is not started.** TAKSH is not a proven coding intelligence; this
milestone qualifies the corpus pipeline, not the model.
