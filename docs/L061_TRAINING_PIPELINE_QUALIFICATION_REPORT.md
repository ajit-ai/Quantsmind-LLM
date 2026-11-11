# TAKSH L0.6.1 — Training Pipeline Qualification Report (DLCAE-Aware)

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `45f0dc912d6f995b4e5486c31d45847f11a97531` (L0.6.0-R1 baseline)
**Milestone type:** Engineering qualification only — **NOT a training authorization, NOT a capability claim.**
**Package version:** remains `1.0.0` (`taksh==1.0.0`) — not bumped, not a release.

---

## 1. Verdict

**L0.6.1 focused qualification: PASS.** All required focused tests pass, relevant
L0.6.0/R1 tests remain green, preservation anchors are byte-identical, and no
training, data acquisition, GPU use, publication, or Karkain execution occurred.

**What this pass means:** the *training-pipeline infrastructure* (configuration,
manifest, provenance, orchestration, checkpoint lifecycle, replay, DLCAE handoff,
Karkain contract, resource guards, observability) is implemented, tested on
synthetic fixtures or local artifacts, and fail-closed.

**What this pass does NOT mean:** TAKSH has **not** trained any model in this
milestone; **0/30** on the frozen Python benchmark remains the honest ceiling
(all milestones); Karkain remains **NOT ELIGIBLE / DISCOVERED**; the host is
**currently resource-blocked** for a full fine-tune (see §8); no capability is
claimed for any language. A passing engineering qualification is not a coding
capability demonstration.

---

## 2. Implementation summary (sections A–J)

### A. Canonical training configuration — `taksh-foundation-111m-training-v1`
- `configs/foundation/l0601_training.yaml` — schema `taksh.training_config/1`, compatible with the
  canonical foundation model (`taksh-foundation-111m`: vocab 5822, d_model 768, 12 layers,
  12 heads, 6 KV heads, FFN 3072, context 2048, tied embeddings, RoPE θ=10000, RMSNorm,
  110,658,816 parameters — unchanged, verified not redefined).
- `evaluation/l0601_training_config.py` — validator: schema identity, exact model-field equality
  with `configs/foundation/l060_foundation.yaml`, explicit dotted-key overrides with type
  coercion (unknown keys rejected), canonical hash `edf857db867329be7b2a174a1c6a48c5b921412db3b08424011c50935833ffc4`,
  reproducibility contract (fixed seeds, determinism required, status `NOT_YET_AUTHORIZED`).

### B+C. Multilingual training manifest + fail-closed provenance/licensing
- `evaluation/l0601_training_manifest.py` — schema `taksh.training_manifest/1`. Each entry carries:
  `id, language_id, class(language|technology), registry_status, dataset_id, source_version,
  fingerprint_sha256, tokenizer_id, tokenizer_version, token_count, sample_count,
  quality_status, excluded, benchmark_contaminated, split, acquisition_state, license,
  provenance{source,revision,license_note}`.
- Registry compatibility: `language_id` validated against the 53-language catalogue + supporting
  technologies in `configs/feasibility/l051_languages.yaml`; Karkain remains in the extension
  registry (`l060r1_dynamic_language_registry.yaml`), not in the 53.
- Fail-closed enforcement: unknown registry ids, unknown licenses (allowlist reused from
  `evaluation/l060_training_readiness.py`), missing provenance, malformed fingerprints,
  duplicate entry ids, conflicting shared fingerprints, non-integer counts → rejected with
  precise reasons. `benchmark_contaminated` is readable and enforced downstream; the example
  manifest is uncontaminated.
- Example manifest (`taksh-local-research-only-v1`, sha256 `0a7c45642295…`): **one** real,
  measured entry — `python_verified_l04`, 1,590,703 verified Python tokens (measured in L0.4,
  content fingerprinted, licenses verified per slice with contaminated slices quarantined).
  No fictional corpus records. The general-English geometry corpus was deliberately **not**
  added (it is not a language slice and would have required a fabricated class).
- Manifest sha: `0a7c45642295db7d2e37177686a6ee4e8eec689eca01c94157d89f6a8bad7c82`.

### D. Incremental learning & orchestration — `training/incremental.py`
- Strategies implemented and comparable (none declared superior): `equal`, `proportional`,
  `quality`, `hybrid` (proportional × quality weight, quality weights verified=1.0 /
  unverified=0.5 / unknown=0.1 / rejected=0.0).
- Deterministic under fixed seeds (Python `random.Random(seed)`); replayable batch sequences;
  exposure accounting per entry with `MAX_EXPOSURE = 4` — oversampling is **reported, never
  silently applied**. Demo run flags `python_verified_l04: 300` under a single-entry manifest
  (300 steps ≫ ceiling), which is exactly the intended alarm.
- Job manifest schema `taksh.incremental_job/1` (config sha, dataset fingerprint, strategy,
  seed, step budget, state `planned`); `resume_validation()` refuses mismatched
  config/tokenizer hashes and never overwrites. No training launched.

### E. Checkpoint lifecycle — `training/checkpoint_lifecycle.py`
- Atomic writes (`.tmp_partial` → fsync → `os.replace`); fingerprint = hash of
  `{model_cfg, tokenizer_sha, step, tokens_processed}`; required-key verification;
  interrupted-write detection (stale `.tmp_partial` → reject); tampered-fingerprint rejection;
  recovery audit records. All verification done on synthetic temporary fixtures.
- **No existing campaign checkpoint read, written, modified, renamed or migrated.**
  `runs/v180_sft/checkpoint_last.pt` remains byte-identical (§9).

### F. Catastrophic forgetting & replay — `training/replay.py`
- `forgetting_metrics(baseline, after, tolerance=0.05)`: per-language absolute/relative deltas,
  regression flags, mean relative delta, `catastrophic_forgetting_detected`.
- `ReplayBuffer` requires provenance (`id, language_id, source, license, provenance_note, hash`)
  — entries without it are rejected. Synthetic python/javascript fixture correctly flags a
  −5% python regression. Interfaces exist and are measured; **no forgetting prevention is
  claimed**. Rollback policy documented: halt adaptation, revert to last validated checkpoint.

### G. DLCAE → Learning Engine — `evaluation/l0601_dlcae_learning_interface.py`
- 10-stage ladder with explicit authorization boundaries:
  `DISCOVERED → COMPREHENSION_IN_PROGRESS → KNOWLEDGE_VALIDATED → ELIGIBLE_FOR_TRAINING →
  TRAINING_AUTHORIZED → TRAINED → FUNCTIONALLY_QUALIFIED | ENGINEERING_QUALIFIED`, plus
  `BLOCKED`/`ROLLED_BACK`. Illegal transitions rejected; reaching `TRAINING_AUTHORIZED` requires
  `owner_authorization_id`; qualification requires recorded evaluation evidence; `BLOCKED` only
  owner-unblockable.
- Walkthrough on the Karkain fixture stops at `KNOWLEDGE_VALIDATED`; `ELIGIBLE_FOR_TRAINING`
  refused ("owner authorization required before any training run"). Discovered material never
  becomes training data automatically.

### H. Karkain learning-integration contract — `configs/foundation/l0601_karkain_learning_contract.yaml`
- Identity `.kark`, registry linkage (extension registry, `DISCOVERED` — never capability),
  spec pinned **1.1.0**, compiler pinned **1.1.0 (windows/amd64, Stable Build)**, KIR v1 (SSA),
  `TAKSH-KARK-BENCH v1.0.0` referenced (0 tasks instantiated), model-only vs compiler-assisted
  always separate, `corpus_manifest: NOT_INSTANTIATED / zero_entries: true`.

**Measured KCC state on this host (unchanged, recorded, not bypassed):** `kcc.exe` exits
`0xC0000135`; all `karkain.exe` compile-pipeline commands (run/build/check/kir/lint/test) exit 3
with zero diagnostics; a freshly `go build ./cmd/karkain` behaves identically. Consequence:
**Karkain training eligibility = NOT MET; zero capability claims; no Karkain execution performed;
Karkain repository untouched.**

### I. Resource safeguards — `evaluation/l0601_resource_preflight.py`
- Extends `evaluation/resource_guard.py` (device allowlist = cpu-only, batch ≤16, context ≤2048,
  steps ≤300, threads ≤4, authorization-by-env-phrase, checkpoint-size cap). Fail-closed on CUDA
  / unknown device, oversized batch/context, unmeasurable memory, **and** when the estimated fp32
  working set (~16 bytes/param ≈ 1.65 GiB) exceeds currently available memory.
- **Measured at report time on this host:** total 4.2 GB, available ≈ 0.66 GiB (83–87% used) →
  `fits_available_memory: false`, **`authorized: false`**. This is a guard report, not a
  performance measurement or an authorization. Network/GPU/payment: not allocated, not queried,
  not provisioned.

### J. Observability aggregate — `evaluation/l0601_observability.py`
- Runs all L0.6.1 tools, classifies each as `IMPLEMENTED_AND_TESTED` / `SIMULATED` / `BLOCKED` /
  `NOT_YET_AUTHORIZED`, and emits `reports/l0601_observability.json` with explicit
  `host_resource_state: BLOCKED (estimated working set exceeds available memory at report time)`.
- 2026-11-11 run: 8/8 tools exit 0; all-exit-zero true; training/acquisition/GPU/publication all
  correctly classified `NOT_YET_AUTHORIZED`; Karkain `NOT ELIGIBLE`.

---

## 3. File inventory (L0.6.1-owned, all new)

| Area | Files |
|---|---|
| A | `configs/foundation/l0601_training.yaml`, `evaluation/l0601_training_config.py` |
| B+C | `evaluation/l0601_training_manifest.py` |
| D | `training/incremental.py` |
| E | `training/checkpoint_lifecycle.py` |
| F | `training/replay.py` |
| G | `evaluation/l0601_dlcae_learning_interface.py` |
| H | `configs/foundation/l0601_karkain_learning_contract.yaml`, `evaluation/l0601_karkain_learning_contract.py` |
| I | `evaluation/l0601_resource_preflight.py` |
| J | `evaluation/l0601_observability.py` |
| Tests | `tests/test_l0601_pipeline.py` |
| Reports | `reports/l0601_{training_config,training_manifest,incremental,checkpoint_lifecycle,replay,dlcae_learning_interface,karkain_learning_contract,resource_preflight,observability}.json` (9 files) |
| This doc | `docs/L061_TRAINING_PIPELINE_QUALIFICATION_REPORT.md` |

**No pre-existing file was modified.** Six pre-existing working-tree modifications
(`serving/code_chat.py`, `serving/local_inference.py`, `training/manifest.py`, `training/pretrain.py`,
`training/sft.py`, `taksh-hf-publication/upload_manifest.json` — CRLF normalization, pre-existing)
are **preserved untouched and not staged**. Nothing was deleted.

---

## 4. Manifest schema and policy (summary)

- Eligibility for inclusion is an explicit `acquisition_state` per entry:
  `candidate → internal_research → internal_verified → owner_approved`. Only `internal_verified`
  (vetted + fingerprinted + license-checked) entries would be usable, and still nothing trains
  without owner authorization.
- `candidate` entries remain candidates only; the L0.6.1 example manifest contains exactly one
  verified internal entry. **No external acquisition occurred or is implied.**
- Contamination: `benchmark_contaminated` flag + fingerprint conflict detection fail closed;
  L0.4's leakage scan already quarantined 4 contaminated slices (recorded in the L0.4 corpus
  PROVENANCE notes).
- Fingerprints: SHA-256, content-based; the validator checks format, uniqueness and conflicts —
  it never claims to fingerprint external corpora (no external data access).
- Dynamic registry compatibility preserved: 53 languages + Karkain (54th, extension registry),
  open-ended, `REGISTERED/DISCOVERED/COMPREHENSION_IN_PROGRESS/FUNCTIONALLY_QUALIFIED/
  ENGINEERING_QUALIFIED/BLOCKED`, regression can demote. Registration never implies capability.

---

## 5. Checkpoint recovery design (summary)

- Atomic single-filesystem rename; verification requires all of
  `{model, optimizer, step, config, schema=quantsmind.checkpoint/1, fingerprint}`; fingerprint
  mismatch (metadata altered, e.g. tampered `tokens_processed`) → reject; stale `.tmp_partial`
  (interrupted write) → reject; missing/corrupt → reject. Resume refuses mismatched
  config/tokenizer hashes. Recovery audit record always states
  `reject; start a fresh run (never overwrite)` on failure. Existing checkpoints are never read
  by the verifier; tests use `tmp_path` fixtures only.

---

## 6. Focused test commands and results

```
python -m pytest tests/test_l0601_pipeline.py -q     # 48 passed (16 required areas covered)
python -m pytest tests/test_l060_foundation.py tests/test_l060r1_architecture_alignment.py tests/test_m6_resources.py -q
                                                      # 61 passed (L0.6.0/R1 remains green)
python -m evaluation.l0601_observability              # 8/8 tools exit 0
```
Python: `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe` (3.14.7). CPU-only, small fixtures.

Test-to-requirement mapping: 1 config schema/validation · 2 manifest validation · 3 registry
compatibility · 4 provenance/license fail-closed · 5 contamination rejection · 6 deterministic
sampling · 7 distribution/exposure reporting · 8 job planning · 9 checkpoint integrity · 10
resume/recovery · 11 replay interfaces · 12 forgetting metrics · 13 DLCAE eligibility transitions ·
14 Karkain contract pinning · 15 resource preflight fail-closed (incl. memory starvation +
unmeasurable memory fixtures, injected deterministically) · 16 checkpoint/tokenizer preservation.

---

## 7. Preservation verification (before and after)

| Artifact | Method | Result |
|---|---|---|
| `evaluation/l02_python_bench_v1.json` | SHA-256 (streaming, working tree) | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — **unchanged** |
| `runs/v180_sft/checkpoint_last.pt` | SHA-256 (streaming, working tree) | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — **unchanged** |
| Tokenizer `tokenizer/taksh_code_bpe_v1.json` (committed artifact) | SHA-256 of `git cat-file blob HEAD:…` (LF content as committed) | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — **unchanged** |
| Tokenizer working-tree copy (CRLF) | SHA-256 of working-tree bytes | `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` — CRLF variant, matches prior milestone report; committed LF artifact is canonical |
| `configs/foundation/l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — **unchanged** |
| Original 53-language registry + Karkain extension + multilingual 64-task benchmark | not modified; L0.6.0/R1 tests green | unchanged |
| `taksh-hf-publication/` | only pre-existing user CRLF modification in `upload_manifest.json`; not staged | preserved |

Hashing method note: committed Git blob hashes and working-tree hashes are reported separately
because of the CRLF working tree; the committed LF artifact is the canonical tokenizer and is
compared against the git blob, never against the CRLF working copy. No Git blob hashes were
confused with content hashes.

---

## 8. Resource situation (measured, not estimated)

Total physical memory ≈ **4.2 GB**; at report time ≈ **0.66–0.71 GiB available** (83–88% used).
Estimated fp32 fine-tune working set ≈ **1.65 GiB** (16 bytes/param × 110,658,816; not a
performance guarantee). `fits_available_memory: false` → **preflight fails closed and prints
`authorized=False`.** Implications: any future authorized full fine-tune on this host would need
substantially more free memory, quantization/offload decisions, or a different host — an owner
decision. No throughput, price, or duration estimates are made. No GPU exists or was queried.

---

## 9. No-training / no-acquisition / no-publication attestation

- No model was trained, fine-tuned, warmed up, or evaluated-with-weights in this milestone.
- No dataset was downloaded, acquired, or ingested; no network access was used by any tool.
- No GPU was allocated or spent; no paid APIs were used; no money was spent.
- Nothing was published; no tags, releases, merges to `develop`/`main`, or Karkain repo changes.
- L0.6.1 is stopped; L0.6.2 is **not** started.

---

## 10. Remaining blockers and owner decisions

1. **Training authorization** — none granted; owner must explicitly authorize (env phrase +
   `--confirm`) before any run; preflight must be re-run and pass at that time.
2. **Host capacity** — current free memory insufficient for the naive fp32 working set; owner
   must decide on quantization/offload/host before any authorized run.
3. **Karkain toolchain** — `kcc.exe` `0xC0000135`; all compile-pipeline commands exit 3 with no
   diagnostics. Karkain remains `DISCOVERED`, contract qualified but execution blocked; needs a
   working compiler on the Karkain side (owner-owned repo) before any Karkain line of work can
   advance. This is an external blocker, not bypassed.
4. **Karkain corpus & bench instantiation** — `TAKSH-KARK-BENCH` has 0 tasks instantiated; a Karkain
   qualification bar must be set by the owner before any Karkain capability claim.
5. **Candidate datasets** — remain candidates only (see `configs/feasibility/l05_*` and
   PROVENANCE files); any acquisition requires owner sign-off and legal review.
6. **Multilingual benchmark** — denominator rule stands (≥16 executed tasks/language and ≥50%
   pass ratio); current qualified status of all 53-language lines is unchanged by L0.6.1.

## 11. Honest status summary

| Area | Status |
|---|---|
| Pipeline implementation + focused tests | **QUALIFIED** (48/48) |
| L0.6.0/R1 regression | **GREEN** (61/61) |
| Canonical model | unchanged, compatible, 110,658,816 params |
| Training authorization | **NOT YET AUTHORIZED** |
| Data acquisition | **NOT AUTHORIZED / NONE PERFORMED** |
| Training | **NOT PERFORMED** |
| Karkain | **NOT ELIGIBLE — DISCOVERED** (compiler broken on host) |
| Frozen benchmark ceiling | **0/30** (memorization 24/24 ≠ generalization) |
| Host resource preflight | **FAIL-CLOSED (authorized=False at report time)** |
| `develop` / `main` | **NOT TOUCHED** |

No TAKSH capability claim is made or implied by this report.
