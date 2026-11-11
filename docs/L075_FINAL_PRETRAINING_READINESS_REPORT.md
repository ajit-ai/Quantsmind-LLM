# TAKSH L0.7.5 — Final Pretraining Authorization & Execution Readiness Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD / ending HEAD:** `b47fafac17338382cee0e2ddb00dfe23b07d5922` (unchanged; one commit will be added)
**Milestone type:** final readiness & decision qualification — **NOT a training milestone.**
**Package version:** remains `1.0.0` — not bumped.

---

## 1. Verdict

**BLOCKED — L0.8.0 Pretraining must not start.** All engineering-workstreams of L0.7.5 pass;
every owner-dependent prerequisite is explicitly **MISSING, CONDITIONAL or NOT_AUTHORIZED**;
no amount of engineering flips the aggregate without owner action. The expected-and-honest
outcome is reproduced verbatim from the L0.7.5 gate:

**`BLOCKED` — training authorization is missing; training-eligible tokens remain zero; the canonical foundation checkpoint does not exist; the host cannot hold the training state; legal and distribution rights beyond internal research do not exist.**

---

## 2. Baseline verification

Branch `feature/l060-foundation-model-qualification` · HEAD == remote == `b47fafac…` ✓ ·
working tree 100 entries (6 pre-existing user CRLF modifications preserved) · L0.7.0–L0.7.4
reports/artifacts present; all L0.7.2 hashes re-verified below; `taksh-hf-publication/` untouched.

| Artifact | Method | Result |
|---|---|---|
| Frozen Python benchmark | SHA-256 | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| `v180_sft` checkpoint | SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged |
| Tokenizer committed blob | git blob hash | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` — unchanged |
| `data/l04_code_corpus/train.jsonl` | SHA-256 | `cf89b8bc803b7363d2643e337a561524644e4924318c90332399f1625dfa1d5f` — unchanged |
| L0.7.2 snapshot `l072-l04-fdb1fee2beaae9c2` | records re-hash → `records_sha256` recompute = match; parent lineage `l071-l04-8de91c22931af682` | **preserved, byte-identical** |

---

## 3. Workstream A — Owner-decision register (authoritative)

`configs/foundation/l075_owner_decision_register.yaml` — validated by `evaluation/l075_decision_register.py`
(9 entries, schema `taksh.owner_decision_register/1`, every entry carries `evidence_reference` + `owner_action`;
register/decision-summary consistency enforced; **zero owner approval fabricated**).

| ID | Question (abbreviated) | Status |
|---|---|---|
| A1.1 | Is any pretraining authorized? | **MISSING** |
| A1.2 | Internal research permission (L0.4 record) | **CONDITIONAL** — internal pretraining only; external release needs fresh counsel |
| A2 | Authorized compute environment available? | **MISSING** — MX150 4 GiB (CUDA 11.1, CPU-only torch) present-but-unusable; no authorization, no budget |
| A3.1 | Internal L0.4 corpus permission | **CONDITIONAL** |
| A3.2 | May external corpora be acquired? (BigCode v1/v2, CodeSearchNet, PyPI) | **HUMAN_LEGAL_REVIEW_REQUIRED** |
| A3.3 | May any checkpoint be distributed commercially/publicly? | **NOT_AUTHORIZED** |
| A3.4 | May artifacts be published (PyPI/HF/docs)? | **NOT_AUTHORIZED** |
| A4 | Can genuine CI execute on this branch? | **OWNER_ACTION_REQUIRED** — workflow invisible on non-default branch (HTTP 404), production CI has no dispatch (HTTP 422) |
| A5 | Full 53+Karkain scope preserved? | **GRANTED** (preservation, not expansion) — Python-only corpus is a *data limitation*, never a scope decision |

Decision summary (machine-readable in register): granted=[A5], conditional=[A1.2, A3.1],
missing=[A1.1, A2], not_authorized=[A3.3, A3.4], human_legal_review_required=[A3.2],
owner_action_required=[A4].

---

## 4. Workstream B — Corpus eligibility reconciliation

| State | Verified (2026-11-11) |
|---|---|
| Snapshot integrity | `l072-l04-fdb1fee2beaae9c2`; records_sha256 recomputed = `b4a127d637cf…`; frozen_for_training=false |
| Technical sample eligibility | 1,050 TECHNICALLY_ELIGIBLE / 1,380,611 tokens |
| Legal training permission | **CONDITIONAL — internal research pretraining only** (L0.4 record); external distribution needs fresh counsel review |
| Owner authorization | **MISSING** — zero runs, zero overrides, no synthetic sign-off anywhere |
| Training-eligible token count | **ZERO** |
| Multilingual coverage | Python only; 52 other languages + Karkain: zero qualified tokens |
| Model/tokenizer compatibility | PASS (artifact checks); tokenizer committed LF blob ✓; `l060_foundation.yaml` ✓ |
| Training configuration validity | PASS as a *specification* (L0.6.1 canonical config); never used to launch a run |

The aggregate is gated on every mandatory prerequisite — none of which is satisfied; hence the
verdict **BLOCKED** and training-eligible tokens remain **ZERO**. Nothing in this milestone converts technical eligibility into training eligibility.

---

## 5. Workstream C — Canonical model & checkpoint readiness

Canonical `taksh-foundation-111m`: vocab 5822 · d_model 768 · 12 layers · 12 heads / 6 KV heads ·
FFN 3072 · context 2048 · tied · RoPE θ=10,000 · RMSNorm · GQA · SwiGLU · **110,658,816 params**.

L0.7.4 spec (`configs/foundation/l074_canonical_checkpoint_readiness.yaml`) — unchanged, still valid.
L0.7.4 validator result re-confirmed: canonical 111M → **MISSING_ARTIFACT**; `v180_sft` →
**GEOMETRY_INCOMPATIBLE**; synthetic fixture → **SYNTHETIC_REJECTED**; nothing materialized; v180_sft
never substituted; existing checkpoints untouched (hashes above). Lifecycle (atomic write,
interrupted-write detection, fingerprint tamper rejection, resume validation) **qualified** and
unchanged — exercised by L0.6.1 tests, still in place for any future authorized run.

Checkpoint lifecycle **specification is ready**; the **artifact does not exist**; **training was not run**.

---

## 6. Workstream D — Compute feasibility (measured this milestone)

| Measure | Value | Label |
|---|---|---|
| CPU logical cores | 8 | measured |
| RAM total | 4,199,780,352 B (3.91 GiB) | measured (2026-11-11) |
| RAM available at measurement | ~0.19–0.21 GiB, 95% used | measured |
| Swap | 8.59 GB total / 6.38 GB free | measured |
| Disk free | ~170 GiB | measured |
| GPU | MX150, 4,096 MiB; driver 457.49; CUDA 11.1; `torch.cuda.is_available()==False` (CPU torch build) | measured |
| fp32 + Adam estimate for canonical 111M | ~1.77 GiB (16 B/param) | **estimate** |
| MX150 vs canonical 111M | 4 GiB VRAM, no bf16, no CUDA torch — unusable | measured + estimate |

**Qualification outcomes:** current host = **NOT VIABLE** (any strategy); CPU-only = not viable;
memory-efficient CPU = unmeasured/impractical; MX150 = unusable (toolchain mismatch + insufficient
as comfort path); owner GPU existence = **NOT QUALIFIED-AVAILABLE**; external compute = **owner-authorized-only, zero provisioning**.
**Minimum recommended (estimate, not qualified):** single GPU ≥ 16 GiB bf16 VRAM + ≥ 16 GiB system RAM + ≥ 80 GiB disk.
**This register does not create, rent, or simulate compute. No prices or durations invented anywhere.**

---

## 7. Workstream E — CI and Python runtime closure

**Seven inherited regressions — unchanged dispositions (all RESOLVED):** adapter availability ≠ transformers ·
deterministic RAM-policy tests · portable manifest paths · synthetic checkpoint geometry fixture ·
tokenizer/model geometry validation · S2 triple inspection · reproducible seeded decoding.
Re-verified via L0.7.4 focused suite: **197 passed, 1 skipped**.

**CI execution:** production workflow trigger = push-to-`develop`/`main` + pull_requests-to-those branches only;
no `workflow_dispatch`; `ci-l0704-manual-qualification.yml` present on this branch (read-only, tests-only) but
GitHub Actions exposes and dispatches workflows **only from the default branch (`main`)** — verified with
HTTP 422 (trigger) and HTTP 404 (visibility) on 2026-11-11. Genuine CI execution on this branch is blocked by
default-branch policy. **Production CI remains `CI_EXECUTION_BLOCKED_OWNER_ACTION_REQUIRED`.**
Owner action (either): push the qualification workflow into `main`, or add `workflow_dispatch:` to production
CI on `main`, or authorize+receive a PR from this branch. **No default-branch changes were made here.**

**Python 3.11 runtime:** static scan of 479 files at `feature_version=(3,11)` — zero real-code syntax
incompatibilities; `requires-python >= 3.10`; all declared dependencies support cp311. Runtime on 3.11
**NOT EXECUTED** — no interpreter installed; download unauthorized. Verdict remains **CONDITIONAL**
(static only). No claim that TAKSH is verified on Python 3.11 runtime.

## 8. Workstream F — Final readiness gate (18 dimensions)

`evaluation/l075_final_readiness.py` → `reports/l075_final_readiness.json`. One row per dimension with
status, evidence, last-verified date (2026-11-11), blocking dependency, and owner-vs-engineering responsibility:

| # | Dimension | Status | Responsibility |
|---|---|---|---|
| 1 | corpus_snapshot_integrity | **PASS** | engineering |
| 2 | data_provenance | **PASS** | engineering (internal); external = owner/counsel |
| 3 | technical_corpus_eligibility | **PASS** (1,050 / 1,380,611) | engineering |
| 4 | legal_training_permission | **PARTIAL** — internal research only | owner + counsel |
| 5 | owner_training_authorization | **MISSING** | **owner** |
| 6 | multilingual_language_coverage | **PARTIAL** — Python only (53+Karkain zero) | engineering fact; scope decisions owner's |
| 7 | tokenizer_model_compatibility | **PASS** | engineering |
| 8 | canonical_checkpoint_availability | **MISSING** | engineering spawn; blocked by owner |
| 9 | hardware_resource_feasibility | **BLOCKED** | **owner** (environment) |
| 10 | benchmark_thresholds | **PASS** — 0/30 stands | engineering |
| 11 | focused_regression_qualification | **PASS** — 7/7 dispositions hold | engineering |
| 12 | full_github_actions_ci_execution | **BLOCKED** | **owner** (default-branch action) |
| 13 | python_runtime_compatibility | **CONDITIONAL** — static only | owner (interpreter), engineering (verify) |
| 14 | reproducibility_and_recovery | **PASS** | engineering |
| 15 | training_artifact_provenance | **MISSING** | **owner**-authorized run → engineering execution |
| 16 | training_environment_authorization | **MISSING** | **owner** |
| 17 | external_checkpoint_distribution_permission | **NOT_AUTHORIZED** | **owner** + counsel |
| 18 | karkain_readiness_and_integration | **BLOCKED** | owner (Karkain repo), engineering otherwise |

Blocking dimensions (owner-dependent unless noted): 5, 8*, 9, 12, 15*, 16, 17, 18** (*spawn of owner deps, **Karkain toolchain)
Aggregate verdict: **BLOCKED** — every mandatory prerequisite is absent; no dimension was overridden.

---

## 9. Workstream G — L0.8.0 Entry Contract

`docs/L075_L080_ENTRY_CONTRACT.md` — explicit, checkable preconditions across Data · Model · Environment ·
Training · Evaluation with a closing list of prohibitions and a strict GO/NO-GO rule: **if any mandatory
condition is unmet, training must not begin — this contract does not check a single box by itself.**

## 10. Roadmap preservation (unchanged, unabridged)

Nothing was merged, deleted or abbreviated: L0 Foundation (L0.8.0–L0.10.3) · L1 Adaptive Coding Intelligence
(L1.0.0–L1.9.0) · L2 Repository-Aware Engineering (L2.0–L2.10) · L3 Advanced Autonomous Software Engineering (L3.0–L3.7).
Original 53-language catalogue and Karkain (54th first-class, DISCOVERED, pinned SPEC/compiler 1.1.0,
TAKSH-KARK-BENCH 0 tasks) are all intact; DLCAE's six engines and eight subsystems intact; L0–L3 ambition
stands at full strength. Python-only corpus is a data limitation, not a reduction of TAKSH's identity or roadmap.

---

## 11. Preservation evidence (before and after)

`taksh-hf-publication/`: untouched (only pre-existing CRLF modifications in `upload_manifest.json`, unstaged).
L0.7.0/L0.7.1/L0.7.2/L0.7.3/L0.7.4 artifacts: byte-identical (one volatile L0.7.0 resource report
re-overwritten by its own test suite was restored byte-identical from git). Working tree: the six pre-existing
user CRLF modifications in `serving/`, `training/`, `taksh-hf-publication/` remain unstaged. `develop` and
`main` untouched and unmerged; no tags, releases, publications or force-pushes; package version `1.0.0`.

---

## 12. Focused tests and CI

```
python -m pytest tests/test_l075_readiness.py -q     → 33 passed (new L0.7.5 suite)
python -m pytest <L0.6.0→L0.7.5 focused chain> -q     → 409 passed (no regressions)
python -m pytest <Seven inherited-failure files> -q   → 197 passed, 1 skipped
python -m pytest tests/test_l074_readiness.py -q      → 26 passed (also rerun post-L0.7.5 edit)
python -m evaluation.l075_decision_register           → register valid (9 entries)
python -m evaluation.l075_final_readiness             → 18 dimensions; verdict BLOCKED
Actions CI  : workflow probes recorded (HTTP 422, HTTP 404); genuine execution unavailable
             without owner default-branch action — integration is documented, not worked around
```

## 13. Exact files created / changed

**New L0.7.5-owned files:** `configs/foundation/l075_owner_decision_register.yaml` ·
`evaluation/l075_decision_register.py` · `evaluation/l075_final_readiness.py` ·
`tests/test_l075_readiness.py` · `docs/L075_L080_ENTRY_CONTRACT.md` ·
`docs/L075_FINAL_PRETRAINING_READINESS_REPORT.md` ·
`reports/l075_owner_decision_register.json` · `reports/l075_final_readiness.json`

**One pre-existing L0.7.4 file modified:** `evaluation/l074_checkpoint_readiness.py` —
one-line ordering fix so a synthetic fixture is reported as `SYNTHETIC_REJECTED` even before
file-existence is checked (previously a synthetic fixture would short-circuit to MISSING_ARTIFACT
regardless of the flag). Test suites re-verified green after the change.

## 14. Attestation

No training or fine-tuning occurred; no external data was acquired, downloaded, cloned, or ingested;
no GPU was used for anything; no paid API or storage; nothing published; no merge to `develop`/`main`;
no tags or releases; frozen benchmark not executed (0/30 stands); tokenizer and checkpoint artifacts
byte-identical; Karkain repository and compiler untouched; Python 3.11 runtime unverified (static-only);
`develop`/`main` unchanged; package version unchanged.

## 15. Remaining owner actions (in strict dependency order)

1. **Decide and record A1** (authorization for a training run) — until then nothing else matters.
2. **Decide and record A2** (compute environment), with data residency and storage boundary.
3. **Engage counsel** for A3.2/A3.3/A3.4 if any external acquisition or distribution is contemplated.
4. **Decide A4** (default-branch workflow change or PR) to obtain genuine CI verification on any future work.
5. Only then may L0.8.0 be considered, via its own milestone with preflight NO-GO → GO flip.

**L0.7.5 stops here. L0.8.0 is not started. Nothing authorizes training, acquisition or publication.**
