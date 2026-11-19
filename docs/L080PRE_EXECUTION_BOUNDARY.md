# TAKSH L0.8.0-PRE — Execution Boundary and Design Document

**Milestone:** L0.8.0-PRE — Foundation Training Preflight (**PREPARATION ONLY**)
**Owner decision:** A1.1-C (preparation/preflight only) · A2-E (no compute authorization)
**Authorization reference:** `TAKSH-OWNER-DECISION-2026-11-11`
**Baseline:** `84888e63720023a561a08dcc0efed4a694b79426`
**Status rule:** A preparation milestone. It prepares the system for a future
training authorization. It never trains, updates weights, acquires data, or
spends money. A preparation authorization is NEVER a training authorization.

---

## 1. Purpose

Produce a reproducible, read-only, evidence-backed preflight that an owner can
review before (ever) considering an authorization decision for L0.8.0.

## 2. Execution boundary

**In scope (read-only preparation):**
- verify canonical model identity/configuration (`taksh-foundation-111m`);
- verify tokenizer artifact and vocabulary parity;
- verify tokenizers-to-model binding (fingerprints);
- verify frozen research snapshot integrity (`l072-l04-fdb1fee2beaae9c2`): identity, provenance,
  splits, duplicates, contamination screens vs frozen benchmarks, exclusion accounting;
- produce a **non-executing, versioned training-configuration proposal**;
- produce resource estimates (measured vs estimated; nothing measured beyond psutil);
- verify checkpoint/recovery contracts (reuse L0.7.x machinery);
- produce a machine-readable aggregate report and a human-readable GO/NO-GO (expected: **NO-GO for training**);
- run focused tests; commit nothing but milestone-owned files.

**Out of scope (prohibited, explicitly):**
- any training, pretraining, fine-tuning, weight update, or checkpoint creation;
- any checkpoint artifact substitution (no `v180_sft` substitution, ever);
- acquisition/download/ingestion of external data;
- compute rental/provisioning/paid APIs;
- modifications to frozen benchmarks, thresholds, tasks, or expected outputs;
- changes to `develop` or `main`; tags, releases, PRs;
- default-branch changes for CI; Karkain repository changes.

## 3. Design principles

1. Every artifact is READ-ONLY with respect to models, tokenizers, corpora and benchmarks.
2. Every machine-readable value is labeled **measured**, **estimated** or **inherited evidence**.
3. Fail-closed everywhere: missing data → NOT_AUTHORIZED/MISSING/BLOCKED.
4. Parent-child provenance flows forward; no silent repair is permitted.
5. Preparation QUALIFIED ≠ training AUTHORIZED. Verdicts are always reported separately:
   `PREPARATION_QUALIFIED` and `TRAINING_GO` are different axes with a hard wall between them.
6. v180_sft is never, in any code path, accepted as the canonical foundation checkpoint.

## 4. Module map (evaluation/l080pre/)

| Module | Purpose |
|---|---|
| `model_identity.py` | canonical config identity, parameter math, provenance, anti-substitution |
| `tokenizer_integrity.py` | tokenizer artifact hash (git blob vs working tree) + vocab parity |
| `corpus_verify.py` | snapshot identity, eligibility, splits, duplicates, contamination re-check |
| `training_config.py` | versioned non-executing training configuration proposal |
| `resource_estimate.py` | measured host vs estimated working set |
| `checkpoint_recovery.py` | lifecycle/recovery readiness from L0.7.4 checkpoint contract |
| `benchmark_safeguard.py` | frozen benchmark hash + contamination checks vs both benchmarks |
| `gate.py` | 18-dimension aggregate fail-closed validator; PREPARATION_QUALIFIED vs TRAINING_GO |
| `observability.py` | aggregate report |

No existing validator is duplicated; all prior machinery is reused by import.

## 5. Refusals and invariants (tests enforce)

- no training command can be invoked by any artifact produced here;
- any attempt to instantiate the `taksh-foundation-111m` initializer fails closed
  (no auto-creation of weights);
- the owner decision record can only make A1.1-C/A2-E greener, never others;
- the aggregate cannot produce TRAINING_GO while mandatory gates (authorization, checkpoint,
  hardware, legal eligibility) are unsatisfied;
- frozen benchmarks never enter training-related manifests or snapshots.
