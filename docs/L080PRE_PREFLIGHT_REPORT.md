# TAKSH L0.8.0-PRE — Foundation Training Preflight Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `84888e63720023a561a08dcc0efed4a694b79426`
**Milestone type:** preparation-only — **no training occurred and none is authorized**.
**Owner decision reference:** `TAKSH-OWNER-DECISION-2026-11-11` (A1.1-C preparation-only · A2-E no compute)

---

## 1. Verdict

**PREPARATION_QUALIFIED — preparation is qualified to proceed as far as preparation goes. TRAINING NO-GO — unchanged and absolute.**

The preflight demonstrates that the canonical foundation configuration is intact and measured,
the corpus is frozen and verified, the tokenizer is untampered, the benchmarks are preserved,
and the training-configuration proposal is complete with every unverified piece explicitly
flagged. Every owner- and environment-gated dimension remains the one and the same answer as
before: **training is not authorized and this milestone creates no authorization**.

---

## 2. Baseline and final git state (before/after)

| Item | Result |
|---|---|
| Starting HEAD | `84888e63720023a561a08dcc0efed4a694b79426` (unchanged) |
| Branch | `feature/l060-foundation-model-qualification` |
| Working tree at start | 100 entries; 6 pre-existing user CRLF modifications preserved byte-identical |
| L0.7.6 roadmap artifacts | preserved; `taksh-v3-roadmap.yaml` unmodified |

---

## 3. Owner decision record

`docs/L075_L080_OWNER_DECISION_RECORD.md` updated as filed by the owner. **Only two boxes checked:
A1.1-C (preparation/preflight only; no training) and A2-E (no compute authorization).**

- A1.1-A, A1.1-B remain unchecked; A2-A…A2-D remain unchecked; A2-D bounding fields remain unspecified and unauthorized.
- §6 now carries: date `2026-11-11`, decisions, the owner-supplied restrictions verbatim, and `Owner authorization reference: TAKSH-OWNER-DECISION-2026-11-11`.
- Explicit restriction text recorded: no training, no checkpoint creation/substitution, no acquisition, no GPU rental/spend, no paid APIs, no publication, no develop/main changes, no tags/releases.
- **A preparation authorization is never a training authorization** — recorded explicitly in the doc and enforced by the gate tests.

---

## 4. Model configuration identity

| Check | Result |
|---|---|
| `configs/foundation/l060_foundation.yaml` sha256 | `01a90464…e4a2fa3` — matches L0.7.0 anchor |
| Config identity (`id`/`name` = `taksh-foundation-111m`) | verified per file (name at document top level) |
| Architecture | Llama-style decoder-only; vocab 5822, d_model 768, 12 layers, 12 heads, 6 KV heads, FFN 3072, context 2048, RoPE θ=10000, RMSNorm, tied embeddings |
| Parameter count (verified) | **110,658,816** (formulaic verification from the config) |
| Architecture fingerprint | `92da3497cec631b4…` — matches L0.6.0 canonical fingerprint |
| `v180_sft` substituted for canonical? | **never** — classified GEOMETRY_INCOMPATIBLE vs `d_model 768` |

## 5. Tokenizer integrity

| Method | Value |
|---|---|
| Committed Git blob (LF, canonical) | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — matches L0.6.0 anchor |
| Working tree (CRLF) | `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` (intentionally different bytes; both reported, never conflated) |
| Vocab size | 5,822 (matches model config) |
| Tokenizer never rewritten, normalized or regenerated | confirmed |

## 6. Corpus snapshot and eligibility

| Check | Result |
|---|---|
| Snapshot `l072-l04-fdb1fee2beaae9c2` identity | verified (records_sha256 recomputed = matches snapshot) |
| Derived children | 1,052 |
| Eligible children | **1,050** |
| Eligible tokens | **1,380,611** (measured) |
| Eligible-set leakage across splits (duplicate groups) | 0 |
| Contamination overlap vs frozen benchmark | **0 exact matches** |
| Language coverage | python-only |
| Frozen verdict | FROZEN_RESEARCH_ONLY |
| frozen_for_training | false |
| Training-eligible tokens | **ZERO** |

Technical eligibility grants nothing beyond technical work. Legal/training eligibility unchanged.

## 7. Training configuration proposal

`reports/l080pre_training_config.json` — **complete, versioned, labelled, NOT_AUTHORIZED.**
Inherited (L0.6.1 canonical): AdamW(β1=0.9, β2=0.95, wd=0.1, clip 1.0), cosine LR, warmup 200,
min_lr_ratio 0.1, micro-batch 4×2048, grad-accum 8, seed 42, checkpoint every 500, eval every 100,
recovery gates. Everything unverified (Adafactor, early stop, full-run memory, CPU feasibility at
seq2048) is marked an unverified hypothesis. Owner authorization field: **NOT PROVIDED / NOT_AUTHORIZED**.
No training process was launched or attempted by any tool produced here.

## 8. Resource measurements vs estimates

Measured (16:0xZ 2026-11-11): 8 CPUs · RAM total 4,199,780,352 B · available 444,612,608 B (10.6% free) · disk ~170 GiB · MX150 4,096 MiB (driver 457.49 / CUDA 11.1, `torch.cuda.is_available()==False`, CPU build).
Estimated (never measured): fp32+Adam state **1,770,541,056 B (~1.65 GiB)** · bf16 variants smaller · activations several GiB at context 2048 · not a guarantee. Host cannot qualify.
**Verdict: NOT FEASIBLE on this host; minimum viable = GPU ≥16 GiB VRAM, ≥16 GiB RAM, ≥80 GiB disk, owner-authorized.**

## 9. Checkpoint and recovery readiness

- Synthetic-fixture lifecycle: **atomic write verified; interrupted-write detection verified; fingerprint-tampered checkpoint rejected** (verified in this milestone).
- Resume-gate contracts from L0.6.0/L0.6.1/L0.7.4 remain in force; optimizer/scheduler state required for real artifacts.
- Canonical 111M artifact: **does not exist**. `v180_sft`: GEOMETRY_INCOMPATIBLE. No artifact materialized; no existing checkpoint read/modified/substituted.

## 10. Evaluation and benchmark safeguards

- Frozen Python benchmark `854c7ca8…` intact; 64-task multilingual `7a839060…` intact; tasks/scoring/expected outputs untouched; **0/30 stands**.
- Read-only contamination probe vs frozen Python benchmark on the 1052 freeze records: **zero exact collisions** (contamination guard proven operational).
- Capability claim: none anywhere; compiler-assisted vs model-only separation unchanged.

## 11. Karkain status (unchanged)

Karkain remains `DISCOVERED` — zero corpus, zero benchmark tasks, `kcc.exe` `0xC0000135` on this host, compiler pinned to 1.1.0 (windows/amd64 Stable). No Karkain artifact, repository, or compiler was modified. No Karkain claim is made or implied.

## 12. CI inherited regression status

No L0.7.3/L0.7.4/L0.7.5/L0.7.6 file content was rewritten except one L0.7.6-era test,
`tests/test_l076_roadmap.py::test_owner_decision_record_remains_blank`, updated to assert the
authorized state recorded by the owner in this milestone (A1.1-C + A2-E only; everything else
unchecked). Seven inherited CI dispositions unchanged (RESOLVED by covering tests, all pass).
Focused milestone tests (`test_l080pre_preflight.py`, 29 pass) exercise model/tokenizer/corpus/
config/checkpoint/benchmark/gate integrity end-to-end. Focused chain L0.6.0→L0.8.0-PRE: **466 passed**.
Actions CI execution remains blocked on the default-branch policy (unchanged; owner action required).

## 13. Test commands and results

```
python -m pytest tests/test_l080pre_preflight.py -q     → 29 passed
python -m pytest <L0.6.0→L0.8.0-PRE chain> -q            → 466 passed
python -m pytest tests/test_l076_roadmap.py -q           → 28 passed (updated for authorized owner decisions)
python -m evaluation.l080pre.observability                → 8/8 exit 0
```

## 14. Changed files (L0.8.0-PRE-owned)

`docs/L075_L080_OWNER_DECISION_RECORD.md` · `docs/L080PRE_EXECUTION_BOUNDARY.md` ·
`docs/L080PRE_PREFLIGHT_REPORT.md` (this file) · `configs/l080pre/l080pre_preflight.yaml` ·
`evaluation/l080pre/{__init__,model_identity,tokenizer_integrity,corpus_verify,training_config,
resource_estimate,checkpoint_recovery,benchmark_safeguard,gate,observability}.py` ·
`tests/test_l080pre_preflight.py` · `tests/test_l076_roadmap.py` (authorized owner-decision alignment) ·
`reports/l080pre_{model_identity,tokenizer_integrity,corpus_verify,training_config,resource_estimate,checkpoint_recovery,benchmark_safeguard,aggregate,observability}.json`

**No prior milestone artifact modified.**

## 15. Preservation evidence (before and after)

| Artifact | Method | Result |
|---|---|---|
| Frozen Python benchmark | SHA-256 | `854c7ca8…` — unchanged |
| Multilingual benchmark | SHA-256 | `7a839060…` — unchanged |
| `v180_sft` checkpoint | streaming SHA-256 | `9b541036…` — unchanged |
| Tokenizer committed blob | git blob | `fbedf003…` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464…` — unchanged |
| 53-language registry | SHA-256 | `075d4b56…` — unchanged |
| L0.7.6 registry (`taksh-v3-roadmap.yaml`) | SHA-256 | `69a38917…` — unchanged |
| L0.7.6 roadmap docs | untouched | preserved |
| `taksh-hf-publication/` | untouched | preserved |
| develop / main | untouched | ✓ |

## 16. Actions not performed

- **No training, pretraining, fine-tuning or weight update**
- **No checkpoint created, modified, migrated or substituted; canonical artifact remains MISSING**
- **No external dataset acquisition, mirroring or ingestion**
- **No GPU rental, cloud provisioning, purchase or spend; no paid API/service use**
- **No publication, tag, release or PR; develop/main untouched**
- **No capability claim of any kind**

## 17. Preparation qualification

**PREPARATION_QUALIFIED** — the preflight machinery itself passed all 18 axes on the preparation side (canonical identity/configuration intact, tokenizer untampered, corpus frozen and verified, benchmarks preserved, reservices measured, safeguards demonstrated, gate fail-closed). This qualifies *preparation to the extent preparation is allowed to go*. It does **not** promote any training act to permitted.

## 18. Final training GO/NO-GO

**NO-GO.** Training is not authorized and cannot be authorized by this report.
A future authorization decision belongs solely to the owner and requires, in order:
1. written A1.1 authorization for a training run (or reaffirmation of A1.1-C with acceptance of preparation-only state), and
2. owner-provided compute authorization (A2 decision re-opened, or an explicit GPU authorization), and
3. an authorized training-eligible corpus (frozen & owner-authorized), and
4. a vetted checkpoint location and toolchain, and
5. every mandatory L0.8.0 entry-contract precondition independently verified before launch.

Until those conditions are met, the frozen benchmark remains 0/30, the checkpoint remains missing, and TAKSH remains not-a-proven coding intelligence.

---
One pre-existing L0.7.6 test (`test_owner_decision_record_remains_blank`) was updated to assert the authorized state recorded by the owner (A1.1-C + A2-E only, everything else unchecked). All other L0.6.x/L0.7.x artifacts are untouched; L0.7.6 remains recorded as IN_PROGRESS in `configs/roadmap/taksh-v3-roadmap.yaml` (ownership decision on finalizing its status left to the owner in a subsequent milestone).
