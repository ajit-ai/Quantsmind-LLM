# TAKSH 1.2.0 — Model Capability Implementation Report

**Status:** Source preparation only. No training executed.

---

## What was implemented

- `evaluation/l120_training_readiness.py` — machine-readable training-readiness report generator.
- `taksh/coding_engine.py` (see 1.3.0) — task orchestration primitives (1.2.0 depends on them for future evaluation runs).
- `taksh/coding_validation.py` (see 1.3.0) — pre-flight validation layer reused by 1.2.0 for honest reporting.
- Checkpoint lifecycle qualification assets already present in repo (`training/checkpoint.py`, `training/geometry.py`, `training/tokenizer_contract.py`).

## Machine-readable readiness report

Generated: `reports/l120_training_readiness.json`

Summary:
- Foundation config: `taksh-foundation-111m`; 110,658,816 parameters — verified
- Tokenizer: `qm_code_bpe_v0.json`, vocab 5,822 — verified
- Corpus snapshot: `artifacts/l072_l04_freeze/snapshot.json` — `FROZEN_RESEARCH_ONLY` (this branch: absent by design)
- Frozen Python benchmark: `evaluation/l02_python_bench_v1.json` — `BLOCKED_ARTIFACT_ABSENT` on this branch
- Training engine: pretrain/SFT implemented; optimizer AdamW; scheduler cosine; **not executed**
- Authorization: A1.1-C preparation-only; A2-E no compute; training_authorized = false

## Explicitly not done

- No training run. No checkpoint creation. No data acquisition. No GPU/cloud access.
- No capability measurement against frozen benchmarks.

## Training authorization packet (Phase A3)

Added `configs/roadmap/l120_training_authorization.yaml`,
`evaluation/l120_training_authorization.py`,
`reports/l120_training_authorization.json` and
[`TAKSH-1.2.0-TRAINING-AUTHORIZATION-PACKET.md`](TAKSH-1.2.0-TRAINING-AUTHORIZATION-PACKET.md).

Current gate result: `BLOCKED`, `TRAINING_NOT_AUTHORIZED`. Model and tokenizer
identity checks pass. Corpus licences, corpus scale, corpus payload, compute
and checkpoint gates remain blocked. Exact blockers and owner decisions are in
the packet; no training was executed.

## What would unlock 1.2.0

Owner-written decisions `D-TRAIN-1`, `D-COMPUTE-2`, `D-CORPUS-3`,
`D-CHECKPOINT-4` and `D-EVAL-5`, plus the exit criteria in the packet.

## Preserved

Same freeze list as 1.1.0: v180_sft GEOMETRY_INCOMPATIBLE; benchmarks intact; owner CRLF preserved; develop/main untouched by this milestone.
