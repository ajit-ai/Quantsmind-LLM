# TAKSH 1.2.0 Training Authorization Packet

**Status:** `BLOCKED` - request for authorization only. Nothing in this document
authorizes training, compute, data acquisition, checkpoint creation,
evaluation, publication, merge, tag or release.

**Machine-readable inputs:**
- `configs/roadmap/l120_training_authorization.yaml`
- `reports/l120_training_authorization.json`
- `python -m evaluation.l120_training_authorization`

**Verdict:** `TRAINING_NOT_AUTHORIZED`; **ready to train:** `false`.

---

## 1. What is ready

| Gate | Status | Evidence |
|---|---|---|
| Model contract | `PASS` | `taksh-foundation-111m`, 110,658,816 parameters, config LF SHA-256 `fd96...` matches the historical L1.1.0 record |
| Tokenizer contract | `PASS` | `tokenizer/qm_code_bpe_v0.json`, vocab 5,822, SHA-256 `90fd...` matches the recorded artifact |
| Training engine source | `PARTIAL` | `training/pretrain.py`, `training/sft.py`, checkpoint/geometry contracts implemented; never executed at target scale |
| Data card | `PASS` (present) | `data/DATA_CARD_code_v1.md` |

These are source and identity checks. They are not evidence of model quality.

## 2. Why training is blocked

| Gate | Status | Exact blocker |
|---|---|---|
| Corpus licences | `BLOCKED` | 9 sources checked; 7 verified, 2 quarantined with `needs_human_review` (`psf-black`, `pypa-setuptools`). Required unresolved count is 0. |
| Corpus scale | `BLOCKED` | 2,239,771 tokens against a 100,000,000 target (97,760,229 short); 9 repositories against 20. |
| Corpus payload | `BLOCKED` | Manifest and token metadata exist, but `data/code_corpus_v2.jsonl`, `data/code_corpus_v2.tokens.bin` and the frozen snapshot `artifacts/l072_l04_freeze/snapshot.json` are absent on this ref. Payload files are gitignored. |
| Compute | `BLOCKED` | No owner authorization. Budget 0 USD, 0 authorized GPU-hours. Local host is CPU-only and not qualified for 111M training. |
| Checkpoint | `BLOCKED` | No canonical `taksh-foundation-111m` checkpoint exists; checkpoint creation is not authorized. `v180_sft` remains `GEOMETRY_INCOMPATIBLE` and must never be substituted. |

The earlier M2 gate already recorded the licence and scale blockers, and the
corpus manifest itself records that its holdouts are not statistically
sufficient for model evaluation. Neither issue can be closed by code.

## 3. Owner decisions required

| ID | Decision | Current status | Evidence required |
|---|---|---|---|
| `D-TRAIN-1` | Training authorization | `NOT GRANTED` | Signed decision naming run, scope and stop conditions |
| `D-COMPUTE-2` | Compute authorization | `NOT GRANTED` | Provider/hardware, budget ceiling, duration, data-egress rules |
| `D-CORPUS-3` | Corpus and licence authorization | `NOT GRANTED` | Human rulings for the two quarantined sources; frozen snapshot approval |
| `D-CHECKPOINT-4` | Checkpoint creation and storage | `NOT GRANTED` | Canonical target path, retention policy, access controls |
| `D-EVAL-5` | Post-training capability evaluation | `NOT GRANTED` | Frozen benchmark set, protocol and disclosure policy |

A code change cannot grant these decisions. The report fails closed if an
authorization flag is set while its corresponding decision is not `GRANTED`.

## 4. Exit criteria for 1.2.0

All of the following must hold before the gate can become `PASS`:

1. Every owner decision above is `GRANTED` and signed.
2. Licence gate has zero unresolved sources and a published data card.
3. Corpus reaches the 100M-token and 20-repository M2.3 targets.
4. A frozen corpus snapshot exists with recorded SHA-256 values.
5. Authorized compute records provider, budget and limits.
6. Config, tokenizer and 110,658,816-parameter identity still match.
7. A checkpoint is written atomically with a manifest and passes integrity and
   geometry load checks.
8. Training records seed, steps, tokens, finite loss and no NaN events.
9. Capability is measured only on frozen benchmarks with raw outputs preserved.
10. No capability, GA or publication claim is made before step 9 passes.

## 5. Regenerate the evidence

```bash
python -m evaluation.l120_training_authorization
python -m pytest tests/test_l120_training_authorization.py -q
```

The command exits non-zero while the gate is closed. That is intentional: a
closed gate must not look like a successful training authorization.

## 6. Non-actions

This packet does not:

- train or fine-tune;
- run an optimizer step;
- acquire or download a dataset;
- install packages or contact a provider;
- create, modify or serialize a checkpoint;
- run capability evaluation or re-score a benchmark;
- upload, publish, merge, tag or release anything.
