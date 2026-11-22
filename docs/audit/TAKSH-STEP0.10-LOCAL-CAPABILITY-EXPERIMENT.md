# TAKSH — Local Capability Experiment (Step 0.10)

**Date:** 2026-11-20
**Status:** `COMPLETED — NEGATIVE RESULT` — 0/20 correct in every condition tested; no capability claim.
**Authorization:** owner-approved bounded CPU experiments, local corpora only, no money, no external data.
**Related evidence:** `docs/audit/TAKSH-CAP-CORPUS-PROVENANCE.md`, `evaluation/native_v01/baseline_results.json`.

---

## 1. Question

Can a 28.8M-parameter TAKSH checkpoint, trained only on locally available open
code and a locally authored SFT set, move the immutable Native v0.1 suite away
from **0/20 correct**?

The experiment does not assume yes. A negative result is a valid outcome and is
recorded as such.

## 2. What changed in the evaluation

The frozen baseline feeds each task prompt to the model **raw**:

> `Write a Python function \`add(a, b)\` that returns the sum of \`a\` and \`b\`.`

The product serving path (`serving/code_chat.py`) renders prompts through
`training.chat_template.render_for`. The earlier I4 SFT was trained under that
template but evaluated raw, so its instruction-following hypothesis was never
actually tested.

A separate, non-invasive runner now measures the **serving-template condition**:

- `evaluation/native_v01/template_runner.py`
- same frozen task set and scoring code;
- separate output files;
- the immutable `baseline_results.json` is not touched.

This is a different condition, not a replacement baseline. Task content and
scoring are identical; prompt bytes are not.

## 3. Result A — I4 checkpoint under its training template

Checkpoint: `runs/i4_aligned_sft_full/checkpoint_last.pt` (28,787,328 params,
130 SFT steps, final train loss 4.86).
Evidence: `evaluation/capability/i4_template_results.json`,
`docs/audit/TAKSH-I4-TEMPLATE-CONDITION.md`.

| Metric | Raw-prompt baseline (immutable) | I4 under serving template |
|---|---:|---:|
| Tasks | 20 | 20 |
| Empty output | 20 / 20 | 0 / 20 |
| Degenerate/repetitive | 20 / 20 | 19 / 20 |
| Syntax valid | 0 | 1 |
| Executable | 0 | 0 |
| Correct | 0 | 0 |

The template fixed the **empty-output collapse**: the model now emits tokens. It
did not produce executable code. Most outputs collapsed to two-token fragments
such as `"  ("`; one emitted `"return"`.

**Interpretation:** the interface mismatch was real and worth fixing, but it was
not the capability blocker. The checkpoint had seen only ~1 epoch (260 examples,
130 optimizer steps) and had not learned to generate full functions reliably.

## 4. Result B — continued pretraining (aborted, no checkpoint)

A continued-pretraining run warm-started from `runs/v160_pretrain` on the clean
local corpus (600 steps planned, lr 5e-5, seq 512, batch 4,096 tokens/step).

It reached step 20 with loss moving 6.54 → 6.80 (noisy, still far above a useful
code-model loss). It was stopped when running it concurrently with SFT pushed
free physical memory to **276 MB** on a 4 GB host. No checkpoint was saved at the
configured 100-step interval. The run is therefore **aborted**, not a result.

This is recorded rather than silently discarded. It also confirms the hardware
ceiling: one 28.8M training job at a time is the safe limit on this machine.

## 5. Result C — extended aligned SFT (1,000 steps)

Run: `runs/i4_aligned_sft_long/`
Config: `configs/train/sft_i4_aligned_long.yaml`
Init: `runs/i4_aligned_sft_full/checkpoint_last.pt`
Data: `data/l02_sft/sft.jsonl` (260 locally authored CC0 pairs, full-function answers)
Steps: 1,000 (7.7 epochs), lr 5e-5, batch 2, CPU.
Final training loss: **2.47** (from 5.73). Checkpoint SHA-256:
`cd1030ac9f5ae620eb29838847bcf6acb4bdf4afa5f702bd96ee3277d4a4fc4f`.

| Metric | Raw prompts | Serving template |
|---|---:|---:|
| Syntax valid | 3 / 20 | 6 / 20 |
| Executable | 2 / 20 | 0 / 20 |
| Correct | **0 / 20** | **0 / 20** |
| Truncated | 10 / 20 | 16 / 20 |
| Degenerate | 11 / 20 | 4 / 20 |
| Empty | 0 / 20 | 0 / 20 |

Raw outputs tended to be a two-token `''` literal: syntactically valid, but no
target function. Template outputs were longer and often hit the 96-token limit.
Neither condition produced a correct function.

**Interpretation:** more instruction tuning moved syntax and output length, not
correctness. A training loss of 2.47 is still far too high to reproduce a
~30-token function exactly. The remaining question is whether that is an
optimization limit or the capacity limit of a 28.8M model.

A cache/no-cache A/B on the same prompt produced identical `defdef…` output, so
this is model behaviour, not a KV-cache or position-offset defect.

## 6. Result D — second SFT continuation (aborted, negative diagnostic)

Run: `runs/i4_aligned_sft_long2/`
Config: `configs/train/sft_i4_aligned_long2.yaml`
Init: `runs/i4_aligned_sft_long/checkpoint_last.pt`
Planned: 2,000 steps, lr 1e-4, batch 2, seed 43.

The higher LR **regressed** the loss instead of improving it:

| Step | Loss | EMA | Note |
|---:|---:|---:|---|
| 0 | 2.32 | 2.32 | start from the 1,000-step checkpoint |
| 50 | 3.51 | 3.48 | peak LR destabilises |
| 100 | 3.05 | 3.05 | partial recovery, still worse than 2.47 |
| 150 | 2.99 | 2.95 | still worse; ~7 s/step under memory pressure |

It was stopped at step 150. No checkpoint was saved (the interval was 500), and
the run is **not** a result about capability. It is a negative diagnostic: a
higher LR did not improve the 28.8M model, and the 4 GB host cannot sustain the
run at useful throughput while paging.

## 7. Verdict

- **Capability: 0/20 correct.** The immutable baseline and every new condition
  tested here agree.
- **Training/serving prompt mismatch is real and was fixed in measurement.**
  It changed emptiness and syntax, not correctness.
- **1,000 steps of aligned SFT moved syntax (3–6/20) but produced 0 correct.**
- **A higher-LR continuation regressed the loss; local CPU scale is exhausted.**
- **No external data, no money, and no GPU were used.** The experiment stayed
  inside the owner's constraints.
- **No capability claim, release, publication, or merge is made by this work.**

## 8. What would change the verdict

A positive result requires at least one task to be **correct**, not merely
non-empty or executable. The next honest levers, in order of likely value:

1. stop claiming local CPU scale can close the gap; it did not;
2. a real capability programme needs authorised GPU compute and a much larger,
   licence-cleared corpus — an owner decision, not an engineering default;
3. until then, TAKSH remains an honestly qualified developer-preview
   infrastructure, not a coding model.
