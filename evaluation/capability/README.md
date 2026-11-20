# Capability Evaluation Evidence

This directory holds **new** capability measurements only. It never replaces the
immutable Native v0.1 baseline at `evaluation/native_v01/baseline_results.json`.

## Conditions

| Evidence file | Checkpoint | Prompt condition | Correct |
|---|---|---|---:|
| `evaluation/native_v01/baseline_results.json` | `runs/v180_sft/checkpoint_last.pt` | raw task prompt (immutable baseline) | **0 / 20** |
| `i4_template_results.json` | `runs/i4_aligned_sft_full/checkpoint_last.pt` | `training.chat_template.render_for` | **0 / 20** |
| `sft1000_raw_results.json` | `runs/i4_aligned_sft_long/checkpoint_last.pt` | raw task prompt | **0 / 20** |
| `sft1000_template_results.json` | `runs/i4_aligned_sft_long/checkpoint_last.pt` | `training.chat_template.render_for` | **0 / 20** |

The template condition is measured by
`evaluation/native_v01/template_runner.py`; it reuses the frozen task set and
scoring code, and writes separate evidence. It is a serving-interface condition,
not a replacement baseline.

Training loss logs are copied here from the local `runs/` directories so the
evidence survives without committing checkpoints:

- `i4_aligned_sft_full_loss.jsonl` — 130-step I4 SFT;
- `i4_aligned_sft_long_loss.jsonl` — 1,000-step continuation;
- `i4_aligned_sft_long2_loss.jsonl` — aborted higher-LR continuation.

## What the results show

- Raw prompts on the unmodified v180 checkpoint: 20/20 empty outputs.
- The same checkpoint under its training template: 20/20 non-empty, but 19/20
  degenerate and 0 correct.
- 1,000 further SFT steps: syntax improves to 3–6/20 depending on condition,
  but **0/20 correct**; outputs collapse to `''` or repeated `def` tokens.
- A higher-LR continuation (`runs/i4_aligned_sft_long2/`) regressed from loss
  2.47 to 2.95–3.05 and was stopped at step 150; it saved no checkpoint.
- No condition has produced a single correct function.

## Non-claims

- No coding capability is claimed.
- No model, checkpoint, dataset, or package is published by this directory.
- The official frozen benchmark remains unchanged.
