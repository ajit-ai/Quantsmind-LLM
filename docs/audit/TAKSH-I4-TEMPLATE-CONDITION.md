# TAKSH Native v0.1 — Chat-Template Condition

- Generated (UTC): 2026-11-20T04:10:18Z
- Condition: **chat_template_serving** (not the immutable raw-prompt baseline)
- Status: **COMPLETED**
- Checkpoint: `runs\i4_aligned_sft_full\checkpoint_last.pt`
- Checkpoint SHA-256: `5b791d0dc6b1424cda901c793b30edd0644dde8e9d670838634fc254eb99918f`
- Tokenizer SHA-256: `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92`
- Load: 48.211 s | Total generation: 11.94 s
- Prompt rendering: `training.chat_template.render_for(tokenizer, SYSTEM_PROMPT, task.prompt)`

## Overall results

- Tasks: **20**
- Syntax valid: **1** (5.00%)
- Executable: **0** (0.00%)
- Correct: **0** (0.00%)
- Instruction-followed: **0** (0.00%)
- Truncated: **0** (0.00%)
- Empty: **0** (0.00%)
- Degenerate/repetitive: **19** (95.00%)
- Inference failures: **0** (0.00%)

> This condition is comparable to the baseline in task content and scoring,
> but not identical in prompt bytes. It measures the serving interface.

## Per-task results

| Task | Syntax | Exec | Correct | Empty | Degenerate |
|---|---:|---:|---:|---:|---:|
| NV01-BASIC-001 | 0 | 0 | 0 | 0 | 1 |
| NV01-BASIC-002 | 0 | 0 | 0 | 0 | 1 |
| NV01-ARITH-003 | 0 | 0 | 0 | 0 | 1 |
| NV01-STRING-004 | 0 | 0 | 0 | 0 | 1 |
| NV01-STRING-005 | 0 | 0 | 0 | 0 | 1 |
| NV01-LIST-006 | 0 | 0 | 0 | 0 | 1 |
| NV01-DICT-007 | 0 | 0 | 0 | 0 | 1 |
| NV01-ITER-008 | 0 | 0 | 0 | 0 | 1 |
| NV01-COND-009 | 0 | 0 | 0 | 0 | 1 |
| NV01-EDGE-010 | 0 | 0 | 0 | 0 | 1 |
| NV01-EDGE-011 | 0 | 0 | 0 | 0 | 1 |
| NV01-ALGO-012 | 0 | 0 | 0 | 0 | 1 |
| NV01-ALGO-013 | 1 | 0 | 0 | 0 | 0 |
| NV01-ALGO-014 | 0 | 0 | 0 | 0 | 1 |
| NV01-COMP-015 | 0 | 0 | 0 | 0 | 1 |
| NV01-COMP-016 | 0 | 0 | 0 | 0 | 1 |
| NV01-INST-017 | 0 | 0 | 0 | 0 | 1 |
| NV01-INST-018 | 0 | 0 | 0 | 0 | 1 |
| NV01-DEBUG-019 | 0 | 0 | 0 | 0 | 1 |
| NV01-DEBUG-020 | 0 | 0 | 0 | 0 | 1 |

## Honesty note

A passing result here would demonstrate that the *serving interface* can elicit
code from the checkpoint. It would not erase the raw-prompt baseline, and it
would not by itself establish general coding capability.
