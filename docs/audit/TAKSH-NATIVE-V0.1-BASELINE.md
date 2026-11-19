# TAKSH Native v0.1 — Baseline Capability Evaluation

**Milestone:** Native Model v0.1 — Baseline Capability Evaluation (engineering/evaluation).
**Scope:** Reproducible benchmark of existing native checkpoint `code_v160_15m` (`v180_sft`) via native inference only. No training/fine-tuning, no checkpoint/tokenizer/dataset changes, no external providers.

## 1. Starting point and git state

- Starting branch / HEAD (inspected): `feature/l060-1-integration-closure` @ `87b876114a698b334ac7d15c6d1eee32f9138ffd`.
- Final branch / HEAD (end of evaluation): `feature/l060-1-integration-closure` @ `87b876114a698b334ac7d15c6d1eee32f9138ffd` (no commits created).
- Pre-existing uncommitted work preserved (not reset/stashed/discarded): `M serving/foundation_inference.py`, `M serving/native_provider.py`, `?? tests/test_native_v160_inference.py` (Native v0).
- New uncommitted evaluation artifacts (this milestone): `evaluation/native_v01/` (tasks/runner/__init__/baseline_results.json), `tests/test_native_v01_evaluation.py`, `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`.
- Protected untracked directories preserved untouched: `data/l02_overfit/`, `data/l02_sft/`, `data/l03_corpus/`, `data/l04_code_corpus/`.
- Primary repo, canonical 111M config (`configs/foundation/l060_foundation.yaml`), existing checkpoints/tokenizer/datasets: not modified, not copied, not deleted, not replaced.

## 2. Exact model, checkpoint, tokenizer, runtime, and inference configuration

- Checkpoint path: `F:\Codes\Git\Quantsmind-LLM\runs\v180_sft\checkpoint_last.pt` (exists: True).
- Checkpoint SHA-256: `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` (matches expected `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666`).
- Tokenizer path: `C:\Users\Lenovo\.local\share\opencode\worktree\545eb7\integration\taksh-l060-1-closure\tokenizer\taksh_code_bpe_v1.json` (exists: True).
- Tokenizer SHA-256: `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` (vocab 5822, BOS 1, EOS 2).
- Model identity/meta: `{"checkpoint": "F:\\Codes\\Git\\Quantsmind-LLM\\runs\\v180_sft\\checkpoint_last.pt", "tokenizer": "C:\\Users\\Lenovo\\.local\\share\\opencode\\worktree\\545eb7\\integration\\taksh-l060-1-closure\\tokenizer\\taksh_code_bpe_v1.json", "vocab_size": 5822, "parameters": 28787328, "total_parameters": 28787328, "device": "cpu", "max_seq_len": 2048, "model_name": "qm-code-v160-15m", "model_config": {"name": "qm-code-v160-15m", "vocab_size": 5822, "d_model": 384, "n_layers": 12, "n_heads": 8, "n_kv_heads": 4, "ffn_dim": 1536, "max_seq_len": 2048, "rope_theta": 10000.0, "norm_eps": 1e-05, "tie_embeddings": true}}`
- Python: `3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26300-SP0` | Device: `cpu`
- Generation settings: `{"max_new_tokens": 96, "temperature": 0.0, "seed": 1, "exec_timeout_s": 5.0}` (bounded CPU, sequential, no parallel task execution).
- Native inference implementation: `serving/foundation_inference.py` (`load_foundation`, `generate_bounded`) / provider: `serving/native_provider.py`; existing inference tests: `tests/test_native_v160_inference.py`.
- No mock provider, no external LLM/Ollama/API keys/network dependencies, no checkpoint downloads, no training/fine-tuning, no weight/tokenizer/dataset modification.

## 3. Benchmark implementation (reproducible)

- Task inventory: `evaluation/native_v01/tasks.py` — **20 stable tasks** (IDs `NV01-*`), each with prompt, category, difficulty, expected behavior, acceptance criteria, and deterministic executable checks (no external/model-generated expected answers).
- Categories covered (all required): basic Python function generation (2), arithmetic/string manipulation (3), lists/dicts/iteration (3), conditional logic & edge cases (3), simple algorithms (3), function completion from partial prefix (2), instruction-following coding tasks (2), code correction/debugging (2).
- Runner: `evaluation/native_v01/runner.py` — real native load+generate, objective scoring (syntax/execution/correctness/instruction/truncation/empty/degenerate/inference failure), isolated subprocess execution in temp dirs with strict timeout (`-I`, constrained env, cwd=temp), JSON+MD outputs, stable per-task IDs/scores for future comparison.
- Automated evaluation tests: `tests/test_native_v01_evaluation.py` (inventory, extraction/scoring, safe execution incl. timeout/syntax handling, completion harness, schema/summarization, determinism, missing-checkpoint block).
- Machine-readable results: `evaluation/native_v01/baseline_results.json`. Human-readable report: `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`. No large generated artifacts or model weights committed.

## 4. Actual native inference execution status

- Status: **COMPLETED** (genuine bounded CPU native inference executed for all 20 tasks; no mock/external substitution).
- Checkpoint load: 19.322 s (loaded checkpoint step=200 reported). Total generation: 2.932 s (~0.147 s/task).
- Inference failures (runtime exceptions): 0/20. Model produced outputs for each prompt under the native engine; this is **model execution evidence, not coding-capability evidence**.

> **Native inference execution ≠ coding capability. Syntactically valid code ≠ correct code.** No tasks were excluded post-hoc; all 20 are retained with scores and reasons.

## 5. Results — overall and by category

- Tasks: **20**
- Valid Python syntax rate: **0 / 20** (0.00%)
- Executable output rate: **0 / 20** (0.00%)
- Task correctness rate (executable tests passed): **0 / 20** (0.00%)
- Instruction-following success rate (target function produced, conservative): **0 / 20** (0.00%)
- Completion vs truncation: truncated (finish_reason=length) **0** (0.00%); completed/non-length stops **20**
- Empty output rate: **16** (80.00%)
- Degenerate/repetitive output rate: **20** (100.00%)
- Inference failures: **0** (0.00%)

| Category | Tasks | Syntax | Executable | Correct | Instruction |
|---|---:|---:|---:|---:|---:|
| basic_function_generation | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| arithmetic_string_manipulation | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| lists_dicts_iteration | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| conditional_logic_edge_cases | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| simple_algorithm | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| function_completion | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| instruction_following | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| code_correction_debugging | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

## 6. Per-task results (all retained)

| Task | Category | Difficulty | Syntax | Exec | Correct | Instr | Trunc | Empty | Degenerate | Error | Gen(s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| NV01-BASIC-001 | basic_function_generation | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.4294 |
| NV01-BASIC-002 | basic_function_generation | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1223 |
| NV01-ARITH-003 | arithmetic_string_manipulation | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0879 |
| NV01-STRING-004 | arithmetic_string_manipulation | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0924 |
| NV01-STRING-005 | arithmetic_string_manipulation | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.108 |
| NV01-LIST-006 | lists_dicts_iteration | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0944 |
| NV01-DICT-007 | lists_dicts_iteration | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.089 |
| NV01-ITER-008 | lists_dicts_iteration | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.098 |
| NV01-COND-009 | conditional_logic_edge_cases | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0911 |
| NV01-EDGE-010 | conditional_logic_edge_cases | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1292 |
| NV01-EDGE-011 | conditional_logic_edge_cases | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1335 |
| NV01-ALGO-012 | simple_algorithm | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1125 |
| NV01-ALGO-013 | simple_algorithm | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1664 |
| NV01-ALGO-014 | simple_algorithm | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.1057 |
| NV01-COMP-015 | function_completion | easy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed | 0.2658 |
| NV01-COMP-016 | function_completion | medium | 0 | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed | 0.1769 |
| NV01-INST-017 | instruction_following | easy | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0743 |
| NV01-INST-018 | instruction_following | medium | 0 | 0 | 0 | 0 | 0 | 1 | 1 | empty_output | 0.0615 |
| NV01-DEBUG-019 | code_correction_debugging | medium | 0 | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed | 0.2868 |
| NV01-DEBUG-020 | code_correction_debugging | medium | 0 | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed | 0.2067 |

## 7. Representative outputs (successful and failed)

- **Successful/correct outputs:** none observed (0/20 correct, 0/20 syntactically valid).
- **Representative failed outputs (verbatim snippets from real native inference):**
  - `NV01-BASIC-001`..`NV01-ALGO-014`, `NV01-INST-017/018`: empty decoded output (`""`), finish_reason `stop`, completion_tokens 0, repair abstained (`no signature available`) — counted as empty/degenerate, non-executable, incorrect.
  - `NV01-COMP-015` (completion, prefix `def count_items(items):`): raw output `(` (syntax error: '(' was never closed), finish_reason `stop`.
  - `NV01-COMP-016` (completion, prefix `def squares(nums):`): raw output `(` (syntax error).
  - `NV01-DEBUG-019` (fix `sum_to_n`): raw output `(` (syntax error).
  - `NV01-DEBUG-020` (fix `contains`): raw output `(` (syntax error).
- Interpretation: the engine/tokenizer/checkpoint executed and produced bounded generations, but decoded content was empty or a single degenerate token/fragment; **no task-level coding behavior demonstrated**.

## 8. Memory and runtime observations

- Load time: 19.322 s on CPU; total generation time: 2.932 s for 20 sequential tasks (avg ~0.147 s/task). Max new tokens 96, temperature 0.0, seed 1.
- Bounded conservative CPU run (sequential, no parallelism) completed without crashes/OOM in this environment; peak RSS not instrumented by the runner (limitation). Inference calls returned normally (0 inference failures).
- Generated-code execution used isolated temp directories, per-case subprocess with timeout (5.0 s), and constrained env; no access to repo secrets/credentials/user files intended.

## 9. Evidence-supported gap classification (demonstrated vs hypotheses)

**Demonstrated by this benchmark (evidence):**
- Native inference path works end-to-end on this machine for `v180_sft` (safe load, bounded decode, 0 runtime inference failures).
- Task performance is 0% across syntax/execution/correctness/instruction-following for all 20 tasks and all 8 required categories; outputs are predominantly empty (80%) and uniformly degenerate by measured repetition/emptiness criteria (100%).
- A consistent failure mode is immediate stop with 0 completion tokens and repair abstention due to missing signature, or emission of a lone `(` fragment on completion/debug prompts — indicating the model is not producing usable code continuations under the current prompt/decoding/eval setup.

**Plausible contributing factors (hypotheses, not proven by this run alone):**
- *Insufficient training/optimization or data-task mismatch:* checkpoint step and SFT objective may not align with these instruction/function-completion/debug prompts (common for undertrained or narrowly trained small models).
- *Prompting/formatting and instruction-following limitations:* natural-language instruction prompts may not match the model’s trained format; empty/early-stop behavior suggests weak instruction conditioning.
- *Tokenization/decoding interaction:* tokenizer is vocab-compatible (5822) but prompt/continuation formatting and EOS/stop behavior may interact poorly with this checkpoint; requires targeted probes.
- *Model capacity/architecture limits:* 31M params can limit general coding competence, but current outputs are too degenerate to isolate capacity as the primary cause from training/format issues.
- *Inference/decoding or repair behavior:* repair abstains without a signature and some tasks stopped immediately; worth inspecting decode settings/logits, but no crash/engine error was observed (engine itself not implicated by failures).

**Not supported as demonstrated root cause:** no evidence of tokenizer incompatibility/load failure, context-length overflow, or execution harness malfunction causing the 0% scores (harness validated by automated tests).

## 10. Recommended next engineering/training milestone (evidence-based)

- **Primary recommendation:** run a **diagnostic milestone before further blind scaling/training** — (a) prompt/format ablation (match prompts to the model’s SFT/chat/code template if one exists; test few-shot and raw completion forms), (b) decoding/logit probes on these tasks (inspect first-token logits/EOS tendency, with/without repair, bounded temperature/top-k sweeps) to distinguish model-predicts-EOS/garbage from prompt/format mismatch, and (c) targeted evaluation on distributions closer to `v180_sft` training data.
- **If diagnostics confirm weak capability rather than format mismatch:** prioritize **training-data quality/coverage and additional targeted SFT/optimization** for code-instruction and function synthesis at 31M, then re-run this identical `native_v01` benchmark as the regression baseline.
- **Defer capability claims:** do not claim coding capability or production readiness from current measurements (0% correctness). Treat Native v0.1 as an honest, reproducible baseline for comparison only.

## 11. Limitations

- Small benchmark (20 tasks) — indicative baseline, not comprehensive coding competence.
- Instruction-following metric is a conservative proxy (target function produced), not full semantic compliance.
- No peak memory instrumentation; single seed/decoding config (temperature 0.0, max_new_tokens 96) — reproducible but narrow.
- Real-checkpoint execution depends on local path `F:/Codes/Git/Quantsmind-LLM/runs/v180_sft/checkpoint_last.pt`; absence would block (documented) — in this run it was present and hash-verified.
- T01 canonical 111M native capability remains **BLOCKED_MISSING_CHECKPOINT** (separate, unchanged). T16 external provider **BLOCKED_EXTERNAL_DEPENDENCY**.

## 12. Reproducibility

- Runner CLI: `python -m evaluation.native_v01.runner --checkpoint <path> --tokenizer tokenizer/taksh_code_bpe_v1.json --max-new-tokens 96 --temperature 0.0 --seed 1 --timeout 5.0 --device cpu`
- Produces stable per-task IDs/scores in `evaluation/native_v01/baseline_results.json` and report in `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md` for comparison against future model versions.

