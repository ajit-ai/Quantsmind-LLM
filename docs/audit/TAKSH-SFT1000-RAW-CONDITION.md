# TAKSH Native v0.1 — Raw-Prompt Condition (new checkpoint)

> This is a raw-prompt re-evaluation of `runs/i4_aligned_sft_long/checkpoint_last.pt`.
> It is **not** the immutable `evaluation/native_v01/baseline_results.json` artifact.
> The official baseline remains 0/20 for the v180 checkpoint.

- Generated (UTC): 2026-11-20T04:45:35Z
- Status: **COMPLETED**
- Checkpoint: `runs\i4_aligned_sft_long\checkpoint_last.pt`
- Checkpoint SHA-256: `cd1030ac9f5ae620eb29838847bcf6acb4bdf4afa5f702bd96ee3277d4a4fc4f`
- Tokenizer: `F:\Codes\Git\Quantsmind-LLM\tokenizer\taksh_code_bpe_v1.json`
- Model meta: `{"checkpoint": "runs\\i4_aligned_sft_long\\checkpoint_last.pt", "checkpoint_sha256": "cd1030ac9f5ae620eb29838847bcf6acb4bdf4afa5f702bd96ee3277d4a4fc4f", "tokenizer": "F:\\Codes\\Git\\Quantsmind-LLM\\tokenizer\\taksh_code_bpe_v1.json", "tokenizer_sha256": "554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92", "vocab_size": 5822, "parameters": 28787328, "total_parameters": 28787328, "device": "cpu", "max_seq_len": 2048, "model_name": "qm-code-v160-15m", "model_config": {"name": "qm-code-v160-15m", "vocab_size": 5822, "d_model": 384, "n_layers": 12, "n_heads": 8, "n_kv_heads": 4, "ffn_dim": 1536, "max_seq_len": 2048, "rope_theta": 10000.0, "norm_eps": 1e-05, "tie_embeddings": true}, "template_markers": {"<bos>": 1, "<eos>": 2}}`
- Load: 7.547 s | Total generation: 37.532 s
- Generation config: `{"max_new_tokens": 96, "temperature": 0.0, "seed": 1, "exec_timeout_s": 5.0}`

## Overall results

- Tasks: **20**
- Syntax valid: **3** (15.00%)
- Executable: **2** (10.00%)
- Correct: **0** (0.00%)
- Instruction-followed (target produced): **0** (0.00%)
- Truncated: **10** (50.00%)
- Empty: **0** (0.00%)
- Degenerate/repetitive: **11** (55.00%)
- Inference failures: **0** (0.00%)

> Native inference execution ≠ coding capability. Syntax validity ≠ correctness.

## Results by category

| Category | Tasks | Syntax | Executable | Correct | Instruction |
|---|---:|---:|---:|---:|---:|
| basic_function_generation | 2 | 1 (50%) | 1 (50%) | 0 (0%) | 0 (0%) |
| arithmetic_string_manipulation | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| lists_dicts_iteration | 3 | 1 (33%) | 1 (33%) | 0 (0%) | 0 (0%) |
| conditional_logic_edge_cases | 3 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| simple_algorithm | 3 | 1 (33%) | 0 (0%) | 0 (0%) | 0 (0%) |
| function_completion | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| instruction_following | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| code_correction_debugging | 2 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

## Per-task results

| Task | Category | Syntax | Exec | Correct | Instr | Trunc | Degenerate | Error |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NV01-BASIC-001 | basic_function_generation | 1 | 1 | 0 | 0 | 0 | 0 | entry_point_not_defined |
| NV01-BASIC-002 | basic_function_generation | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: unmatched ')' |
| NV01-ARITH-003 | arithmetic_string_manipulation | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: invalid syntax |
| NV01-STRING-004 | arithmetic_string_manipulation | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: unmatched ')' |
| NV01-STRING-005 | arithmetic_string_manipulation | 0 | 0 | 0 | 0 | 0 | 0 | SyntaxError: unmatched ')' |
| NV01-LIST-006 | lists_dicts_iteration | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: '(' was never closed |
| NV01-DICT-007 | lists_dicts_iteration | 1 | 1 | 0 | 0 | 0 | 0 | entry_point_not_defined |
| NV01-ITER-008 | lists_dicts_iteration | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: invalid syntax |
| NV01-COND-009 | conditional_logic_edge_cases | 0 | 0 | 0 | 0 | 0 | 0 | SyntaxError: invalid syntax |
| NV01-EDGE-010 | conditional_logic_edge_cases | 0 | 0 | 0 | 0 | 0 | 0 | SyntaxError: '(' was never closed |
| NV01-EDGE-011 | conditional_logic_edge_cases | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: unmatched ')' |
| NV01-ALGO-012 | simple_algorithm | 1 | 0 | 0 | 0 | 1 | 0 | NameError: name 'joinjoinjoinjoinjoinjoinjoinjoinjoinjoinjoi |
| NV01-ALGO-013 | simple_algorithm | 0 | 0 | 0 | 0 | 1 | 1 | IndentationError: unexpected indent |
| NV01-ALGO-014 | simple_algorithm | 0 | 0 | 0 | 0 | 0 | 0 | SyntaxError: unmatched ')' |
| NV01-COMP-015 | function_completion | 0 | 0 | 0 | 0 | 1 | 1 | SyntaxError: '(' was never closed |
| NV01-COMP-016 | function_completion | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed |
| NV01-INST-017 | instruction_following | 0 | 0 | 0 | 0 | 0 | 0 | SyntaxError: invalid syntax |
| NV01-INST-018 | instruction_following | 0 | 0 | 0 | 0 | 1 | 0 | SyntaxError: invalid syntax |
| NV01-DEBUG-019 | code_correction_debugging | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed |
| NV01-DEBUG-020 | code_correction_debugging | 0 | 0 | 0 | 0 | 0 | 1 | SyntaxError: '(' was never closed |

## Representative outputs

### NV01-BASIC-001 (not correct)

```text
 ''
```

### NV01-BASIC-002 (not correct)

```text
 '')'):
    return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return:
    return return return
    return
    return
    return
   
   
```

### NV01-ARITH-003 (not correct)

```text

    return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return
    return return
    return return
    return
   
   
   
   
   
   
  
```

### NV01-STRING-004 (not correct)

```text
 sum):
    return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return:def    return return
    return return return
    return
    return
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
 
```

### NV01-STRING-005 (not correct)

```text
 chjoin):
```

### NV01-LIST-006 (not correct)

```text
 sum(1
    return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return return:def    return
    return
    return
    return
    return
   
   
   
   
   
   
   
  
```

