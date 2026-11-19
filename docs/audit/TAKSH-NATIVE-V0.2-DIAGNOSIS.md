# TAKSH Native v0.2 — Generation Diagnosis and Training Readiness

**Mission:** Investigate why `code_v160_15m` (`v180_sft`) produced empty output or lone `(` in Native v0.1; distinguish prompt-format mismatch, EOS prediction, tokenization/decoding behavior, inference defects, and genuinely weak learned capability. Controlled diagnostics only — **no training/fine-tuning**, no weight/tokenizer/checkpoint changes, no external providers.

## 1. Baseline and preservation (inspected)
- Starting branch: `feature/l060-1-integration-closure` | Expected/actual HEAD: `87b876114a698b334ac7d15c6d1eee32f9138ffd` | Ending HEAD: `87b876114a698b334ac7d15c6d1eee32f9138ffd` (no commits).
- Preserved uncommitted work: Native v0 (`M serving/foundation_inference.py`, `M serving/native_provider.py`, `?? tests/test_native_v160_inference.py`) and Native v0.1 benchmark/results (`evaluation/native_v01/`, `tests/test_native_v01_evaluation.py`, `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`, `evaluation/native_v01/baseline_results.json` **unchanged**).
- Protected untracked data preserved untouched: `data/l02_overfit/`, `data/l02_sft/`, `data/l03_corpus/`, `data/l04_code_corpus/` (and other existing data dirs untouched). Checkpoints/tokenizer/training corpora unchanged; no branches/worktrees created; no reset/clean/stash.

## 2. Recovered training-format evidence (existing evidence only)
**Verified (from configs/code/manifests/data):**
- **C:\Users\Lenovo\.local\share\opencode\worktree\545eb7\integration\taksh-l060-1-closure\configs\train\sft_v180.yaml**: v180_sft SFT config: run qm-code-v180-sft, base_checkpoint runs/v160_pretrain/checkpoint_last.pt, base_model_config configs/model/code_v160_15m.yaml, tokenizer tokenizer/taksh_code_bpe_v1.json, train/val data/v180_sft/sft.jsonl, max_length 512, 200 steps, seed 42.
- **training/sft.py + training/chat_template.py + training/tokenizer_contract.py**: SFTPairDataset uses contract detected from tokenizer (taksh_code_bpe_v1 is v1 plain_instruction with <bos>/<eos>, no role markers). Training renders render_completion(SYSTEM_PROMPT, instruction, response, contract_id=v1) => '<bos>{SYSTEM_PROMPT}\n\n{instruction}\n\n{response}<eos>' and masks prompt tokens via prompt_token_length (prompt half omits response/EOS). Loss is on response (+EOS) tokens; prompt masked.
- **data/v180_sft/sft.jsonl (present locally)**: v180_sft records are instruction-completion pairs (id/task/instruction/input/response). Observed style: instruction asks to complete a function/return expression and includes code prefix; response is short completion (e.g. 'a + b', 'x * y') — completion-oriented targets in an instruction wrapper.
- **data/v180_corpus/taksh_code_v1.manifest.json**: v180 corpus manifest indicates project/docs/scripts code corpus (cleared code text) used for pretraining context; separate from SFT pairs.
- **data/l02_sft/dataset_manifest.json**: l02_sft is a different, later cleared synthetic SFT set (260 records, full function responses, CC0) with ### Instruction/Response-style formatting via training/instruction_format.py in that pipeline; v180_sft checkpoint under test was trained per sft_v180.yaml, not on l02_sft.

**Representative local samples (small, unmodified evidence):**
- `data/v180_sft/sft.jsonl` (present): e.g. instruction `"Complete the following Python function... def add(a,b):\n    return "` with response `"a + b"` (id p01_add), `"x * y"` (p02_mul) — completion-oriented, short targets in instruction wrapper. (No large data copied into artifacts.)
- `data/l02_sft/sft.jsonl`/manifest: different later cleared SFT set (260 full-function responses, CC0, ### Instruction/Response pipeline) — **not** the training set of `v180_sft` checkpoint.

**Unresolved / uncertainty:**
- Exact v180_sft training run logs/optimizer states beyond checkpoint step/config presence were not exhaustively recovered in this environment; checkpoint config payload read via strict safe allowlist encountered additional numpy dtype globals during metadata inspection (weights not modified). Core training recipe is evidenced by configs/train/sft_v180.yaml and training code/data.
- Whether v180_sft used any additional chat/role markers beyond tokenizer contract v1 is constrained by tokenizer evidence (no role special tokens registered) — consistent with plain bos/eos prompt; definitive per-example tokenization matches SFTPairDataset behavior.
- Provenance that cannot be fully re-derived here is not claimed; conclusions distinguish evidenced facts from uncertainty.

## 3. Experiment inventory and exact matrix
- Checkpoint: `F:\Codes\Git\Quantsmind-LLM\runs\v180_sft\checkpoint_last.pt` SHA-256 `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` (expected match). Tokenizer: `C:\Users\Lenovo\.local\share\opencode\worktree\545eb7\integration\taksh-l060-1-closure\tokenizer\taksh_code_bpe_v1.json` SHA-256 `554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92` (v1 bos/eos, vocab 5822). Model meta: `{"checkpoint": "F:\\Codes\\Git\\Quantsmind-LLM\\runs\\v180_sft\\checkpoint_last.pt", "tokenizer": "C:\\Users\\Lenovo\\.local\\share\\opencode\\worktree\\545eb7\\integration\\taksh-l060-1-closure\\tokenizer\\taksh_code_bpe_v1.json", "vocab_size": 5822, "parameters": 28787328, "total_parameters": 28787328, "device": "cpu", "max_seq_len": 2048, "model_name": "qm-code-v160-15m", "model_config": {"name": "qm-code-v160-15m", "vocab_size": 5822, "d_model": 384, "n_layers": 12, "n_heads": 8, "n_kv_heads": 4, "ffn_dim": 1536, "max_seq_len": 2048, "rope_theta": 10000.0, "norm_eps": 1e-05, "tie_embeddings": true}}`
- Runtime: Python `3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]` | `Windows-11-10.0.26300-SP0` | device `cpu` | load 14.855s | total generation 127.393s | max_new_tokens=64, sequential CPU.
- Task subset (stable v0.1, semantics/criteria unchanged): ['NV01-BASIC-001', 'NV01-ARITH-003', 'NV01-LIST-006', 'NV01-COMP-015', 'NV01-INST-017', 'NV01-DEBUG-019'] covering basic function generation, arithmetic, list/dict operations, function completion, instruction-following, code correction.
- Prompt variants (controlled): ['v01_benchmark', 'raw_completion_prefix', 'recovered_sft_format', 'minimal_instruction', 'few_shot'] — (1) existing v0.1 benchmark prompt; (2) raw code-completion prefix; (3) recovered SFT format (contract-correct v1 prompt half `<bos>{SYSTEM_PROMPT}\n\n{instruction}\n\n`, with v180_sft-style completion instructions mapped per task, based on training/sft.py+chat_template.py+v180_sft evidence); (4) minimal instruction; (5) small few-shot (compatibility probe, not verified as original training format).
- Decoding configs (fixed seeds, bounded): [{"name": "greedy_v01_baseline", "temperature": 0.0, "top_k": 50, "top_p": 1.0, "seed": 1}, {"name": "temp_0.2", "temperature": 0.2, "top_k": 50, "top_p": 1.0, "seed": 1}, {"name": "temp_0.7", "temperature": 0.7, "top_k": 50, "top_p": 1.0, "seed": 1}, {"name": "topk_10_temp_0.7", "temperature": 0.7, "top_k": 10, "top_p": 1.0, "seed": 1}]. v0.1 greedy baseline config preserved as `greedy_v01_baseline`.
- Total experiments: **120** (6 tasks × 5 variants × 4 decodings), all executed and recorded (no cases removed).
- Diagnostics captured per experiment (without modifying weights): prompt/prompt token IDs, first-step logits top candidates, EOS token rank & probability (and greedy pick), per-step chosen token info/raw IDs, raw token IDs before decoding, decoded text before repair and after repair (separately observable), finish reason, generated token count, repair diagnostic.

## 4. Aggregate observations
- Immediate EOS by greedy/first-step evidence: **13.33%** overall; by variant — v01_benchmark **66.67%** (16/24), raw_completion_prefix 0%, recovered_sft_format 0%, minimal_instruction 0%, few_shot 0%. (Rate computed over all experiments; in affected v01_benchmark task cases greedy selects EOS at step 0 across configs.)
- Zero completion tokens: **13.33%** (all v01_benchmark cases where greedy EOS selected). Empty decoded before repair: **13.33%**.
- Finish stop: **86.67%**; raw token IDs consistent with decoded-before-repair: **100.00%** (100%).
- Decoding sensitivity (bounded sweep): changing temperature/top-k produced different sampled strings in some non-v01/non-empty cases but **did not produce correct/useful code** and **did not eliminate immediate-EOS empty outputs on v01_benchmark prompts** nor yield meaningful quality gains (by_decoding immediate-EOS/zero/empty rates remain 13.33% each, aligned to the same affected prompt cases).

## 5. Representative raw token/logit and decode observations (greedy_v01_baseline)
### NV01-BASIC-001 | v01_benchmark — immediate EOS (empty) on natural-language v0.1 prompt
- finish=stop generated_tokens=0 immediate_eos_greedy=True prompt_tokens=24
- first-step: greedy_pick=2 chosen_id=2 chosen_text=''; EOS rank=1 prob=0.05879371985793114; top5=[id=2 p=0.0588, id=12 p=0.0448, id=11 p=0.0348, id=17 p=0.0268, id=202 p=0.0237]
- generated_token_ids (prefix): []
- decoded_before_repair: ""
- decoded_after_repair: ""
- repair: `{"enabled": true, "applied": false, "status": "abstained_no_signature", "reason": "no signature available", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-BASIC-001 | recovered_sft_format — recovered v1 SFT-style prompt avoids immediate EOS but output degenerate/invalid
- finish=stop generated_tokens=2 immediate_eos_greedy=False prompt_tokens=205
- first-step: greedy_pick=263 chosen_id=263 chosen_text='   '; EOS rank=5 prob=0.017119379714131355; top5=[id=263 p=0.2657, id=264 p=0.1759, id=202 p=0.0466, id=279 p=0.0459, id=2 p=0.0171]
- generated_token_ids (prefix): [263, 12]
- decoded_before_repair: "   )"
- decoded_after_repair: "   )"
- repair: `{"enabled": true, "applied": false, "status": "abstained_parse_error", "reason": "syntax error: unmatched ')' (<unknown>, line 2)", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-BASIC-001 | raw_completion_prefix — raw prefix avoids immediate EOS but output degenerate
- finish=length generated_tokens=64 immediate_eos_greedy=False prompt_tokens=11
- first-step: greedy_pick=11 chosen_id=11 chosen_text='('; EOS rank=6 prob=0.022199073806405067; top5=[id=11 p=0.0714, id=12 p=0.0595, id=17 p=0.0501, id=202 p=0.0253, id=29 p=0.0227]
- generated_token_ids (prefix): [11, 12, 263, 11, 12, 263, 11, 12, 263, 11, 12, 263, 11, 12, 263, 11]
- decoded_before_repair: "()   ()   ()   ()   ()   ()   (((((((((((((((((((((((((((((((((((((((((((((("
- decoded_after_repair: "()   ()   ()   ()   ()   ()   (((((((((((((((((((((((((((((((((((((((((((((("
- repair: `{"enabled": true, "applied": false, "status": "abstained_parse_error", "reason": "syntax error: '(' was never closed (<unknown>, line 2)", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-COMP-015 | v01_benchmark — lone '(' selected (explains v0.1 syntax error)
- finish=stop generated_tokens=1 immediate_eos_greedy=False prompt_tokens=35
- first-step: greedy_pick=11 chosen_id=11 chosen_text='('; EOS rank=12 prob=0.013176633045077324; top5=[id=11 p=0.1759, id=17 p=0.0791, id=12 p=0.0707, id=29 p=0.0385, id=66 p=0.0309]
- generated_token_ids (prefix): [11]
- decoded_before_repair: "("
- decoded_after_repair: "("
- repair: `{"enabled": true, "applied": false, "status": "abstained_parse_error", "reason": "syntax error: '(' was never closed (<unknown>, line 2)", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-COMP-015 | recovered_sft_format — recovered format avoids '(' but still degenerate
- finish=stop generated_tokens=2 immediate_eos_greedy=False prompt_tokens=204
- first-step: greedy_pick=263 chosen_id=263 chosen_text='   '; EOS rank=5 prob=0.017240069806575775; top5=[id=263 p=0.2659, id=264 p=0.1760, id=202 p=0.0468, id=279 p=0.0459, id=2 p=0.0172]
- generated_token_ids (prefix): [263, 12]
- decoded_before_repair: "   )"
- decoded_after_repair: "   )"
- repair: `{"enabled": true, "applied": false, "status": "abstained_parse_error", "reason": "syntax error: unmatched ')' (<unknown>, line 2)", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-DEBUG-019 | v01_benchmark — lone '(' on debug prompt
- finish=stop generated_tokens=1 immediate_eos_greedy=False prompt_tokens=66
- first-step: greedy_pick=11 chosen_id=11 chosen_text='('; EOS rank=12 prob=0.012680242769420147; top5=[id=11 p=0.1747, id=17 p=0.0797, id=12 p=0.0703, id=29 p=0.0380, id=66 p=0.0311]
- generated_token_ids (prefix): [11]
- decoded_before_repair: "("
- decoded_after_repair: "("
- repair: `{"enabled": true, "applied": false, "status": "abstained_parse_error", "reason": "syntax error: '(' was never closed (<unknown>, line 2)", "mapping": {}, "changed": false, "raw_preserved": true}`

### NV01-INST-017 | v01_benchmark — immediate EOS on instruction-following prompt
- finish=stop generated_tokens=0 immediate_eos_greedy=True prompt_tokens=37
- first-step: greedy_pick=2 chosen_id=2 chosen_text=''; EOS rank=1 prob=0.06441180408000946; top5=[id=2 p=0.0644, id=12 p=0.0487, id=11 p=0.0400, id=17 p=0.0296, id=202 p=0.0258]
- generated_token_ids (prefix): []
- decoded_before_repair: ""
- decoded_after_repair: ""
- repair: `{"enabled": true, "applied": false, "status": "abstained_no_signature", "reason": "no signature available", "mapping": {}, "changed": false, "raw_preserved": true}`

## 6. Answers to required questions (with classification)
1. **Does the model immediately predict EOS?** — **Demonstrated (prompt-dependent).** On v0.1 natural-language benchmark prompts, greedy first step selects EOS token id 2 (EOS rank 1, e.g. NV01-BASIC-001 prob ~0.0588, NV01-INST-017 ~0.0644) yielding 0 completion tokens/empty decode in affected subset cases (16/24 v01_benchmark experiments). On other v0.1 cases in subset (e.g. NV01-COMP-015/DEBUG-019) it instead selects token `(` (id 11, rank 1) — not immediate EOS but immediate degenerate token. On recovered SFT/raw/minimal/few-shot variants, greedy first step does **not** select EOS immediately in these experiments (EOS rank drops, e.g. rank 5/~0.017).
2. **Does the recovered SFT format improve generation?** — **Demonstrated: avoids immediate EOS/empty stop and changes first-token behavior; not demonstrated to yield correct/capable code.** Recovered v1 prompt-half format (and other non-v0.1 variants) eliminated immediate-EOS empties and lone-`(` first-token in these controlled runs, but outputs remained degenerate/invalid fragments (e.g. whitespace+`)`, repeated punctuation/parens, nonsense) across tasks/decodings — i.e. improved continuation onset/format alignment signal but **no task-correct solutions** observed.
3. **Are raw token IDs consistent with decoded output?** — **Demonstrated.** `raw_ids_match_decoded_before_repair_rate` = 100%; per-experiment decoded_before_repair equals decoding of recorded `generated_token_ids`. Raw IDs, decoded-before-repair, and decoded-after-repair are separately recorded (repair mostly abstained on no-signature/parse-error). Lone `(` in v0.1 cases is a real selected token (id 11) present in raw IDs and decoded text, not a decoding artifact.
4. **Does controlled decoding change output quality?** — **Demonstrated: sampling changes surface strings in some cases; not demonstrated to improve task quality/correctness.** Greedy vs temp 0.2/0.7/top-k produced varied degenerate strings for some non-empty cases, but no correct code, no removal of immediate-EOS behavior on affected v01 prompts, and aggregate empty/EOS rates unchanged across configs. Degeneracy is largely robust/misaligned, not fixed by conservative decoding tweaks.
5. **Is the v0.1 benchmark distribution reasonably aligned with training?** — **Likely misaligned (evidence-supported).** v180_sft training evidence shows instruction-completion pairs with short completion targets given code-prefix instructions (e.g. complete `return ` → `a + b`) rendered under v1 chat_template SFT format; v0.1 benchmark used natural-language requests like 'Write a Python function...' without that completion framing. This distribution/format mismatch is consistent with observed immediate EOS or degenerate first tokens on v0.1 prompts and improved (but still incompetent) continuation under recovered/completion-style prompts. **Unknown:** exact full training distribution coverage beyond sampled/manifest/config evidence.
6. **Is the failure primarily an inference/configuration issue, or does the evidence point toward inadequate learned capability?** — **Demonstrated:** not a raw inference-engine/token-decoding consistency defect (raw IDs match decodes; engine runs stably; 0 runtime errors). **Demonstrated:** prompt/format strongly affects first-token/EOS behavior (configuration/distribution sensitivity). **Likely primary limitation: inadequate learned capability and/or strong train-eval distribution mismatch** for these coding tasks — even when immediate EOS is avoided via better-aligned prompts, the model emits degenerate, incorrect continuations rather than valid solutions. A narrow inference/config issue (prompt formatting) explains part of v0.1 empties/`(` but **not** the absence of correct code under aligned prompts.
7. **What training intervention, if any, is justified?** — **Likely justified (conditional):** targeted SFT on data that matches desired inference format/tasks (instruction-to-function synthesis, full function bodies, debug/completion) with consistent template, after confirming template/decoding setup; evaluate with `native_v01` as immutable baseline. **Not justified by this evidence:** moving to a larger architecture as the first response (current failures are degenerate/misaligned at 31M and not yet capacity-diagnosed). **Unknown until small aligned probes/training run:** magnitude of gain achievable at 31M.

## 7. Conclusions (classified)
**Demonstrated:** (a) v0.1 empty outputs in affected cases caused by immediate EOS selection (greedy step 0, token id 2, high rank/prob) on v0.1 prompt style; (b) v0.1 lone `(` outputs are real selected token id 11 in raw IDs/decoded text (e.g. NV01-COMP-015/DEBUG-019), not decode/repair artifacts; (c) raw token IDs are consistent with decoded-before-repair (100%); repair is separately observable and generally abstains; (d) recovered SFT-style/completion/raw-prefix/minimal prompts avoid immediate EOS and change first-token distribution but still produce degenerate/incorrect outputs; (e) controlled decoding (temp/top-k) varies samples but yields no task-correct improvement; (f) training evidence shows v180_sft used v1 bos/eos SFT prompt with completion-style short responses (v180_sft data) — distribution/format differs from v0.1 natural-language prompts.
**Likely:** v0.1 benchmark prompts are materially misaligned with v180_sft training distribution/format; principal blocker to useful coding outputs is weak learned capability and/or distribution mismatch rather than an inference engine defect.
**Unknown:** exact full extent of v180_sft training distribution/coverage; how much capability 31M can reach with better-aligned SFT data/template; residual metadata details not fully extractable under strict safe-load allowlist in this environment.

## 8. Resource observations
- CPU sequential, bounded (max_new_tokens 64), 120 experiments: load 14.855s, total generation 127.393s. No crashes/OOM observed. Peak memory not measured (no claim). No external calls/network/downloads; no unrestricted subprocess execution.

## 9. Recommended next milestone (evidence-based)
- **Immediate (high value, low risk):** standardize and validate an inference prompt/template aligned to the model’s actual SFT/training contract (v1 completion-style as evidenced) and add a prompt-alignment evaluation track alongside the immutable `native_v01` benchmark — to separate format mismatch from capability.
- **If pursuing capability gains:** run a small, controlled SFT/alignment experiment on 31M (`code_v160_15m` family) using instruction-to-function/full-body data matching target tasks and the verified template, then re-evaluate on `native_v01` (immutable) + aligned track. Do **not** justify larger architecture from this checkpoint’s poor performance alone.
- **Defer:** capability/production claims; architecture scaling decisions until aligned-format baseline is measured.

## 10. Reproducibility and artifacts
- Runner: `python -m evaluation.native_v02.diagnose --checkpoint <path> --tokenizer tokenizer/taksh_code_bpe_v1.json --max-new-tokens 64 --device cpu`
- Machine-readable: `evaluation/native_v02/diagnosis_results.json` | Report: `docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md` | Tests: `tests/test_native_v02_diagnosis.py` | Inventory/runner: `evaluation/native_v02/`.
- Native v0.1 baseline preserved unchanged (`evaluation/native_v01/baseline_results.json`, `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`).

