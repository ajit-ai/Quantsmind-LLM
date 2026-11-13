> **Historical report — preserved unchanged.** This L0.6.0 qualification was
> produced at commit `c601a73716d5` on
> `feature/l060-foundation-model-qualification`. Its results are historical and
> are not re-asserted at the current HEAD. The current L0.6.1 closure status is
> recorded separately in `docs/L060-1-INTEGRATION-CLOSURE.md` and
> `reports/l061_integration_qualification.json`. Native model capability remains
> `NATIVE_CAPABILITY_NOT_EVALUATED`; no compatible canonical checkpoint has been
> produced or promoted.

# TAKSH L0.6.0 â€” Foundation Model Qualification Report

**Verdict: CONDITIONAL PASS** â€” core engineering foundation is qualified; explicit gaps remain (listed). No coding capability is claimed. No data acquisition, GPU spend, training, or publication is authorized by this milestone.

## 1. Starting and ending state
- Branch `feature/l060-foundation-model-qualification` @ `c601a73716d5` (created from `main` @ `c601a73716d5` per work-order Â§7)
- `main` = `c601a73716d5`; `develop` = `8c553465216d`
- Working tree entries preserved: 98 (all L0.1â€“L0.5.1 user work intact)
- Frozen Python benchmark SHA-256: `854c7ca833c224ebâ€¦` (unchanged)
- Tokenizer: vocab 5,822 (unchanged); package `taksh==1.0.0` (unchanged)
- L0.1 default checkpoint: `runs/v180_sft/checkpoint_last.pt` (`9b541036â€¦`) â€” untouched

## 2. Files created (modified: none except shared test expectations)
- `configs/foundation/l060_foundation.yaml` + `evaluation/l060_config_qualification.py`
- `evaluation/l060_{model,training,inference,multilingual,evidence,foundation_report}.py`
- `serving/foundation_inference.py`; `evaluation/resource_guard.py`
- `evaluation/l051_multilingual_benchmark_v1.json` (additive qualification block)
- `evaluation/l051_multilingual_benchmark.py` (qualification surfaced); `docs/L060_STRATEGY_C_INTERFACE.md`
- `tests/test_l060_foundation.py`; `tests/test_l05*/l051_feasibility.py` (branch-tolerance updates only)

## 3. Architecture and parameter-count evidence (WP1)
- Config: `taksh-foundation-111m` v1 â€” vocab 5,822 (fixed), d_model 768, 12 layers, 12 heads (6 KV, GQA 2:1), ffn 3072, context 2048, tied embeddings
- **Parameter count: 110,658,816 (exact match: True)**
- Head dim: 64; GQA: 12q / 6kv (2:1, matching 28.8M baseline)
- Canonical config sha256: `cde34e52cddf6642174b0940cea5cd69dd9567f7be58d57463da89b260ffda90`
- Roundtrip identical (JSON): True; file roundtrip: True
- Invalid configs are rejected (bad vocab, divisibility, GQA ratio, seq_len, rope_theta, eps, params)

## 4. Model implementation test evidence (WP2)
- 110M foundation model instantiates cleanly: **110,658,816 params == analytic 110,658,816**
- Forward pass shapes ok=True; logits finite=True; loss finite=True (loss=4.1748)
- Causal attention exact: prefix-vs-full max |diff| = **0.00e+00** (True)
- Deterministic init: same seed identical=True; different seed differs=True
- state_dict roundtrip (tiny fixture): True; loads with weights_only=True
- Padding note: padding masks are not applied by the current implementation (right-pad + causal attention + ignore_index=-100 in the loss makes it safe for right-padded batches); documented limitation, not a silent defect

## 5. Training pipeline readiness (WP3)
- Interfaces qualified: 11 (config load, token batch, loss, grad accumulation, optimizer/scheduler, checkpoint save/resume, seed, provenance, failure recovery, manifest validation)
- Manifest enforcement: unknown license, unapproved source, missing provenance, benchmark contamination (path + sha256), unverified language label â€” all rejected; compound documented licenses pass
- In-repo manifests re-validated: l03 corpus pass, l04 code corpus pass
- No training performed; nothing acquired; no data downloaded

## 6. Local inference evidence (WP5)
- Load path ok: True; meta params = 10816 (synthetic fixture)
- Bounded generation: completion_tokens=8 â‰¤ 8; finish_reason=length; max_new_tokens enforced=True
- Context enforcement: 200-token prompt truncated to max_seq_len=24 without error
- Determinism: temperature=0 identical across two calls: True
- Structured errors: missing ckpt â†’ ok: CheckpointIncompatibleError; bad limit â†’ ok: ConfigurationError
- L0.1 default checkpoint preserved and not loaded in this probe

## 7. Multilingual benchmark correction (WP6)
- **Denominator inconsistency resolved:** per-language success = proportion of executed tasks passing all unit assertions; **qualification requires â‰¥ 16 tasks per language AND â‰¥ 50% pass ratio**
- Starter benchmark: 64 tasks, 16 languages, families 1â€“4 instantiated, 5â€“10 specified (not instantiated â€” disclosed)
- Shortfall: +12 tasks per language (+192 total) + families 5â€“10 instances to reach full qualification coverage
- Toolchain: bash, c, cpp, csharp, go, java, javascript, kotlin, python, rust, sql, typescript; missing: ['php', 'ruby', 'swift', 'dart']
- Python frozen benchmark untouched: sha256 854c7ca833c224ebâ€¦ identical
- Scoring interfaces: ['score_task', 'score_language', 'score_benchmark']

## 8. Resource-control verification (WP7)
- `evaluation/resource_guard.py`: max steps/batch/context/checkpoint-size/threads; explicit device selection; `--confirm <exact-phrase>` gate; env-var authorization gate; **fail-closed** â€” no code path in the module grants authorization (verified by tests)
- Guards demand an external operator decision recorded outside the module (env var + confirmation phrase); absence of either raises AuthorizationRequired
- Non-cpu devices (e.g. 'cuda') raise AuthorizationRequired on this host

## 9. Test results and skipped tests (WP9)
- **236/236 focused tests pass** (12 files): L0.6.0 29/29 (config validation, param count, forward, causal, determinism, roundtrip, manifest licence enforcement, inference limits, auth fail-closed, denominator validation, preservation)
- `tests/test_training_correctness.py` 39 tests, training e2e 6, L0.5.1 24, L0.5 22, L0.4 15, L0.3 9, L0.2 15, L0.1 18, legacy-serving 17, code-serving 4, sft 13
- Skipped: none skipped in the focused battery (all checkpoint-dependent tests run against real checkpoints on disk)

## 10. Remaining blockers (WP10)
- no 111M checkpoint has been instantiated outside of tiny synthetic fixtures by design; real artifact requires owner-authorized compute
- strategy C remains unproven; no training performed at any scale in L0.6.0
- per-language sandboxed execution harness is NOT implemented (L1.0 work item); multilingual benchmark cannot be scored yet
- families 5-10 of the multilingual benchmark are specified but not instantiated; qualification for non-Python languages remains impossible today
- 12 of 16 L1.0 toolchains exist; php/ruby/swift/dart unavailable and cannot be installed without authorization
- C/C++ require a vsdevcmd environment; TypeScript compiles only via Node type stripping (no tsc installed)
- qualified code corpus remains 1.59M Python tokens; multilingual training remains infeasible without licensed data acquisition

## 11. Exact owner approvals required for L0.6.1 / L0.7
- written authorization to acquire licensed multilingual data (named corpora + legal sign-off)
- written authorization for GPU compute budget (class and duration)
- decision on first L1.0 language execution (recommendation: python + javascript + go + rust + java + bash)
- decision on TypeScript toolchain (install tsc or drop TS from executable set)
- confirmation that frozen Python benchmark + multilingual 50%-of-executed-tasks-per-language bar are binding
- confirmation that L0.1 default checkpoint remains untouched until a candidate passes the frozen benchmark

## 12. Exit decision

### CONDITIONAL PASS

Core foundation engineering is qualified and preserved; nothing here authorizes data acquisition, GPU purchase/rental, training, publication or release. L0.6.1/L0.7 requires the approvals in section 11.

*Generated by `python -m evaluation.l060_foundation_report`. The L0.1 default checkpoint remains `runs/v180_sft/checkpoint_last.pt`. The frozen Python benchmark remains `854c7ca8â€¦`. The 53-language registry is unchanged.*
