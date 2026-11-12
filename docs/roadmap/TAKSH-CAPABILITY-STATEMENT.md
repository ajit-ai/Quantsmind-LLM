# TAKSH Capability Statement

**Date:** 2026-11-12
**Product:** Taksh (formerly Quantsmind-LLM)
**Scope:** `feature/taksh-five-version-ga-program`
**Evidence:** `reports/taksh_publication_readiness.json`, `reports/l110_native_model_foundation.json`, `reports/l110_config_hash_reconciliation.json`, `reports/l111_native_inference.json`, `reports/l120_training_readiness.json`, `reports/l120_training_authorization.json`, `reports/taksh_frozen_benchmarks.json`, `reports/taksh_tool_calling_contract.json`, `reports/taksh_capability_evaluation.json`, `reports/taksh_multifile_engineering.json`, `reports/taksh_ga_security_contract.json`, `reports/M7_OUTCOME.json`

---

## 1. Verdict

**Taksh is an engineering platform with a preserved negative research result.
It is not a demonstrated coding model, a released model artifact, a released
dataset, or a production service.**

| Fact | Value |
|---|---|
| Model availability | `MODEL_UNAVAILABLE` |
| Actual coding capability | `NOT EVALUATED` |
| Canonical checkpoint | **MISSING** |
| Historical M7 result | **0/40** syntax-valid, **0/8** functional, evidence level **0** |
| Frozen Python benchmark | **0/30** preserved; artifact `ABSENT` on this ref; not re-scored |
| IDE integration | Local client extension implemented; not autonomous capability |
| Karkain | `DISCOVERED`; KCC `0xC0000135`; KARK-BENCH 0 tasks |
| Production GA | `NOT STARTED` |
| Training authorization | **NOT GRANTED** (A1.1-C) |
| Compute authorization | **NOT GRANTED** (A2-E) |
| Publication authorization | **NOT GRANTED** for this branch |

The historical v1.0.0 release is a **source package** release. Its own release
notes say it is "not a model release", "not a dataset release" and "not evidence
of coding capability". That distinction still holds.

---

## 2. What is actually demonstrated

These are engineering facts, verified by tests or committed configuration. They
are **not** model-capability claims.

| Capability | Status | Evidence |
|---|---|---|
| Native 111M architecture definition | `PASS` | `configs/model/foundation_111m.yaml`; exactly **110,658,816** parameters; 12 layers, d_model 768, 12/6 GQA, context 2,048, tied embeddings |
| Tokenizer binding | `PASS` | `tokenizer/qm_code_bpe_v0.json`; vocab **5,822**; SHA-256 recorded |
| Generation-parameter contract | `PASS` | `serving/generation_contract.py`; fail-closed validation |
| Native inference mechanics | `PARTIAL` | `serving/local_inference.py`; focused tests; **no canonical checkpoint** |
| Training pipeline source | `PARTIAL` | `training/pretrain.py`, `training/sft.py`, checkpoint/geometry contracts; never authorized at target scale |
| Evaluation harness | `PARTIAL` | `evaluation/eval_code.py` and frozen-benchmark infrastructure; no measured model |
| Repository understanding | `PARTIAL` | `taksh/project_understanding.py`; file/symbol/dependency indexing; no quality measurement |
| Code validation | `PARTIAL` | `taksh/coding_validation.py`; syntax/import/semantic heuristics plus test discovery; no model-quality claim |
| Coding-task orchestration | `PARTIAL` | `taksh/coding_engine.py`; explicit stages; caller-supplied generator |
| Python specialists | `PARTIAL` | `taksh/specialists.py`; ten interfaces; template/heuristic outputs, no model |
| OpenAI-compatible chat surface | `PARTIAL` | `/health`, `/v1/models`, non-streaming `/v1/chat/completions`; `--ckpt` required; tool calls and streaming rejected |
| Tool calling | `BLOCKED` | HTTP 400 `unsupported_parameter` |
| Agentic coding (Cline/OpenCode) | `BLOCKED` | no canonical checkpoint and no tool calling |
| Hugging Face export preparation | `PARTIAL` | `evaluation/hf_export.py`; dry-run by default, strict compatibility gate, honest model-card template; no checkpoint, so no export written |
| Hugging Face model artifact | `BLOCKED` | no canonical checkpoint; exporter implemented but never exercised against real weights; no publication authorization |
| Hugging Face dataset | `BLOCKED` | licence, provenance, PII and contamination review incomplete |
| Karkain | `BLOCKED` | KCC `0xC0000135`; 0 benchmark tasks |
| Production security/deployment | `NOT STARTED` | no audit, deployment automation or release decision |
| 2.0.0 GA | `NOT STARTED` | upstream model and owner decisions outstanding |

Focused regression evidence from the implementation slice:
`276 passed, 3 skipped` across the foundation, inference, model, config,
five-version component, publication/HF-export, SFT, tokenizer, serving,
training-core and release-metadata tests. This proves the code paths under
test; it does not prove model quality.

The apparent 1.1.0 foundation-config hash drift is line-ending normalisation,
not a configuration-content change: the historical `fd96…` hash matches the
config's LF-normalised bytes, while the Windows working tree hashes the CRLF
bytes to `744e…`. `reports/l110_config_hash_reconciliation.json` records both
forms and confirms that the historical report was not rewritten.

---

## 3. Version status

The program labels below are execution labels, not package versions. The
package remains `1.0.0` until an authorized release. See
[`TAKSH-PROGRAM-VERSION-GATES.md`](TAKSH-PROGRAM-VERSION-GATES.md) and
`reports/taksh_version_gates.json`; the current gate report says
`NO_RELEASE_AUTHORIZED`.

| Version | Status | Meaning |
|---|---|---|
| 1.1.0 Native Foundation | `PASS` | Source/config/model/inference contracts are engineering-qualified. No model artifact exists. |
| 1.2.0 Native Model Capability | `BLOCKED` | Authorization packet: 2 unresolved licences, 2.24M/100M tokens, no compute authorization and no canonical checkpoint. |
| 1.3.0 Coding Intelligence | `PARTIAL` | Orchestration, validation and specialist source exists; capability is not evaluated. |
| 1.4.0 Autonomous Engineering | `PARTIAL` | Project understanding and CLI foundations exist; agentic multi-file editing is not implemented and is blocked by model/tool calling. |
| 2.0.0 GA | `NOT STARTED` | Requires a qualified model, security/deployment evidence and owner release decisions. |

> These labels follow the five-version consolidation. The repository's
> long-range `TAKSH_MASTER_DEVELOPMENT_ROADMAP.md` uses a longer version index;
> see the version-label note in `TAKSH-FIVE-VERSION-IMPLEMENTATION-BASELINE.md`.

---

## 4. What Taksh can do today

With the current branch and no model artifact:

- run its test suite and release-preparation checks;
- build the source package (code only, no weights or data);
- index a Python repository (files, symbols, imports, dependency graph);
- run heuristic syntax/import/semantic validation on candidate Python source;
- orchestrate coding-task stages when a caller supplies a generator;
- start an OpenAI-compatible **chat** server only if a checkpoint is supplied
  with `--ckpt` (there is no canonical checkpoint);
- inspect the canonical 111M configuration and tokenizer binding;
- plan a local Hugging Face export and generate an honest model card only when a
  compatible canonical checkpoint exists (currently blocked).

It cannot:

- generate usable code (no trained canonical model);
- follow instructions or hold a reliable multi-turn conversation (no qualified
  instruction-tuned checkpoint; historical chat markers are not a capability);
- call tools, edit files, run commands or complete an agentic coding loop;
- serve streaming responses;
- be used from Cline or OpenCode as a coding agent;
- be published as a Hugging Face model or dataset;
- be listed on OpenRouter;
- be described as production-ready, secure or GA.

---

## 5. What would change this answer

This is not a promise of capability; it is the minimum evidence path.

1. **Owner authorization:** the five decisions in
   [`TAKSH-1.2.0-TRAINING-AUTHORIZATION-PACKET.md`](TAKSH-1.2.0-TRAINING-AUTHORIZATION-PACKET.md)
   - training (A1.1-A), compute (A2), corpus/licence, checkpoint storage and
   post-training evaluation - plus a licence-reviewed corpus at the required
   scale.
2. **Canonical checkpoint:** train or otherwise produce the
   `taksh-foundation-111m` artifact and record its SHA-256.
3. **Compatibility:** geometry/tokenizer checks pass.
4. **Capability evaluation:** run a frozen, pre-registered benchmark and publish
   raw outputs, including failures.
5. **Instruction/tool qualification:** if agentic use is intended, implement and
   evaluate tool calling and streaming, then test Cline/OpenCode end to end.
6. **Security and deployment review:** sandboxing, secret handling, prompt
   injection, network exposure.
7. **Owner release decision:** per channel, with a rollback plan.

Until then, the honest labels are `MODEL_UNAVAILABLE` and
`ACTUAL_CODING_CAPABILITY = NOT EVALUATED`.

---

## 6. Non-claims

Taksh does **not** claim:

- that it writes correct code;
- that it beats, matches or approaches any named model;
- that its historical 0/30, 0/40 or 0/8 results are anything other than negative;
- that engineering tests are capability evidence;
- that an HTTP 200 from a serving endpoint means correct output;
- that a tokenizer, configuration or architecture is a trained model;
- that Karkain is supported;
- that the package is production-ready, secure, or GA;
- that any model, dataset or commercial listing has been published.
