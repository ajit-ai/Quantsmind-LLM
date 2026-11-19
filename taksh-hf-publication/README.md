---
license: apache-2.0
language:
- en
- code
library_name: taksh
tags:
- coding
- code-generation
- python
- javascript
- typescript
- from-scratch
- sovereign
- research
- controlled-pilot
pipeline_tag: text-generation
task_categories:
- code-generation
- code-completion
size_categories:
- 1M<n_params<10M
widget:
- text: "def add(a, b):\n    return a + b\n"
- text: "def compute_total(x, y):\n"
model-index:
- name: Taksh v1.4.1 ALPHA
  results: []
---

# Taksh v1.4.1 ALPHA — Public Release Preparation

> **ALPHA release. Controlled pilot only. Not a production/GA release.**
>
> This is the source code package for **Taksh v1.4.1 ALPHA**, a sovereign,
> from-scratch coding LLM research release. The package version is **1.0.0**
> (Apache License 2.0). Model weights and training data are **not** part of this
> release and are **not cleared for redistribution**. See the
> [License & Provenance](#license--provenance-verification) section below.

## What TAKSH is

Taksh (formerly Quantsmind-LLM) is a sovereign-weights coding LLM project: own
architecture, own tokenizer, own training engine, own data pipeline. Same
Transformer family as Llama (RMSNorm / RoPE / GQA / SwiGLU), trained from
scratch on Python + JavaScript/TypeScript code.

This public ALPHA artifact publishes the **source code and evaluation harness**
under the Apache License 2.0. It does **not** include model checkpoint weights
or training data, which remain under separate provenance review and are not
cleared for redistribution.

## Package version

`taksh.__version__ == "1.0.0"` (status: `final`). This is the Python package
version, **not** a model-capability claim. It is not a model weight version.

## Supported capabilities and limitations

### Qualified hybrid software (deterministic synthesis + planner validation)

The deterministic hybrid generator (rule-based operation synthesis with
structured planner validation) was qualified on frozen evaluation suites:

| Metric | Result | Gate |
|---|---|---|
| Copy exact | 10/13 | ≥10/13 |
| Supplemental functional | 33/48 | ≥33/48 |
| Generalization24 | 21/24 | ≥21/24 |
| Independent functional | **39/40** | ≥39/40 |
| Independent abstention | 5/6 | ≥5/6 |
| AD-1 binding | 23 | ≥23 |
| C3 binding | **28/28** | 28/28 |
| C3 functional | 10 | ≥10 |

Full test suite: **1,649 passed, 1 skipped, 0 failed**. Full qualification
results are in `TAKSH_V141_FINAL_QUALIFICATION_REPORT.md`.

### Neural model-only capability — NOT QUALIFIED

The frozen neural checkpoint contributes **0 functional** across every
evaluation suite (copy, supplemental, generalization, AD-1, C3). Model-only
operation-selection probes never rank the correct operation at useful
probability. All demonstrated correctness comes from deterministic synthesis,
not from the neural model. Do not describe hybrid results as standalone neural
model results.

### Hybrid deployment — CONTROLLED PILOT ONLY

The hybrid architecture (bounded hybrid with structured planner validation) is
suitable for controlled pilot use only. It is **not** production/GA-ready.
Remaining open items include remote CI verification (monthly CI usage
exhausted), checkpoint-loader legacy fallback policy, and the documented
capability ceiling.

## Installation

The package is distributed as a wheel and source distribution:

```bash
pip install taksh==1.0.0
taksh --version
# Taksh 1.0.0 (final)
```

Build from source:

```bash
python -m pip install --no-build-isolation .
```

Runtime dependencies (see `requirements.txt`):

- `torch>=2.2,<3`
- `tokenizers>=0.15`
- `numpy>=1.26`
- `pyyaml>=6.0`
- `gradio>=4.44`
- `openai>=1.40`
- `pillow>=10.0`
- `fastapi>=0.104`
- `uvicorn>=0.24`
- `pydantic>=2.0`

Optional (dev): `pytest>=8.0`

## Quick-start

```bash
# Install the package
pip install .

# Verify version
taksh --version

# Import the supported interfaces
import taksh
assert taksh.__version__ == "1.0.0"

from evaluation.operation_synth import synthesize
from evaluation.operation_planner import plan, validate, compile_plan

# Deterministic operation synthesis (general rule, not fixture-keyed)
result = synthesize("Return the longer of two strings.",
                    "def pick(word_a, word_b):")
print(result["body"])

# Structured planner validation
plan_result = plan("Add the two numbers, then divide by four.",
                   "def calc(a, b):")
assert validate(plan_result, "def calc(a, b):")
compiled = compile_plan(plan_result, "def calc(a, b):")
assert compiled["action"] == "planned"
```

## Supported hybrid workflow

1. Accept the original instruction + function signature.
2. Use the existing generation backend (neural draft) **or** the deterministic
   synthesizer when the operation is unambiguous.
3. Construct an operation plan where supported (ordered steps, explicit
   dependencies, intermediate values distinct from final output).
4. Validate operation selection, dependencies, ordering, and arguments.
5. Execute only safe, validated candidates under existing restrictions.
6. Preserve provenance; record validation and execution outcomes.
7. Return a correct result or explicitly abstain.

Planner validation statuses: `VALIDATED`, `INVALID_PLAN`,
`UNSUPPORTED_PLAN`, `VALIDATION_UNAVAILABLE`. Unsupported planner coverage is
**never** treated as proven-invalid; proven-invalid plans are never executed.

## Error, unsupported-operation, and abstention behavior

- Unsupported operations (no plan in vocabulary) → `UNSUPPORTED_PLAN`, abstain,
  never guessed.
- Proven-invalid plans (undefined identifiers, duplicate results, cycles,
  forward references) → `INVALID_PLAN`, never executed.
- Ambiguous descriptions → abstain, never invent behavior.
- No broad coercion that silently changes program meaning.

## Known limitations

- Independent functional qualification: 39/40 — one case (Q-p1, underspecified
  "normalize" transform) correctly abstained; one abstention case (Q-x3)
  remains missed.
- Model-only functional: 0 in all suites.
- Unseen phrasings abstain or fall to the 0-functional model.
- Rule vocabulary is the ceiling; coverage trails the synthesizer rule surface
  by design (favors transparency).
- 512-token context window for served checkpoints.
- No KV cache; latency grows with context.

## Security and safety disclosures

- Checkpoint loading uses `torch.load(weights_only=True)` by default.
  `weights_only=False` is reachable **only** via an explicit trusted-artifact
  boundary (`allow_unsafe_trusted=True` from operator startup configuration),
  never from a request.
- Serving binds `127.0.0.1` by default; remote binding requires explicit
  `--allow-remote` and is only safe behind auth/firewall.
- Optional bearer auth via `QM_API_KEY`.
- No request-triggered checkpoint loading (`POST /load` removed).
- No secrets, credentials, or telemetry in the package.
- This is research software. Serving HTTP success does not establish output
  correctness. Generated code is not execution-certified.

## License and provenance verification

Taksh source code is licensed under the Apache License, Version 2.0. See
`LICENSE` and `NOTICE`.

**Model weights and training data are NOT part of this release and carry no
license.** The frozen S2 checkpoint (`artifacts/taksh-s2-experimental/taksh-s2.pt`,
SHA-256 `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`)
is **NOT APPROVED FOR DISTRIBUTION**: training-data license not cleared,
model-weights license not cleared, `distribution_approved: false`,
`publication_status: not published`. The frozen tokenizer vocab
(`tokenizer/qm_code_bpe_v0.json`,
SHA-256 `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`)
was trained on training data whose license is not cleared and is likewise
**excluded** from this public release.

Third-party dependencies remain under their own licenses (see
`requirements.txt`).

See `LICENSE_PROVENANCE.md` for full license and provenance verification, and
`EXCLUDED_FILES.md` for the excluded-file inventory and reasons.

## Version and release information

- Designation: **TAKSH v1.4.1 ALPHA**
- Python package version: `1.0.0` (package version is **not** changed for this
  ALPHA release)
- Release status: `final` (package metadata); ALPHA (product-facing maturity)
- Baseline commit: `a654e5d` on branch `develop`
- Qualification: `TAKSH_V141_FINAL_QUALIFICATION_REPORT.md`
- Publication: **NOT PERFORMED** — awaiting explicit owner authorization. This
  artifact directory is upload-ready but has not been uploaded to Hugging Face.

## Citation

If you use this software, please cite it as a research release of the Taksh
sovereign coding LLM project, package `taksh` version 1.0.0, Apache-2.0.

## Repository

https://github.com/ajit-ai/Quantsmind-LLM

## Hugging Face

This ALPHA artifact is prepared for upload to the Hugging Face model hub as
`ajit-ai/taksh` (source code package only — no model weights). Upload requires
separate explicit owner authorization.