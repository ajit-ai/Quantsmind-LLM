# License and Provenance Verification — TAKSH v1.4.1 ALPHA

## Package license (CLEARED for distribution)

The Taksh source code package is licensed under the **Apache License, Version 2.0**.
See `LICENSE` and `NOTICE` in this directory.

- Package name: `taksh`
- Package version: `1.0.0`
- Release status: `final`
- Author: Ajit Kumar
- Repository: https://github.com/ajit-ai/Quantsmind-LLM

Third-party dependencies remain under their own licenses (see `requirements.txt`):
`torch`, `tokenizers`, `numpy`, `pyyaml`, `gradio`, `openai`, `pillow`,
`fastapi`, `uvicorn`, `pydantic`, `pytest`.

## Model weights and training data license (NOT CLEARED)

The frozen S2 checkpoint and tokenizer vocabulary are **excluded** from this
public release. Provenance status (from
`artifacts/taksh-s2-experimental/artifact_manifest.json`):

| Asset | SHA-256 | License | Distribution cleared? |
|---|---|---|---|
| Checkpoint `taksh-s2.pt` | `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` | not cleared | **No** |
| Tokenizer vocab `qm_code_bpe_v0.json` | `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` | not cleared (training-data license not cleared) | **No** |

Provenance fields from `artifact_manifest.json`:

- `training_data_license`: **not cleared**
- `model_weights_license`: **not cleared**
- `distribution_approved`: **false**
- `publication_status`: **not published**

The artifact's `LICENSE_STATUS.txt` states:

> NOT APPROVED FOR DISTRIBUTION pending license and provenance approval.
> This directory is a locally staged experimental artifact candidate. It is not
> a Hugging Face release, not a published model, not a license declaration, and
> not cleared for redistribution.

## What this means for the ALPHA publication

This Hugging Face publication directory contains **source code only** (Apache-2.0).
It deliberately excludes:

1. **Model checkpoint weights** — not cleared for distribution.
2. **Tokenizer vocabulary** — training-data license not cleared.
3. **Training data / corpora** — license not cleared.
4. **Private datasets and caches** — not part of the source release.

No license declaration is made for model weights or training data. No claim is
made that redistributing the checkpoint or tokenizer is permitted. Separate
owner/legal decisions on licensed deployment, training-data redistribution,
nested license conflicts, and tokenizer-input provenance remain open (see
`EXCLUDED_FILES.md` and the v1.4.1 completion report).

No training, fine-tuning, optimizer steps, or checkpoint writes were performed
as part of this publication preparation.

## Verification

- Package metadata (`pyproject.toml`): `license = "Apache-2.0"`,
  `license-files = ["LICENSE", "NOTICE"]`
- `LICENSE` and `NOTICE` are present in the publication directory.
- Package version `1.0.0` matches `taksh.__version__` (version consistency
  check passes).
