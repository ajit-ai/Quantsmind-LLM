---
license: unknown
language: en
pipeline_tag: text-generation
---

# Taksh S2 Experimental — Draft Model Card

**Status: DRAFT. NOT APPROVED FOR DISTRIBUTION. This is not an approved public model card.**

## Purpose

Experimental locally staged artifact candidate for the original from-scratch Taksh S2 model. It is intended for research, debugging, and evaluation only.

## Architecture and specifications

- Model ID: `taksh-s2`
- Architecture: `qm-code-m3-9m`
- Parameters: **9,359,104**
- Vocabulary: **5,822**
- Context length: **512**
- Weight file: `taksh-s2.pt`
- Weight size: **37,463,045 bytes**
- Weight SHA-256: **`02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`**
- Tokenizer: `tokenizer/qm_code_bpe_v0.json`
- Tokenizer size: **410,135 bytes**
- Tokenizer SHA-256: **`90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`**

## Inference based on the existing Taksh implementation

From the repository root:

```python
import torch
from tokenizers import Tokenizer
from serving.local_inference import load_model, generate_ids

torch.set_num_threads(4)
model, _config = load_model("artifacts/taksh-s2-experimental/taksh-s2.pt", "cpu")
tokenizer = Tokenizer.from_file("artifacts/taksh-s2-experimental/tokenizer/qm_code_bpe_v0.json")

result = generate_ids(
    model,
    tokenizer,
    "Write a function that returns the sum of two integers.\ndef compute_total(x, y):\n",
    max_new_tokens=32,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    device="cpu",
    seed=20261023,
    binding_repair=False,
)
print(result["text"])
```

Verified sample output:

```text
    return left + right
```

This output parses as Python but uses the wrong parameter names for the prompt. It does not demonstrate coding competence.

## Hardware requirements

CPU-only inference is supported by the existing Taksh implementation. The tested checkpoint loaded and generated on the local Windows CPU environment. No GPU or CUDA requirement is claimed for this experimental artifact.

## Evaluation results

- AD-1 raw: **2/29 exact**, **12/29 structural**, **6/29 binding**
- C3 held-out: **0/54 exact**, **0/54 binding**, **0/45 functional**

Do not describe this model as a general Python coding model.

## License and provenance

**Weights are not cleared for distribution.** This card does not declare a license and does not constitute permission to publish, redistribute, or commercially use the weights.

Training-data and model-weight licensing/provenance remain unresolved in the repository. A separate owner/legal decision and data/model provenance package are required before publication.

## Limitations

- Weak parameter binding.
- Weak composition and held-out mapping.
- No demonstrated general Python capability.
- 512-token context.
- No KV-cache performance profile in this artifact.
- Generated code is not execution-certified.

## Reproducibility

Use the hashes above and the smoke test in `tests/test_taksh_s2_experimental_artifact_smoke.py` to verify the artifact before any future distribution decision.
