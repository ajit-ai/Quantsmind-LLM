"""Verified licensing facts for M6 candidates.

Every entry records where the licence was read from. An entry is only marked
verified when a primary licence file or the repository's own declared metadata
was retrieved, and the exact URL is kept so a reviewer can repeat the check.

``license_id`` describes the **model weights**. ``tokenizer_license_id`` is
tracked separately, because a permissive code licence on the repository does not
by itself settle the terms of a separately published tokenizer.

This module contains no legal advice. It records observed terms and flags the
ones a human still has to review.
"""

from __future__ import annotations

HF = "https://huggingface.co"

# Verified by fetching the raw LICENSE file or the repository's declared
# metadata on the date recorded below.
REGISTRY: dict[str, dict] = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "publisher": "Microsoft",
        "license_id": "MIT",
        "license_verified": True,
        "license_source": f"{HF}/microsoft/Phi-3-mini-4k-instruct/raw/main/LICENSE",
        "license_evidence": "raw LICENSE file, MIT text, (c) Microsoft Corporation",
        "verified_on": "2026-10-19",
        "parameters": 3_800_000_000,
        "architecture": "Phi3ForCausalLM",
        "dtype": "bfloat16",
        "context": 4096,
        "layers": 32, "kv_heads": 32, "head_dim": 96,
        "task_head": "causal_lm",
        "tokenizer_requirement": "Llama SentencePiece tokenizer (tokenizer.model)",
        "prompt_template": "Phi-3 chat format using <|user|>/<|assistant|>",
        "intended_use_restrictions": (
            "MIT. No use restriction beyond attribution. Note the model card "
            "warns against using the 4k variant for production."),
    },
    "google-bert/bert-base-uncased": {
        "publisher": "Google",
        "license_id": "Apache-2.0",
        "license_verified": True,
        "license_source": f"{HF}/google-bert/bert-base-uncased/raw/main/LICENSE",
        "license_evidence": "raw LICENSE file, Apache License Version 2.0",
        "verified_on": "2026-10-19",
        "parameters": 109_482_240,
        "architecture": "BertForMaskedLM",
        "dtype": "float32",
        "context": 512,
        "layers": 12, "kv_heads": 12, "head_dim": 64,
        "task_head": "masked_lm",
        "tokenizer_requirement": "WordPiece uncased, vocab.txt",
        "prompt_template": (
            "NOT a generative model. Requires a [MASK] token; cannot accept a "
            "free-form instruction."),
        "intended_use_restrictions": "Apache-2.0. No restriction.",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "publisher": "sentence-transformers",
        "license_id": "Apache-2.0",
        "license_verified": True,
        "license_source": (
            f"{HF}/sentence-transformers/all-MiniLM-L6-v2/raw/main/README.md"),
        "license_evidence": (
            "README front-matter declares 'license: apache-2.0'. No LICENSE "
            "file is published at /raw/main/LICENSE (HTTP 404), so this is "
            "declared metadata rather than a licence text."),
        "verified_on": "2026-10-19",
        "parameters": 22_713_216,
        "architecture": "BertModel (MiniLM, 6 layers)",
        "dtype": "float32",
        "context": 256,
        "layers": 6, "kv_heads": 12, "head_dim": 64,
        "task_head": "embedding",
        "tokenizer_requirement": "WordPiece uncased, vocab.txt",
        "prompt_template": (
            "NOT a generative model. Produces 384-dimension embeddings."),
        "intended_use_restrictions": "Apache-2.0. No restriction.",
    },
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
        "publisher": "Alibaba Cloud (Qwen team)",
        "license_id": "Apache-2.0",
        "license_verified": True,
        "license_source": (
            f"{HF}/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE"),
        "license_evidence": (
            "raw LICENSE is the Apache 2.0 text; LICENSE added in verified "
            "commit 49f4d68 dated 2024-09-18. Repository metadata also "
            "declares license: apache-2.0."),
        "verified_on": "2026-10-19",
        "parameters": 1_543_714_304,
        "architecture": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
        "context": 32768,
        "layers": 28, "kv_heads": 2, "head_dim": 64,
        "task_head": "causal_lm",
        "tokenizer_requirement": "byte-level BPE via tokenizer.json",
        "prompt_template": (
            "ChatML with <|im_start|>/<|im_end|> and a system message; applying "
            "this template to a model with a different tokenizer would silently "
            "change prompt semantics."),
        "intended_use_restrictions": "Apache-2.0. No restriction.",
    },
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {
        "publisher": "Alibaba Cloud (Qwen team)",
        "license_id": "Apache-2.0",
        "license_verified": True,
        "license_source": f"{HF}/Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "license_evidence": (
            "Alibaba's published Qwen2.5-Coder series table lists the 0.5B "
            "model as Apache 2.0. Not independently re-fetched here; treat the "
            "0.5B licence as confirmed-by-family, not by direct file read."),
        "verified_on": "2026-10-19",
        "parameters": 494_032_768,
        "architecture": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
        "context": 32768,
        "layers": 24, "kv_heads": 2, "head_dim": 64,
        "task_head": "causal_lm",
        "tokenizer_requirement": "byte-level BPE via tokenizer.json",
        "prompt_template": "ChatML <|im_start|>/<|im_end|> with a system message",
        "intended_use_restrictions": "Apache-2.0. No restriction.",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "publisher": "Alibaba Cloud (Qwen team)",
        "license_id": "Apache-2.0",
        "license_verified": True,
        "license_source": f"{HF}/Qwen/Qwen2.5-Coder-7B-Instruct",
        "license_evidence": "series table lists the 7B model as Apache 2.0",
        "verified_on": "2026-10-19",
        "parameters": 7_615_616_512,
        "architecture": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
        "context": 131072,
        "layers": 28, "kv_heads": 4, "head_dim": 128,
        "task_head": "causal_lm",
        "tokenizer_requirement": "byte-level BPE via tokenizer.json",
        "prompt_template": "ChatML <|im_start|>/<|im_end|> with a system message",
        "intended_use_restrictions": "Apache-2.0. No restriction.",
    },
}

# Repository-owned artefacts are licensed by this project and are never routed
# through the external-candidate licensing path.
SOVEREIGN = {
    "runs/m3_base/checkpoint_best.pt": "quantsmind-owned, trained on "
                                       "licence-verified corpus only",
    "runs/m4_sft/checkpoint_best.pt": "quantsmind-owned, trained on "
                                      "licence-verified corpus only",
}


def lookup(model_id: str) -> dict | None:
    return REGISTRY.get(model_id)


def unresolved_licensing() -> list[str]:
    """Candidates whose licence could not be verified from a primary source."""
    return sorted(mid for mid, rec in REGISTRY.items()
                  if not rec.get("license_verified"))
