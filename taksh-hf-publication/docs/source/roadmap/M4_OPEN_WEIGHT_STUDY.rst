Open-Weight Model Strategy Study
================================

.. contents:: On this page
   :local:
   :depth: 2

Method and honesty rules
------------------------

Every row below is labelled with its evidence class:

``VERIFIED``
   read directly from the model's licence file or official model card.
``INFERENCE``
   a conclusion drawn from verified facts; it is reasoning, not a measurement.
``ESTIMATE``
   arithmetic or a third-party claim that has not been measured here.
``UNKNOWN``
   not established, and not guessed.

**No model was downloaded or run for this study.** Downloading a candidate to
experiment with it is a separate, authorised step. The "QuantsMind suitability"
column is therefore reasoning about constraints, not a benchmark result.

Nothing here is a comparison of QuantsMind against another model. No such
comparison was run, and none is implied.

Verified licensing facts
------------------------

.. list-table:: Verified from the licence files / model cards
   :header-rows: 1
   :widths: 34 24 42

   * - Family
     - Licence
     - Evidence
   * - Qwen3-Coder (incl. 480B-A35B, 30B-A3B, Coder-Next)
     - **Apache-2.0**
     - VERIFIED: the repository LICENSE file is the Apache License 2.0 text,
       Copyright 2024 Alibaba Cloud
   * - Qwen2.5-Coder 0.5B / 1.5B / 7B / 14B / 32B
     - **Apache-2.0**
     - VERIFIED: family is Apache 2.0 across these sizes
   * - Qwen2.5-Coder **3B**
     - **Qwen Research licence**
     - VERIFIED: the 3B size is the documented exception. **Not a commercial
       licence** -- it must not be used for a commercial product without
       explicit review
   * - Community fine-tunes of Qwen2.5-Coder (various)
     - Apache-2.0 claimed on their cards
     - ESTIMATE: third-party self-declared. Using someone else's fine-tune adds
       a supply-chain and trust question that the base model does not

Candidate assessment
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 20 30

   * - Candidate
     - Params
     - Licence
     - Runs on this host?
     - QuantsMind suitability
   * - Qwen3-Coder-Next
     - 80B (3B active MoE)
     - Apache-2.0
     - **No**
     - VERIFIED: 256k native context; strongest agentic coding of the family.
       Far beyond 3.91 GB RAM. MoE + hybrid attention means a new
       implementation
   * - Qwen3-Coder-30B-A3B
     - 30B (3B active)
     - Apache-2.0
     - **No**
     - MoE. Not trainable here
   * - Qwen2.5-Coder-7B-Instruct
     - 7B
     - Apache-2.0
     - Inference only, with a quantised build
     - VERIFIED: ~5.5T tokens of code pretraining; GGUF Q4_K_M builds exist at
       roughly 4.4 GB. **Does not fit in 3.91 GB total RAM**
   * - Qwen2.5-Coder-1.5B-Instruct
     - 1.5B
     - Apache-2.0
     - **Yes**, at ~941 MB (GGUF Q4_K_M)
     - VERIFIED: base and instruct variants, FIM support. The realistic
       local candidate
   * - Qwen2.5-Coder-0.5B-Instruct
     - 0.5B
     - Apache-2.0
     - **Yes**, at ~380 MB (GGUF Q4_K_M)
     - The only candidate comfortably inside current RAM. Weakest code ability
       of the family
   * - Mistral / DeepSeek code-capable
     - varies
     - varies
     - not assessed
     - UNKNOWN: not investigated here. Choosing a candidate requires the same
       licence and size analysis before it can be recommended

Hardware reality
----------------

.. list-table:: Measured
   :header-rows: 1
   :widths: 40 60

   * - Item
     - Value
   * - RAM
     - 3.91 GiB total; 31-337 MiB available during M3/M4 (contended)
   * - Device
     - CPU only, no CUDA, 4 torch threads
   * - Training throughput
     - ~2,500 tokens/s on a 2.85M model; ~44 min for 400 steps

INFERENCE: a 7B model would not fit for *training* on this host at any
quantisation, because the optimiser states alone exceed available RAM. Q4 GGUF
suits *inference*, not fine-tuning.

The compatibility problem
--------------------------

INFERENCE, from two verified facts:

1. ``training/model.py`` implements a Llama-family decoder -- RMSNorm, RoPE,
   grouped-query attention, SwiGLU, tied embeddings, no biases.
2. The Qwen families use different architectures (Qwen2 adds biases and uses
   QKV-projection norms; Qwen3-Next is MoE with hybrid attention).

So adopting **any** Qwen checkpoint is an architecture change, not a weight
swap. It requires either a new model implementation or a switch to HF
``transformers``. It is also a tokenizer change: QuantsMind's own checkpoints
embed 5822 dimensions and would no longer load.

This is the single most important practical finding of the study, and it is not
visible from the licence tables.

Recommendation
--------------

**Hybrid, staged.** The two tracks are not alternatives; they answer different
questions.

1. **Keep the from-scratch track, but stop treating it as the product path for
   now.** It proves the pipeline (M3) and the adaptation machinery (M4), and
   those are reusable regardless of base. Its binding constraints are data
   volume (1.2% of target) and licence authorisation, not code.

2. **Evaluate ``Qwen2.5-Coder-1.5B-Instruct`` as the practical product base.** It
   is Apache-2.0, has both base and instruct variants, has a quantised build that
   fits this class of hardware, and is small enough to fine-tune with LoRA once
   more memory is available.

3. **Do not download anything yet.** Adopting an external base changes the
   product's identity, the tokenizer contract, the checkpoint format and the
   licence obligations. That is a product decision for the owner, not an
   engineering detail.

4. **Resolve the licence questions first.** Two corpus sources
   (``psf-black``, ``pypa-setuptools``) still await human judgement, and the
   source allowlist needs authorisation. Those block corpus growth whichever
   base is chosen.

What would change the recommendation
------------------------------------

* If a GPU host with >=24 GB became available: Qwen2.5-Coder-7B base with LoRA
  becomes credible, and the comparison against QuantsMind's own model becomes
  worth running.
* If the owner requires QuantsMind's weights to be entirely independently
  trained: Strategy A remains the only path, and the corpus scale problem must
  be solved first.
* If licence review rejects the external models: the decision is made for us,
  and Strategy A is the whole plan.
