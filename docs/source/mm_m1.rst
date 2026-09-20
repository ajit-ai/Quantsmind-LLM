Multimodal M1 — architecture fit-check and real-data preparation
================================================================

.. _mm-m1-status:

Status
------

.. warning::

   **M1 is not promoted.** M1.1 establishes synthetic multimodal architecture and
   optimization evidence only. M1 promotion still requires real-data evaluation:
   caption NLL improvement against the no-image baseline, retrieval improvement,
   text regression within the defined tolerance, no NaNs, and reproducible CPU
   behaviour. The synthetic result (``10.747 -> 0.2821``) must not be presented as
   M1 training evidence.

Milestones
----------

* **M1.1 — COMPLETED.** Synthetic multimodal architecture fit-check.
* **M1.2 — in preparation.** Real-data infrastructure (no large training run).
* **M2 — NOT STARTED.** No image generation, no VQ-VAE, no image-token vocab.

M1.1 — architecture completed
=============================

Components
----------
* ``training/vision.py`` — from-scratch ViT: ``PatchEmbed``, GQA-capable
  ``VitAttention``, pre-norm SwiGLU ``VitBlock``, ``VisionEncoder``. Patch 16,
  ``112 x 112`` input -> 49 patches + CLS = 50 image tokens. Learned positional
  embeddings (no RoPE), non-causal self-attention.
* ``training/multimodal.py`` — ``QMMultimodal``: VisionEncoder -> Projector
  (``Linear(128,256) -> SiLU -> Linear(256,128)``) -> shared QM decoder. MM
  attention mask semantics: image->image visible, text->image visible, text->text
  causal, image->text blocked. Vocab 49156 = 49152 BPE + 4 specials:
  ``IMG_START`` (49152), ``IMG_END`` (49153), ``CAPTION_START`` (49154),
  ``TEXT_EOS`` (49155). Future image-token vocab boundary 51200 is documented only.
* ``training/model.py`` — additive M0 refactor: ``embed``/``decode``/
  ``forward_embeds`` plus an optional ``attn_mask`` in ``Attention``; the M0
  ``forward`` and its causal behaviour are unchanged.
* ``training/image_data.py`` — ``ImageTextDataset`` synthetic loader (unchanged
  path in M1.2).
* ``training/train_mm.py`` — fit-check training loop.
* ``scripts/synth_images.py`` — deterministic hermetic synthetic pairs
  (6 colors x 5 shapes x 6 positions x 2 sizes).

Synthetic fit-check completed
-----------------------------
Ran 500 steps on CPU (1 thread) warm-started from the real ``local_15m`` M0
checkpoint, on 256 synthetic pairs:

* initial loss ``10.747`` ~= ``ln(49156)`` (uniform-loss baseline ``~10.80``)
* final loss ``0.2821`` (target ``< 8.0``) — synthetic memorization achieved
* ViT and Projector gradients non-zero and finite throughout
* conditioning probe: max |logit diff| ``~0.578`` — the image changes next-token
  distributions
* CPU wall-clock ``~1411 s`` for 500 steps at ``~345 tok/s`` (1 thread)
* 9,513,216 params total (text 7,768,192 + vision 1,679,488 + projector 65,536)

Tests
-----
* ``python -m pytest tests -q`` — **65 passed at M1.1**, 85 after M1.2.
* ``tests/test_vision.py``, ``tests/test_multimodal.py`` (mask semantics,
  special-token layout, gradients reach ViT/projector),
  ``tests/test_m0_compat.py`` (vocab-extension preserves rows, bit-identical
  logits on the original 49152 columns).

M0 compatibility result
-----------------------
* Warm-start loads the text backbone with 0 new/ignored params.
* ``torch.equal`` holds across all 49152 text-logit columns vs the untouched M0
  model; tok_emb rows ``0..49151`` identical; tied lm_head identical.
* Existing checkpoint keys (``model``/``optimizer``/``step``/``config``/``cfg``)
  preserved; ``metadata`` added additively.

CPU fit-check result
--------------------
See *Synthetic fit-check completed*. Result artifact:
``runs/mm_fit_check/checkpoint_last.pt`` (gitignored).

Limitations
-----------
* Synthetic, hermetic data only — no real images, no caption quality.
* No M1 training evidence; memorization of 256 triple-patterns is not a skill.
* Single image span per sequence; multi-span is masked but not exercised.
* No image generation or continuous-embedding decoding (M2).
* M1.1 is architecture + optimization evidence only.

M1.2 — real-data preparation
=============================

Dataset structure
-----------------
.. code-block:: text

   data/mm_m1/
       MANIFEST.json      # provenance-aware sample index (schema version 1)
       DATA_CARD.rst      # data card
       train/             # split directories; image files are never committed
       val/
       test/

Real images are not committed; ``data/mm_m1/.gitignore`` keeps them out. Nothing is
downloaded automatically.

Manifest format
---------------
Top-level: ``schema_version``, ``preprocessing_version``, ``dataset`` (name,
splits, image resolution), ``samples``. Each sample:

* ``sample_id`` (required), ``split`` in ``{train, val, test}`` (required)
* ``image_path`` (required, relative to the manifest directory)
* ``caption`` (required)
* ``source`` (required)
* ``source_url``, ``license``, ``attribution``, ``width``, ``height`` (optional)
* ``preprocessing_version``

Parser: ``training/mm_manifest.py`` — ``load_manifest`` validates required fields
and raises ``ValueError`` on malformed input.

Provenance requirements
-----------------------
``source`` is required for every sample. ``license`` must be the actual license
when known and ``null``/absent when unknown. ``source_url`` and ``attribution``
are recorded when known. The project never invents provenance or licensing
information; additions must be backed by a documented acquisition log.

Preprocessing
-------------
``training/image_data.py::preprocess_image`` — deterministic, no augmentation:

1. RGB conversion
2. cover-resize (LANCZOS)
3. center crop to the M1 resolution ``112 x 112``
4. float32 ``(112, 112, 3)`` tensor in ``[0, 1]``
5. optional per-channel normalization when configured

``RealImageDataset`` drives the flow ``MANIFEST.json -> image file -> RGB ->
resize/crop -> M1 tensor -> caption/token sequence``, exposes the same batch
contract as ``ImageTextDataset``, and fails loudly (``FileNotFoundError`` with the
sample id) on missing files. Pillow is required for image decoding.

Training configuration
----------------------
``configs/train/mm_m1_local.yaml`` — model, train/val/test manifests, image
resolution, sequence length, batch, gradient accumulation, learning rate, warmup,
scheduler, gradient clipping, seed, max steps, checkpoint directory, CPU thread
limits. It replaces neither ``configs/train/mm_fit_check.yaml`` (synthetic) nor the
M0 ``local_15m.yaml``.

Reproducibility metadata
------------------------
Checkpoints additionally record (additively, all existing keys preserved):
configuration, random seed, dataset manifest identity/hash, training step,
optimizer and model state, preprocessing version, and the git commit when
available. No network access is required.