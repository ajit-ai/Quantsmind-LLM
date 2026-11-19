============
Evaluation
============

This package contains **measurement infrastructure only**.  It reports model
outputs and metrics; it does not make, change, or enforce promotion decisions.
The promotion gate and its rules live in ``promotion.yaml`` and in
``../docs/ROADMAP.md``.

Nothing here acquires, licenses, or bundles real image data.  Every evaluator
accepts whatever inputs you point it at and reports without judging quality.

Gate context (M1)
=================

The M1 promotion gate requires (among other things):

1. **zero-shot caption** measurably works,
2. **VQA smoke** passes,
3. **no text-task regression** vs the text-only checkpoint, and
4. **loss sanity** (manual review) with no unrecovered loss spikes in the
   final 20% of training.

The entry points below are the tools used to *measure* those criteria on real
data once it is available.  They produce JSON reports and/or human-readable
review packets; they never output a pass/fail promotion verdict themselves.

Entry points
============

All commands are run from the repository root.  ``--device`` defaults to
``cpu`` (pass ``cuda`` if appropriate).  ``--tokenizer`` defaults to
``tokenizer/qm_bpe_v0.json``.

Zero-shot caption (generation)
------------------------------

``python -m evaluation.eval_caption --ckpt runs/mm_m1_smoke/checkpoint_last.pt --image path/to/img.jpg --out report.json``

or evaluate a manifest-driven image set::

``python -m evaluation.eval_caption --ckpt runs/mm_m1_smoke/checkpoint_last.pt --manifest data/mm_m1/MANIFEST.json --split val --max-samples 100 --out report.json``

Output: per-image generated caption, generated token ids, mean per-token
log-probability of the generated sequence (nil if the model immediately emits
EOS).  Reports are measurements; they do not judge whether a caption is
"correct" (a human/another metric does that).

VQA smoke
---------

``python -m evaluation.eval_vqa --ckpt runs/mm_m1_smoke/checkpoint_last.pt --records path/to/records.jsonl --out report.json``

Input records (JSON array, single object, or JSONL)::

    {"image": "imgs/x.jpg", "question": "What is the total?", "expected_answer": "12"}

For each record the model generates an answer conditioned on the image and
question; the normalized answer (lower-cased, whitespace-collapsed) is
compared to the normalized expected answer.  Exact match = pass.  This is a
**smoke** test: it catches grossly broken pipelines, not graded quality.
Missing/malformed records are reported per-record and skipped; the process
always exits ``0`` because it is measurement.

Text-task regression / forgetting
---------------------------------

``python -m evaluation.eval_text --ckpt runs/mm_m1_smoke/checkpoint_last.pt --ref runs/local_15m/checkpoint_last.pt --tokens data/corpus_v0.tokens.bin --seed evaluation/india_finance/seed_v0.jsonl --out report.json``

Reuses the pilot metrics (held-out perplexity on a uint16 memmap, and the
seed question-set NLL by category) on the **text component** of a checkpoint,
so a multimodal checkpoint can be compared directly against the text-only
checkpoint it was warm-started from.  ``--tokens`` and ``--seed`` are both
optional; missing inputs are reported as ``null`` rather than fabricated.
Deltas vs ``--ref`` are reported numerically; thresholds (if any) belong in
``promotion.yaml``, not here.

Loss sanity (manual review helper)
----------------------------------

``python -m evaluation.loss_sanity --log runs/mm_m1_smoke/log.txt --out review.json``

Scans a ``train_mm``-style log (``step N/M loss=...``) for the promotion rule
"no unrecovered loss spikes in the final 20% of training": a spike is a step
whose loss exceeds 1.5x the median of the first 80% of steps; a spike is
recovered if a later step is at or below 1.25x that median.  The output is a
review packet for a **human** to confirm; this tool does not promote anything.

Shared checkpoint loader
========================

``evaluation/mm_checkpoint.load_checkpoint(path, device, vision_cfg=None)``
loads either a text-only ``QMModel`` checkpoint or a multimodal
``QMMultimodal`` checkpoint produced by ``training/train_mm.py``.  For
multimodal checkpoints the vision config is resolved from
``config["run"]["vision_config"]`` (a YAML path, resolved relative to the
checkpoint) or from an explicit inline dict (hermetic tests).  The checkpoint
schema is read-only; nothing here modifies state dicts.

Testing
=======

The harness has hermetic tests (no network, no real data, small CPU models):

``python -m pytest tests/test_eval_mm.py -q``

What is and is not proven
=========================

These evaluators prove: the measurement pipeline runs end-to-end on a
checkpoint, images, questions, token streams, and training logs.

They do NOT prove: that captions/VQA answers are correct, that the model did
not regress on text tasks (deltas are reports), that a checkpoint satisfies
any promotion rule, or that training data provenance is complete.  Those
judgments are made separately (data acquisition+licensing, human review, and
``promotion.yaml``).