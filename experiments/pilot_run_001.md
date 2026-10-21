# Experiment: pilot_run_001 — 113M pilot on Kaggle T4

> **Status corrected 2026-10-14: NO RESULTS EXIST.** This entry has read "RUNNING"
> since 2026-08-24 with every result field blank. No `pilot_report.json`, no
> `loss_curve.png`, and no `qm_pilot_final.pt` exist anywhere in the repository or under
> `runs/`. The run either never started or was abandoned without recording an outcome.
> **Do not cite this experiment as evidence of anything.**
>
> Also note: `scripts/kaggle_pilot.ipynb` is a **self-contained fork** of the
> architecture. It defines its own `RMSNorm`/GQA/SwiGLU and its own training loop
> (cells 6 and 10) and never imports `training/`. Even if completed, it would have
> validated a copy of the code, not this repository's training engine
> (Open Issue E5).

Status: **NO RECORDED RESULT** (previously: "RUNNING", started 2026-08-24)
Notebook: scripts/kaggle_pilot.ipynb (fixed: RoPE + 0.02 init)
Corpus: corpus_v0 (9,993,983 tokens) · Tokenizer: qm_bpe_v0 (49,152 BPE)
Model: `configs/model/pilot_300m.yaml`, which computes to **113,145,408 params** —
the "113M" in the title is correct; the `pilot_300m` filename is not

## Result fields — ALL EMPTY, NO DATA RECORDED

| Field | Value |
|---|---|
| GPU | unknown |
| Steps completed | — |
| Tokens seen (M) | — |
| Initial eval loss (expect ~10.8) | — |
| Final eval loss (pass < 5.0) | — |
| Throughput (tok/s) | — |
| Wall clock (min) | — |
| Crashes / NaN | — |

## Artifacts
- [ ] pilot_report.json — **absent**
- [ ] loss_curve.png — **absent**
- [ ] qm_pilot_final.pt — **absent**

## Verdict
- Held-out perplexity: **unmeasured**
- India-finance seed NLL: **unmeasured**
- Next action: rerun against `training/pretrain.py` (not the forked notebook) once
  `configs/train/pretrain_pilot.yaml` is made executable (Open Issue B4)
