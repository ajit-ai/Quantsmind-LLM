# Kaggle Pilot Run Guide (₹0)

Goal: train your **113M-param pilot** on **your** 10M-token corpus using Kaggle's free GPU.
No credit card. ~30 free GPU-hours/week. This validates the entire training pipeline.

## Step 1 — Kaggle account (2 min)
1. Go to https://www.kaggle.com → Sign up (or Google login)
2. Verify your phone (needed to unlock GPU quota): Settings → Phone verification

## Step 2 — Upload your corpus as a Dataset (5 min)
1. Kaggle → **Datasets** → **New Dataset**
2. Upload these two files from your repo folder `data/`:
   - `corpus_v0.tokens.bin`
   - `corpus_v0.tokens.meta.json`
3. Dataset name: **`qm-corpus-v0`** → Publish (visibility Private is fine)
4. Note your dataset slug — it looks like `yourusername/qm-corpus-v0`

## Step 3 — Create the notebook (3 min)
1. Kaggle → **Code** → **New Notebook**
2. **File → Import Notebook** → upload `scripts/kaggle_pilot.ipynb` from this repo
3. Right panel → **Session options**:
   - Accelerator: **GPU T4 x2** (or P100)
   - Internet: **Off** (not needed — everything is preinstalled)
4. In the config cell (cell 2), set:
   - `tokens_path = "/kaggle/input/qm-corpus-v0/corpus_v0.tokens.bin"`
   - `meta_path = "/kaggle/input/qm-corpus-v0/corpus_v0.tokens.meta.json"`
   (Kaggle mounts datasets at `/kaggle/input/<dataset-name>/...` — cell 1 prints
   the exact paths it finds, so you can copy-paste from its output)

## Step 4 — Run (8–9 hours, unattended)
- **Run All**. The notebook:
  - prints model params (~113M) and initial eval loss (~ln(49152) ≈ 10.8 expected at init)
  - trains with fp16 autocast, cosine LR, NaN watchdog
  - **self-stops at 8.5h** (Kaggle sessions cap at 12h) and saves everything
- You can close the browser — use **Save Version → Save & Run All** for background execution
- Expected: loss falling from ~10.8 toward 4–5 range; tokens/sec printed every 25 steps

## Step 5 — Collect results (5 min)
From the notebook output (`/kaggle/working`), download:
- `qm_pilot_final.pt` — **your trained weights** (keep safe; ~450MB)
- `pilot_report.json` — attach to `experiments/` in the repo
- `loss_curve.png` — your first training curve 🎉

## Success criteria (Phase-1 exit for this task)
| Signal | Pass |
|---|---|
| Initial loss | ≈ 10.8 (ln of vocab) |
| Final eval loss | < 5.0 and still trending down |
| No NaN / crashes | watchdog silent |
| Throughput | ≥ 15k tok/s on T4 (fp16) |

## Troubleshooting
| Problem | Fix |
|---|---|
| "input dataset not found" | Check slug path; re-run cell 1 to list mounted inputs |
| OOM (rare on 16GB) | Set `batch_size=8` in config cell |
| Session died mid-run | Checkpoints save every 500 steps to output; re-run continues from scratch (resume support comes later) |
| Quota exhausted | Wait for weekly reset (Sat); 30h is enough for this pilot |

## What this proves
Pipeline end-to-end: your tokenizer → your tokens → your architecture → your trained
weights → your eval numbers. The 12B-token / 1.5B runs use the **same code at bigger scale**.
