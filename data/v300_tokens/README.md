# TAKSH v3.0.0 — Training Tokens Directory

This directory contains the pretraining token binaries for qm-code-v300-70b.

## Structure

| File | Description |
|---|---|
| `train.tokens.bin` | Training token IDs (50B+ tokens) |
| `val.tokens.bin` | Validation token IDs |
| `train.tokens.meta.json` | Training metadata |
| `val.tokens.meta.json` | Validation metadata |
| `manifest.json` | Corpus provenance |

## Generation

Run `scripts/build_v300_corpus.py` to collect cleared code corpora (50B+ tokens),
then `data/encode_dataset.py` to encode to token bins.

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
