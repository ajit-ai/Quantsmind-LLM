# Data Card — corpus_v0 (pilot-scale, ₹0 tier)

Created: 2026-08-24 · Pipeline: `data/build_corpus.py` · Tokenizer: `tokenizer/qm_bpe_v0.json`

## Contents

| Field | Value |
|---|---|
| Chunks (train-ready) | 9,311 |
| Tokens (own 49,152-BPE tokenizer) | ~10.0M |
| Sources | wikipedia-en (8,764 chunks), gutenberg-econ (537), wikipedia-hi (10) |
| Licenses | CC-BY-SA-3.0 (Wikipedia), Public Domain (Gutenberg) |
| Chunk size | ≤6,000 chars, split at paragraph boundaries |
| Filters applied | quality (length, alpha-ratio ≥0.6) · exact dedup (SHA-1) · PII scrub (Aadhaar/PAN/phone/email) |
| Rejections | 1,113 chunks (quality), 0 dups |

## Known limitations (v0)
- ~10M tokens is **pilot-probe scale only** — 300M pilot needs ~12B tokens (≈1,200× more)
- Hindi underrepresented (Hindi Wikipedia stubs are short) — plan: longer Hindi sources (NCERT texts, Hindi news archives)
- Exact dedup only; fuzzy MinHash dedup lands before the 12B-token build
- No decontamination pass yet vs eval sets (required before any training run)

## Reproduce
```powershell
python -m data.sources.wikipedia --lang en --max-docs 2000 --out data/raw/wikipedia/wiki_en.jsonl
python -m data.sources.wikipedia --lang hi --max-docs 500  --out data/raw/wikipedia/wiki_hi.jsonl
python -m data.sources.gutenberg --out-dir data/raw/gutenberg
python -m data.build_corpus
python -m tokenizer.train_tokenizer --input data/corpus_v0.txt --out tokenizer/qm_bpe_v0.json --vocab-size 49152
python -m data.encode_dataset --corpus data/corpus_v0.txt --tokenizer tokenizer/qm_bpe_v0.json --out data/corpus_v0.tokens.bin
```

## Provenance ledger note
Every chunk row carries `source` + `license`. Raw downloads carry `MANIFEST.json`
(Gutenberg) or dataset citation (Wikimedia 20231101 dump via HF `datasets` 5.0.1).
