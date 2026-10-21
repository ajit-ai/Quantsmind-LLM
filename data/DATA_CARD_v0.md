# Data Card — corpus_v0 (pilot-scale, ₹0 tier)

Created: 2026-08-24 · Pipeline: `data/build_corpus.py` · Tokenizer: `tokenizer/qm_bpe_v0.json`

## Contents

| Field | Value |
|---|---|
| Chunks (train-ready) | 9,308 |
| Tokens (own 49,152-BPE tokenizer) | 9,993,983 |
| Sources | wikipedia-en (8,764 chunks), gutenberg-econ (534), wikipedia-hi (10) |
| Licenses | CC-BY-SA-3.0 (Wikipedia), Public Domain (Gutenberg) |
| Chunk size | ≤6,000 chars, split at paragraph boundaries |
| Filters applied | quality (length, alpha-ratio ≥0.6) · exact dedup (SHA-1) · PII scrub (Aadhaar/PAN/phone/email) |
| Rejections | 1,113 chunks (quality), 0 dups |

## Known limitations (v0)
- 9,993,983 tokens is **pilot-probe scale only**. The `pilot_300m.yaml` config (actually 113,145,408 params) targets ~12B tokens (≈1,200× more)
- Hindi underrepresented (Hindi Wikipedia stubs are short) — plan: longer Hindi sources (NCERT texts, Hindi news archives)
- Exact dedup plus MinHash-LSH was applied, but LSH registers a signature only under the first unmatched band, so near-duplicate recall is degraded (Open Issue B7)
- No decontamination pass yet vs eval sets (required before any training run). `docs/PAUSED.md` previously described this corpus as "decontaminated"; that was wrong and has been corrected
- **No data card exists for the code corpus.** `data/code_corpus_v0.jsonl` (77 chunks, 100,244 tokens) is sourced entirely from this repository's own files and has no licence provenance record

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
