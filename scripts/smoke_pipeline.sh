#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m data.prepare_sample --out data/sample_corpus.txt --lines 2000
python -m tokenizer.train_tokenizer --input data/sample_corpus.txt --out runs/smoke/tokenizer.json --vocab-size 500
python -m data.encode_dataset --corpus data/sample_corpus.txt --tokenizer runs/smoke/tokenizer.json --out runs/smoke/tokens.bin
python -m training.pretrain --config configs/train/smoke_toy.yaml --out-dir runs/smoke --tokens runs/smoke/tokens.bin

test -f runs/smoke/checkpoint_last.pt
echo "SMOKE OK"
