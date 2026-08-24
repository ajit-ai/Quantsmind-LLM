$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m data.prepare_sample --out data/sample_corpus.txt --lines 2000
if (-not $?) { exit 1 }
python -m tokenizer.train_tokenizer --input data/sample_corpus.txt --out runs/smoke/tokenizer.json --vocab-size 500
if (-not $?) { exit 1 }
python -m data.encode_dataset --corpus data/sample_corpus.txt --tokenizer runs/smoke/tokenizer.json --out runs/smoke/tokens.bin
if (-not $?) { exit 1 }
python -m training.pretrain --config configs/train/smoke_toy.yaml --out-dir runs/smoke --tokens runs/smoke/tokens.bin
if (-not $?) { exit 1 }

if (Test-Path "runs/smoke/checkpoint_last.pt") {
    Write-Host "SMOKE OK"
} else {
    Write-Host "checkpoint missing"
    exit 1
}
