# TAKSH Local API — Windows launcher.
#
# Starts the OpenAI-compatible local coding backend on 127.0.0.1. No external
# service is contacted; weights are loaded once from the local checkpoint.
#
# Usage:
#   .\scripts\run_taksh_api.ps1
#   .\scripts\run_taksh_api.ps1 -Checkpoint runs\v180_sft\checkpoint_last.pt
#   .\scripts\run_taksh_api.ps1 -Port 8010 -StrictHistory
#   $env:QM_API_KEY = "secret"; .\scripts\run_taksh_api.ps1
#
# Stop with Ctrl+C.

[CmdletBinding()]
param(
    [string]$Checkpoint = "runs/i4_aligned_sft_long/checkpoint_last.pt",
    [string]$Tokenizer  = "tokenizer/taksh_code_bpe_v1.json",
    [int]   $Port       = 8002,
    [string]$Host       = "127.0.0.1",
    [switch]$StrictHistory,
    [switch]$AllowRemote
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot "taksh-env\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
    Write-Host "Using repository virtual environment: $Python"
}
else {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        $Python = "py"
        Write-Host "Using Python launcher: py"
    }
    else {
        $Fallback = Get-Command python -ErrorAction SilentlyContinue
        if (-not $Fallback) {
            throw "No Python interpreter found. Create taksh-env or install Python 3.10+."
        }
        $Python = "python"
        Write-Host "Using system python: $Fallback"
    }
}

foreach ($Path in @($Checkpoint, $Tokenizer)) {
    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path (relative to $RepoRoot). " +
              "Pass -Checkpoint/-Tokenizer or train/select a local checkpoint."
    }
}

$ServerArgs = @(
    "-m", "serving.openai_server",
    "--ckpt", $Checkpoint,
    "--tokenizer", $Tokenizer,
    "--host", $Host,
    "--port", "$Port"
)

if (-not $StrictHistory) {
    $ServerArgs += "--flatten-history"
}
if ($AllowRemote) {
    $ServerArgs += "--allow-remote"
}

Write-Host ""
Write-Host "TAKSH Local API"
Write-Host "  base URL   : http://${Host}:$Port/v1"
Write-Host "  checkpoint : $Checkpoint"
Write-Host "  tokenizer  : $Tokenizer"
Write-Host "  auth       : $(if ($env:QM_API_KEY) { 'bearer key set' } else { 'off (loopback only)' })"
Write-Host "  history    : $(if ($StrictHistory) { 'strict contract (multi-turn refused)' } else { 'flattened compatibility mode' })"
Write-Host ""
Write-Host "Capability is not claimed: the frozen Native v0.1 baseline is 0/20."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

if ($Python -eq "py") {
    & py -3 @ServerArgs
}
else {
    & $Python @ServerArgs
}
exit $LASTEXITCODE
