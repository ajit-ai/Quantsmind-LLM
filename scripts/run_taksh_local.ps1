# TAKSH Local Coding Assistant (L0.1) — Windows launcher.
#
# Starts the experimental, local-only Taksh inference UI on 127.0.0.1.
# No external services are called; CPU is the default device.
#
# Usage:
#   .\scripts\run_taksh_local.ps1
#   .\scripts\run_taksh_local.ps1 -Port 7862 -Device cpu
#   .\scripts\run_taksh_local.ps1 -MaxNewTokens 64
#
# Stop with Ctrl+C.

[CmdletBinding()]
param(
    [string]$Checkpoint = "runs/v180_sft/checkpoint_last.pt",
    [string]$Tokenizer  = "tokenizer/taksh_code_bpe_v1.json",
    [int]   $Port       = 7861,
    [ValidateSet("cpu", "cuda")]
    [string]$Device     = "cpu",
    [int]   $MaxNewTokens = 64
)

$ErrorActionPreference = "Stop"

# Repository root = parent of this script's directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Prefer the repository-local virtual environment if it exists; otherwise fall
# back to the system Python launcher. No downloads are performed either way.
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
              "Checkout the repository with its model artifacts, or pass -Checkpoint/-Tokenizer."
    }
}

Write-Host "Launching TAKSH Local Coding Assistant on http://127.0.0.1:$Port (loopback only)"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Tokenizer : $Tokenizer"
Write-Host "Device    : $Device"
Write-Host "Press Ctrl+C to stop.`n"

if ($Python -eq "py") {
    & py -3 -m serving.local_assistant --ckpt $Checkpoint --tokenizer $Tokenizer `
        --host 127.0.0.1 --port $Port --device $Device --max-new-tokens $MaxNewTokens
}
else {
    & $Python -m serving.local_assistant --ckpt $Checkpoint --tokenizer $Tokenizer `
        --host 127.0.0.1 --port $Port --device $Device --max-new-tokens $MaxNewTokens
}
exit $LASTEXITCODE
