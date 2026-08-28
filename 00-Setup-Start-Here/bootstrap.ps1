# One-stop setup: installs uv (if needed) and syncs this repo's Python
# environment. Safe to re-run any time.
#
# Usage (from anywhere), in PowerShell:
#   powershell -ExecutionPolicy ByPass -File 00-Setup-Start-Here\bootstrap.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "== Designing Quantum Applications: environment bootstrap =="

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "-> uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    $UvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $UvBin) {
        $env:Path = "$UvBin;$env:Path"
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "uv was installed but isn't on PATH in this session."
        Write-Host "Close and reopen PowerShell, then re-run:"
        Write-Host "  powershell -ExecutionPolicy ByPass -File 00-Setup-Start-Here\bootstrap.ps1"
        exit 1
    }
} else {
    Write-Host "-> uv already installed ($(uv --version))"
}

Write-Host "-> Setting up the environment (uv sync)..."
Set-Location $RepoRoot
uv sync

Write-Host "-> Verifying the install..."
uv run 00-Setup-Start-Here/01_check_install.py

Write-Host ""
Write-Host "== Bootstrap complete =="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Save your IBM Quantum API token:"
Write-Host "       uv run 00-Setup-Start-Here/02_save_token.py"
Write-Host "  2. Verify the token and check your instance:"
Write-Host "       uv run 00-Setup-Start-Here/03_check_token.py"
Write-Host "  3. Run a real circuit:"
Write-Host "       uv run 00-Setup-Start-Here/04_test_quantum.py"
Write-Host ""
Write-Host "See README.md for full details."
