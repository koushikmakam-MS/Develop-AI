#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Pre-push quality gate — runs lint + tests before allowing a git push.
.DESCRIPTION
    This script runs ruff linting and the full pytest suite.
    Exit code 0 = all clear, non-zero = push should be blocked.
    Usage:
        .\run_tests.ps1           # lint + test
        .\run_tests.ps1 -NoCov   # skip coverage
        .\run_tests.ps1 -Quick   # just tests, no lint
#>
param(
    [switch]$NoCov,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = ".venv/Scripts/python.exe"
if (-not (Test-Path $py)) {
    Write-Host "❌  Virtual-env not found at $py — run setup.ps1 first." -ForegroundColor Red
    exit 1
}

$failed = 0

# ── Step 1: Lint with ruff ──────────────────────────────────────────────
if (-not $Quick) {
    Write-Host "`n🔍  Running ruff lint..." -ForegroundColor Cyan
    & $py -m ruff check brain_ai/ tests/ --no-fix
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Lint issues found (see above)." -ForegroundColor Yellow
        $failed = 1
    } else {
        Write-Host "✅  Lint passed." -ForegroundColor Green
    }
}

# ── Step 2: Run pytest ──────────────────────────────────────────────────
Write-Host "`n🧪  Running tests..." -ForegroundColor Cyan
$pytestArgs = @("-m", "pytest", "tests/", "-v", "--tb=short")
if (-not $NoCov) {
    $pytestArgs += @("--cov=brain_ai", "--cov-report=term-missing")
}
& $py @pytestArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌  Tests failed." -ForegroundColor Red
    $failed = 1
} else {
    Write-Host "✅  All tests passed." -ForegroundColor Green
}

# ── Summary ─────────────────────────────────────────────────────────────
Write-Host ""
if ($failed -ne 0) {
    Write-Host "🚫  Quality gate FAILED — fix issues before pushing." -ForegroundColor Red
    exit 1
} else {
    Write-Host "🎉  Quality gate PASSED — safe to push." -ForegroundColor Green
    exit 0
}
