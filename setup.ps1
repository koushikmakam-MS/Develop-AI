#!/usr/bin/env pwsh
<#
.SYNOPSIS
    BCRD DeveloperAI — One-command setup & initialization.

.DESCRIPTION
    Runs ALL prerequisite steps in order:
      1. Create Python virtual environment (if missing)
      2. Install pip dependencies
      3. Copy config templates (if config.yaml / config.local.yaml don't exist)
      4. Sync docs from Azure DevOps
      5. Index docs into ChromaDB
      6. Sync + index source code into ChromaDB
      7. (Optional) Start the Kusto MCP server
      8. (Optional) Run the Doc Improver bootstrap

.PARAMETER Force
    Force re-sync and re-index everything from scratch.

.PARAMETER SkipKusto
    Don't start the Kusto MCP server at the end.

.PARAMETER SkipDocImprover
    Don't run the Doc Improver cycle.

.PARAMETER ConfigPath
    Path to config.yaml (default: config.yaml in project root).

.EXAMPLE
    .\setup.ps1                     # Normal setup
    .\setup.ps1 -Force              # Force re-sync + re-index everything
    .\setup.ps1 -SkipKusto          # Skip starting the Kusto server
#>

param(
    [switch]$Force,
    [switch]$SkipKusto,
    [switch]$SkipDocImprover,
    [string]$ConfigPath = "config.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# ── Helpers ──────────────────────────────────────────────────────────

function Write-Step($num, $msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Step $num — $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Ok($msg) {
    Write-Host "  ✅ $msg" -ForegroundColor Green
}

function Write-Skip($msg) {
    Write-Host "  ⏭️  $msg" -ForegroundColor Yellow
}

function Write-Fail($msg) {
    Write-Host "  ❌ $msg" -ForegroundColor Red
}

# ── Banner ───────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  🧠 BCRD DeveloperAI Setup & Initialization" -ForegroundColor Magenta
Write-Host "  ==================================" -ForegroundColor Magenta
Write-Host "  Project root: $ProjectRoot"
Write-Host "  Config:       $ConfigPath"
if ($Force) { Write-Host "  Mode:         FORCE (re-sync + re-index all)" -ForegroundColor Yellow }
Write-Host ""

$forceArg = if ($Force) { "--force" } else { "" }

# ── Step 1: Python Virtual Environment ───────────────────────────────

Write-Step 1 "Python virtual environment"

$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "  Creating virtual environment..."
    python -m venv .venv
    Write-Ok "Virtual environment created at .venv/"
} else {
    Write-Ok "Virtual environment already exists."
}

# Activate it
$activateScript = Join-Path $venvPath "Scripts" "Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Ok "Virtual environment activated."
} else {
    Write-Fail "Could not find activation script at $activateScript"
    exit 1
}

# ── Step 2: Install Dependencies ─────────────────────────────────────

Write-Step 2 "Install Python dependencies"

Write-Host "  Installing from requirements.txt..."
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install failed."; exit 1 }

Write-Host "  Installing package in editable mode..."
pip install -q -e .
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install -e . failed."; exit 1 }

Write-Ok "All dependencies installed."

# ── Step 3: Config Files ─────────────────────────────────────────────

Write-Step 3 "Configuration"

# 3a. config.yaml (generic settings — safe to commit)
if (-not (Test-Path $ConfigPath)) {
    if (Test-Path "config.yaml.template") {
        Copy-Item "config.yaml.template" $ConfigPath
        Write-Ok "config.yaml created from template."
    } else {
        Write-Fail "No config.yaml or config.yaml.template found."
        exit 1
    }
} else {
    Write-Ok "config.yaml found."
}

# 3b. config.local.yaml (secrets — gitignored)
$localConfigPath = Join-Path (Split-Path $ConfigPath) "config.local.yaml"
if (-not (Test-Path $localConfigPath)) {
    if (Test-Path "config.local.yaml.template") {
        Copy-Item "config.local.yaml.template" $localConfigPath
        Write-Host ""
        Write-Host "  ⚠️  config.local.yaml created from template." -ForegroundColor Yellow
        Write-Host "  ⚠️  Please edit config.local.yaml and fill in your secrets:" -ForegroundColor Yellow
        Write-Host "       - Azure DevOps PAT"
        Write-Host "       - Azure AI Foundry endpoint & API key"
        Write-Host "       - Kusto cluster URL & database"
        Write-Host ""
        Write-Host "  Then re-run this script." -ForegroundColor Yellow
        exit 0
    } else {
        Write-Fail "No config.local.yaml or config.local.yaml.template found."
        exit 1
    }
} else {
    Write-Ok "config.local.yaml found."
}

# Quick validation: check that placeholder values are not present in local config
$localContent = Get-Content $localConfigPath -Raw
if ($localContent -match "<YOUR_") {
    Write-Host ""
    Write-Host "  ⚠️  config.local.yaml still has placeholder values (<YOUR_...>)." -ForegroundColor Yellow
    Write-Host "  ⚠️  Please fill in your actual secrets before continuing." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# ── Step 4: Sync Docs from Azure DevOps ──────────────────────────────

Write-Step 4 "Sync docs from Azure DevOps"

$syncArgs = @("run_sync.py")
if ($Force) { $syncArgs += "--force" }
if ($ConfigPath -ne "config.yaml") { $syncArgs += "--config", $ConfigPath }

python @syncArgs
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Doc sync failed — check your Azure DevOps PAT and repo URL."
    Write-Host "  Continuing anyway (you can re-run sync later)..." -ForegroundColor Yellow
} else {
    Write-Ok "Docs synced."
}

# ── Step 5: Index Docs into ChromaDB ─────────────────────────────────

Write-Step 5 "Index docs into ChromaDB"

$indexArgs = @("run_index.py")
if ($Force) { $indexArgs += "--force" }
if ($ConfigPath -ne "config.yaml") { $indexArgs += "--config", $ConfigPath }

python @indexArgs
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Doc indexing failed."
} else {
    Write-Ok "Docs indexed into ChromaDB."
}

# ── Step 6: Sync + Index Source Code ─────────────────────────────────

Write-Step 6 "Index source code into ChromaDB"

$codeArgs = @("run_code_index.py")
if ($Force) { $codeArgs += "--force" }
if ($ConfigPath -ne "config.yaml") { $codeArgs += "--config", $ConfigPath }

python @codeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Code indexing failed."
} else {
    Write-Ok "Source code indexed into ChromaDB."
}

# ── Step 7: Doc Improver (optional) ──────────────────────────────────

Write-Step 7 "Doc Improver (bootstrap / improvement cycle)"

if ($SkipDocImprover) {
    Write-Skip "Skipped (use -SkipDocImprover:$false to enable)."
} else {
    # Check if doc_improver is enabled in config
    $docImproverEnabled = python -c "from brain_ai.config import get_config; c=get_config(); print(c.get('doc_improver',{}).get('enabled',False))" 2>$null
    if ($docImproverEnabled -eq "True") {
        Write-Host "  Running Doc Improver cycle (this may take a while)..."
        python run_doc_improver.py --force
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Doc Improver cycle failed."
        } else {
            Write-Ok "Doc Improver cycle complete."
        }
    } else {
        Write-Skip "Doc Improver is disabled in config. Set doc_improver.enabled: true to activate."
    }
}

# ── Step 8: Kusto MCP Server (optional) ─────────────────────────────

Write-Step 8 "Kusto MCP Server"

if ($SkipKusto) {
    Write-Skip "Skipped (use -SkipKusto:$false to enable)."
} else {
    Write-Host "  Starting Kusto MCP server in background..."
    $kustoJob = Start-Job -ScriptBlock {
        param($root, $cfg)
        Set-Location $root
        & (Join-Path $root ".venv" "Scripts" "Activate.ps1")
        python run_kusto_server.py
    } -ArgumentList $ProjectRoot, $ConfigPath
    Start-Sleep -Seconds 3

    if ($kustoJob.State -eq "Running") {
        Write-Ok "Kusto MCP server running in background (Job ID: $($kustoJob.Id))."
        Write-Host "  Use 'Stop-Job $($kustoJob.Id)' to stop it."
    } else {
        Write-Fail "Kusto MCP server failed to start."
    }
}

# ── Done ─────────────────────────────────────────────────────────────

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  🎉 BCRD DeveloperAI setup complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    python run_chat.py              # Start CLI chat"
Write-Host "    python run_bot.py               # Start Teams Bot"
Write-Host "    python run_doc_improver.py      # Run Doc Improver manually"
Write-Host "    python run_daily.py             # Run daily sync + re-index"
Write-Host ""
