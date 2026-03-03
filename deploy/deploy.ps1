#!/usr/bin/env pwsh
# ============================================================
# deploy.ps1 — BCDR DeveloperAI Azure Deployment Script
# ============================================================
# Deploys the BCDR DeveloperAI Teams Bot to Azure Container Apps.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker Desktop running (for image build)
#   - Bot Framework registration (az bot create, or via portal)
#
# Usage:
#   .\deploy\deploy.ps1                                 # Interactive — prompts for values
#   .\deploy\deploy.ps1 -ResourceGroup BCDR-devai-rg `
#       -BotAppId "xxx" -BotAppPassword "yyy" `
#       -LlmApiKey "zzz" -AzureDevOpsPat "pat"
# ============================================================

param(
    [string]$ResourceGroup = "BCDR-devai-rg",
    [string]$Location = "eastus2",
    [string]$BaseName = "BCDR-devai",
    [string]$BotAppId,
    [string]$BotAppPassword,
    [string]$LlmApiKey,
    [string]$AzureDevOpsPat,
    [switch]$SkipBuild,
    [switch]$SkipDeploy
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  BCDR DeveloperAI Azure Deployment" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# ── Step 0: Validate prerequisites ──────────────────────────

Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI not found. Install from https://aka.ms/install-azure-cli"
}
if (-not $SkipBuild -and -not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker not found. Install Docker Desktop."
}

# Ensure logged in
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged into Azure. Running 'az login'..." -ForegroundColor Yellow
    az login
}
Write-Host "  Azure account: $($account.name) ($($account.id))" -ForegroundColor Green

# ── Step 1: Prompt for missing secrets ───────────────────────

if (-not $BotAppId) {
    $BotAppId = Read-Host "Enter Bot Framework App ID (from Azure Bot registration)"
}
if (-not $BotAppPassword) {
    $BotAppPassword = Read-Host "Enter Bot Framework App Password" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}
if (-not $LlmApiKey) {
    $LlmApiKey = Read-Host "Enter Azure OpenAI API Key" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}
if (-not $AzureDevOpsPat) {
    $AzureDevOpsPat = Read-Host "Enter Azure DevOps PAT" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}

# ── Step 2: Create resource group ────────────────────────────

Write-Host "`n[1/5] Creating resource group '$ResourceGroup'..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output none
Write-Host "  Done" -ForegroundColor Green

# ── Step 3: Deploy Azure infrastructure via Bicep ────────────

if (-not $SkipDeploy) {
    Write-Host "`n[2/5] Deploying Azure infrastructure (Bicep)..." -ForegroundColor Yellow

    $deployResult = az deployment group create `
        --resource-group $ResourceGroup `
        --template-file deploy/main.bicep `
        --parameters baseName=$BaseName `
                     botAppId=$BotAppId `
                     botAppPassword=$BotAppPassword `
                     llmApiKey=$LlmApiKey `
                     azureDevOpsPat=$AzureDevOpsPat `
        --output json | ConvertFrom-Json

    $acrLoginServer = $deployResult.properties.outputs.acrLoginServer.value
    $appUrl = $deployResult.properties.outputs.appUrl.value
    $botEndpoint = $deployResult.properties.outputs.botEndpoint.value

    Write-Host "  ACR: $acrLoginServer" -ForegroundColor Green
    Write-Host "  App URL: $appUrl" -ForegroundColor Green
    Write-Host "  Bot Endpoint: $botEndpoint" -ForegroundColor Green
} else {
    # Get existing values
    $acrName = ($BaseName -replace '-','') + 'acr'
    $acrLoginServer = az acr show --name $acrName --query loginServer -o tsv
    $appName = "$BaseName-bot"
    $appUrl = "https://$(az containerapp show --name $appName --resource-group $ResourceGroup --query 'properties.configuration.ingress.fqdn' -o tsv)"
    $botEndpoint = "$appUrl/api/messages"
}

# ── Step 4: Build & push Docker image ────────────────────────

if (-not $SkipBuild) {
    Write-Host "`n[3/5] Building Docker image..." -ForegroundColor Yellow

    # Login to ACR
    az acr login --name ($acrLoginServer -split '\.')[0]

    $imageTag = "${acrLoginServer}/${BaseName}:latest"
    docker build -t $imageTag .
    Write-Host "  Built: $imageTag" -ForegroundColor Green

    Write-Host "`n[4/5] Pushing image to ACR..." -ForegroundColor Yellow
    docker push $imageTag
    Write-Host "  Pushed" -ForegroundColor Green
} else {
    Write-Host "`n[3/5] Skipping Docker build (--SkipBuild)" -ForegroundColor DarkGray
    Write-Host "[4/5] Skipping Docker push (--SkipBuild)" -ForegroundColor DarkGray
}

# ── Step 5: Update container app to use new image ────────────

Write-Host "`n[5/5] Updating container app with new image..." -ForegroundColor Yellow

$appName = "$BaseName-bot"
az containerapp update `
    --name $appName `
    --resource-group $ResourceGroup `
    --image "${acrLoginServer}/${BaseName}:latest" `
    --output none

Write-Host "  Updated" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────────────

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Bot URL:      $appUrl" -ForegroundColor White
Write-Host "  Bot Endpoint: $botEndpoint" -ForegroundColor White
Write-Host "  Health Check: $appUrl/api/health" -ForegroundColor White
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "  1. Go to Azure Portal > Bot Service > '$BaseName-bot-service'" -ForegroundColor White
Write-Host "  2. Channels > Microsoft Teams > click 'Open in Teams'" -ForegroundColor White
Write-Host "  3. Or install the Teams app manifest from brain_ai/bot/teams_manifest/" -ForegroundColor White
Write-Host ""
Write-Host "  To update:" -ForegroundColor Yellow
Write-Host "    .\deploy\deploy.ps1 -SkipDeploy -ResourceGroup $ResourceGroup" -ForegroundColor White
Write-Host "  To redeploy infra:" -ForegroundColor Yellow
Write-Host "    .\deploy\deploy.ps1 -ResourceGroup $ResourceGroup" -ForegroundColor White
Write-Host ""
