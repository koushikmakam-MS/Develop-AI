#!/usr/bin/env bash
# ============================================================
# BCRD DeveloperAI — One-command setup & initialization (Linux / Mac / WSL)
# ============================================================
#
# Runs ALL prerequisite steps in order:
#   1. Create Python virtual environment (if missing)
#   2. Install pip dependencies
#   3. Copy config templates (if config.yaml / config.local.yaml don't exist)
#   4. Sync docs from Azure DevOps
#   5. Index docs into ChromaDB
#   6. Sync + index source code into ChromaDB
#   7. (Optional) Run the Doc Improver bootstrap
#   8. (Optional) Start the Kusto MCP server
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                    # Normal setup
#   ./setup.sh --force            # Force re-sync + re-index everything
#   ./setup.sh --skip-kusto       # Don't start Kusto server
#   ./setup.sh --skip-doc-improver
# ============================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────

FORCE=false
SKIP_KUSTO=false
SKIP_DOC_IMPROVER=false
CONFIG_PATH="config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)           FORCE=true; shift ;;
        --skip-kusto)      SKIP_KUSTO=true; shift ;;
        --skip-doc-improver) SKIP_DOC_IMPROVER=true; shift ;;
        --config)          CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./setup.sh [--force] [--skip-kusto] [--skip-doc-improver] [--config path]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

FORCE_ARG=""
if $FORCE; then FORCE_ARG="--force"; fi

CONFIG_ARG=""
if [[ "$CONFIG_PATH" != "config.yaml" ]]; then CONFIG_ARG="--config $CONFIG_PATH"; fi

step() {
    echo ""
    echo "============================================================"
    echo "  Step $1 — $2"
    echo "============================================================"
}

ok()   { echo "  ✅ $1"; }
skip() { echo "  ⏭️  $1"; }
fail() { echo "  ❌ $1"; }

# ── Banner ───────────────────────────────────────────────────────────

echo ""
echo "  🧠 BCRD DeveloperAI Setup & Initialization"
echo "  =================================="
echo "  Project root: $PROJECT_ROOT"
echo "  Config:       $CONFIG_PATH"
if $FORCE; then echo "  Mode:         FORCE (re-sync + re-index all)"; fi
echo ""

# ── Step 1: Python Virtual Environment ───────────────────────────────

step 1 "Python virtual environment"

if [[ ! -d ".venv" ]]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
    ok "Virtual environment created at .venv/"
else
    ok "Virtual environment already exists."
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment activated."

# ── Step 2: Install Dependencies ─────────────────────────────────────

step 2 "Install Python dependencies"

echo "  Installing from requirements.txt..."
pip install -q -r requirements.txt

echo "  Installing package in editable mode..."
pip install -q -e .

ok "All dependencies installed."

# ── Step 3: Config Files ─────────────────────────────────────────────

step 3 "Configuration"

# 3a. config.yaml (generic settings — safe to commit)
if [[ ! -f "$CONFIG_PATH" ]]; then
    if [[ -f "config.yaml.template" ]]; then
        cp config.yaml.template "$CONFIG_PATH"
        ok "config.yaml created from template."
    else
        fail "No config.yaml or config.yaml.template found."
        exit 1
    fi
else
    ok "config.yaml found."
fi

# 3b. config.local.yaml (secrets — gitignored)
LOCAL_CONFIG_DIR="$(dirname "$CONFIG_PATH")"
LOCAL_CONFIG_PATH="${LOCAL_CONFIG_DIR}/config.local.yaml"
if [[ ! -f "$LOCAL_CONFIG_PATH" ]]; then
    if [[ -f "config.local.yaml.template" ]]; then
        cp config.local.yaml.template "$LOCAL_CONFIG_PATH"
        echo ""
        echo "  ⚠️  config.local.yaml created from template."
        echo "  ⚠️  Please edit config.local.yaml and fill in your secrets:"
        echo "       - Azure DevOps PAT"
        echo "       - Azure AI Foundry endpoint & API key"
        echo "       - Kusto cluster URL & database"
        echo ""
        echo "  Then re-run this script."
        exit 0
    else
        fail "No config.local.yaml or config.local.yaml.template found."
        exit 1
    fi
else
    ok "config.local.yaml found."
fi

# Quick validation: check for placeholder values in local config
if grep -q "<YOUR_" "$LOCAL_CONFIG_PATH"; then
    echo ""
    echo "  ⚠️  config.local.yaml still has placeholder values (<YOUR_...>)."
    echo "  ⚠️  Please fill in your actual secrets before continuing."
    echo ""
    exit 0
fi

# ── Step 4: Sync Docs from Azure DevOps ──────────────────────────────

step 4 "Sync docs from Azure DevOps"

if python run_sync.py $FORCE_ARG $CONFIG_ARG; then
    ok "Docs synced."
else
    fail "Doc sync failed — check your Azure DevOps PAT and repo URL."
    echo "  Continuing anyway (you can re-run sync later)..."
fi

# ── Step 5: Index Docs into ChromaDB ─────────────────────────────────

step 5 "Index docs into ChromaDB"

if python run_index.py $FORCE_ARG $CONFIG_ARG; then
    ok "Docs indexed into ChromaDB."
else
    fail "Doc indexing failed."
fi

# ── Step 6: Sync + Index Source Code ─────────────────────────────────

step 6 "Index source code into ChromaDB"

if python run_code_index.py $FORCE_ARG $CONFIG_ARG; then
    ok "Source code indexed into ChromaDB."
else
    fail "Code indexing failed."
fi

# ── Step 7: Doc Improver (optional) ──────────────────────────────────

step 7 "Doc Improver (bootstrap / improvement cycle)"

if $SKIP_DOC_IMPROVER; then
    skip "Skipped (remove --skip-doc-improver to enable)."
else
    DOC_IMPROVER_ENABLED=$(python -c "from brain_ai.config import get_config; c=get_config(); print(c.get('doc_improver',{}).get('enabled',False))" 2>/dev/null || echo "False")
    if [[ "$DOC_IMPROVER_ENABLED" == "True" ]]; then
        echo "  Running Doc Improver cycle (this may take a while)..."
        if python run_doc_improver.py --force; then
            ok "Doc Improver cycle complete."
        else
            fail "Doc Improver cycle failed."
        fi
    else
        skip "Doc Improver is disabled in config. Set doc_improver.enabled: true to activate."
    fi
fi

# ── Step 8: Kusto MCP Server (optional) ─────────────────────────────

step 8 "Kusto MCP Server"

if $SKIP_KUSTO; then
    skip "Skipped (remove --skip-kusto to enable)."
else
    echo "  Starting Kusto MCP server in background..."
    nohup python run_kusto_server.py > kusto_server.log 2>&1 &
    KUSTO_PID=$!
    sleep 3

    if kill -0 "$KUSTO_PID" 2>/dev/null; then
        ok "Kusto MCP server running in background (PID: $KUSTO_PID)."
        echo "  Use 'kill $KUSTO_PID' to stop it."
    else
        fail "Kusto MCP server failed to start. Check kusto_server.log"
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  🎉 BCRD DeveloperAI setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    python run_chat.py              # Start CLI chat"
echo "    python run_bot.py               # Start Teams Bot"
echo "    python run_doc_improver.py      # Run Doc Improver manually"
echo "    python run_daily.py             # Run daily sync + re-index"
echo ""
