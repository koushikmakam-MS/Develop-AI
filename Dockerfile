# ============================================================
# BCDR DeveloperAI — Multi-stage Docker build
# ============================================================
# Stage 1: Build layer with all build deps
# Stage 2: Slim runtime with only what's needed
# ============================================================

FROM python:3.13-slim AS builder

WORKDIR /build

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY brain_ai/ brain_ai/
COPY run_bot.py run_daily.py run_sync.py run_index.py run_kusto_server.py ./

RUN pip install --no-cache-dir --prefix=/install .

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.13-slim AS runtime

LABEL maintainer="BCDR DeveloperAI Team"
LABEL description="BCDR DeveloperAI Teams Bot + Background Services"

WORKDIR /app

# Install minimal runtime deps (git needed for repo sync)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Azure CLI (needed for Kusto az_cli auth)
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy application code
COPY brain_ai/ brain_ai/
COPY run_bot.py run_daily.py run_sync.py run_index.py run_kusto_server.py ./
COPY docs/ docs/
COPY config.yaml.template config.yaml.template

# Create directories for persistent data
RUN mkdir -p .chromadb .repo_cache logs

# Expose ports
# 3978 = Teams Bot (Bot Framework)
# 8701 = Kusto MCP server
EXPOSE 3978 8701

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:3978/api/health || exit 1

# Default: start the bot (which includes all background services)
ENTRYPOINT ["python", "run_bot.py"]
CMD ["--port", "3978"]
