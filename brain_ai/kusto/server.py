"""
Kusto MCP Server — exposes KQL query execution as MCP-style tool endpoints.

Runs as a lightweight local HTTP service (FastAPI/uvicorn) so that:
  - BCRD DeveloperAI agents can call KQL queries via simple HTTP POST
  - Authentication (az cli, managed identity, device code) is handled once at startup
  - Queries are validated (read-only) before execution
  - Results are returned as structured JSON

Usage:
    python -m brain_ai.kusto.server                 # default config.yaml
    python -m brain_ai.kusto.server --port 8701     # custom port
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional

from brain_ai.config import get_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kusto client helpers (extracted from debug_agent.py)
# ---------------------------------------------------------------------------

DANGEROUS_COMMANDS = [".set", ".drop", ".alter", ".delete", ".create", ".append", ".replace"]


def create_kusto_client(cfg: dict):
    """Create an authenticated Kusto client based on config."""
    from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

    kusto_cfg = cfg["kusto"]
    cluster = kusto_cfg["cluster_url"]
    auth_method = kusto_cfg.get("auth_method", "az_cli")

    if auth_method == "az_cli":
        kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
    elif auth_method == "managed_identity":
        kcsb = KustoConnectionStringBuilder.with_aad_managed_service_identity_authentication(
            cluster
        )
    elif auth_method == "device_code":
        kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
    else:
        raise ValueError(f"Unknown Kusto auth method: {auth_method}")

    return KustoClient(kcsb)


def validate_query(query: str) -> Optional[str]:
    """
    Validate that a KQL query is read-only.
    Returns an error message if blocked, or None if safe.
    """
    query_stripped = query.strip().lower()
    for cmd in DANGEROUS_COMMANDS:
        if query_stripped.startswith(cmd):
            return f"BLOCKED: Refusing to execute mutating command: {cmd}"
    return None


def execute_kql(kusto_client, database: str, query: str) -> Dict[str, Any]:
    """
    Execute a KQL query and return structured results.

    Returns:
        {
            "success": bool,
            "columns": [...],
            "rows": [[...], ...],
            "row_count": int,
            "truncated": bool,
            "error": str | None,
            "formatted": str        # human-readable table
        }
    """
    MAX_ROWS = 50

    # Safety check
    error = validate_query(query)
    if error:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": error,
            "formatted": error,
        }

    try:
        response = kusto_client.execute(database, query)
        primary = response.primary_results[0] if response.primary_results else None

        if primary is None:
            return {
                "success": True,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": None,
                "formatted": "(No results)",
            }

        columns = [col.column_name for col in primary.columns]
        rows = []
        for row in primary:
            rows.append([str(row[col]) for col in columns])

        total = len(rows)
        truncated = total > MAX_ROWS
        display_rows = rows[:MAX_ROWS]

        # Build human-readable table
        header = " | ".join(columns)
        separator = "-+-".join("-" * max(len(c), 3) for c in columns)
        body = "\n".join(" | ".join(r) for r in display_rows)
        formatted = f"{header}\n{separator}\n{body}"
        if truncated:
            formatted += f"\n... ({total - MAX_ROWS} more rows truncated)"
        if total == 0:
            formatted = "(Query returned 0 rows)"

        return {
            "success": True,
            "columns": columns,
            "rows": display_rows,
            "row_count": total,
            "truncated": truncated,
            "error": None,
            "formatted": formatted,
        }

    except Exception as e:
        log.error("KQL execution error: %s", e)
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(e),
            "formatted": f"KQL Error: {e}",
        }


# ---------------------------------------------------------------------------
# MCP Server — lightweight HTTP API
# ---------------------------------------------------------------------------

def create_app(cfg: dict | None = None):
    """Create the FastAPI app for the Kusto MCP server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the Kusto MCP server. "
            "Install it with: pip install fastapi uvicorn"
        )

    if cfg is None:
        cfg = get_config()

    app = FastAPI(
        title="BCRD DeveloperAI Kusto MCP Server",
        description="MCP-style tool server for executing KQL queries against Azure Data Explorer",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- State ---
    state: Dict[str, Any] = {
        "kusto_client": None,
        "database": cfg["kusto"]["database"],
        "cluster_url": cfg["kusto"]["cluster_url"],
        "connected": False,
        "error": None,
    }

    # --- Pydantic models ---
    class QueryRequest(BaseModel):
        query: str
        database: str | None = None  # override default database

    class QueryResponse(BaseModel):
        success: bool
        columns: List[str] = []
        rows: List[List[str]] = []
        row_count: int = 0
        truncated: bool = False
        error: str | None = None
        formatted: str = ""

    class ToolDefinition(BaseModel):
        name: str
        description: str
        parameters: Dict[str, Any]

    class HealthResponse(BaseModel):
        status: str
        connected: bool
        cluster_url: str
        database: str
        error: str | None = None

    # --- Lazy connect ---
    def _get_client():
        if state["kusto_client"] is None:
            try:
                state["kusto_client"] = create_kusto_client(cfg)
                state["connected"] = True
                state["error"] = None
                log.info("Kusto MCP: Connected to %s", state["cluster_url"])
            except Exception as e:
                state["connected"] = False
                state["error"] = str(e)
                log.error("Kusto MCP: Connection failed: %s", e)
                raise HTTPException(
                    status_code=503,
                    detail=f"Kusto connection failed: {e}. "
                    "Ensure you are logged in (az login) or check auth_method in config.yaml.",
                )
        return state["kusto_client"]

    # --- MCP Tool Listing (discovery) ---
    @app.get("/tools", response_model=List[ToolDefinition])
    def list_tools():
        """List available MCP tools (KQL query execution)."""
        return [
            ToolDefinition(
                name="execute_kql",
                description=(
                    "Execute a read-only KQL (Kusto Query Language) query against "
                    f"Azure Data Explorer cluster: {state['cluster_url']}, "
                    f"database: {state['database']}. "
                    "Mutating commands (.set, .drop, .alter, etc.) are blocked."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The KQL query to execute",
                        },
                        "database": {
                            "type": "string",
                            "description": f"Database name (default: {state['database']})",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="health_check",
                description="Check Kusto connection health",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    # --- Execute KQL tool ---
    @app.post("/tools/execute_kql", response_model=QueryResponse)
    def tool_execute_kql(req: QueryRequest):
        """Execute a KQL query and return results."""
        client = _get_client()
        db = req.database or state["database"]
        log.info("Kusto MCP: Executing query on %s: %s", db, req.query[:200])
        result = execute_kql(client, db, req.query)
        return QueryResponse(**result)

    # --- Health check ---
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        """Check server and Kusto connection health."""
        # Try to connect if not already
        try:
            _get_client()
        except Exception:
            pass
        return HealthResponse(
            status="ok" if state["connected"] else "degraded",
            connected=state["connected"],
            cluster_url=state["cluster_url"],
            database=state["database"],
            error=state["error"],
        )

    return app


def main():
    """Run the Kusto MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="BCRD DeveloperAI Kusto MCP Server")
    parser.add_argument("--port", type=int, default=8701, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. Install with: pip install uvicorn")
        return

    app = create_app()
    print(f"\n🔌 BCRD DeveloperAI Kusto MCP Server starting on http://{args.host}:{args.port}")
    print(f"   Tools endpoint: http://{args.host}:{args.port}/tools")
    print(f"   Health check:   http://{args.host}:{args.port}/health")
    print(f"   Execute KQL:    POST http://{args.host}:{args.port}/tools/execute_kql\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
