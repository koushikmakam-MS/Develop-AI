"""
Kusto MCP Client — lightweight HTTP client for calling the Kusto MCP server.

Used by DebugAgent and other components to execute KQL queries
without directly managing Kusto authentication.

Falls back to direct Kusto SDK if the MCP server is not running.
"""

import json
import logging
from typing import Any, Dict, Optional

from brain_ai.config import get_config

log = logging.getLogger(__name__)

DEFAULT_MCP_URL = "http://127.0.0.1:8701"


class KustoMCPClient:
    """
    Client for the Kusto MCP server.

    Tries the MCP server first; if unavailable, falls back to direct SDK.
    """

    def __init__(self, cfg: dict | None = None, mcp_url: str | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.mcp_url = mcp_url or cfg.get("kusto", {}).get("mcp_url", DEFAULT_MCP_URL)
        self.database = cfg["kusto"]["database"]
        self._direct_client = None  # lazy fallback

    def execute_kql(self, query: str, database: str | None = None) -> Dict[str, Any]:
        """
        Execute a KQL query via MCP server, with fallback to direct SDK.

        Returns:
            {"success": bool, "formatted": str, "columns": [...], "rows": [...], ...}
        """
        db = database or self.database

        # Try MCP server first
        result = self._try_mcp_server(query, db)
        if result is not None:
            return result

        # Fallback to direct SDK
        log.info("MCP server unavailable, falling back to direct Kusto SDK...")
        return self._try_direct_sdk(query, db)

    def _try_mcp_server(self, query: str, database: str) -> Optional[Dict[str, Any]]:
        """Try executing via MCP HTTP server."""
        import urllib.error
        import urllib.request

        url = f"{self.mcp_url}/tools/execute_kql"
        payload = json.dumps({"query": query, "database": database}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                log.info("KQL executed via MCP server (rows=%d)", body.get("row_count", 0))
                return body
        except urllib.error.URLError as e:
            log.debug("MCP server not reachable at %s: %s", self.mcp_url, e)
            return None
        except Exception as e:
            log.warning("MCP server error: %s", e)
            return None

    def _try_direct_sdk(self, query: str, database: str) -> Dict[str, Any]:
        """Fallback: execute KQL directly via azure-kusto-data SDK."""
        from brain_ai.kusto.server import create_kusto_client, execute_kql

        if self._direct_client is None:
            try:
                self._direct_client = create_kusto_client(self.cfg)
                log.info("Direct Kusto SDK connected to %s", self.cfg["kusto"]["cluster_url"])
            except Exception as e:
                log.error("Direct Kusto SDK connection failed: %s", e)
                return {
                    "success": False,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": (
                        f"Could not connect to Kusto: {e}\n\n"
                        "Options to fix:\n"
                        "1. Start the MCP server: python -m brain_ai.kusto.server\n"
                        "2. Run 'az login' for CLI authentication\n"
                        "3. Check kusto.auth_method in config.yaml"
                    ),
                    "formatted": (
                        f"❌ Kusto connection failed: {e}\n\n"
                        "**To fix, try one of:**\n"
                        "1. Start the MCP server: `python -m brain_ai.kusto.server`\n"
                        "2. Run `az login` for CLI authentication\n"
                        "3. Check `kusto.auth_method` in config.yaml"
                    ),
                }

        return execute_kql(self._direct_client, database, query)

    def health_check(self) -> Dict[str, Any]:
        """Check if the MCP server is reachable."""
        import urllib.error
        import urllib.request

        url = f"{self.mcp_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {
                "status": "unreachable",
                "connected": False,
                "cluster_url": self.cfg["kusto"]["cluster_url"],
                "database": self.database,
                "error": f"MCP server not reachable: {e}",
            }
