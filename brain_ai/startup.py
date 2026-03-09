"""
Startup checks and dependency launcher for BCDR DeveloperAI.

Ensures all required services and prerequisites are ready before
starting the chat interface:
  1. Config file exists and is valid
  2. ChromaDB has indexed documents (warns if empty)
  3. Kusto MCP server is running (auto-starts if configured)

Usage:
    from brain_ai.startup import preflight_check
    cfg = preflight_check()   # returns config dict, raises on fatal errors
"""

import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console

from brain_ai.config import get_config

log = logging.getLogger(__name__)
console = Console()

DEFAULT_MCP_PORT = 8701
MCP_STARTUP_TIMEOUT = 15  # seconds to wait for MCP server to become healthy


def _check_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate that config.yaml exists and loads correctly."""
    console.print("[dim]  Checking config...[/dim]", end=" ")
    try:
        cfg = get_config(config_path)
        console.print("[green]✓[/green]")
        return cfg
    except FileNotFoundError as e:
        console.print("[red]✗[/red]")
        console.print(f"[red]  Config not found: {e}[/red]")
        console.print("[dim]  Run: cp config.yaml.template config.yaml[/dim]")
        raise
    except Exception as e:
        console.print("[red]✗[/red]")
        console.print(f"[red]  Config error: {e}[/red]")
        raise


def _check_vectorstore(cfg: Dict[str, Any]) -> Tuple[bool, int]:
    """Check if ChromaDB has indexed documents. Returns (ok, doc_count)."""
    console.print("[dim]  Checking ChromaDB index...[/dim]", end=" ")
    try:
        import chromadb

        vs_cfg = cfg["vectorstore"]
        persist_dir = Path(vs_cfg["persist_directory"]).resolve()

        if not persist_dir.exists():
            console.print("[yellow]⚠ not found[/yellow]")
            console.print(
                "[yellow]  No index directory. Run:[/yellow]\n"
                "[dim]    python run_sync.py && python run_index.py[/dim]"
            )
            return False, 0

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection_name = vs_cfg.get("collection_name", "agent_kt_docs")

        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            console.print("[yellow]⚠ empty[/yellow]")
            console.print(
                "[yellow]  Index is empty. Run:[/yellow]\n"
                "[dim]    python run_sync.py && python run_index.py[/dim]"
            )
            return False, 0

        console.print(f"[green]✓[/green] ({count} chunks)")
        return True, count

    except Exception as e:
        console.print(f"[yellow]⚠ error: {e}[/yellow]")
        return False, 0


def _is_mcp_server_running(port: int = DEFAULT_MCP_PORT) -> bool:
    """Check if the Kusto MCP server is reachable."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") in ("ok", "degraded")
    except Exception:
        return False


def _start_mcp_server(cfg: Dict[str, Any], port: int = DEFAULT_MCP_PORT) -> Optional[subprocess.Popen]:
    """
    Start the Kusto MCP server as a background subprocess.

    Returns the Popen handle if started, or None if it fails.
    """
    console.print("[dim]  Starting Kusto MCP server...[/dim]", end=" ")

    try:
        # Start as a detached background process
        proc = subprocess.Popen(
            [sys.executable, "-m", "brain_ai.kusto.server", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # On Windows, CREATE_NEW_PROCESS_GROUP detaches the child
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Wait for it to become healthy
        deadline = time.time() + MCP_STARTUP_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process exited early — something went wrong
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                console.print("[red]✗ exited[/red]")
                if stderr:
                    console.print(f"[red]  {stderr[:300]}[/red]")
                return None

            if _is_mcp_server_running(port):
                console.print(f"[green]✓[/green] (port {port})")
                return proc

            time.sleep(0.5)

        console.print("[yellow]⚠ timeout[/yellow]")
        console.print(
            f"[yellow]  MCP server did not respond within {MCP_STARTUP_TIMEOUT}s.[/yellow]"
        )
        return proc  # still return — it may still be starting

    except Exception as e:
        console.print(f"[red]✗ failed: {e}[/red]")
        return None


def _check_kusto_mcp(cfg: Dict[str, Any], auto_start: bool = True) -> Optional[subprocess.Popen]:
    """
    Ensure the Kusto MCP server is running.

    If auto_start is True, starts it as a background process when not running.
    Returns the subprocess handle (if we started it), or None.
    """
    port = cfg.get("kusto", {}).get("mcp_port", DEFAULT_MCP_PORT)

    console.print("[dim]  Checking Kusto MCP server...[/dim]", end=" ")

    if _is_mcp_server_running(port):
        console.print(f"[green]✓[/green] (port {port})")
        return None

    console.print("[yellow]⚠ not running[/yellow]")

    if not auto_start:
        console.print(
            "[dim]  Start manually: python run_kusto_server.py[/dim]"
        )
        console.print(
            "[dim]  Debug Agent will fall back to direct SDK (requires az login).[/dim]"
        )
        return None

    return _start_mcp_server(cfg, port)


def _check_llm(cfg: Dict[str, Any]) -> bool:
    """Validate that LLM config has required fields."""
    console.print("[dim]  Checking LLM config...[/dim]", end=" ")
    llm_cfg = cfg.get("llm", {})

    endpoint = llm_cfg.get("endpoint", "")
    api_key = llm_cfg.get("api_key", "")

    if not endpoint or endpoint.startswith("<"):
        console.print("[red]✗ missing endpoint[/red]")
        console.print("[red]  Set llm.endpoint in config.local.yaml or BCDR_DEVAI_LLM_API_KEY env var[/red]")
        return False

    if not api_key or api_key.startswith("<"):
        console.print("[red]✗ missing api_key[/red]")
        console.print("[red]  Set llm.api_key in config.local.yaml or BCDR_DEVAI_LLM_API_KEY env var[/red]")
        return False

    console.print(f"[green]✓[/green] ({llm_cfg.get('model', 'unknown')})")
    return True


def _check_code_index(cfg: Dict[str, Any]) -> Tuple[bool, int]:
    """Check if the code index has any indexed source code."""
    console.print("[dim]  Checking code index...[/dim]", end=" ")
    try:
        import chromadb

        vs_cfg = cfg["vectorstore"]
        persist_dir = Path(vs_cfg["persist_directory"]).resolve()

        if not persist_dir.exists():
            console.print("[yellow]⚠ not found[/yellow]")
            return False, 0

        client = chromadb.PersistentClient(path=str(persist_dir))
        code_cfg = cfg.get("code_index", {})
        collection_name = code_cfg.get("collection_name", "bms_code")

        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            console.print("[yellow]⚠ empty[/yellow]")
            console.print(
                "[dim]  Coder Agent needs code indexed. Run:[/dim]\n"
                "[dim]    python run_sync.py && python run_code_index.py[/dim]"
            )
            return False, 0

        console.print(f"[green]✓[/green] ({count} chunks)")
        return True, count

    except Exception as e:
        console.print(f"[yellow]⚠ error: {e}[/yellow]")
        return False, 0


def preflight_check(
    config_path: Optional[str] = None,
    auto_start_kusto: bool = True,
) -> Tuple[Dict[str, Any], Optional[subprocess.Popen]]:
    """
    Run all pre-flight checks before starting BCDR DeveloperAI.

    Args:
        config_path: Optional path to config.yaml
        auto_start_kusto: If True, auto-start the Kusto MCP server when not running.

    Returns:
        (cfg, mcp_process) — config dict and optional MCP subprocess handle.

    Raises:
        SystemExit if fatal checks fail.
    """
    console.print()
    console.print("[bold]🔍 BCDR DeveloperAI Pre-flight Checks[/bold]")
    console.print("─" * 40)

    fatal = False
    mcp_proc = None

    # 1. Config
    try:
        cfg = _check_config(config_path)
    except Exception:
        sys.exit(1)

    # 2. LLM config
    if not _check_llm(cfg):
        fatal = True

    # 3. ChromaDB
    vs_ok, doc_count = _check_vectorstore(cfg)
    if not vs_ok:
        console.print(
            "[dim]  Knowledge Agent will have limited functionality without indexed docs.[/dim]"
        )

    # 4. Code index (for Coder Agent)
    _check_code_index(cfg)

    # 5. Kusto MCP server
    mcp_proc = _check_kusto_mcp(cfg, auto_start=auto_start_kusto)

    console.print("─" * 40)

    if fatal:
        console.print("[red]❌ Fatal errors found. Fix the above issues and try again.[/red]\n")
        sys.exit(1)

    console.print("[green]✅ Ready![/green]\n")
    return cfg, mcp_proc
