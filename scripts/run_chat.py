#!/usr/bin/env python3
"""
Start the BCDR DeveloperAI CLI chat interface.

Runs pre-flight checks first (config, ChromaDB, Kusto MCP server)
then launches the interactive chat.

Usage:
    python run_chat.py
    python run_chat.py --config path/to/config.yaml
    python run_chat.py --no-kusto          # skip auto-starting Kusto MCP server
"""

import argparse
import atexit
import signal
import sys

from brain_ai.startup import preflight_check
from brain_ai.cli.chat import run_chat


def main():
    parser = argparse.ArgumentParser(description="BCDR DeveloperAI CLI Chat")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--no-kusto",
        action="store_true",
        help="Don't auto-start the Kusto MCP server",
    )
    args = parser.parse_args()

    # Run pre-flight checks and auto-start dependencies
    cfg, mcp_proc = preflight_check(
        config_path=args.config,
        auto_start_kusto=not args.no_kusto,
    )

    # Ensure the MCP server subprocess is cleaned up on exit
    if mcp_proc is not None:
        def _cleanup():
            try:
                mcp_proc.terminate()
                mcp_proc.wait(timeout=5)
            except Exception:
                mcp_proc.kill()

        atexit.register(_cleanup)

    # Launch the chat (pass the already-loaded config)
    run_chat(config_path=args.config)


if __name__ == "__main__":
    main()
