#!/usr/bin/env python3
"""
Start the BCDR DeveloperAI Kusto MCP Server.

This runs a local HTTP service that exposes KQL query execution as
MCP-style tool endpoints. The Debug Agent connects to this automatically.

Usage:
    python run_kusto_server.py
    python run_kusto_server.py --port 8701
    python run_kusto_server.py --config path/to/config.yaml
"""

import argparse
from brain_ai.kusto.server import main

if __name__ == "__main__":
    main()
