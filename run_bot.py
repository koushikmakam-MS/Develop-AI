#!/usr/bin/env python3
"""
Start the BCRD DeveloperAI Microsoft Teams Bot with all background services.

Services started:
  - Teams Bot (Bot Framework, port 3978)
  - Kusto MCP server (background, port 8701)
  - Daily sync scheduler (every 24h)
  - Unanswered-message auto-reply monitor (10 min)

Usage:
    python run_bot.py                    # default port 3978
    python run_bot.py --port 8080        # custom port
    python run_bot.py --config my.yaml   # custom config
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="BCRD DeveloperAI Teams Bot")
    parser.add_argument("--port", type=int, default=3978, help="Port (default: 3978)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    try:
        from brain_ai.bot.app import run
    except ImportError as e:
        print(
            f"❌ Missing dependency: {e}\n\n"
            "Install Teams bot dependencies:\n"
            '  pip install "bcrd-devai[teams]"\n'
            "  or: pip install botbuilder-core aiohttp\n"
        )
        sys.exit(1)

    cfg = None
    if args.config:
        from brain_ai.config import get_config
        cfg = get_config(args.config)

    run(port=args.port, cfg=cfg)


if __name__ == "__main__":
    main()
