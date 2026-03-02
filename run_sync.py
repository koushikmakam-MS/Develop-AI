#!/usr/bin/env python3
"""
Run the repo sync - clones/pulls the Azure DevOps repo and copies
markdown files into docs/agentKT/.

Usage:
    python run_sync.py              # Incremental (only changed files)
    python run_sync.py --force      # Force copy all files
"""

import argparse
import logging
import json

from brain_ai.sync.repo_sync import sync_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Sync docs from Azure DevOps repo")
    parser.add_argument("--force", action="store_true", help="Force copy all files")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)

    result = sync_docs(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
