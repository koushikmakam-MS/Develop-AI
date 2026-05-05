#!/usr/bin/env python3
"""
Re-index all markdown docs in docs/agentKT/ into ChromaDB.

Usage:
    python run_index.py             # Incremental (skip unchanged files)
    python run_index.py --force     # Force re-index all files
"""

import argparse
import logging
import json

from brain_ai.vectorstore.indexer import DocIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Index docs into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-index all files")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)

    indexer = DocIndexer()
    result = indexer.index_all(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
