#!/usr/bin/env python3
"""
Index BMS source code into ChromaDB for the Coder Agent.

Usage:
    python run_code_index.py             # Incremental (skip unchanged files)
    python run_code_index.py --force     # Force re-index all files

Prerequisites:
    Run `python run_sync.py` first to clone/pull the BMS repo.
"""

import argparse
import logging
import json

from brain_ai.vectorstore.code_indexer import CodeIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Index BMS source code into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-index all files")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)

    indexer = CodeIndexer()
    print(f"\n📦 Code collection currently has {indexer.collection.count()} chunks.\n")

    result = indexer.index_all(force=args.force)
    print(f"\n✅ Code indexing complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
