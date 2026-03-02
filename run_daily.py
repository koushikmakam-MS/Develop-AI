#!/usr/bin/env python3
"""
Daily Sync & Index - run this via cron / Task Scheduler.

Performs:
  1. Git pull from Azure DevOps repo
  2. Copy changed markdown files to docs/agentKT/
  3. Re-index changed files into ChromaDB

Example cron (daily at 2 AM):
  0 2 * * * cd /path/to/bcrd_devai && python run_daily.py >> daily_sync.log 2>&1

Example Windows Task Scheduler:
  Action: python
  Arguments: run_daily.py
  Start in: C:\...\bcrd_devai
"""

import logging
import json
from datetime import datetime, timezone

from brain_ai.sync.repo_sync import sync_docs
from brain_ai.vectorstore.indexer import DocIndexer
from brain_ai.vectorstore.code_indexer import CodeIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("daily_sync.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("Daily sync started at %s", datetime.now(timezone.utc).isoformat())
    log.info("=" * 60)

    # Step 1: Sync docs from repo
    try:
        sync_result = sync_docs()
        log.info("Sync result: %s", json.dumps(sync_result, indent=2))
    except Exception as e:
        log.error("Sync failed: %s", e, exc_info=True)
        return

    # Step 2: Re-index docs into ChromaDB
    try:
        indexer = DocIndexer()
        index_result = indexer.index_all()
        log.info("Doc index result: %s", json.dumps(index_result, indent=2))
    except Exception as e:
        log.error("Doc indexing failed: %s", e, exc_info=True)
        return

    # Step 3: Re-index source code into ChromaDB (for Coder Agent)
    try:
        code_indexer = CodeIndexer()
        code_result = code_indexer.index_all()
        log.info("Code index result: %s", json.dumps(code_result, indent=2))
    except Exception as e:
        log.error("Code indexing failed: %s", e, exc_info=True)
        # Non-fatal: docs are still indexed

    log.info("Daily sync complete.")


if __name__ == "__main__":
    main()
