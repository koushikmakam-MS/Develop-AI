#!/usr/bin/env python
"""
run_doc_improver.py — Entry point for the Doc Improver background workflow.

Usage:
  python run_doc_improver.py                    # Run one improvement cycle
  python run_doc_improver.py --config cfg.yaml  # Custom config
  python run_doc_improver.py --force             # Run even if interval hasn't elapsed
  python run_doc_improver.py --iterations 5     # Override max iterations
  python run_doc_improver.py --daemon           # Run as a background daemon (repeat on schedule)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap ────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from brain_ai.config import load_config, get_config
from brain_ai.agents.doc_improver_agent import DocImproverAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("doc_improver")

LAST_RUN_FILE = ROOT / ".doc_improver_last_run"


# ── Helpers ──────────────────────────────────────────────────────────

def _should_run(cfg: dict, force: bool) -> bool:
    """Check whether enough time has passed since the last run."""
    if force:
        return True

    interval_hours = cfg.get("doc_improver", {}).get("run_interval_hours", 72)
    if LAST_RUN_FILE.exists():
        try:
            last_run = datetime.fromisoformat(
                LAST_RUN_FILE.read_text().strip()
            )
            elapsed = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
            if elapsed < interval_hours:
                log.info(
                    "Last run %.1f hours ago (interval=%dh) — skipping.",
                    elapsed, interval_hours,
                )
                return False
        except Exception:
            pass  # corrupt file — just run
    return True


def _save_last_run():
    """Record that we ran just now."""
    LAST_RUN_FILE.write_text(datetime.now(timezone.utc).isoformat())


# ── Main ─────────────────────────────────────────────────────────────

def run_once(cfg: dict, force: bool = False, max_iterations: int | None = None):
    """Execute a single doc improvement cycle."""
    if not cfg.get("doc_improver", {}).get("enabled", False):
        log.info("Doc improver is disabled in config — exiting.")
        return

    if not _should_run(cfg, force):
        return

    # Override max_iterations if provided
    if max_iterations is not None:
        cfg.setdefault("doc_improver", {})["max_iterations"] = max_iterations

    agent = DocImproverAgent(cfg)
    result = agent.run_improvement_cycle()

    _save_last_run()

    log.info("=" * 50)
    log.info("RESULT: %s", json.dumps(result, indent=2, default=str))
    log.info("=" * 50)
    return result


def run_daemon(cfg: dict, max_iterations: int | None = None):
    """Run repeatedly on the configured interval."""
    interval_hours = cfg.get("doc_improver", {}).get("run_interval_hours", 72)
    log.info("Starting doc improver daemon (interval=%dh)", interval_hours)

    while True:
        try:
            run_once(cfg, force=True, max_iterations=max_iterations)
        except Exception as e:
            log.error("Cycle failed: %s", e, exc_info=True)

        sleep_seconds = interval_hours * 3600
        log.info("Sleeping for %d hours until next cycle...", interval_hours)
        time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="BCDR DeveloperAI Doc Improver")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Run even if interval hasn't elapsed")
    parser.add_argument("--iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--daemon", action="store_true", help="Run as a repeating background daemon")
    args = parser.parse_args()

    load_config(args.config)
    cfg = get_config()

    if args.daemon:
        run_daemon(cfg, max_iterations=args.iterations)
    else:
        run_once(cfg, force=args.force, max_iterations=args.iterations)


if __name__ == "__main__":
    main()
