#!/usr/bin/env python3
"""
Unified CLI for BCDR DeveloperAI.

Consolidates all run_*.py scripts into one entry point:

    brainai chat          — Interactive chat (with pre-flight checks)
    brainai index         — Index markdown docs into ChromaDB
    brainai code-index    — Index source code into ChromaDB
    brainai hive-index    — Index code for all configured hives
    brainai status        — Show hive index status, topics & staleness
    brainai gateway-test  — Test gateway routing with sample questions
    brainai sync          — Sync docs from Azure DevOps repo
    brainai daily         — Run daily sync + index cycle
    brainai kusto-server  — Start the Kusto MCP server
    brainai bot           — Start the Teams bot
    brainai doc-improver  — Run the doc improvement cycle
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configure logging consistently across all subcommands."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )
    # Quiet noisy loggers
    for noisy in ("chromadb", "httpx", "anthropic", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Subcommand handlers ─────────────────────────────────────────

def cmd_chat(args):
    """Launch interactive chat with pre-flight checks."""
    from brain_ai.startup import preflight_check
    from brain_ai.cli.chat import run_chat
    import atexit

    cfg, mcp_proc = preflight_check(
        config_path=args.config,
        auto_start_kusto=not args.no_kusto,
    )

    if mcp_proc is not None:
        def _cleanup():
            try:
                mcp_proc.terminate()
                mcp_proc.wait(timeout=5)
            except Exception:
                mcp_proc.kill()
        atexit.register(_cleanup)

    run_chat(config_path=args.config)


def cmd_index(args):
    """Index markdown docs into ChromaDB."""
    _setup_logging()
    from brain_ai.vectorstore.indexer import DocIndexer
    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)
    indexer = DocIndexer()
    result = indexer.index_all(force=args.force)
    print(json.dumps(result, indent=2))


def cmd_code_index(args):
    """Index source code into ChromaDB."""
    _setup_logging()
    from brain_ai.vectorstore.code_indexer import CodeIndexer
    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)
    indexer = CodeIndexer()
    print(f"\n📦 Code collection currently has {indexer.collection.count()} chunks.\n")
    result = indexer.index_all(force=args.force)
    print(f"\n✅ Code indexing complete:")
    print(json.dumps(result, indent=2))


def cmd_hive_index(args):
    """Index code for all configured hives."""
    _setup_logging()
    from brain_ai.config import get_config
    from brain_ai.vectorstore.code_indexer import CodeIndexer
    import copy

    cfg = get_config(args.config)

    # --list mode
    hive_defs = cfg.get("hives", {}).get("definitions", {})
    default = cfg.get("hives", {}).get("default_hive", "")

    if args.list:
        print(f"\n🐝 Configured Hives ({len(hive_defs)} total):")
        for name, hdef in hive_defs.items():
            display = hdef.get("display_name", name)
            repo = hdef.get("paths", {}).get(
                "repo_clone_dir",
                cfg.get("paths", {}).get("repo_clone_dir", "?"),
            )
            marker = " ⭐ (default)" if name == default else ""
            exists = "✅" if Path(repo).resolve().exists() else "❌ missing"
            print(f"  {name}{marker} — {display}  [{exists}]")
        return

    if not hive_defs:
        print("❌ No hives configured in config.yaml → hives.definitions")
        sys.exit(1)

    if args.hive:
        if args.hive not in hive_defs:
            print(f"❌ Unknown hive: '{args.hive}'. Available: {list(hive_defs.keys())}")
            sys.exit(1)
        hives_to_index = {args.hive: hive_defs[args.hive]}
    else:
        hives_to_index = hive_defs

    print(f"\n🐝 Indexing {len(hives_to_index)} hive(s): {list(hives_to_index.keys())}")
    results = []
    total_start = time.time()

    for name, hdef in hives_to_index.items():
        merged = copy.deepcopy(cfg)
        for key in ("paths", "vectorstore", "code_index"):
            if key in hdef:
                merged.setdefault(key, {}).update(hdef[key])

        repo_dir = merged.get("paths", {}).get("repo_clone_dir", ".repo_cache")
        repo_path = Path(repo_dir).resolve()
        if not repo_path.exists():
            print(f"  ⏭️  {name}: repo not found at {repo_path}")
            results.append({"hive": name, "status": "skipped"})
            continue

        collection = hdef.get("code_index", {}).get("collection_name", f"{name}_code")
        print(f"\n  🐝 {name}: indexing from {repo_path} → {collection}")

        try:
            indexer = CodeIndexer(merged)
            before = indexer.collection.count()
            start = time.time()
            result = indexer.index_all(force=args.force)
            elapsed = time.time() - start
            after = indexer.collection.count()
            result.update(hive=name, elapsed=f"{elapsed:.1f}s", chunks=after)
            results.append(result)
            print(f"  ✅ {name}: {after} chunks ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            results.append({"hive": name, "status": "error", "error": str(e)})

    total = time.time() - total_start
    print(f"\n🐝 Done — {len(results)} hive(s) in {total:.1f}s")

    # Auto-extract topics if requested
    if getattr(args, "refresh_topics", False):
        print("\n🔄 Refreshing topics for indexed hives...")
        try:
            from brain_ai.hive.topic_extractor import TopicExtractor
            extractor = TopicExtractor(cfg)
            for name in hives_to_index:
                hdef = hives_to_index[name]
                display = hdef.get("display_name", name)
                desc = hdef.get("description", "")
                docs_col = hdef.get("vectorstore", {}).get("collection_name")
                code_col = hdef.get("code_index", {}).get("collection_name")
                print(f"  Extracting topics for {name}...", end=" ", flush=True)
                topics = extractor.extract_topics(
                    hive_name=name,
                    display_name=display,
                    description=desc,
                    docs_collection_name=docs_col,
                    code_collection_name=code_col,
                )
                print(f"→ {len(topics)} topics")
        except Exception as e:
            print(f"  ❌ Topic extraction error: {e}")


def cmd_sync(args):
    """Sync docs from Azure DevOps repo."""
    _setup_logging()
    from brain_ai.sync.repo_sync import sync_docs
    if args.config:
        from brain_ai.config import load_config
        load_config(args.config)
    result = sync_docs(force=args.force)
    print(json.dumps(result, indent=2))


def cmd_daily(args):
    """Run daily sync + index cycle."""
    _setup_logging(log_file="daily_sync.log")
    log = logging.getLogger("daily")
    log.info("Daily sync started at %s", datetime.now(timezone.utc).isoformat())

    from brain_ai.sync.repo_sync import sync_docs
    from brain_ai.vectorstore.indexer import DocIndexer
    from brain_ai.vectorstore.code_indexer import CodeIndexer

    try:
        sync_result = sync_docs()
        log.info("Sync: %s", json.dumps(sync_result, indent=2))
    except Exception as e:
        log.error("Sync failed: %s", e, exc_info=True)
        return

    try:
        index_result = DocIndexer().index_all()
        log.info("Doc index: %s", json.dumps(index_result, indent=2))
    except Exception as e:
        log.error("Doc indexing failed: %s", e, exc_info=True)
        return

    try:
        code_result = CodeIndexer().index_all()
        log.info("Code index: %s", json.dumps(code_result, indent=2))
    except Exception as e:
        log.error("Code indexing failed (non-fatal): %s", e)

    log.info("Daily sync complete.")


def cmd_kusto_server(args):
    """Start the Kusto MCP server."""
    from brain_ai.kusto.server import main as kusto_main
    kusto_main()


def cmd_bot(args):
    """Start the Teams bot."""
    try:
        from brain_ai.bot.app import run
    except ImportError as e:
        print(
            f"❌ Missing dependency: {e}\n\n"
            'Install: pip install "BCDR-devai[teams]"'
        )
        sys.exit(1)

    cfg = None
    if args.config:
        from brain_ai.config import get_config
        cfg = get_config(args.config)
    run(port=args.port, cfg=cfg)


def cmd_doc_improver(args):
    """Run doc improvement cycle."""
    _setup_logging()
    from brain_ai.config import load_config, get_config
    from brain_ai.agents.doc_improver_agent import DocImproverAgent

    load_config(args.config or "config.yaml")
    cfg = get_config()

    if not cfg.get("doc_improver", {}).get("enabled", False):
        print("Doc improver is disabled in config.")
        return

    if args.iterations is not None:
        cfg.setdefault("doc_improver", {})["max_iterations"] = args.iterations

    agent = DocImproverAgent(cfg)
    result = agent.run_improvement_cycle()
    print(json.dumps(result, indent=2, default=str))


def cmd_status(args):
    """Show hive index status, topic counts, and staleness."""
    from brain_ai.hive.discovery_store import DiscoveryStore
    ds = DiscoveryStore()
    all_meta = ds.get_all_metadata()
    stale = ds.get_stale_hives()
    stale_names = {s["hive_name"] for s in stale}

    if not all_meta:
        print("No index metadata yet. Run: brainai hive-index")
        return

    print("\n\U0001f41d BrainAI \u2014 Hive Status")
    print("=" * 70)
    total_code = total_docs = total_topics = 0
    for m in all_meta:
        name = m["hive_name"]
        code = m.get("code_chunks", 0)
        docs = m.get("doc_chunks", 0)
        topics = ds.get_topics(name)
        last = (m.get("last_code_index") or "never")[:19].replace("T", " ")
        marker = " \u26a0\ufe0f  STALE" if name in stale_names else " \u2713"
        total_code += code
        total_docs += docs
        total_topics += len(topics)
        print(f"  {name:<12s}{marker}")
        print(f"    Code: {code:>7,} chunks | Docs: {docs:>5,} chunks | Topics: {len(topics):>3}")
        print(f"    Last indexed: {last}")

    print(f"\n  {'TOTAL':<12s}")
    print(f"    Code: {total_code:>7,} chunks | Docs: {total_docs:>5,} chunks | Topics: {total_topics:>3}")
    if stale:
        print(f"\n  \u26a0\ufe0f  {len(stale)} stale hive(s) \u2014 run: brainai hive-index")
    print()
    ds.close()


def cmd_gateway_test(args):
    """Run gateway routing test with sample questions."""
    from brain_ai.config import get_config
    from brain_ai.hive.registry import HiveRegistry
    from brain_ai.llm_client import LLMClient
    from brain_ai.hive.gateway import Gateway

    cfg = get_config(args.config)
    reg = HiveRegistry(cfg)
    llm = LLMClient(cfg)
    gw = Gateway(reg, llm, cfg.get("hives", {}).get("default_hive", "dpp"))

    tests = [
        "How does cross-region restore work?",
        "What are protection containers?",
        "How does PIT Catalog manage recovery points?",
        "Explain snapshot coordination",
        "What common utilities exist for serialization?",
        "How does datamover transfer data?",
        "What monitoring alerts fire on backup failure?",
        "How does the regional RP handle geo-redundancy?",
        "What is a backup vault?",
        "How do backup policies work?",
    ]

    print("\n\U0001f500 Gateway Routing Test")
    print("=" * 100)
    print(f"{'Question':<55} {'Hive':<12} {'Method':<18} {'Conf':>5}  Matched")
    print("-" * 100)
    for q in tests:
        r = gw.route(q)
        matched = r["matched_topics"].get(r["hive"], [])
        m_str = ", ".join(matched[:2]) if matched else ""
        print(f"{q:<55} {r['hive']:<12} {r['method']:<18} {r['confidence']:>4.0%}  {m_str}")
    print()


# ── CLI definition ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="brainai",
        description="BCDR DeveloperAI — Unified CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # chat
    p = sub.add_parser("chat", help="Interactive chat")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--no-kusto", action="store_true", help="Skip auto-starting Kusto MCP")
    p.set_defaults(func=cmd_chat)

    # index
    p = sub.add_parser("index", help="Index markdown docs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_index)

    # code-index
    p = sub.add_parser("code-index", help="Index source code")
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_code_index)

    # hive-index
    p = sub.add_parser("hive-index", help="Index code for all hives")
    p.add_argument("--force", action="store_true")
    p.add_argument("--hive", type=str, default=None, help="Index only this hive")
    p.add_argument("--list", action="store_true", help="List hives and exit")
    p.add_argument("--refresh-topics", action="store_true", help="Auto-extract routing topics after indexing")
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_hive_index)

    # status
    p = sub.add_parser("status", help="Show hive index status, topics & staleness")
    p.set_defaults(func=cmd_status)

    # gateway-test
    p = sub.add_parser("gateway-test", help="Test gateway routing with sample questions")
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_gateway_test)

    # sync
    p = sub.add_parser("sync", help="Sync docs from Azure DevOps")
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_sync)

    # daily
    p = sub.add_parser("daily", help="Run daily sync + index")
    p.set_defaults(func=cmd_daily)

    # kusto-server
    p = sub.add_parser("kusto-server", help="Start Kusto MCP server")
    p.set_defaults(func=cmd_kusto_server)

    # bot
    p = sub.add_parser("bot", help="Start Teams bot")
    p.add_argument("--port", type=int, default=3978)
    p.add_argument("--config", type=str, default=None)
    p.set_defaults(func=cmd_bot)

    # doc-improver
    p = sub.add_parser("doc-improver", help="Run doc improvement cycle")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--daemon", action="store_true")
    p.set_defaults(func=cmd_doc_improver)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
