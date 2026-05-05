#!/usr/bin/env python3
"""
Index source code for all configured hives into separate ChromaDB collections.

Each hive gets its own code collection, isolated from other hives.
This script reads hive definitions from config.yaml and indexes the
repo_clone_dir + sync_paths for each hive.

Usage:
    python run_hive_index.py                    # Index all hives (incremental)
    python run_hive_index.py --force             # Force re-index all
    python run_hive_index.py --hive bms          # Index only the 'bms' hive
    python run_hive_index.py --hive bms --force  # Force re-index 'bms'
    python run_hive_index.py --list              # List all hives and their status
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from brain_ai.config import get_config
from brain_ai.vectorstore.code_indexer import CodeIndexer
from brain_ai.hive.discovery_store import DiscoveryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def _build_hive_config(global_cfg: dict, hive_name: str, hive_def: dict) -> dict:
    """Merge hive-level overrides into the global config (same logic as Hive class)."""
    import copy

    merged = copy.deepcopy(global_cfg)

    if "paths" in hive_def:
        merged.setdefault("paths", {})
        merged["paths"].update(hive_def["paths"])

    if "vectorstore" in hive_def:
        merged.setdefault("vectorstore", {})
        merged["vectorstore"].update(hive_def["vectorstore"])

    if "code_index" in hive_def:
        merged.setdefault("code_index", {})
        merged["code_index"].update(hive_def["code_index"])

    return merged


def index_hive(global_cfg: dict, hive_name: str, hive_def: dict, force: bool = False) -> dict:
    """Index a single hive's code repo."""
    display_name = hive_def.get("display_name", hive_name)
    repo_dir = hive_def.get("paths", {}).get("repo_clone_dir", global_cfg.get("paths", {}).get("repo_clone_dir", ".repo_cache"))
    code_cfg = hive_def.get("code_index", {})
    collection = code_cfg.get("collection_name", f"{hive_name}_code")
    sync_paths = code_cfg.get("sync_paths", ["src"])

    repo_path = Path(repo_dir).resolve()
    if not repo_path.exists():
        log.warning("Repo path does not exist for hive '%s': %s", hive_name, repo_path)
        return {"hive": hive_name, "status": "skipped", "reason": f"repo not found: {repo_path}"}

    print(f"\n{'='*60}")
    print(f"🐝 Indexing hive: {hive_name} ({display_name})")
    print(f"   Repo: {repo_path}")
    print(f"   Collection: {collection}")
    print(f"   Sync paths: {sync_paths}")
    print(f"{'='*60}")

    merged_cfg = _build_hive_config(global_cfg, hive_name, hive_def)

    try:
        indexer = CodeIndexer(merged_cfg)
        before_count = indexer.collection.count()
        print(f"   Collection currently has {before_count} chunks.")

        start = time.time()
        result = indexer.index_all(force=force)
        elapsed = time.time() - start

        after_count = indexer.collection.count()
        result["hive"] = hive_name
        result["collection"] = collection
        result["elapsed_seconds"] = round(elapsed, 1)
        result["chunks_before"] = before_count
        result["chunks_after"] = after_count

        print(f"   ✅ Done in {elapsed:.1f}s — {after_count} chunks total")

        # Record in discovery store
        try:
            ds = DiscoveryStore()
            ds.update_index_metadata(
                hive_name=hive_name,
                index_type="code",
                chunks=after_count,
                display_name=display_name,
                duration_s=round(elapsed, 1),
            )
            ds.close()
        except Exception as ds_err:
            log.warning("Could not update discovery store: %s", ds_err)

        return result

    except Exception as e:
        log.error("Failed to index hive '%s': %s", hive_name, e, exc_info=True)
        return {"hive": hive_name, "status": "error", "error": str(e)}


def list_hives(global_cfg: dict):
    """Print a summary of all configured hives."""
    hive_defs = global_cfg.get("hives", {}).get("definitions", {})
    default = global_cfg.get("hives", {}).get("default_hive", "")

    print(f"\n🐝 Configured Hives ({len(hive_defs)} total):")
    print(f"{'─'*70}")

    for name, hdef in hive_defs.items():
        display = hdef.get("display_name", name)
        repo = hdef.get("paths", {}).get("repo_clone_dir", global_cfg.get("paths", {}).get("repo_clone_dir", "?"))
        collection = hdef.get("code_index", {}).get("collection_name", f"{name}_code")
        sync_paths = hdef.get("code_index", {}).get("sync_paths", ["src"])
        agents = hdef.get("agents", {}).get("enabled", [])
        topics = hdef.get("scope", {}).get("topics", [])
        repo_exists = Path(repo).resolve().exists()

        marker = " ⭐ (default)" if name == default else ""
        status = "✅" if repo_exists else "❌ repo missing"

        print(f"\n  {name}{marker}")
        print(f"    Display:    {display}")
        print(f"    Repo:       {repo}  [{status}]")
        print(f"    Collection: {collection}")
        print(f"    Sync:       {', '.join(sync_paths)}")
        print(f"    Agents:     {', '.join(agents)}")
        print(f"    Topics:     {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")

    print(f"\n{'─'*70}")


def main():
    parser = argparse.ArgumentParser(description="Index code for all hives")
    parser.add_argument("--force", action="store_true", help="Force re-index all files")
    parser.add_argument("--hive", type=str, default=None, help="Index only this hive")
    parser.add_argument("--list", action="store_true", help="List all hives and exit")
    parser.add_argument("--refresh-topics", action="store_true",
                        help="Auto-extract topics from indexed content after indexing")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = get_config(args.config)

    if args.list:
        list_hives(cfg)
        return

    hive_defs = cfg.get("hives", {}).get("definitions", {})

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
        result = index_hive(cfg, name, hdef, force=args.force)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"🐝 HIVE INDEXING SUMMARY ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")
    for r in results:
        status = r.get("status", "ok")
        if status == "skipped":
            print(f"  ⏭️  {r['hive']}: skipped — {r.get('reason', '?')}")
        elif status == "error":
            print(f"  ❌ {r['hive']}: error — {r.get('error', '?')}")
        else:
            print(f"  ✅ {r['hive']}: {r.get('chunks_after', '?')} chunks ({r.get('elapsed_seconds', '?')}s)")

    # Auto-extract topics if requested
    if args.refresh_topics:
        print(f"\n🔄 Refreshing topics for indexed hives...")
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
            log.error("Topic extraction failed: %s", e)
            print(f"  ❌ Topic extraction error: {e}")

    print()


if __name__ == "__main__":
    main()
