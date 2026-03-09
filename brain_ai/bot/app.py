"""
BCDR DeveloperAI Teams Bot — aiohttp web application.

Exposes:
  POST /api/messages     — Bot Framework channel connector endpoint
  GET  /api/health       — Health check
  POST /api/daily-sync   — Trigger doc + code re-indexing (webhook-friendly)

Runs all services in one process:
  - Teams Bot (via Bot Framework)
  - Kusto MCP server (background, on configurable port)
  - Daily sync scheduler (background task, configurable interval)
"""

import asyncio
import logging

from aiohttp import web
from botbuilder.core import BotFrameworkAdapter
from botbuilder.schema import Activity

from brain_ai.bot.adapter import create_adapter
from brain_ai.bot.teams_bot import BrainAIBot
from brain_ai.config import get_config

log = logging.getLogger(__name__)


# ── Background services ─────────────────────────────────────────────

async def _run_kusto_server(cfg: dict):
    """Start the Kusto MCP server in a background thread."""
    import uvicorn

    from brain_ai.kusto.server import create_app as create_kusto_app

    kusto_port = cfg.get("kusto", {}).get("mcp_port", 8701)
    kusto_app = create_kusto_app(cfg)

    config = uvicorn.Config(
        kusto_app,
        host="0.0.0.0",
        port=kusto_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    log.info("Starting Kusto MCP server on port %d (background)", kusto_port)
    await server.serve()


async def _daily_sync_loop(cfg: dict, interval_hours: int = 24):
    """Run daily sync + re-index on a repeating schedule."""
    from brain_ai.sync.repo_sync import sync_docs
    from brain_ai.vectorstore.code_indexer import CodeIndexer
    from brain_ai.vectorstore.indexer import DocIndexer

    log.info("Daily sync scheduler started (interval: %dh)", interval_hours)
    while True:
        await asyncio.sleep(interval_hours * 3600)
        log.info("Running scheduled sync & re-index...")
        try:
            sync_docs()
            DocIndexer(cfg).index_all()
            CodeIndexer(cfg).index_all()
            log.info("Scheduled sync & re-index complete")
        except Exception as e:
            log.error("Scheduled sync failed: %s", e, exc_info=True)


async def _doc_improver_loop(cfg: dict):
    """Run the Doc Improver agent on a repeating schedule."""
    from brain_ai.agents.doc_improver_agent import DocImproverAgent

    imp_cfg = cfg.get("doc_improver", {})
    if not imp_cfg.get("enabled", False):
        log.info("Doc improver is disabled — background loop will not start.")
        return

    interval_hours = imp_cfg.get("run_interval_hours", 72)
    log.info("Doc Improver background loop started (interval: %dh)", interval_hours)

    # Wait a bit on startup so other services initialize first
    await asyncio.sleep(60)

    while True:
        log.info("Running Doc Improver cycle...")
        try:
            agent = DocImproverAgent(cfg)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, agent.run_improvement_cycle)
            log.info("Doc Improver cycle complete: %s", result)
        except Exception as e:
            log.error("Doc Improver cycle failed: %s", e, exc_info=True)
        await asyncio.sleep(interval_hours * 3600)


def _trigger_sync(cfg: dict):
    """Run sync + re-index once (called from the /api/daily-sync webhook)."""
    from brain_ai.sync.repo_sync import sync_docs
    from brain_ai.vectorstore.code_indexer import CodeIndexer
    from brain_ai.vectorstore.indexer import DocIndexer

    sync_docs()
    DocIndexer(cfg).index_all()
    CodeIndexer(cfg).index_all()


# ── Web application factory ─────────────────────────────────────────

def create_app(cfg: dict | None = None) -> web.Application:
    """Build and return the aiohttp web application."""
    if cfg is None:
        cfg = get_config()

    bot_cfg = cfg.get("teams_bot", {})
    app_id = bot_cfg.get("app_id", "")
    app_password = bot_cfg.get("app_password", "")

    adapter: BotFrameworkAdapter = create_adapter(app_id, app_password)
    bot = BrainAIBot(cfg, adapter=adapter)

    # ── Route handlers ──────────────────────────────────────────────

    async def messages(req: web.Request) -> web.Response:
        """Main endpoint for Bot Framework messages."""
        if "application/json" not in (req.content_type or ""):
            return web.Response(status=415, text="Unsupported media type")
        body = await req.json()
        activity = Activity().deserialize(body)
        auth_header = req.headers.get("Authorization", "")
        response = await adapter.process_activity(activity, auth_header, bot.on_turn)
        if response:
            return web.json_response(data=response.body, status=response.status)
        return web.Response(status=201)

    async def health(req: web.Request) -> web.Response:
        """Health-check endpoint."""
        return web.json_response({
            "status": "ok",
            "service": "BCDR DeveloperAI Teams Bot",
            "sessions": len(bot._sessions),
            "pending_auto_replies": sum(1 for p in bot._pending.values() if not p.answered),
        })

    async def daily_sync(req: web.Request) -> web.Response:
        """Trigger a manual sync + re-index (webhook-friendly)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _trigger_sync, cfg)
            return web.json_response({"status": "ok", "message": "Sync complete"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # ── Lifecycle hooks ─────────────────────────────────────────────

    async def on_startup(app_obj: web.Application):
        """Start background services when the app starts."""
        # Start the unanswered-message monitor
        bot.start_monitor()

        # Start Kusto MCP server in background
        if bot_cfg.get("start_kusto_server", True):
            asyncio.ensure_future(_run_kusto_server(cfg))

        # Start daily sync scheduler
        sync_hours = bot_cfg.get("sync_interval_hours", 24)
        if sync_hours > 0:
            asyncio.ensure_future(_daily_sync_loop(cfg, sync_hours))

        # Start Doc Improver background loop
        asyncio.ensure_future(_doc_improver_loop(cfg))

    app = web.Application()
    app.router.add_post("/api/messages", messages)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/daily-sync", daily_sync)
    app.on_startup.append(on_startup)

    log.info("BCDR DeveloperAI app created (app_id=%s)", app_id or "(local dev)")
    return app


# ── Entry point ──────────────────────────────────────────────────────

def run(port: int = 3978, cfg: dict | None = None):
    """Start the bot web server with all background services."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler("BCDR_devai_bot.log"), logging.StreamHandler()],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    if cfg is None:
        cfg = get_config()

    app = create_app(cfg)

    print(f"\n{'='*60}")
    print("  🤖 BCDR DeveloperAI Teams Bot")
    print(f"  Bot endpoint:   http://localhost:{port}/api/messages")
    print(f"  Health check:   http://localhost:{port}/api/health")
    print(f"  Manual sync:    POST http://localhost:{port}/api/daily-sync")
    print(f"{'='*60}")
    print("\n  Background services:")
    print(f"  • Kusto MCP server (port {cfg.get('kusto', {}).get('mcp_port', 8701)})")
    print(f"  • Daily sync scheduler ({cfg.get('teams_bot', {}).get('sync_interval_hours', 24)}h)")
    print("  • Unanswered-message monitor (10 min)")
    imp = cfg.get("doc_improver", {})
    if imp.get("enabled", False):
        print(f"  • Doc Improver ({imp.get('run_interval_hours', 72)}h cycle, "
              f"max {imp.get('max_iterations', 3)} iterations)")
    print(f"{'='*60}\n")

    web.run_app(app, host="0.0.0.0", port=port)
