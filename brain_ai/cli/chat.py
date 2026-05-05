"""
CLI Chat Interface for BCDR DeveloperAI.

Provides a rich terminal chat experience with:
- Colored output (agent identification)
- Multi-line input support
- Conversation history
- Commands: /clear, /agents, /quit, /help
"""

import logging
import sys
from typing import Optional

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from brain_ai.agents.brain_agent import BrainAgent
from brain_ai.config import get_config
from brain_ai.hive.router import HiveRouter

log = logging.getLogger(__name__)

console = Console()

# Module-level reference so /logs command can toggle verbosity
_console_handler: Optional[logging.StreamHandler] = None

AGENT_COLORS = {
    "knowledge": "cyan",
    "debug": "yellow",
    "coder": "magenta",
    "knowledge_updater": "green",
    "brain": "green",
}

HIVE_COLORS = {
    "bms": "blue",
    "common": "green",
    "protection": "red",
    "core": "magenta",
}

WELCOME_TEXT = """
# ?? BCDR DeveloperAI - Azure Backup Management Assistant

Welcome! I can help you with:
- **Project questions** → Knowledge Agent (architecture, features, config, etc.)
- **Debugging issues** → Debug Agent (error analysis, KQL queries, troubleshooting)
- **Code tracing** → Coder Agent (code paths, root cause analysis, source code flows)
- **Correcting docs** → Knowledge Updater (fix docs & create a PR automatically)

**Commands:**
- `/help`    - Show this message
- `/clear`   - Clear conversation history
- `/agents`  - List available agents
- `/kusto`   - Check Kusto MCP server status
- `/code`    - Check code index status
- `/pending` - Show staged doc corrections
- `/hives`   - List available hives & their scopes
- `/logs`    - Toggle verbose logging on/off (for testing)
- `/quit`    - Exit

Type your question below to get started!
"""


def _get_updater(brain: BrainAgent | HiveRouter):
    """Safely retrieve the knowledge_updater agent, if available."""
    if isinstance(brain, HiveRouter):
        # In hive mode, check the default hive's BrainAgent
        default = brain.registry.default_hive
        if default:
            return default.brain._agents.get("knowledge_updater")
        return None
    return brain._agents.get("knowledge_updater")


def _print_routing_diagram(result: dict):
    """Show a visual routing diagram when cross-hive work happened."""
    hive = result.get("hive")
    consulted = result.get("consulted_hives", [])
    chain = result.get("delegation_chain", [])
    agent = result.get("agent", "brain")

    # Only show diagram when there's cross-hive activity
    if not hive or (not consulted and not chain):
        return

    tree = Tree(
        Text("📨 Your Question", style="bold white"),
        guide_style="dim",
    )

    # Router node
    router_node = tree.add(Text("🔀 HiveRouter", style="bold cyan"))

    # Primary hive
    hive_color = HIVE_COLORS.get(hive, "white")
    primary_label = Text(f"🐝 {hive.upper()} ", style=f"bold {hive_color}")
    primary_label.append("← primary", style="dim")
    primary_node = router_node.add(primary_label)
    primary_node.add(Text(f"⚙️  {agent.capitalize()} Agent", style="dim"))

    # Delegation chain (explicit [DELEGATE:] hops)
    if chain:
        for hop in chain:
            hop_color = HIVE_COLORS.get(hop, "white")
            hop_label = Text(f"🔗 {hop.upper()} ", style=f"bold {hop_color}")
            hop_label.append("← delegated", style="dim yellow")
            router_node.add(hop_label)

    # Consulted hives (proactive discovery)
    if consulted:
        for c in consulted:
            c_color = HIVE_COLORS.get(c, "white")
            c_label = Text(f"🐝 {c.upper()} ", style=f"bold {c_color}")
            c_label.append("← consulted", style="dim green")
            router_node.add(c_label)

    # Synthesis step
    all_sources = [hive] + chain + consulted
    if len(all_sources) > 1:
        synth_label = Text("✨ Synthesized Response ", style="bold white")
        synth_label.append(f"({len(all_sources)} hives)", style="dim")
        router_node.add(synth_label)

    console.print()
    console.print(Panel(tree, title="🗺️  Routing Flow", border_style="dim", padding=(0, 1)))


def _print_response(agent_name: str, response: str, hive_name: str | None = None):
    """Pretty-print an agent response."""
    color = AGENT_COLORS.get(agent_name, "white")
    if hive_name:
        hive_color = HIVE_COLORS.get(hive_name, "white")
        title = f"🐝 [{hive_name.upper()}] {agent_name.capitalize()} Agent"
    else:
        title = f"🤖 {agent_name.capitalize()} Agent"
    console.print()
    console.print(
        Panel(
            Markdown(response),
            title=title,
            title_align="left",
            border_style=color,
            padding=(1, 2),
        )
    )


def _handle_command(command: str, brain: BrainAgent | HiveRouter) -> bool:
    """
    Handle slash commands. Returns True if the command was handled.
    """
    cmd = command.strip().lower()

    # /hives — list available hives (only in hive mode)
    if cmd == "/hives":
        if isinstance(brain, HiveRouter):
            console.print("\n[bold]🐝 Active Hives:[/bold]")
            for hive in brain.registry:
                agents_str = ", ".join(brain.get_hive_agents(hive.name))
                topics_str = ", ".join(hive.topics[:5])
                if len(hive.topics) > 5:
                    topics_str += f" (+{len(hive.topics) - 5} more)"
                marker = " ← active" if brain._last_hive == hive.name else ""
                console.print(
                    f"  [bold]{hive.name}[/bold] — {hive.display_name}{marker}\n"
                    f"    Agents: {agents_str}\n"
                    f"    Scope: {topics_str}"
                )
        else:
            console.print("[dim]Hive mode is not enabled. Set hives.enabled: true in config.yaml[/dim]")
        return True

    if cmd in ("/quit", "/exit", "/q"):
        updater = _get_updater(brain)
        if updater and updater.pending_count > 0:
            console.print(
                f"\n[yellow]⚠️  You have {updater.pending_count} unsaved doc correction(s):[/yellow]"
            )
            for i, pc in enumerate(updater._pending_corrections, 1):
                console.print(f"  [yellow]{i}. {pc['doc_source']} — {pc['summary']}[/yellow]")
            console.print()
            try:
                answer = prompt(
                    '  Submit as PR before exiting? (yes/no/cancel) > '
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                answer = "cancel"
            if answer in ("cancel", "c"):
                return True  # stay in the loop
            if answer in ("yes", "y"):
                with console.status("[bold green]Creating PR...[/bold green]", spinner="dots"):
                    result = brain.chat("submit")
                _print_response(result["agent"], result["response"])
        console.print("\n[dim]Goodbye! 👋[/dim]\n")
        sys.exit(0)

    if cmd in ("/help", "/h"):
        console.print(Markdown(WELCOME_TEXT))
        return True

    if cmd == "/clear":
        updater = _get_updater(brain)
        if updater and updater.pending_count > 0:
            console.print(
                f"[yellow]⚠️  You have {updater.pending_count} unsaved doc correction(s). "
                f"Clearing will discard them.[/yellow]"
            )
            try:
                answer = prompt('  Continue? (yes/no) > ').strip().lower()
            except (KeyboardInterrupt, EOFError):
                answer = "no"
            if answer not in ("yes", "y"):
                console.print("[dim]Clear cancelled.[/dim]")
                return True
            updater.clear_pending()
        if isinstance(brain, HiveRouter):
            brain.reset_conversation()
        else:
            brain.reset_conversation()
        console.print("[dim]Conversation history cleared.[/dim]")
        return True

    if cmd == "/agents":
        if isinstance(brain, HiveRouter):
            for hive in brain.registry:
                agents = list(hive.brain._agents.keys())
                console.print(f"[dim]{hive.name}: {', '.join(agents)}[/dim]")
        else:
            agents = list(brain._agents.keys())
            console.print(f"[dim]Available agents: {', '.join(agents)}[/dim]")
        return True

    if cmd == "/kusto":
        from brain_ai.kusto.client import KustoMCPClient
        kusto = KustoMCPClient(brain.cfg)
        health = kusto.health_check()
        if health.get("connected"):
            console.print(
                f"[green]Kusto MCP: Connected[/green] "
                f"({health['cluster_url']} / {health['database']})"
            )
        elif health.get("status") == "unreachable":
            console.print(
                "[yellow]Kusto MCP server not running.[/yellow] "
                "Start it with: [bold]python run_kusto_server.py[/bold]\n"
                "[dim]Debug Agent will fall back to direct SDK (requires az login).[/dim]"
            )
        else:
            console.print(
                f"[red]Kusto MCP: {health.get('status', 'error')}[/red] "
                f"- {health.get('error', 'unknown error')}"
            )
        return True

    if cmd == "/code":
        try:
            from brain_ai.vectorstore.code_indexer import CodeIndexer
            code_idx = CodeIndexer(brain.cfg)
            count = code_idx.collection.count()
            if count > 0:
                console.print(
                    f"[green]Code index: {count} chunks indexed[/green] "
                    f"(collection: {code_idx.collection_name})"
                )
            else:
                console.print(
                    "[yellow]Code index is empty.[/yellow] "
                    "Run: [bold]python run_code_index.py[/bold]"
                )
        except Exception as e:
            console.print(f"[red]Code index error: {e}[/red]")
        return True

    if cmd == "/pending":
        updater = _get_updater(brain)
        if updater and updater.pending_count > 0:
            console.print(
                f"[green]📋 {updater.pending_count} pending correction(s):[/green]\n"
                f"{updater.pending_summary()}\n\n"
                f'[dim]Say "submit" or "agree" to create a PR.[/dim]'
            )
        else:
            console.print("[dim]No pending corrections.[/dim]")
        return True

    if cmd == "/logs":
        if _console_handler is None:
            console.print("[red]Logging not initialised yet.[/red]")
            return True
        if _console_handler.level <= logging.DEBUG:
            # Currently verbose -> switch to quiet
            _console_handler.setLevel(logging.WARNING)
            console.print("[dim]Verbose logging OFF — only warnings/errors shown in chat.[/dim]")
        else:
            # Currently quiet -> switch to verbose
            _console_handler.setLevel(logging.DEBUG)
            console.print(
                "[green]Verbose logging ON — all logs will appear in chat.[/green]\n"
                "[dim]Use /logs again to turn off.[/dim]"
            )
        return True

    return False


def run_chat(config_path: Optional[str] = None):
    """Main CLI chat loop."""
    # Setup logging — file gets everything, console only gets warnings/errors
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler: verbose (INFO+)
    file_handler = logging.FileHandler("BCDR_devai.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    # Console handler: quiet (WARNING+ only, so logs don't flood the chat)
    global _console_handler
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.WARNING)
    _console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    root_logger.addHandler(_console_handler)

    # Quiet down noisy third-party loggers
    for noisy in ("chromadb", "httpx", "anthropic", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    console.print(Markdown(WELCOME_TEXT))

    try:
        cfg = get_config(config_path)
        hives_enabled = cfg.get("hives", {}).get("enabled", False)
        if hives_enabled:
            brain = HiveRouter(cfg)
            console.print("[dim]🐝 Hive mode active — "
                          f"{len(brain.registry)} hive(s): "
                          f"{', '.join(brain.registry.names)}[/dim]")
        else:
            brain = BrainAgent(cfg)
    except Exception as e:
        console.print(f"[red]Failed to initialize BCDR DeveloperAI: {e}[/red]")
        console.print("[dim]Check your config.yaml and try again.[/dim]")
        return

    history = InMemoryHistory()

    while True:
        try:
            user_input = prompt(
                "\n?? You > ",
                history=history,
                multiline=False,
            ).strip()

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                if _handle_command(user_input, brain):
                    continue

            # Send to brain agent / hive router
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                result = brain.chat(user_input)

            _print_routing_diagram(result)
            _print_response(result["agent"], result["response"], result.get("hive"))

        except KeyboardInterrupt:
            updater = _get_updater(brain)
            if updater and updater.pending_count > 0:
                console.print(
                    f"\n[yellow]⚠️  {updater.pending_count} unsaved correction(s). "
                    f"Use /quit to save or /pending to review.[/yellow]"
                )
            else:
                console.print("\n[dim]Use /quit to exit.[/dim]")
            continue
        except EOFError:
            break
