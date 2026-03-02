"""
CLI Chat Interface for BCRD DeveloperAI.

Provides a rich terminal chat experience with:
- Colored output (agent identification)
- Multi-line input support
- Conversation history
- Commands: /clear, /agents, /quit, /help
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from brain_ai.config import get_config
from brain_ai.agents.brain_agent import BrainAgent

log = logging.getLogger(__name__)

console = Console()

AGENT_COLORS = {
    "knowledge": "cyan",
    "debug": "yellow",
    "coder": "magenta",
    "knowledge_updater": "green",
    "brain": "green",
}

WELCOME_TEXT = """
# ?? BCRD DeveloperAI - Azure Backup Management Assistant

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
- `/quit`    - Exit

Type your question below to get started!
"""


def _print_response(agent_name: str, response: str):
    """Pretty-print an agent response."""
    color = AGENT_COLORS.get(agent_name, "white")
    title = f"?? {agent_name.capitalize()} Agent"
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


def _handle_command(command: str, brain: BrainAgent) -> bool:
    """
    Handle slash commands. Returns True if the command was handled.
    """
    cmd = command.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        updater = brain._agents.get("knowledge_updater")
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
        updater = brain._agents.get("knowledge_updater")
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
        brain.reset_conversation()
        console.print("[dim]Conversation history cleared.[/dim]")
        return True

    if cmd == "/agents":
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
        updater = brain._agents.get("knowledge_updater")
        if updater and updater.pending_count > 0:
            console.print(
                f"[green]📋 {updater.pending_count} pending correction(s):[/green]\n"
                f"{updater.pending_summary()}\n\n"
                f'[dim]Say "submit" or "agree" to create a PR.[/dim]'
            )
        else:
            console.print("[dim]No pending corrections.[/dim]")
        return True

    return False


def run_chat(config_path: Optional[str] = None):
    """Main CLI chat loop."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler("bcrd_devai.log"), logging.StreamHandler()],
    )
    # Quiet down noisy loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    console.print(Markdown(WELCOME_TEXT))

    try:
        cfg = get_config(config_path)
        brain = BrainAgent(cfg)
    except Exception as e:
        console.print(f"[red]Failed to initialize BCRD DeveloperAI: {e}[/red]")
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

            # Send to brain agent
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                result = brain.chat(user_input)

            _print_response(result["agent"], result["response"])

        except KeyboardInterrupt:
            updater = brain._agents.get("knowledge_updater")
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
