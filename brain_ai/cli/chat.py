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
from datetime import datetime
from pathlib import Path
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
    "dpp": "blue",
    "rsv": "bright_blue",
    "dataplane": "magenta",
    "common": "green",
    "protection": "red",
    "regional": "cyan",
    "pitcatalog": "yellow",
    "datamover": "bright_red",
    "monitoring": "bright_green",
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
- `/status`  - Show hive index status, staleness & topic counts
- `/save`    - Save last response as rich Markdown (with Mermaid routing diagram)
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
    """Show a visual routing diagram with Gateway routing info."""
    hive = result.get("hive")
    consulted = result.get("consulted_hives", [])
    chain = result.get("delegation_chain", [])
    agent = result.get("agent", "brain")
    routing = result.get("routing", {})

    if not hive:
        return

    # Always show at least a compact routing line
    hive_color = HIVE_COLORS.get(hive, "white")
    method = routing.get("method", "?")
    confidence = routing.get("confidence", 0)
    conf_style = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"

    # Compact routing summary
    route_text = Text()
    route_text.append("🔀 Gateway → ", style="bold cyan")
    route_text.append(f"{hive.upper()}", style=f"bold {hive_color}")
    route_text.append(f"  [{method}, ", style="dim")
    route_text.append(f"{confidence:.0%}", style=conf_style)
    route_text.append(" confidence]", style="dim")

    # Show matched topics if any
    matched = routing.get("matched_topics", {}).get(hive, [])
    if matched:
        route_text.append(f"  matched: {', '.join(matched[:3])}", style="dim")

    console.print(route_text)

    # Show [ASK:] callbacks if any
    cross_asks = result.get("cross_hive_asks", [])
    if cross_asks:
        for ask in cross_asks:
            ask_text = Text()
            target = ask.get("target", "?")
            ask_color = HIVE_COLORS.get(target, "white")
            ask_text.append("  ↪ ", style="dim")
            ask_text.append(f"[ASK:{target.upper()}]", style=f"bold {ask_color}")
            question = ask.get("question", "")
            if len(question) > 60:
                question = question[:60] + "..."
            ask_text.append(f" {question}", style="dim")
            console.print(ask_text)

    # Show code boundary crossings if any
    code_boundaries = result.get("code_boundaries", [])
    if code_boundaries:
        for bnd in code_boundaries:
            bnd_text = Text()
            target = bnd.get("target_hive", "?")
            bnd_color = HIVE_COLORS.get(target, "white")
            pattern = bnd.get("pattern", "?")
            source = bnd.get("source_file", "")
            bnd_text.append("  🔗 ", style="dim")
            bnd_text.append(f"[BOUNDARY:{target.upper()}]", style=f"bold {bnd_color}")
            bnd_text.append(f" {pattern}", style="dim italic")
            if source:
                bnd_text.append(f" (from {source})", style="dim")
            console.print(bnd_text)

    # Show full tree only for cross-hive activity
    if not consulted and not chain and not cross_asks and not code_boundaries:
        return

    tree = Tree(
        Text("📨 Your Question", style="bold white"),
        guide_style="dim",
    )

    # Gateway node
    gw_label = Text("🔀 Gateway", style="bold cyan")
    gw_node = tree.add(gw_label)

    # Primary hive
    hive_color = HIVE_COLORS.get(hive, "white")
    primary_label = Text(f"🐝 {hive.upper()} ", style=f"bold {hive_color}")
    primary_label.append(f"({agent} agent)", style="dim")
    primary_node = gw_node.add(primary_label)

    # [ASK:] callbacks (agent-initiated, inline)
    if cross_asks:
        for ask in cross_asks:
            target = ask.get("target", "?")
            a_color = HIVE_COLORS.get(target, "white")
            question = ask.get("question", "")
            if len(question) > 70:
                question = question[:70] + "..."
            a_label = Text(f"↩️  [ASK:{target.upper()}] ", style=f"bold {a_color}")
            a_label.append(question, style="dim italic")
            primary_node.add(a_label)

    # Delegation chain (explicit [DELEGATE:] hops)
    if chain:
        for hop in chain:
            hop_color = HIVE_COLORS.get(hop, "white")
            hop_label = Text(f"🔗 {hop.upper()} ", style=f"bold {hop_color}")
            hop_label.append("← delegated", style="dim yellow")
            gw_node.add(hop_label)

    # Code boundary crossings (coder-detected, evidence-based)
    if code_boundaries:
        for bnd in code_boundaries:
            target = bnd.get("target_hive", "?")
            b_color = HIVE_COLORS.get(target, "white")
            pattern = bnd.get("pattern", "?")
            source = bnd.get("source_file", "")
            b_label = Text(f"🔗 [BOUNDARY:{target.upper()}] ", style=f"bold {b_color}")
            b_label.append(pattern, style="dim italic")
            if source:
                b_label.append(f" ← {source}", style="dim")
            primary_node.add(b_label)

    # Consulted hives (proactive discovery) — nested under primary hive
    consulted_details = result.get("consulted_details", [])
    if consulted_details:
        for cd in consulted_details:
            c_name = cd.get("hive", "?")
            c_question = cd.get("question", "")
            c_color = HIVE_COLORS.get(c_name, "white")
            if len(c_question) > 70:
                c_question = c_question[:70] + "..."
            c_label = Text(f"🐝 {c_name.upper()} ", style=f"bold {c_color}")
            c_label.append(c_question, style="dim italic")
            primary_node.add(c_label)
    elif consulted:
        for c in consulted:
            c_color = HIVE_COLORS.get(c, "white")
            c_label = Text(f"🐝 {c.upper()} ", style=f"bold {c_color}")
            c_label.append("← consulted", style="dim green")
            primary_node.add(c_label)

    # Synthesis step
    asked_hives = list({a.get("target", "") for a in cross_asks})
    boundary_hives = list({b.get("target_hive", "") for b in code_boundaries})
    all_sources = [hive] + chain + consulted + asked_hives + boundary_hives
    if len(all_sources) > 1:
        synth_label = Text("✨ Response ", style="bold white")
        synth_label.append(f"({len(all_sources)} hives combined)", style="dim")
        gw_node.add(synth_label)

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

    if cmd.startswith("/save"):
        return _handle_save(cmd, brain)

    if cmd == "/status":
        try:
            from brain_ai.hive.discovery_store import DiscoveryStore
            ds = DiscoveryStore()
            all_meta = ds.get_all_metadata()
            stale = ds.get_stale_hives()
            stale_names = {s["hive_name"] for s in stale}

            if not all_meta:
                console.print("[dim]No index metadata yet. Run an index to populate.[/dim]")
            else:
                console.print("\n[bold]📊 Hive Index Status:[/bold]")
                for m in all_meta:
                    name = m["hive_name"]
                    code_chunks = m.get("code_chunks", 0)
                    doc_chunks = m.get("doc_chunks", 0)
                    last_code = m.get("last_code_index", "never")
                    if last_code and last_code != "never":
                        last_code = last_code[:19].replace("T", " ")
                    marker = " [yellow]⚠️ stale[/yellow]" if name in stale_names else " [green]✓[/green]"
                    topics = ds.get_topics(name)
                    topic_count = len(topics)
                    console.print(
                        f"  [bold]{name}[/bold]{marker}\n"
                        f"    Code: {code_chunks:,} chunks | Docs: {doc_chunks:,} chunks\n"
                        f"    Last indexed: {last_code} | Topics: {topic_count}"
                    )
            ds.close()
        except Exception as e:
            console.print(f"[red]Status error: {e}[/red]")
        return True

    return False


# ── Module-level state for /save ──────────────────────────────────
_last_question: Optional[str] = None
_last_result: Optional[dict] = None


def _build_mermaid(result: dict) -> str:
    """Build a Mermaid flowchart from routing result."""
    hive = result.get("hive", "unknown")
    agent = result.get("agent", "brain")
    consulted = result.get("consulted_hives", [])
    chain = result.get("delegation_chain", [])
    cross_asks = result.get("cross_hive_asks", [])

    lines = ["```mermaid", "flowchart TD"]
    lines.append('    Q["📨 User Question"]')
    lines.append('    R["🔀 HiveRouter"]')
    lines.append(f'    P["{hive.upper()} 🐝<br/>{agent} agent"]')
    lines.append("    Q --> R")
    lines.append("    R -->|primary| P")

    for i, hop in enumerate(chain):
        node = f"D{i}"
        lines.append(f'    {node}["{hop.upper()} 🔗<br/>delegated"]')
        lines.append(f"    R -->|delegate| {node}")

    for i, c in enumerate(consulted):
        node = f"C{i}"
        lines.append(f'    {node}["{c.upper()} 🐝<br/>consulted"]')
        lines.append(f"    R -->|consult| {node}")

    for i, ask in enumerate(cross_asks):
        target = ask.get("target", "?")
        node = f"A{i}"
        lines.append(f'    {node}["{target.upper()} ↩️<br/>asked"]')
        lines.append(f"    P -->|ASK| {node}")
        lines.append(f"    {node} -->|answer| P")

    asked_hives = [f"A{i}" for i in range(len(cross_asks))]
    all_sources = ["P"] + [f"D{i}" for i in range(len(chain))] + [f"C{i}" for i in range(len(consulted))]
    if len(all_sources) > 1 or asked_hives:
        lines.append('    S["✨ Synthesized Response"]')
        for src in all_sources:
            lines.append(f"    {src} --> S")

    lines.append("```")
    return "\n".join(lines)


def _handle_save(cmd: str, brain) -> bool:
    """Save the last response as a rich Markdown file."""
    global _last_question, _last_result

    if _last_result is None:
        console.print("[dim]Nothing to save yet — ask a question first.[/dim]")
        return True

    # Determine output path
    parts = cmd.split(maxsplit=1)
    if len(parts) > 1:
        out_path = Path(parts[1].strip())
    else:
        save_dir = Path("saved_responses")
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hive = _last_result.get("hive", "unknown")
        out_path = save_dir / f"{hive}_{ts}.md"

    result = _last_result
    hive = result.get("hive", "unknown")
    agent = result.get("agent", "unknown")
    consulted = result.get("consulted_hives", [])
    chain = result.get("delegation_chain", [])
    response = result.get("response", "")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = [
        f"# 🐝 BrainAI Response — {hive.upper()}",
        "",
        "## Metadata",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| **Date** | {now} |",
        f"| **Primary Hive** | {hive} |",
        f"| **Agent** | {agent} |",
    ]
    routing = result.get("routing", {})
    if routing:
        md_lines.append(f"| **Routing Method** | {routing.get('method', '?')} |")
        md_lines.append(f"| **Confidence** | {routing.get('confidence', 0):.0%} |")
        matched = routing.get("matched_topics", {}).get(hive, [])
        if matched:
            md_lines.append(f"| **Matched Topics** | {', '.join(matched[:5])} |")
    if chain:
        md_lines.append(f"| **Delegated To** | {', '.join(chain)} |")
    if consulted:
        md_lines.append(f"| **Consulted Hives** | {', '.join(consulted)} |")

    md_lines += [
        "",
        "## Question",
        "",
        f"> {_last_question}",
        "",
    ]

    # Add Mermaid routing diagram if cross-hive
    if consulted or chain:
        md_lines += [
            "## Routing Flow",
            "",
            _build_mermaid(result),
            "",
        ]

    md_lines += [
        "## Response",
        "",
        response,
        "",
        "---",
        f"*Generated by BrainAI — {len(consulted) + len(chain) + 1} hive(s) involved*",
    ]

    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    console.print(f"[green]💾 Saved to {out_path}[/green]")
    return True


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
            # Check for stale hives
            try:
                from brain_ai.hive.discovery_store import DiscoveryStore
                ds = DiscoveryStore()
                stale = ds.get_stale_hives()
                ds.close()
                if stale:
                    for s in stale:
                        reason = s.get("reason", "unknown")
                        console.print(
                            f"  [yellow]⚠️  {s['hive_name']}: {reason}[/yellow]"
                        )
                    console.print(
                        "[dim]  Run: python scripts/run_hive_index.py --hive <name> --force[/dim]"
                    )
            except Exception:
                pass  # Discovery store not yet initialized — that's fine
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

            # Track for /save
            global _last_question, _last_result
            _last_question = user_input
            _last_result = result

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


if __name__ == "__main__":
    run_chat()
