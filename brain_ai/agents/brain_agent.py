"""
Brain Agent - the central orchestrator/router.

Receives user messages and routes them to the appropriate sub-agent:
  - Knowledge Agent: project questions, architecture, how-to
  - Debug Agent: error debugging, Kusto queries, troubleshooting

Extensible: add new agents by registering them in AGENT_REGISTRY.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from brain_ai.agents.coder_agent import CoderAgent
from brain_ai.agents.debug_agent import DebugAgent
from brain_ai.agents.knowledge_agent import KnowledgeAgent
from brain_ai.agents.knowledge_updater_agent import KnowledgeUpdaterAgent
from brain_ai.config import get_config
from brain_ai.llm_client import LLMClient

log = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the Brain Agent, a routing orchestrator for the Azure Backup Management AI assistant.

Your ONLY job is to classify the user's message and decide which specialized agent should handle it.

Available agents:
- "knowledge": For questions about the project, architecture, features, processes, documentation, how-to, configuration, design, etc.  The user is ASKING for information; they are not providing corrections or managing docs.
- "debug": For debugging errors, troubleshooting issues, running Kusto/KQL queries, investigating failures, analyzing logs, or any request that involves diagnosing a problem.  Also handles ANY follow-up to a debug conversation (e.g. providing a time range, answering a clarifying question from the debug agent, saying "go ahead", "yes", "30 days", etc.).
- "coder": For tracing code paths, understanding source code flows, finding which code handles a specific operation or error, reviewing how a feature is implemented in the BMS service codebase, identifying root causes at the code level, explaining class/method call chains, or any question about the actual source code (C#, Python, etc.).
- "knowledge_updater": For ALL documentation management operations.  This includes:
  • Correcting information ("that's wrong", "actually it works like", "the doc is incorrect")
  • Updating / fixing docs ("please update", "fix the documentation", "this should say")
  • Creating new docs ("create doc", "new doc", "draft a doc", "save doc", "add doc")
  • Submitting changes ("submit", "agree", "create pr", "push it", "lgtm", "ship it")
  • Discarding changes ("discard", "cancel changes", "drop corrections")
  • Reviewing pending changes ("what's pending", "show corrections")
  • Any follow-up to a knowledge_updater conversation where the user is confirming, accepting, or providing additional content for a doc operation.

IMPORTANT ROUTING RULES:
1. Look at the conversation history.  If the previous assistant response came from a specific agent, the current message is very likely a follow-up — route to the SAME agent unless the user clearly changes topic.
2. Short replies like time ranges ("30 days"), confirmations ("yes", "go ahead", "proceed"), or GUIDs are almost always follow-ups to the currently active agent.
3. If the knowledge_updater previously offered to create a doc and the user responds with "yes", "create doc", "save doc", "go ahead", etc., route to "knowledge_updater".
4. If there are pending corrections and the user says "submit", "agree", "discard", etc., route to "knowledge_updater".
5. Route to "coder" ONLY when the user is specifically asking about source code, call chains, or implementation details in the codebase — NOT for doc operations.
6. Only route to "knowledge" if the user is clearly asking a NEW informational question.

{state_context}

Respond with EXACTLY one of these formats (no other text):
ROUTE:knowledge
ROUTE:debug
ROUTE:coder
ROUTE:knowledge_updater

If the message is a greeting or unclear AND there is no active conversation, default to:
ROUTE:knowledge
"""


class BrainAgent:
    """
    Central orchestrator that routes messages to sub-agents.

    Usage:
        brain = BrainAgent()
        response = brain.chat("How does the backup policy work?")
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.llm = LLMClient(cfg)

        # Hive identity (set when this BrainAgent is owned by a Hive)
        self.hive_name: Optional[str] = None
        self.hive_display_name: Optional[str] = None
        self._peer_hive_names: List[str] = []  # other hives available

        # Agent registry - add new agents here
        self._agents: Dict[str, Any] = {}
        enabled = cfg.get("agents", {}).get("enabled", ["knowledge", "debug"])

        if "knowledge" in enabled:
            self._agents["knowledge"] = KnowledgeAgent(cfg)
        if "debug" in enabled:
            self._agents["debug"] = DebugAgent(cfg)
        if "coder" in enabled:
            self._agents["coder"] = CoderAgent(cfg)
        if "knowledge_updater" in enabled:
            self._agents["knowledge_updater"] = KnowledgeUpdaterAgent(cfg)

        # Conversation history for multi-turn context
        self._conversation_history: List[Dict] = []
        # Track which agent handled the last turn (for routing continuity)
        self._last_agent: Optional[str] = None

        log.info("BrainAgent initialized with agents: %s", list(self._agents.keys()))

    def register_agent(self, name: str, agent_instance: Any):
        """Register a new agent dynamically."""
        self._agents[name] = agent_instance
        log.info("Registered new agent: %s", name)

    def set_hive_context(
        self,
        hive_name: str,
        display_name: str,
        peer_hive_names: List[str],
    ):
        """Set this BrainAgent's hive identity (called by Hive.__init__)."""
        self.hive_name = hive_name
        self.hive_display_name = display_name
        self._peer_hive_names = [h for h in peer_hive_names if h != hive_name]
        log.info(
            "BrainAgent hive context set: %s (peers: %s)",
            hive_name,
            self._peer_hive_names,
        )

    def _build_delegation_hint(self) -> str:
        """Build a delegation instruction for sub-agent system prompts.

        If this BrainAgent is part of a hive and peers exist, agents
        are taught to emit ``[DELEGATE:<hive>]`` when they detect
        the answer requires another domain.
        """
        if not self._peer_hive_names:
            return ""
        peers = ", ".join(self._peer_hive_names)
        return (
            f"\n\nCROSS-DOMAIN DELEGATION:\n"
            f"You are part of the '{self.hive_name}' domain. "
            f"Other domains available: [{peers}].\n"
            f"If you determine the user's question is OUTSIDE your domain "
            f"or requires knowledge from another domain, include this signal "
            f"at the END of your response:\n"
            f"[DELEGATE:<domain_name>] <brief context for the other domain>\n"
            f"Example: [DELEGATE:protection] The user is asking about VM "
            f"snapshot failures during backup — need workload-specific analysis.\n"
            f"Only delegate when truly necessary — most questions should be "
            f"answered within your domain."
        )

    def _route(self, message: str) -> str:
        """Use LLM to classify which agent should handle the message."""
        try:
            # Build compact routing history: agent tags + truncated content
            # so the router sees which agent handled each turn without being
            # overwhelmed by long KQL results or doc text.
            routing_history = []

            # Lead with the last-agent context so the LLM sees it first
            if self._last_agent:
                routing_history.append({
                    "role": "system",
                    "content": (
                        f"[CONTEXT: The MOST RECENT turn was handled by the '{self._last_agent}' agent. "
                        f"If the user's new message looks like a follow-up, route to '{self._last_agent}'.]"
                    ),
                })

            # Append conversation history with truncated content for the router
            for msg in self._conversation_history:
                content = msg["content"]
                # Keep first 300 chars — enough for agent tag + gist, but
                # won't flood the router with KQL tables or full doc text
                if len(content) > 300:
                    content = content[:300] + "...(truncated)"
                routing_history.append({"role": msg["role"], "content": content})

            # Build dynamic state context so the router knows about pending ops
            state_lines = []
            updater = self._agents.get("knowledge_updater")
            if updater and updater.pending_count > 0:
                state_lines.append(
                    f"STATE: The knowledge_updater agent has {updater.pending_count} "
                    f"pending correction(s) waiting to be submitted or discarded."
                )
            if updater and updater._pending_new_doc:
                state_lines.append(
                    "STATE: The knowledge_updater agent previously offered to create a new document "
                    "and is waiting for the user to confirm (e.g. 'create doc', 'yes', 'save doc')."
                )
            state_context = "\n".join(state_lines) if state_lines else ""

            # Inject state into the system prompt
            system_prompt = ROUTER_SYSTEM_PROMPT.format(state_context=state_context)

            text = self.llm.generate(
                message=message,
                system=system_prompt,
                history=routing_history,
            ).strip()

            if text.startswith("ROUTE:"):
                agent_name = text.split(":", 1)[1].strip().lower()
                if agent_name in self._agents:
                    return agent_name

            log.warning("Router returned unexpected: '%s', defaulting to knowledge.", text)
            return "knowledge"

        except Exception as e:
            log.error("Routing error: %s, defaulting to knowledge.", e)
            return "knowledge"

    @property
    def _knowledge_confidence_threshold(self) -> float:
        """Minimum similarity for knowledge agent; below this → coder fallback."""
        return float(
            self.cfg.get("agents", {}).get("knowledge_confidence_threshold", 0.35)
        )

    @property
    def _doc_gap_threshold(self) -> float:
        """Below this confidence, flag the topic as a documentation gap."""
        return float(
            self.cfg.get("agents", {}).get("doc_gap_threshold", 0.50)
        )

    _DOC_GAPS_FILE = "doc_gaps.json"

    def _log_doc_gap(self, question: str, confidence: float) -> None:
        """Append a documentation-gap entry to the local doc_gaps.json file."""
        gap_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "confidence": round(confidence, 3),
        }
        path = os.path.join(
            self.cfg.get("paths", {}).get("docs_dir", "docs/agentKT"),
            "..",
            self._DOC_GAPS_FILE,
        )
        path = os.path.normpath(path)
        try:
            existing: list = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(gap_entry)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            log.info("Logged doc gap to %s", path)
        except Exception as e:
            log.warning("Failed to write doc gap log: %s", e)

    def chat(self, message: str) -> Dict[str, str]:
        """
        Process a user message:
        1. Route to the appropriate agent.
        2. Get the response.
        3. Return {agent, response} dict.
        """
        # Pre-route check: if knowledge_updater has pending corrections
        # and the user says "agree"/"submit"/etc., go straight there
        # (Removed: we now let the LLM router handle this via state context
        #  injected into ROUTER_SYSTEM_PROMPT.)
        agent_name = self._route(message)

        log.info("Routed to: %s", agent_name)

        # Guard: warn if leaving knowledge_updater with pending corrections
        updater = self._agents.get("knowledge_updater")
        if (
            updater
            and updater.pending_count > 0
            and agent_name != "knowledge_updater"
            and self._last_agent == "knowledge_updater"
        ):
            pending_warning = (
                f"⚠️ **You have {updater.pending_count} unsaved doc correction(s):**\n"
                f"{updater.pending_summary()}\n\n"
                f"Say **\"submit\"** to create a PR, **\"discard\"** to drop them, "
                f"or I'll continue with your new question.\n\n---\n\n"
            )
        else:
            pending_warning = ""

        agent = self._agents.get(agent_name)
        if agent is None:
            return {
                "agent": "brain",
                "response": f"Agent '{agent_name}' is not available. Available: {list(self._agents.keys())}",
            }

        # Dispatch to the right agent
        try:
            if agent_name == "knowledge":
                response_text, confidence = agent.answer_with_confidence(
                    message, self._conversation_history
                )
                threshold = self._knowledge_confidence_threshold
                gap_threshold = self._doc_gap_threshold
                log.info("Knowledge confidence: %.3f (coder threshold: %.2f, doc-gap threshold: %.2f)",
                         confidence, threshold, gap_threshold)

                # Flag documentation gap if below doc_gap_threshold
                doc_gap_notice = ""
                if confidence < gap_threshold:
                    self._log_doc_gap(message, confidence)
                    doc_gap_notice = (
                        f"\n\n---\n"
                        f"📝 **Documentation gap detected** — confidence was only "
                        f"**{confidence:.0%}** (threshold: {gap_threshold:.0%}).\n"
                        f"This topic may not be covered in the docs yet. "
                        f"Say **\"create doc\"** to draft a new document for this topic, "
                        f"or I've logged it to `doc_gaps.json` for later review."
                    )

                # Fallback: if knowledge docs are low-relevance, automatically
                # route to coder agent for a source-code-based answer.
                if confidence < threshold and "coder" in self._agents:
                    log.info(
                        "Knowledge confidence %.3f below threshold %.2f "
                        "— auto-routing to coder agent.",
                        confidence, threshold,
                    )
                    coder = self._agents["coder"]
                    coder_response = coder.analyze(message, self._conversation_history)
                    if coder_response and not coder_response.startswith("I don't have any indexed source code"):
                        agent_name = "coder"
                        response_text = coder_response
                    else:
                        # Neither docs nor code had a good answer
                        response_text = (
                            "I couldn't find relevant information in the project documentation "
                            "or the source code for this query. You could try:\n"
                            "- Rephrasing your question\n"
                            "- Checking if the docs/code have been indexed "
                            "(`python run_index.py` / `python run_code_index.py`)\n"
                            "- Asking the **debug agent** if this is a runtime issue"
                        )
                # Append the doc-gap notice to whichever response we ended up with
                if doc_gap_notice:
                    response_text += doc_gap_notice
            elif agent_name == "debug":
                response_text = agent.debug(message, self._conversation_history)
            elif agent_name == "coder":
                response_text = agent.analyze(message, self._conversation_history)
            elif agent_name == "knowledge_updater":
                response_text = agent.handle(message, self._conversation_history)
            else:
                # Generic fallback for future agents that have a `handle` method
                if hasattr(agent, "handle"):
                    response_text = agent.handle(message, self._conversation_history)
                else:
                    response_text = f"Agent '{agent_name}' does not have a handler."
        except Exception as e:
            log.error("Agent '%s' error: %s", agent_name, e, exc_info=True)
            response_text = f"Error from {agent_name} agent: {e}"

        # Prepend pending-corrections warning if switching context
        if pending_warning:
            response_text = pending_warning + response_text

        # Update conversation history (keep last 10 turns)
        # Tag assistant responses with agent name so the router knows who handled each turn
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({
            "role": "assistant",
            "content": f"[Agent: {agent_name}] {response_text}",
        })
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # Track which agent handled this turn for routing continuity
        self._last_agent = agent_name

        return {
            "agent": agent_name,
            "response": response_text,
        }

    def reset_conversation(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        self._last_agent = None
        log.info("Conversation history cleared.")
