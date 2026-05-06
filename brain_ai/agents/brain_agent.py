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

ROUTER_SYSTEM_PROMPT = """You are a message classifier for an Azure Backup technical assistant.

Classify the user's message into one category based on intent.

Categories:
- "knowledge": Questions about architecture, features, documentation, design, configuration.
- "debug": Troubleshooting errors, Kusto queries, log analysis, diagnosing failures.
- "coder": Questions about source code, call chains, class/method implementations.
- "knowledge_updater": Correcting docs, creating docs, submitting doc changes, confirming pending edits.

Rules:
1. If conversation history shows a previous agent handled the last turn, route follow-ups to the same agent.
2. Short replies (time ranges, confirmations, GUIDs) are follow-ups to the active agent.
3. Route to "coder" only for explicit source code questions.
4. Route to "knowledge_updater" for doc corrections, submissions, or confirmations of pending changes.

{state_context}

Respond with EXACTLY one line:
ROUTE:knowledge
ROUTE:debug
ROUTE:coder
ROUTE:knowledge_updater
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

        # Wire coder into debug agent for code enrichment
        if "debug" in self._agents and "coder" in self._agents:
            self._agents["debug"].set_coder(self._agents["coder"])

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

    def set_boundary_config(
        self,
        boundary_map: Dict[str, str],
        boundary_callback,
    ):
        """Wire code-boundary detection into the coder agent.

        Parameters
        ----------
        boundary_map : dict
            Maps boundary patterns (namespace prefixes, class names) to hive names.
        boundary_callback : callable(target_hive, question) -> dict
            Called when the coder detects a cross-hive code reference.
        """
        coder = self._agents.get("coder")
        if coder is None:
            return
        coder.set_boundary_map(boundary_map, self.hive_name or "unknown")
        coder.set_boundary_callback(boundary_callback)
        log.info(
            "BrainAgent boundary config set: %d patterns",
            len(boundary_map),
        )

    def _build_delegation_hint(self) -> str:
        """Build delegation + callback instructions for sub-agent system prompts.

        If this BrainAgent is part of a hive and peers exist, agents
        are taught two cross-domain protocols:

        1. ``[ASK:<hive>] <question>`` — **callback**: ask another domain
           a question and get the answer back so you can write ONE cohesive
           response.  Use this when you NEED information from another domain
           to complete your answer.

        2. ``[DELEGATE:<hive>] <context>`` — **hand-off**: transfer the
           entire question to another domain (you stop, they take over).
           Use this only when the question is completely outside your domain.
        """
        if not self._peer_hive_names:
            return ""
        peers = ", ".join(self._peer_hive_names)
        return (
            f"\n\nCROSS-DOMAIN COLLABORATION:\n"
            f"You are part of the '{self.hive_name}' domain. "
            f"Other domains available: [{peers}].\n\n"
            f"OPTION 1 — ASK (preferred, gets you the answer inline):\n"
            f"If you need specific information from another domain to enrich "
            f"your answer, emit this signal INLINE in your response:\n"
            f"[ASK:<domain_name>] <specific question for that domain>\n"
            f"Example: [ASK:dataplane] How does the data plane handle "
            f"incremental snapshots for DPP backup instances?\n"
            f"The system will fetch the answer and let you rewrite your "
            f"response with the combined knowledge. Max 2 [ASK:] signals per response.\n\n"
            f"OPTION 2 — DELEGATE (full hand-off, you stop):\n"
            f"If the question is ENTIRELY outside your domain, include "
            f"this at the END of your response:\n"
            f"[DELEGATE:<domain_name>] <brief context for the other domain>\n"
            f"Example: [DELEGATE:protection] The user is asking about VM "
            f"snapshot failures during backup — need workload-specific analysis.\n\n"
            f"RULES:\n"
            f"- Prefer [ASK:] when you can still provide useful context yourself.\n"
            f"- Use [DELEGATE:] only when the question is 100% another domain's territory.\n"
            f"- Most questions should be answered within your domain without either signal."
        )

    def _route(self, message: str) -> str:
        """Use LLM to classify which agent should handle the message."""
        try:
            # Build minimal routing history — strip agent tags and keep only
            # short summaries to avoid triggering content filters
            routing_history = []
            for msg in self._conversation_history[-6:]:  # Last 3 turns max
                content = msg["content"]
                # Strip [Agent: ...] prefix
                if content.startswith("[Agent:"):
                    content = content.split("]", 1)[-1].strip()
                # Truncate heavily for the router
                if len(content) > 150:
                    content = content[:150] + "..."
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

            # Add last-agent context to system prompt (not as mid-conversation system message)
            if self._last_agent:
                state_context += f"\nLast turn was handled by: {self._last_agent}. Follow-ups should route there."

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

    # ── Doc + Code synthesis ──────────────────────────────────────

    def _synthesize_doc_and_code(
        self, question: str, doc_response: str, code_response: str,
    ) -> str:
        """Combine documentation and code analysis into one deep response."""
        prompt = (
            f"You have TWO expert analyses of the same question.\n\n"
            f"## Question\n{question}\n\n"
            f"## Documentation Analysis\n{doc_response[:4000]}\n\n"
            f"## Source Code Analysis\n{code_response[:4000]}\n\n"
            f"Create ONE comprehensive response that:\n"
            f"1. Leads with the high-level architecture/concepts from the docs\n"
            f"2. Adds concrete implementation details from the code (class names, methods, data flows)\n"
            f"3. Highlights API details ONLY if they appear in the source analysis: "
            f"REST endpoints, HTTP methods, controller actions, request/response "
            f"contracts, and route patterns. **Do NOT invent endpoints, routes, "
            f"class names, or method signatures.** If a detail isn't in the "
            f"provided analyses, omit it rather than guessing.\n"
            f"4. Highlights anything the code reveals that the docs don't cover\n"
            f"5. Includes a Mermaid diagram if the answer involves a flow or architecture\n"
            f"6. References both doc sources and code file paths exactly as they appear\n\n"
            f"Do NOT mention that two analyses were combined — present it as one unified answer."
        )
        try:
            return self.llm.generate(
                message=prompt,
                system="You are a senior engineer synthesizing documentation and source code into a deep technical explanation.",
                history=[],
            )
        except Exception as e:
            log.warning("Doc+code synthesis failed: %s — returning doc response.", e)
            return doc_response

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
        extra_meta: Dict[str, Any] = {}
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

                # Always combine doc + code for deeper analysis
                if "coder" in self._agents:
                    log.info(
                        "Enriching with code context (knowledge conf=%.3f).",
                        confidence,
                    )
                    try:
                        coder = self._agents["coder"]
                        coder_response, boundaries = coder.analyze_with_boundaries(
                            message, self._conversation_history
                        )
                        if boundaries:
                            extra_meta["code_boundaries"] = boundaries
                        has_code = (
                            coder_response
                            and not coder_response.startswith("I don't have any indexed source code")
                        )

                        if has_code and confidence >= threshold:
                            # Good docs + code → synthesize both
                            response_text = self._synthesize_doc_and_code(
                                message, response_text, coder_response,
                            )
                            agent_name = "knowledge+coder"
                        elif has_code and confidence < threshold:
                            # Weak docs + good code → synthesize but lead with code
                            response_text = self._synthesize_doc_and_code(
                                message, response_text, coder_response,
                            )
                            agent_name = "knowledge+coder"
                        elif not has_code and confidence < threshold:
                            # Weak docs + no code
                            response_text = (
                                "I couldn't find strong matches in the project documentation "
                                "or source code for this query. You could try:\n"
                                "- Rephrasing your question\n"
                                "- Checking if the docs/code have been indexed "
                                "(`python run_index.py` / `python run_code_index.py`)\n"
                                "- Asking the **debug agent** if this is a runtime issue"
                            )
                        # else: good docs, no code → keep knowledge response as-is
                    except Exception as e:
                        log.warning("Code enrichment failed (non-fatal): %s", e)
                # Append the doc-gap notice to whichever response we ended up with
                if doc_gap_notice:
                    response_text += doc_gap_notice
            elif agent_name == "debug":
                response_text = agent.debug(message, self._conversation_history)
            elif agent_name == "coder":
                response_text, boundaries = agent.analyze_with_boundaries(
                    message, self._conversation_history
                )
                if boundaries:
                    agent_name = "coder"
                    # Attach boundary metadata so CLI can display them
                    extra_meta["code_boundaries"] = boundaries
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

        result = {
            "agent": agent_name,
            "response": response_text,
        }
        result.update(extra_meta)
        return result

    def reset_conversation(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        self._last_agent = None
        log.info("Conversation history cleared.")
