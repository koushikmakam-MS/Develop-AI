"""
HiveRouter — top-level router for the Agent Hive Architecture.

Sits above all Hives and:
  1. Classifies the user's intent to determine which hive(s) to dispatch to.
  2. Delegates to the selected hive's BrainAgent.
  3. Detects cross-hive delegation signals ([DELEGATE:<hive>]) and
     orchestrates multi-hive collaboration.
  4. Synthesizes cross-hive results into a unified response.

The router NEVER does domain work — it only classifies, dispatches,
and synthesizes.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from brain_ai.hive.gateway import Gateway
from brain_ai.hive.hive import Hive
from brain_ai.hive.registry import HiveRegistry
from brain_ai.llm_client import LLMClient

log = logging.getLogger(__name__)

# Maximum delegation depth to prevent infinite loops
MAX_DELEGATION_DEPTH = 3

HIVE_ROUTER_SYSTEM_PROMPT = """You are the Top-Level Router Agent for a multi-domain AI assistant.

Your ONLY job is to decide which domain hive(s) should handle the user's message.
You do NOT answer questions — you only route.

Available hives:
{hive_scope_summary}

ROUTING RULES:
1. Look at the user's message and determine which domain it belongs to.
2. If the question clearly fits ONE hive, route to that hive.
3. If the question spans multiple domains, route to the PRIMARY hive
   (the one most responsible for the answer). That hive can delegate
   to other hives if needed.
4. If you can't determine the domain, route to the default hive: "{default_hive}".
5. Look at conversation history — if the user is following up on a previous
   question, route to the SAME hive that handled the last turn.

{state_context}

Respond with EXACTLY this format (no other text):
HIVE:<hive_name>

Examples:
- "How does backup policy work?" → HIVE:bms
- "What SDK utilities are available for serialization?" → HIVE:common
- "VM snapshot failed during backup" → HIVE:bms
- "How does the SQL workload plugin handle quiesce?" → HIVE:protection
"""

SYNTHESIS_PROMPT = """You are synthesizing results from multiple domain experts
into a single unified answer for the user.

The user asked: "{question}"

The primary hive ({primary_hive}) responded:
{primary_response}

The delegated hive ({delegated_hive}) provided additional context:
{delegated_response}

Synthesize these into ONE coherent, well-structured answer that:
1. Combines insights from both domains
2. Explains any cross-domain dependencies clearly
3. Uses proper markdown formatting
4. Does NOT mention "hives" or internal routing — the user sees one assistant
"""

# ── Proactive cross-hive discovery ──────────────────────────────

DISCOVERY_PROMPT = """You are an expert architect analyzing a technical response
about a BCDR (Backup & Disaster Recovery) service.

The user asked: "{question}"

The primary service ({primary_hive} — {primary_display}) answered:
---
{primary_response}
---

Other available services you can consult:
{other_hives_summary}

Your job: Identify which OTHER services are referenced or would be needed
to build a complete end-to-end understanding. For each, generate a
targeted question that will retrieve the most relevant information.

RULES:
- Only include services that are GENUINELY needed for a complete answer.
- Do NOT re-ask the same question — make each sub-question SPECIFIC to
  what that service uniquely knows (e.g. its internal implementation,
  API contracts, data flow).
- If the primary answer is fully self-contained, respond with NONE.
- Maximum 3 cross-service lookups.

Respond in EXACTLY this format (one per line, no other text):
ASK:<hive_name>|<specific question for that service>

Or if no cross-service context is needed:
NONE

Examples:
ASK:dataplane|How does the data plane handle soft delete state for backup items? What APIs or internal methods manage the soft-deleted flag?
ASK:monitoring|What alerts or monitoring events are triggered when a backup item is soft-deleted or when soft delete retention expires?
"""

MULTI_SYNTHESIS_PROMPT = """You are creating a comprehensive, unified technical
response by combining insights from multiple BCDR service teams.

The user asked: "{question}"

## Primary service: {primary_hive} ({primary_display})
{primary_response}

## Additional service inputs:
{additional_sections}

Create ONE unified, well-structured response that:
1. Starts with a high-level summary of the end-to-end flow
2. Walks through each service's role in sequence (like a real request flow)
3. Highlights cross-service interactions (API calls, events, data contracts)
4. Uses clear section headers for each service's contribution
5. Includes a "Cross-Service Flow" section showing how data moves between services
6. Uses proper markdown formatting with headers, bullet points, and code references
7. Does NOT mention "hives", "agents", or internal routing — present this as
   expert knowledge from across the BCDR platform
"""


class HiveRouter:
    """Top-level router that dispatches to domain-specific hives."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = LLMClient(cfg)

        # Build the hive registry
        self.registry = HiveRegistry(cfg)

        # Gateway agent for two-stage routing
        self.gateway = Gateway(
            registry=self.registry,
            llm=self.llm,
            default_hive=self.registry.default_hive_name,
        )

        # Conversation history for routing context
        self._conversation_history: List[Dict[str, str]] = []
        self._last_hive: Optional[str] = None

        log.info(
            "HiveRouter initialized — %d hive(s): %s",
            len(self.registry),
            self.registry.names,
        )

    # ── Public interface ────────────────────────────────────────────

    def chat(self, message: str, deep: bool = True) -> Dict[str, Any]:
        """Process a user message through the hive system.

        Parameters
        ----------
        message : str
            The user's question.
        deep : bool
            If True (default), perform proactive cross-hive discovery
            after the primary answer — i.e. the primary hive's response
            is analysed for references to other services and targeted
            sub-questions are fanned out automatically.
            Set to False for simple/follow-up questions.

        Returns:
            dict with keys: hive, agent, response
            (and optionally: delegated_to, delegation_chain, consulted_hives)
        """
        # Step 1: Route to a hive (via Gateway)
        routing = self._route_to_hive(message)
        hive_name = routing["hive"]
        hive = self.registry.get(hive_name)

        if hive is None:
            hive = self.registry.default_hive
            if hive is None:
                return {
                    "hive": "none",
                    "agent": "router",
                    "response": (
                        "No hives are available. Check your config.yaml "
                        "hives.definitions section."
                    ),
                }
            hive_name = hive.name

        log.info(
            "Routed to hive: %s (%s) via %s [confidence=%.2f]",
            hive_name, hive.display_name,
            routing.get("method", "?"), routing.get("confidence", 0),
        )

        # Step 2: Dispatch to the hive
        result = hive.chat(message)

        # Step 3: Check for explicit [DELEGATE:] signals (backward compat)
        result = self._handle_delegation(
            result, message, source_hive=hive, depth=0
        )

        # Step 4: Proactive cross-hive discovery
        if deep and len(self.registry) > 1:
            result = self._proactive_discovery(
                result, message, primary_hive=hive
            )

        # Step 5: Update conversation history
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({
            "role": "assistant",
            "content": f"[Hive: {result['hive']}] {result['response']}",
        })
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        self._last_hive = result["hive"]

        # Attach routing metadata
        result["routing"] = {
            "method": routing.get("method"),
            "confidence": routing.get("confidence"),
            "scores": routing.get("scores"),
            "matched_topics": routing.get("matched_topics"),
        }

        return result

    def reset_conversation(self):
        """Clear all conversation state across all hives."""
        self._conversation_history.clear()
        self._last_hive = None
        for hive in self.registry:
            hive.reset()
        log.info("All hive conversations cleared.")

    # ── Routing ─────────────────────────────────────────────────────

    def _route_to_hive(self, message: str) -> Dict[str, Any]:
        """Use the Gateway agent for two-stage routing.

        Returns a dict with hive name plus routing metadata
        (confidence, method, scores, matched_topics).
        """
        # If only one hive, skip routing
        if len(self.registry) == 1:
            return {
                "hive": self.registry.names[0],
                "confidence": 1.0,
                "method": "single_hive",
                "scores": {},
                "matched_topics": {},
            }

        return self.gateway.route(
            message=message,
            last_hive=self._last_hive,
            conversation_history=self._conversation_history,
        )

    # ── Proactive cross-hive discovery ───────────────────────────────

    def _proactive_discovery(
        self,
        result: Dict[str, Any],
        question: str,
        primary_hive: Hive,
    ) -> Dict[str, Any]:
        """Analyze the primary response for cross-service references and
        proactively consult other hives to build a complete picture.

        Flow:
          1. Ask LLM to identify which other services are referenced
          2. Fan out targeted sub-questions to those hives
          3. Synthesize everything into a unified response
        """
        # Step 1: Discover cross-hive references
        sub_questions = self._discover_cross_hive_refs(
            question=question,
            primary_hive=primary_hive,
            primary_response=result.get("response", ""),
        )

        if not sub_questions:
            log.info("No cross-hive references found — primary answer is self-contained.")
            return result

        log.info(
            "Cross-hive discovery: %d service(s) to consult: %s",
            len(sub_questions),
            [hive for hive, _ in sub_questions],
        )

        # Step 2: Fan out sub-questions
        additional_responses = self._fan_out(sub_questions)

        if not additional_responses:
            return result

        # Step 3: Multi-hive synthesis
        synthesized = self._multi_synthesize(
            question=question,
            primary_hive=primary_hive,
            primary_response=result.get("response", ""),
            additional_responses=additional_responses,
        )

        consulted = [hive_name for hive_name, _, _ in additional_responses]
        return {
            "hive": primary_hive.name,
            "agent": result.get("agent", "unknown"),
            "response": synthesized,
            "consulted_hives": consulted,
            "delegation_chain": result.get("delegation_chain", []),
        }

    def _discover_cross_hive_refs(
        self,
        question: str,
        primary_hive: Hive,
        primary_response: str,
    ) -> List[Tuple[str, str]]:
        """Use LLM to analyze the primary response and identify which
        other hives should be consulted, with targeted sub-questions.

        Returns list of (hive_name, sub_question) tuples.
        """
        # Build summary of other available hives
        other_lines = []
        for hive in self.registry:
            if hive.name == primary_hive.name:
                continue
            topics = ", ".join(hive.topics[:6])
            other_lines.append(
                f"- **{hive.name}** ({hive.display_name}): {hive.description} "
                f"[topics: {topics}]"
            )

        if not other_lines:
            return []

        prompt = DISCOVERY_PROMPT.format(
            question=question,
            primary_hive=primary_hive.name,
            primary_display=primary_hive.display_name,
            primary_response=primary_response[:3000],
            other_hives_summary="\n".join(other_lines),
        )

        try:
            text = self.llm.generate(
                message=prompt,
                system="Analyze cross-service dependencies. Be precise and selective.",
                history=[],
            ).strip()

            if text.upper().startswith("NONE"):
                return []

            sub_questions: List[Tuple[str, str]] = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("ASK:"):
                    parts = line[4:].split("|", 1)
                    if len(parts) == 2:
                        hive_name = parts[0].strip().lower()
                        sub_q = parts[1].strip()
                        if hive_name in self.registry and hive_name != primary_hive.name:
                            sub_questions.append((hive_name, sub_q))

            return sub_questions[:3]  # Cap at 3

        except Exception as e:
            log.error("Cross-hive discovery failed: %s", e)
            return []

    def _fan_out(
        self, sub_questions: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Dispatch sub-questions to target hives and collect responses.

        Returns list of (hive_name, sub_question, response) tuples.
        """
        results: List[Tuple[str, str, str]] = []

        for hive_name, sub_question in sub_questions:
            hive = self.registry.get(hive_name)
            if hive is None:
                continue

            log.info(
                "  Fan-out → %s: %s",
                hive_name,
                sub_question[:80] + ("..." if len(sub_question) > 80 else ""),
            )

            try:
                hive_result = hive.chat(sub_question)
                response = hive_result.get("response", "")
                if response:
                    results.append((hive_name, sub_question, response))
                # Reset so this sub-question doesn't pollute the hive's conversation
                hive.reset()
            except Exception as e:
                log.error("Fan-out to %s failed: %s", hive_name, e)

        return results

    def _multi_synthesize(
        self,
        question: str,
        primary_hive: Hive,
        primary_response: str,
        additional_responses: List[Tuple[str, str, str]],
    ) -> str:
        """Synthesize the primary response with multiple cross-hive inputs."""
        # Build additional sections
        sections = []
        for hive_name, sub_q, response in additional_responses:
            target_hive = self.registry.get(hive_name)
            display = target_hive.display_name if target_hive else hive_name
            sections.append(
                f"### {display} ({hive_name})\n"
                f"*Question asked:* {sub_q}\n\n"
                f"{response[:3000]}"
            )

        prompt = MULTI_SYNTHESIS_PROMPT.format(
            question=question,
            primary_hive=primary_hive.name,
            primary_display=primary_hive.display_name,
            primary_response=primary_response[:3000],
            additional_sections="\n\n".join(sections),
        )

        try:
            return self.llm.generate(
                message=prompt,
                system=(
                    "Create a comprehensive cross-service technical response. "
                    "Show the end-to-end flow across services clearly."
                ),
                history=[],
            )
        except Exception as e:
            log.error("Multi-synthesis failed: %s — returning primary response.", e)
            return primary_response

    # ── Explicit cross-hive delegation (backward compat) ────────────

    def _handle_delegation(
        self,
        result: Dict[str, Any],
        original_question: str,
        source_hive: Hive,
        depth: int,
    ) -> Dict[str, Any]:
        """Check if the response contains a delegation signal and handle it.

        The protocol:
          Agent response contains ``[DELEGATE:<hive_name>] <context>``
          → Router dispatches context to the target hive
          → Synthesizes both responses into one unified answer
        """
        if depth >= MAX_DELEGATION_DEPTH:
            log.warning(
                "Max delegation depth (%d) reached, stopping.", MAX_DELEGATION_DEPTH
            )
            return result

        delegation = Hive.extract_delegation(result.get("response", ""))
        if delegation is None:
            return result

        target_name = delegation["target_hive"]
        delegation_context = delegation["context"]

        target_hive = self.registry.get(target_name)
        if target_hive is None:
            log.warning(
                "Agent requested delegation to unknown hive '%s', ignoring.",
                target_name,
            )
            # Strip the delegation signal from the response
            response = result["response"]
            idx = response.find(Hive.DELEGATE_PREFIX)
            if idx > 0:
                result["response"] = response[:idx].rstrip()
            return result

        log.info(
            "Cross-hive delegation: %s → %s (depth=%d)",
            source_hive.name,
            target_name,
            depth + 1,
        )

        # Dispatch to target hive with the delegation context
        delegated_result = target_hive.chat(delegation_context)

        # Recursively check if the delegated hive also delegates
        delegated_result = self._handle_delegation(
            delegated_result, original_question, target_hive, depth + 1
        )

        # Synthesize the two responses
        synthesized = self._synthesize(
            question=original_question,
            primary_hive=source_hive.name,
            primary_response=result["response"],
            delegated_hive=target_name,
            delegated_response=delegated_result["response"],
        )

        # Build the combined result
        chain = result.get("delegation_chain", [])
        chain.append({
            "from": source_hive.name,
            "to": target_name,
            "context": delegation_context[:200],
        })

        return {
            "hive": source_hive.name,
            "agent": result.get("agent", "unknown"),
            "response": synthesized,
            "delegated_to": target_name,
            "delegation_chain": chain,
        }

    def _synthesize(
        self,
        question: str,
        primary_hive: str,
        primary_response: str,
        delegated_hive: str,
        delegated_response: str,
    ) -> str:
        """Use LLM to synthesize cross-hive responses into one answer."""
        try:
            # Strip the delegation signal from the primary response
            idx = primary_response.find(Hive.DELEGATE_PREFIX)
            if idx > 0:
                primary_clean = primary_response[:idx].rstrip()
            else:
                primary_clean = primary_response

            prompt = SYNTHESIS_PROMPT.format(
                question=question,
                primary_hive=primary_hive,
                primary_response=primary_clean[:3000],
                delegated_hive=delegated_hive,
                delegated_response=delegated_response[:3000],
            )

            return self.llm.generate(
                message=prompt,
                system="Synthesize multi-domain answers clearly and concisely.",
                history=[],
            )
        except Exception as e:
            log.error("Synthesis failed: %s — returning primary response.", e)
            idx = primary_response.find(Hive.DELEGATE_PREFIX)
            if idx > 0:
                return primary_response[:idx].rstrip()
            return primary_response

    # ── Convenience accessors ───────────────────────────────────────

    @property
    def active_hive(self) -> Optional[Hive]:
        """The hive that handled the last turn."""
        if self._last_hive:
            return self.registry.get(self._last_hive)
        return self.registry.default_hive

    def get_hive_agents(self, hive_name: str) -> List[str]:
        """List agent names available in a given hive."""
        hive = self.registry.get(hive_name)
        if hive:
            return list(hive.brain._agents.keys())
        return []

    def __repr__(self) -> str:
        return (
            f"HiveRouter(hives={self.registry.names}, "
            f"last_hive={self._last_hive!r})"
        )
