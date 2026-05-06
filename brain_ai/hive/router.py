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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DISCOVERY_PROMPT = """You are an expert architect deciding which other services
must be consulted to properly answer the user's question.

The user asked: "{question}"

The primary service ({primary_hive} — {primary_display}) answered:
---
{primary_response}
---

Other available services:
{other_hives_summary}

Consult another service ONLY when at least one of these is true:

1. **Explicit dependency**: The primary response names another service,
   calls its API, or describes a hand-off/data flow TO that service.
2. **Essential ownership**: The other service OWNS a key part of the
   workflow the user asked about (e.g. for "data mobility", a data-mover
   service owns the actual transfer logic; for "cross-region restore",
   a regional service owns the cross-region coordination).

Do NOT consult a service when:
- It only shares a loosely related concept (e.g. don't pull in a
  "protection" service just because the question involves backup).
- The primary response already adequately covers the concept.
- The service's knowledge would be redundant with the primary answer.

RULES:
- Make each sub-question SPECIFIC to what ONLY that service uniquely knows.
- Maximum 2 cross-service lookups.

Respond in EXACTLY this format (one per line, no other text):
ASK:<hive_name>|<specific question for that service>

Or if no cross-service context is needed:
NONE
"""

MULTI_SYNTHESIS_PROMPT = """You are creating a focused, unified technical
response for the user's question.

The user asked: "{question}"

## Primary service: {primary_hive} ({primary_display})
{primary_response}

## Additional service inputs:
{additional_sections}

Create ONE focused response that:
1. Leads with the primary service's answer — this is the core content
2. Weaves in additional service details ONLY where they directly clarify
   or extend the primary answer (drop tangential info)
3. Keeps the response concise — do NOT pad with loosely related details
4. Uses proper markdown formatting with headers, bullet points, code refs
5. Does NOT mention "hives", "agents", or internal routing

CRITICAL ANTI-FABRICATION RULES:
- **Do NOT invent code, class names, method names, file paths, REST
  endpoints, route attributes, or API contracts.** Only include such
  details if they appear verbatim in the provided service inputs.
- If a service's input doesn't mention a specific API or class, do NOT
  add one to make the response "look complete". Omit instead.
- Quote file paths, class names, and code exactly as provided.

IMPORTANT: If an additional service's input is not directly relevant to
the user's specific question, OMIT it entirely rather than including it.
"""


class HiveRouter:
    """Top-level router that dispatches to domain-specific hives."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = LLMClient(cfg)

        # Build the hive registry
        self.registry = HiveRegistry(cfg)

        # Gateway agent for two-stage routing
        hives_cfg = cfg.get("hives", {})
        self.gateway = Gateway(
            registry=self.registry,
            llm=self.llm,
            default_hive=self.registry.default_hive_name,
            primary_hives=hives_cfg.get("primary_hives"),
        )

        # Conversation history for routing context
        self._conversation_history: List[Dict[str, str]] = []
        self._last_hive: Optional[str] = None

        # Wire cross-hive callback into every hive so agents can [ASK:]
        for hive in self.registry:
            hive.set_cross_hive_callback(self._resolve_cross_hive_ask)

        # Build boundary map: pattern → hive_name (for code-boundary detection)
        boundary_map = self._build_boundary_map()
        if boundary_map:
            for hive in self.registry:
                hive.set_boundary_config(boundary_map, self._resolve_cross_hive_ask)
            log.info(
                "Code boundary map: %d patterns across %d hive(s)",
                len(boundary_map),
                len({v for v in boundary_map.values()}),
            )

        log.info(
            "HiveRouter initialized — %d hive(s): %s",
            len(self.registry),
            self.registry.names,
        )

    # ── Public interface ────────────────────────────────────────────

    # Cheap relevance check — runs once per hive in parallel
    _RELEVANCE_PROMPT = (
        "You are the '{hive_name}' service ({display_name}).\n"
        "Your scope: {description}\n"
        "Your topics: {topics}\n\n"
        "User's question: \"{question}\"\n\n"
        "Is this question RELEVANT to your scope? Answer in EXACTLY this format:\n"
        "DECISION: YES|NO\n"
        "REASON: <one short sentence>\n\n"
        "Answer YES only if your service owns part of the answer. Be strict — say NO\n"
        "if the question is mostly about another service's domain."
    )

    ALL_HIVES_SYNTHESIS_PROMPT = (
        "You are creating a unified technical response from multiple service experts.\n\n"
        "User's question: \"{question}\"\n\n"
        "## Service responses (each grounded in their own indexed code/docs):\n\n"
        "{sections}\n\n"
        "Create ONE focused response that:\n"
        "1. Combines insights from the services that have substantive content\n"
        "2. Drops/skips any service input that says \"no relevant code/docs found\"\n"
        "   or appears to be filler — do NOT pad the answer with their content\n"
        "3. Uses clear section headers when multiple services contribute\n"
        "4. Uses proper markdown with code blocks, bullet points, and references\n"
        "5. Does NOT mention \"hives\", \"agents\", or internal routing\n\n"
        "CRITICAL ANTI-FABRICATION RULES:\n"
        "- Do NOT invent code, class names, method names, file paths, REST\n"
        "  endpoints, or API contracts. Only include details that appear\n"
        "  verbatim in the provided service responses.\n"
        "- If no service has concrete code/docs for part of the question, say\n"
        "  so explicitly rather than fabricating.\n"
        "- Quote file paths, class names, and code exactly as provided."
    )

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

    def chat_all_hives(self, message: str) -> Dict[str, Any]:
        """Hybrid mode: cheap relevance pre-filter on ALL hives, then deep
        pass on YES hives only, then synthesize.

        Flow:
          1. Parallel YES/NO relevance check on every hive (~9 small calls)
          2. Deep chat() on hives that said YES (parallel)
          3. Synthesize the multi-hive answers with strict anti-fab rules

        Returns the same shape as chat() with extra keys:
          - relevance_decisions: list of {hive, decision, reason}
          - consulted_hives: list of hive names that contributed
        """
        if len(self.registry) == 0:
            return {
                "hive": "none",
                "agent": "router",
                "response": "No hives are available.",
            }

        # ── Step 1: parallel relevance pre-filter ────────────────────
        log.info("chat_all_hives: running relevance check on %d hive(s)", len(self.registry))
        relevance = self._relevance_filter(message)

        yes_hives = [r for r in relevance if r["decision"] == "YES"]
        log.info(
            "Relevance filter: %d/%d hive(s) said YES: %s",
            len(yes_hives), len(relevance),
            [r["hive"] for r in yes_hives],
        )

        if not yes_hives:
            # Fallback to normal routing if nobody claims relevance
            log.info("No hives marked YES — falling back to normal chat()")
            result = self.chat(message)
            result["relevance_decisions"] = relevance
            return result

        # ── Step 2: parallel deep chat on YES hives ──────────────────
        deep_results = self._fan_out_to_hives([r["hive"] for r in yes_hives], message)

        if not deep_results:
            return {
                "hive": "none",
                "agent": "router",
                "response": "All relevant hives failed to respond.",
                "relevance_decisions": relevance,
            }

        # ── Step 3: synthesize ──────────────────────────────────────
        # If only one YES hive responded, just return its result directly
        if len(deep_results) == 1:
            hive_name, hive_result = deep_results[0]
            hive_result["relevance_decisions"] = relevance
            hive_result["consulted_hives"] = [hive_name]
            hive_result["mode"] = "all_hives"
            return hive_result

        synthesized = self._synthesize_all_hives(message, deep_results)

        primary_name, primary_result = deep_results[0]
        consulted = [name for name, _ in deep_results]

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({
            "role": "assistant",
            "content": f"[All-hives mode: {consulted}] {synthesized[:500]}",
        })
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return {
            "hive": primary_name,
            "agent": primary_result.get("agent", "multi"),
            "response": synthesized,
            "relevance_decisions": relevance,
            "consulted_hives": consulted,
            "consulted_details": [
                {"hive": name, "question": message}
                for name, _ in deep_results
            ],
            "mode": "all_hives",
        }

    def _relevance_filter(self, message: str) -> List[Dict[str, str]]:
        """Run a cheap parallel YES/NO relevance check on every hive."""
        hives = list(self.registry)
        results: List[Dict[str, str]] = []

        def check_one(hive: Hive) -> Dict[str, str]:
            prompt = self._RELEVANCE_PROMPT.format(
                hive_name=hive.name,
                display_name=hive.display_name,
                description=hive.description[:300],
                topics=", ".join(hive.topics[:15]),
                question=message,
            )
            try:
                text = self.llm.generate(
                    message=prompt,
                    system="You are a strict relevance classifier. Answer YES only if your service owns part of the answer.",
                    history=[],
                ).strip()
                # Parse DECISION + REASON
                decision = "NO"
                reason = ""
                for line in text.splitlines():
                    line = line.strip()
                    if line.upper().startswith("DECISION:"):
                        val = line.split(":", 1)[1].strip().upper()
                        if val.startswith("YES"):
                            decision = "YES"
                        elif val.startswith("NO"):
                            decision = "NO"
                    elif line.upper().startswith("REASON:"):
                        reason = line.split(":", 1)[1].strip()
                return {"hive": hive.name, "decision": decision, "reason": reason}
            except Exception as e:
                log.warning("Relevance check failed for %s: %s", hive.name, e)
                return {"hive": hive.name, "decision": "NO", "reason": f"check failed: {e}"}

        # Run in parallel — bounded thread pool
        max_workers = min(len(hives), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(check_one, h): h.name for h in hives}
            for fut in as_completed(futures):
                results.append(fut.result())

        # Stable order by hive name
        results.sort(key=lambda r: r["hive"])
        for r in results:
            log.info("  [%s] %s — %s", r["hive"], r["decision"], r["reason"][:80])
        return results

    def _fan_out_to_hives(
        self, hive_names: List[str], message: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Run deep chat() on each hive in parallel. Returns list of
        (hive_name, result_dict) for hives that produced a response."""
        results: List[Tuple[str, Dict[str, Any]]] = []

        def run_one(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
            hive = self.registry.get(name)
            if hive is None:
                return None
            try:
                result = hive.chat(message)
                # Reset so this isolated turn doesn't pollute hive history
                hive.reset()
                if result.get("response"):
                    return (name, result)
            except Exception as e:
                log.error("Deep chat failed for %s: %s", name, e)
            return None

        max_workers = min(len(hive_names), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_one, n): n for n in hive_names}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)

        # Order: keep stable by original hive order in registry
        order = {h.name: i for i, h in enumerate(self.registry)}
        results.sort(key=lambda x: order.get(x[0], 999))
        return results

    def _synthesize_all_hives(
        self,
        question: str,
        deep_results: List[Tuple[str, Dict[str, Any]]],
    ) -> str:
        """Synthesize multiple hive responses into a single answer."""
        sections = []
        for name, result in deep_results:
            hive = self.registry.get(name)
            display = hive.display_name if hive else name
            response = result.get("response", "")[:4000]
            sections.append(f"### {display} ({name})\n{response}")

        prompt = self.ALL_HIVES_SYNTHESIS_PROMPT.format(
            question=question,
            sections="\n\n".join(sections),
        )

        try:
            return self.llm.generate(
                message=prompt,
                system=(
                    "Create a focused, evidence-grounded response. "
                    "Drop sections with no real content. Never fabricate code."
                ),
                history=[],
            )
        except Exception as e:
            log.error("All-hives synthesis failed: %s — concatenating responses.", e)
            return "\n\n".join(s for _, s in [(n, r.get("response", "")) for n, r in deep_results] if s)

    # ── Clarifier ───────────────────────────────────────────────────

    _CLARIFIER_PROMPT = (
        "You are a clarifier. Decide whether the user's question is specific\n"
        "enough to give a focused, accurate answer, or whether you should ask\n"
        "1-2 short clarifying questions first.\n\n"
        "Available service domains: {hives_summary}\n\n"
        "User's question: \"{question}\"\n\n"
        "Recent conversation context:\n{history}\n\n"
        "Decide:\n"
        "- If the question is specific enough (names a concrete feature, error,\n"
        "  TaskId, file, or follows up on prior context), respond: CLEAR\n"
        "- If it's broad/ambiguous and could mean multiple things across the\n"
        "  available domains, respond with 1-2 SHORT clarifying questions.\n\n"
        "Format:\n"
        "CLEAR\n"
        "  -- OR --\n"
        "ASK:\n"
        "1. <first clarifying question>\n"
        "2. <optional second question>\n\n"
        "Keep clarifying questions short (one line each) and concrete. Offer\n"
        "specific options when possible (e.g. \"Do you mean A, B, or C?\")."
    )

    def clarify_question(self, message: str) -> Dict[str, Any]:
        """Decide if the user's question needs clarification.

        Returns a dict:
          {"status": "clear"}  → proceed
          {"status": "ask", "questions": ["...", "..."]}  → ask user, then merge
        """
        # Build short hive summary for context
        hive_lines = []
        for h in list(self.registry)[:12]:
            hive_lines.append(f"- {h.name}: {h.display_name}")
        hives_summary = "\n".join(hive_lines)

        # Recent history (last 4 messages)
        hist_lines = []
        for turn in self._conversation_history[-4:]:
            role = turn.get("role", "?")
            content = turn.get("content", "")[:200]
            hist_lines.append(f"{role}: {content}")
        history = "\n".join(hist_lines) if hist_lines else "(none)"

        prompt = self._CLARIFIER_PROMPT.format(
            question=message,
            hives_summary=hives_summary,
            history=history,
        )

        try:
            text = self.llm.generate(
                message=prompt,
                system="Be a strict clarifier. Default to CLEAR unless genuinely ambiguous.",
                history=[],
            ).strip()
        except Exception as e:
            log.warning("Clarifier failed: %s — proceeding without clarification.", e)
            return {"status": "clear"}

        if text.upper().startswith("CLEAR"):
            return {"status": "clear"}

        # Parse "ASK:\n1. ...\n2. ..."
        questions: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Match "1. ..." or "2. ..." or "- ..."
            m = re.match(r"^(?:\d+\.|-)\s*(.+)$", line)
            if m:
                q = m.group(1).strip()
                if q and not q.upper().startswith("ASK"):
                    questions.append(q)

        if not questions:
            return {"status": "clear"}

        return {"status": "ask", "questions": questions[:2]}

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

    # ── Cross-hive callback (for [ASK:] protocol) ──────────────────

    def _build_boundary_map(self) -> Dict[str, str]:
        """Build a mapping of namespace/pattern → hive name for boundary detection.

        First tries the discovery store (auto-extracted namespaces from code indexing).
        Falls back to ``scope.boundary_patterns`` from config if the store is empty.
        """
        # Try auto-detected namespaces from discovery store
        try:
            from brain_ai.hive.discovery_store import DiscoveryStore
            ds = DiscoveryStore()
            ns_map = ds.get_namespace_map()
            ds.close()
            if ns_map:
                log.info("Boundary map: %d namespaces from discovery store", len(ns_map))
                return ns_map
        except Exception as e:
            log.debug("Could not read discovery store for namespaces: %s", e)

        # Fallback: config-based boundary_patterns
        boundary_map: Dict[str, str] = {}
        hive_defs = self.cfg.get("hives", {}).get("definitions", {})
        for hive_name, hive_cfg in hive_defs.items():
            patterns = hive_cfg.get("scope", {}).get("boundary_patterns", [])
            for pattern in patterns:
                boundary_map[pattern] = hive_name
        return boundary_map

    def _resolve_cross_hive_ask(
        self, target_hive: str, question: str
    ) -> Dict[str, Any]:
        """Resolve a cross-hive callback from an agent.

        Used for both [ASK:] protocol and boundary resolution.
        Calls the target hive's coder agent directly (no routing, no
        further boundary detection) to prevent infinite recursion.
        """
        hive = self.registry.get(target_hive)
        if hive is None:
            log.warning("[ASK:%s] — unknown hive, ignoring.", target_hive)
            return {
                "hive": target_hive,
                "agent": "none",
                "response": f"Domain '{target_hive}' is not available.",
            }

        log.info("[ASK:%s] resolving: %s", target_hive, question[:80])

        # Call the target hive's coder directly to avoid recursion.
        # Use analyze_simple() — no boundary detection, no callbacks.
        coder = hive.brain._agents.get("coder")
        if coder:
            try:
                response = coder.analyze_simple(question, [])
                return {"hive": target_hive, "agent": "coder", "response": response}
            except Exception as e:
                log.warning("[ASK:%s] coder failed: %s, falling back to knowledge", target_hive, e)

        # Fallback to knowledge agent directly
        knowledge = hive.brain._agents.get("knowledge")
        if knowledge:
            try:
                response, _ = knowledge.answer_with_confidence(question, [])
                return {"hive": target_hive, "agent": "knowledge", "response": response}
            except Exception as e:
                log.warning("[ASK:%s] knowledge failed: %s", target_hive, e)

        return {"hive": target_hive, "agent": "none", "response": "No agents available."}

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

        consulted = [
            {"hive": hive_name, "question": sub_q}
            for hive_name, sub_q, _ in additional_responses
        ]
        return {
            "hive": primary_hive.name,
            "agent": result.get("agent", "unknown"),
            "response": synthesized,
            "consulted_hives": [c["hive"] for c in consulted],
            "consulted_details": consulted,
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
                system="Only identify cross-service lookups when the primary answer is genuinely incomplete. Default to NONE.",
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

            return sub_questions[:2]  # Cap at 2

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
