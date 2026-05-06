"""
Hive — a self-contained domain agent cluster.

Each Hive owns:
  - A BrainAgent (domain-local router + sub-agents)
  - Its own config scope (docs_dir, vector collections, code index)
  - A scope definition (topics it handles, Kusto tables it can query)

The Hive merges its per-hive config overrides on top of the global config
so that agents inside see a domain-scoped view of the world.
"""

import copy
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from brain_ai.agents.brain_agent import BrainAgent

log = logging.getLogger(__name__)

# Max [ASK:] round-trips per chat() call to prevent runaway loops
MAX_ASK_ROUNDS = 3


class Hive:
    """A self-contained domain agent cluster."""

    # ── Cross-hive delegation signal ────────────────────────────────
    # Agents embed this in their response when they detect the answer
    # requires another domain.  The HiveRouter intercepts it.
    DELEGATE_PREFIX = "[DELEGATE:"
    DELEGATE_SUFFIX = "]"

    # ── Cross-hive callback signal ──────────────────────────────────
    # Agents emit [ASK:<hive>] <question> mid-response to query another
    # hive synchronously.  The Hive intercepts it, resolves it via the
    # Gateway callback, and re-feeds the answer so the agent can
    # produce a single cohesive response.
    ASK_PATTERN = re.compile(
        r'\[ASK:([\w]+)\]\s*(.+?)(?=\[ASK:|\[DELEGATE:|$)',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, name: str, hive_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        """
        Parameters
        ----------
        name : str
            Hive identifier (e.g. "bms", "common", "protection").
        hive_cfg : dict
            The hive-specific config block from ``config.yaml → hives.definitions.<name>``.
        global_cfg : dict
            The full top-level config.  Hive-level overrides are merged on top.
        """
        self.name = name
        self.display_name: str = hive_cfg.get("display_name", name)
        self.description: str = hive_cfg.get("description", "")
        self.scope: Dict[str, Any] = hive_cfg.get("scope", {})

        # Topics this hive handles (used by router for classification)
        # Merge: config.yaml topics (base) + auto-extracted topics from discovery store
        self.topics: List[str] = self._load_topics(self.scope.get("topics", []))
        self.kusto_tables: List[str] = self.scope.get("kusto_tables", [])
        self.anchor_terms: List[str] = [
            t.lower() for t in self.scope.get("anchor_terms", [])
        ]

        # Build a merged config: global config + hive overrides
        self._merged_cfg = self._build_merged_config(global_cfg, hive_cfg)

        # Collect peer hive names for delegation
        peer_names = list(global_cfg.get("hives", {}).get("definitions", {}).keys())

        # Cross-hive callback — set by HiveRouter after construction
        self._cross_hive_callback: Optional[
            Callable[[str, str], Dict[str, Any]]
        ] = None

        # Create the domain-local BrainAgent
        self.brain = BrainAgent(self._merged_cfg)
        self.brain.set_hive_context(name, self.display_name, peer_names)

        log.info(
            "Hive '%s' (%s) initialized — agents: %s, topics: %d",
            self.name,
            self.display_name,
            list(self.brain._agents.keys()),
            len(self.topics),
        )

    # ── Config merging ──────────────────────────────────────────────

    @staticmethod
    def _build_merged_config(
        global_cfg: Dict[str, Any], hive_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep-merge hive-level overrides into a copy of the global config.

        Hive-level keys that are merged:
          - ``paths`` → overrides ``global.paths``
          - ``vectorstore`` → overrides ``global.vectorstore``
          - ``code_index`` → overrides ``global.code_index``
          - ``agents`` → overrides ``global.agents``
          - ``kusto`` → overrides ``global.kusto`` (if present)
        """
        merged = copy.deepcopy(global_cfg)

        # Override paths
        if "paths" in hive_cfg:
            merged.setdefault("paths", {})
            merged["paths"].update(hive_cfg["paths"])

        # Override vectorstore
        if "vectorstore" in hive_cfg:
            merged.setdefault("vectorstore", {})
            merged["vectorstore"].update(hive_cfg["vectorstore"])

        # Override code_index
        if "code_index" in hive_cfg:
            merged.setdefault("code_index", {})
            merged["code_index"].update(hive_cfg["code_index"])

        # Override agents
        if "agents" in hive_cfg:
            merged.setdefault("agents", {})
            merged["agents"].update(hive_cfg["agents"])

        # Override kusto
        if "kusto" in hive_cfg:
            merged.setdefault("kusto", {})
            merged["kusto"].update(hive_cfg["kusto"])

        return merged

    # ── Chat interface ──────────────────────────────────────────────

    def set_cross_hive_callback(
        self,
        callback: Callable[[str, str], Dict[str, Any]],
    ):
        """Register a callback for resolving [ASK:<hive>] signals.

        Parameters
        ----------
        callback : callable(target_hive: str, question: str) -> dict
            Must return ``{"response": "...", "hive": "...", ...}``.
        """
        self._cross_hive_callback = callback

    def set_boundary_config(
        self,
        boundary_map: Dict[str, str],
        boundary_callback: Callable[[str, str], Dict[str, Any]],
    ):
        """Wire code-boundary detection into this hive's coder agent.

        Parameters
        ----------
        boundary_map : dict
            Maps boundary patterns to hive names.
        boundary_callback : callable
            Called when coder detects a cross-hive code reference.
        """
        self.brain.set_boundary_config(boundary_map, boundary_callback)

    def chat(self, message: str) -> Dict[str, Any]:
        """Send a message to this hive's BrainAgent.

        If the response contains ``[ASK:<hive>] <question>`` signals,
        they are resolved via the cross-hive callback and the answers
        are re-injected so the agent can produce a single cohesive
        response.

        Returns the standard ``{agent, response}`` dict, augmented with
        ``hive`` name and ``cross_hive_asks`` metadata.
        """
        result = self.brain.chat(message)
        result["hive"] = self.name

        # ── [ASK:] resolution loop ──────────────────────────────────
        cross_hive_asks: List[Dict[str, str]] = []
        for round_num in range(MAX_ASK_ROUNDS):
            asks = self.extract_asks(result.get("response", ""))
            if not asks:
                break  # No more [ASK:] signals — done

            if self._cross_hive_callback is None:
                log.warning(
                    "Hive '%s' got [ASK:] signal but no callback registered.",
                    self.name,
                )
                break

            log.info(
                "Hive '%s' round %d: resolving %d [ASK:] signal(s)",
                self.name, round_num + 1, len(asks),
            )

            # Resolve each [ASK:] and collect answers
            answers: List[str] = []
            for ask in asks:
                target = ask["target_hive"]
                question = ask["question"]
                log.info("  [ASK:%s] %s", target, question[:80])

                try:
                    sub_result = self._cross_hive_callback(target, question)
                    answer_text = sub_result.get("response", "No answer.")
                    answers.append(
                        f"--- Answer from {target.upper()} ---\n{answer_text}"
                    )
                    cross_hive_asks.append({
                        "target": target,
                        "question": question,
                        "answer_preview": answer_text[:200],
                    })
                except Exception as e:
                    log.error("[ASK:%s] failed: %s", target, e)
                    answers.append(
                        f"--- {target.upper()}: could not be reached ({e}) ---"
                    )

            # Strip [ASK:] signals from the original response
            clean_response = self.ASK_PATTERN.sub("", result["response"]).strip()

            # Re-feed: give the agent its own response + the answers
            followup = (
                f"You previously answered:\n{clean_response}\n\n"
                f"Cross-domain answers you requested:\n"
                + "\n\n".join(answers)
                + "\n\nNow write a SINGLE, complete, cohesive response that "
                f"incorporates ALL of the above. Do NOT emit any more "
                f"[ASK:] or [DELEGATE:] signals."
            )
            result = self.brain.chat(followup)
            result["hive"] = self.name

        if cross_hive_asks:
            result["cross_hive_asks"] = cross_hive_asks

        return result

    def reset(self):
        """Clear conversation state."""
        self.brain.reset_conversation()

    # ── Delegation detection ────────────────────────────────────────

    @staticmethod
    def extract_asks(response_text: str) -> List[Dict[str, str]]:
        """Parse all ``[ASK:<hive>] <question>`` signals from a response.

        Returns a list of ``{"target_hive": "...", "question": "..."}`` dicts.
        """
        results = []
        for m in Hive.ASK_PATTERN.finditer(response_text):
            target = m.group(1).strip().lower()
            question = m.group(2).strip()
            if question:  # skip empty asks
                results.append({"target_hive": target, "question": question})
        return results

    @staticmethod
    def extract_delegation(response_text: str) -> Optional[Dict[str, str]]:
        """Parse a ``[DELEGATE:<hive>] <context>`` signal from a response.

        Returns ``{"target_hive": "...", "context": "..."}`` or None.
        """
        idx = response_text.find(Hive.DELEGATE_PREFIX)
        if idx == -1:
            return None
        end = response_text.find(Hive.DELEGATE_SUFFIX, idx + len(Hive.DELEGATE_PREFIX))
        if end == -1:
            return None
        target = response_text[idx + len(Hive.DELEGATE_PREFIX): end].strip().lower()
        context = response_text[end + 1:].strip()
        return {"target_hive": target, "context": context}

    # ── Scope matching ──────────────────────────────────────────────

    def _load_topics(self, config_topics: List[str]) -> List[str]:
        """Merge config.yaml topics with auto-extracted topics from discovery store.

        Priority: config topics (always kept) + discovery store auto-topics.
        Duplicates are removed, config topics come first.
        """
        try:
            from brain_ai.hive.discovery_store import DiscoveryStore
            ds = DiscoveryStore()
            auto_topics = ds.get_topics(self.name, source="auto")
            ds.close()
        except Exception:
            auto_topics = []

        # Merge: config first (as base), then auto-extracted (deduplicated)
        seen = {t.lower() for t in config_topics}
        merged = list(config_topics)
        for t in auto_topics:
            if t.lower() not in seen:
                merged.append(t)
                seen.add(t.lower())

        if auto_topics:
            log.info(
                "Hive '%s' topics: %d from config + %d auto-extracted = %d total",
                self.name, len(config_topics), len(auto_topics), len(merged),
            )

        return merged

    def matches_topic(self, query: str) -> float:
        """Return a simple keyword overlap score (0.0–1.0) for quick hive matching.

        This is a fast heuristic — the HiveRouter also uses the LLM for
        accurate classification when needed.
        """
        query_lower = query.lower()
        if not self.topics:
            return 0.0
        hits = sum(1 for t in self.topics if t.lower() in query_lower)
        return hits / len(self.topics)

    # ── Repr ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Hive(name={self.name!r}, display_name={self.display_name!r}, "
            f"agents={list(self.brain._agents.keys())})"
        )
