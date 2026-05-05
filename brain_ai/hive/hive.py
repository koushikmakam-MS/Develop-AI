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
from typing import Any, Dict, List, Optional

from brain_ai.agents.brain_agent import BrainAgent

log = logging.getLogger(__name__)


class Hive:
    """A self-contained domain agent cluster."""

    # ── Cross-hive delegation signal ────────────────────────────────
    # Agents embed this in their response when they detect the answer
    # requires another domain.  The HiveRouter intercepts it.
    DELEGATE_PREFIX = "[DELEGATE:"
    DELEGATE_SUFFIX = "]"

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
        self.topics: List[str] = self.scope.get("topics", [])
        self.kusto_tables: List[str] = self.scope.get("kusto_tables", [])

        # Build a merged config: global config + hive overrides
        self._merged_cfg = self._build_merged_config(global_cfg, hive_cfg)

        # Collect peer hive names for delegation
        peer_names = list(global_cfg.get("hives", {}).get("definitions", {}).keys())

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

    def chat(self, message: str) -> Dict[str, Any]:
        """Send a message to this hive's BrainAgent.

        Returns the standard ``{agent, response}`` dict, augmented with
        ``hive`` name.
        """
        result = self.brain.chat(message)
        result["hive"] = self.name
        return result

    def reset(self):
        """Clear conversation state."""
        self.brain.reset_conversation()

    # ── Delegation detection ────────────────────────────────────────

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
